from fastapi import APIRouter, Depends, Request, Form, UploadFile, File, HTTPException, Path, Query, Body, Response as FastResponse
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func, exists, and_, not_, update, delete, desc
from sqlalchemy.orm import selectinload, noload
from sqlalchemy.orm.attributes import flag_modified
from typing import List, Any, Optional, Dict
import os, re, asyncio, mimetypes, json, random, uuid, shutil, secrets, pathlib, string, hmac, hashlib, time, logging
from datetime import datetime, timedelta, timezone, date
from pathlib import Path as FSPath
from sqlalchemy.exc import IntegrityError
from urllib.parse import urlencode
from .database import get_db, async_session_maker
from .models import Prompt, PromptMedia, Response, SupportingMedia, Invite, User, Tag, UserWeeklyPrompt, UserWeeklySkip, ResponseSegment, PromptSuggestion, UserProfile, ChapterMeta, prompt_tags, Person, RelationshipEdge, ResponsePerson, KinGroup, KinMembership, PersonShare, UserPrompt, WeeklyToken, WeeklyTokenStatus, WeeklyState, ResponseVersion, AdminEditLog, ResponseNotificationTarget
from .utils import require_authenticated_user, require_admin_user, slugify, slug_person, slug_role, slug_place, require_authenticated_html_user, get_current_user
from .schemas import PromptRead, ResponseSegmentRead, ReorderSegmentsRequest
from .transcription import transcribe_file, enrich_after_transcription
from .llm_client import polish_text, OllamaError, curate_prompts_for_user
from .users import current_active_user, get_jwt_strategy, cookie_transport, SECRET
from fastapi.templating import Jinja2Templates
from sqlalchemy.dialects.postgresql import insert as pg_insert
from .media_pipeline import MediaPipeline, UserBucketsStrategy
from app.services.scheduler import schedule_bulk_send, set_weekly_cron
from app.services.scheduler import schedule_bulk_send
from app.services.mailer import send_weekly_email
from app.services.notifications import notify_new_response
from app.services.scheduler import schedule_bulk_send
from app.services.assignment import tag_based_prompts, persist_suggestions
from passlib.hash import bcrypt
from pydantic import BaseModel
from app.services.auto_tag import suggest_tags_rule_based, suggest_tags_for_prompt, _is_prompt_like
from app.services.assignment_core import score_prompt
from app.services.assignment import _eligible, _user_role_like_slugs, ensure_weekly_prompt, skip_current_prompt as _skip, get_on_deck_candidates, rotate_to_next_unanswered as _rotate
from app.services.people import upsert_person_for_user, resolve_person
from app.services.auto_tag import WHITELIST, reload_whitelist
from sqlalchemy.dialects.postgresql import insert
from app.services.utils_weekly import get_or_refresh_active_token, _now, mark_opened, mark_clicked, mark_used, mark_completed_and_close, parse_token, expire_active_tokens
from app.services.chapter_compile import compile_chapter, chapter_status
from app.services.kinship import classify_kinship
from app.services.infer import infer_edges_for_person, commit_inferred_edges
from app.schemas import ChapterCompilationDTO, ChapterStatusDTO
from app.background import spawn
templates = Jinja2Templates(directory='templates')
router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
STATIC_DIR = FSPath(BASE_DIR) / 'static'
PIPELINE = MediaPipeline(static_root=STATIC_DIR, path_strategy=UserBucketsStrategy())
DATA_DIR = os.path.join(BASE_DIR, 'app', 'data')
from pathlib import Path as _PathAlias
TAG_WL_PATH = _PathAlias(DATA_DIR) / 'tag_whitelist.json'

def _join_code(n: int=8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return ''.join((secrets.choice(alphabet) for _ in range(n)))

def _to_uploads_rel_for_playable(file_path_under_uploads: str | None) -> str | None:
    """
    DB stores legacy paths relative to 'uploads/...'.
    The pipeline’s delete_artifacts expects paths relative to static root (i.e. 'uploads/...').
    Ensure we always return 'uploads/...'.
    """
    if not file_path_under_uploads:
        return None
    rel = file_path_under_uploads.strip().lstrip('/').replace('\\', '/')
    return rel if rel.startswith('uploads/') else f'uploads/{rel}'

def _slugify(s: str) -> str:
    s = (s or '').strip().lower()
    s = re.sub('[^a-z0-9:/\\-]+', '-', s)
    return re.sub('-+', '-', s).strip('-')

def _slugify_labelled(prefix: str, s: str) -> str:
    """
    Produce 'prefix:<slug>' and normalize
    """
    s = (s or '').strip()
    if not s:
        return ''
    if prefix == 'person':
        return slug_person(s)
    if prefix == 'role':
        return slug_role(s)
    if prefix == 'place':
        return slug_place(s)
    return f'{prefix}:{slugify(s)}'

def _user_bucket_name(u: User) -> str:
    try:
        base = (u.username or '').strip()
        return slugify(base) if base else str(u.id)
    except Exception:
        return str(getattr(u, 'id', 'user'))

def _text_for_tagging(resp) -> str:
    """
    Build a single text blob from the pieces we care about.
    """
    parts = []
    try:
        if getattr(resp, 'prompt', None) and getattr(resp.prompt, 'text', None):
            parts.append(resp.prompt.text)
    except Exception:
        pass
    if getattr(resp, 'title', None):
        parts.append(resp.title)
    if getattr(resp, 'response_text', None):
        parts.append(resp.response_text)
    if getattr(resp, 'transcription', None):
        parts.append(resp.transcription)
    return ' \n'.join((p for p in parts if p and str(p).strip()))

async def _get_or_create_tag(db: AsyncSession, name: str):
    nm = (name or '').strip()
    if not nm:
        return None
    slug = _slugify(nm)
    existing = (await db.execute(select(Tag).where(or_(Tag.slug == slug, func.lower(Tag.name) == name.strip().lower())))).scalar_one_or_none()
    if existing:
        return existing
    t = Tag(name=nm, slug=slug)
    db.add(t)
    try:
        await db.flush()
        return t
    except IntegrityError:
        await db.rollback()
        return (await db.execute(select(Tag).where(or_(Tag.slug == slug, func.lower(Tag.name) == nm.lower())))).scalar_one_or_none()

async def _max_order_index(db: AsyncSession, response_id: int) -> int:
    res = await db.execute(select(func.max(ResponseSegment.order_index)).where(ResponseSegment.response_id == response_id))
    m = res.scalar_one_or_none()
    return int(m or 0)

async def _ensure_response_owned(db: AsyncSession, response_id: int, user_id: int) -> Response:
    resp = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user_id))).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    return resp

async def _process_primary_async(response_id: int, tmp: FSPath, filename: str, content_type: str, user_slug: str):
    try:
        loop = asyncio.get_running_loop()
        art = await loop.run_in_executor(None, lambda: PIPELINE.process_upload(temp_path=tmp, logical='response', role='primary', user_slug_or_id=user_slug, prompt_id=None, response_id=response_id, media_id=None, original_filename=filename, content_type=content_type))
        async with async_session_maker() as s:
            resp = await s.get(Response, response_id)
            if not resp:
                return
            playable_rel = (art.playable_rel or '').lstrip('/').replace('\\', '/')
            resp.primary_media_url = playable_rel[len('uploads/'):] if playable_rel.startswith('uploads/') else playable_rel
            resp.primary_thumbnail_path = art.thumb_rel
            resp.primary_mime_type = art.mime_type
            resp.primary_duration_sec = int(art.duration_sec or 0)
            resp.primary_sample_rate = art.sample_rate
            resp.primary_channels = art.channels
            resp.primary_width = art.width
            resp.primary_height = art.height
            resp.primary_size_bytes = art.size_bytes
            resp.primary_codec_audio = art.codec_a
            resp.primary_codec_video = art.codec_v
            await s.commit()
        if playable_rel:
            spawn(transcribe_and_update(response_id, playable_rel, False), name='transcribe_prompt_preview')
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except:
            pass
from app.services.assignment import ensure_weekly_prompt, skip_current_prompt, build_pool_for_user, _eligible, _iso_year_week

@router.get('/people/graph', response_class=HTMLResponse)
async def people_graph_page(request: Request, user=Depends(require_authenticated_html_user), db: AsyncSession=Depends(get_db)):
    me_person_id = None
    try:
        rows = (await db.execute(select(Person.id, Person.meta, Person.display_name).where(Person.owner_user_id == user.id))).all()
        for pid, meta, dname in rows:
            if isinstance(meta, dict) and meta.get('connect_to_owner'):
                rh = str(meta.get('role_hint', '')).strip().lower()
                if rh in {'you', 'self', 'me'}:
                    me_person_id = pid
                    break
        if me_person_id is None:
            uname = (user.username or '').strip().lower()
            uemail = (user.email or '').strip().lower()
            for pid, meta, dname in rows:
                dn = (dname or '').strip().lower()
                if dn and (dn == uname or dn == uemail or dn == 'you'):
                    me_person_id = pid
                    break
    except Exception:
        me_person_id = None
    return templates.TemplateResponse('people_graph.html', {'request': request, 'user': user, 'me_person_id': me_person_id})

@router.get('/api/people/graph')
async def api_people_graph(user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    own_q = select(Person.id, Person.display_name).where(Person.owner_user_id == user.id)
    shared_exists = exists(select(PersonShare.id).join(KinMembership, KinMembership.group_id == PersonShare.group_id).where(PersonShare.person_id == Person.id).where(KinMembership.user_id == user.id))
    shared_q = select(Person.id, Person.display_name).where((Person.visibility == 'groups') & shared_exists)
    prows = (await db.execute(own_q.union(shared_q))).all()
    try:
        rows_all = (await db.execute(select(Person.id, Person.meta, Person.display_name).where(Person.owner_user_id == user.id))).all()
        me_pid = None
        me_name = None
        for pid, meta, dname in rows_all:
            if isinstance(meta, dict) and meta.get('connect_to_owner'):
                rh = str(meta.get('role_hint', '')).strip().lower()
                if rh in {'you', 'self', 'me'}:
                    me_pid = pid
                    me_name = dname
                    break
        if me_pid is None:
            uname = (user.username or '').strip().lower()
            uemail = (user.email or '').strip().lower()
            for pid, meta, dname in rows_all:
                dn = (dname or '').strip().lower()
                if dn and (dn == uname or dn == uemail or dn == 'you'):
                    me_pid = pid
                    me_name = dname
                    break
        if me_pid is not None and all((pid != me_pid for pid, _ in prows)):
            prows.append((me_pid, me_name or 'You'))
    except Exception:
        pass
    if not prows:
        try:
            prof = (await db.execute(select(UserProfile).where(UserProfile.user_id == user.id))).scalars().first()
            tw = (getattr(prof, 'tag_weights', None) or {}).get('tagWeights', {}) if prof else {}
            from app.services.people import upsert_person_for_user
            seeded = False
            for k in tw.keys() if isinstance(tw, dict) else []:
                if isinstance(k, str) and k.startswith('person:'):
                    label = k.split(':', 1)[1].replace('-', ' ').strip()
                    if label:
                        await upsert_person_for_user(db, owner_user_id=user.id, display_name=label)
                        seeded = True
            if seeded:
                await db.commit()
                prows = (await db.execute(own_q.union(shared_q))).all()
        except Exception:
            pass
    pmap = {rid: name for rid, name in prows}
    node_ids = set(pmap.keys())
    dead_map: dict[int, bool] = {}
    color_map: dict[int, str | None] = {}
    hidden_map: dict[int, bool] = {}
    inferred_map: dict[int, bool] = {}
    rolehint_map: dict[int, str | None] = {}
    connect_map: dict[int, bool] = {}
    if node_ids:
        rows = (await db.execute(select(Person.id, Person.death_year, Person.meta).where(Person.id.in_(node_ids)))).all()
        for pid, dy, meta in rows:
            dflag = bool(dy is not None)
            if isinstance(meta, dict) and bool(meta.get('deceased')):
                dflag = True
            dead_map[pid] = dflag
            try:
                if isinstance(meta, dict):
                    color_map[pid] = meta.get('dot_color')
                    hidden_map[pid] = bool(meta.get('hidden'))
                    inferred_map[pid] = bool(meta.get('inferred'))
                    rolehint_map[pid] = meta.get('role_hint') or None
                    connect_map[pid] = bool(meta.get('connect_to_owner'))
            except Exception:
                color_map[pid] = None
    mention_map: dict[int, int] = {}
    if node_ids:
        mrows = (await db.execute(select(ResponsePerson.person_id, func.count(ResponsePerson.id)).join(Response, Response.id == ResponsePerson.response_id).where(Response.user_id == user.id, ResponsePerson.person_id.in_(node_ids)).group_by(ResponsePerson.person_id))).all()
        mention_map = {pid: int(cnt) for pid, cnt in mrows or []}
    nodes = []
    own_ids_rows = (await db.execute(select(Person.id).where(Person.owner_user_id == user.id))).scalars().all()
    own_ids = set((int(x) for x in own_ids_rows or []))
    for rid, name in prows:
        if hidden_map.get(rid):
            continue
        if rid in own_ids and inferred_map.get(rid) and (mention_map.get(rid, 0) == 0) and (not (rolehint_map.get(rid) or connect_map.get(rid))):
            continue
        nodes.append({'id': rid, 'name': name, 'kind': 'person', 'dead': bool(dead_map.get(rid)), 'color': color_map.get(rid)})
    ers = (await db.execute(select(RelationshipEdge).where(RelationshipEdge.user_id == user.id))).scalars().all()
    edges = [{'src': e.src_id, 'dst': e.dst_id, 'rel': e.rel_type, 'confidence': getattr(e, 'confidence', None), 'generation': int((e.meta or {}).get('generation')) if isinstance(getattr(e, 'meta', None), dict) and (e.meta or {}).get('generation') is not None else None} for e in ers if e.src_id in node_ids and e.dst_id in node_ids]
    try:
        me_pid = None
        rows = (await db.execute(select(Person.id, Person.meta, Person.display_name).where(Person.owner_user_id == user.id))).all()
        for pid, meta, dname in rows:
            if isinstance(meta, dict) and meta.get('connect_to_owner'):
                rh = str(meta.get('role_hint', '')).strip().lower()
                if rh in {'you', 'self', 'me'}:
                    me_pid = pid
                    break
        if me_pid is None:
            uname = (user.username or '').strip().lower()
            uemail = (user.email or '').strip().lower()
            for pid, meta, dname in rows:
                dn = (dname or '').strip().lower()
                if dn and (dn == uname or dn == uemail or dn == 'you'):
                    me_pid = pid
                    break
        if me_pid is not None:
            owned_ids_only = [pid for pid, _m in rows] or []
            mention_map: dict[int, int] = {}
            if owned_ids_only:
                mrows = (await db.execute(select(ResponsePerson.person_id, func.count(ResponsePerson.id)).join(Response, Response.id == ResponsePerson.response_id).where(Response.user_id == user.id, ResponsePerson.person_id.in_(owned_ids_only)).group_by(ResponsePerson.person_id))).all()
                mention_map = {pid: int(cnt) for pid, cnt in mrows or []}

            async def _persist_edge_pair(f_src: int, f_dst: int, f_rel: str):
                f_rel = (f_rel or '').strip().lower()
                if not f_rel:
                    return
                stmt_fwd = pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=f_src, dst_id=f_dst, rel_type=f_rel, confidence=0.9).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type'])
                await db.execute(stmt_fwd)
                inv = {'mother-of': 'child-of', 'father-of': 'child-of', 'parent-of': 'child-of', 'child-of': 'parent-of', 'son-of': 'parent-of', 'daughter-of': 'parent-of', 'step-parent-of': 'child-of', 'adoptive-parent-of': 'child-of', 'grandparent-of': 'grandchild-of', 'grandchild-of': 'grandparent-of', 'sibling-of': 'sibling-of', 'half-sibling-of': 'half-sibling-of', 'step-sibling-of': 'step-sibling-of', 'aunt-of': 'niece-of', 'uncle-of': 'nephew-of', 'niece-of': 'aunt-of', 'nephew-of': 'uncle-of', 'cousin-of': 'cousin-of', 'spouse-of': 'spouse-of', 'partner-of': 'partner-of', 'ex-partner-of': 'ex-partner-of', 'friend-of': 'friend-of', 'neighbor-of': 'neighbor-of', 'coworker-of': 'coworker-of', 'mentor-of': 'student-of', 'teacher-of': 'student-of', 'student-of': 'teacher-of', 'coach-of': 'student-of'}.get(f_rel, f_rel)
                stmt_inv = pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=f_dst, dst_id=f_src, rel_type=inv, confidence=0.9).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type'])
                await db.execute(stmt_inv)
            for pid, meta in rows:
                if pid == me_pid:
                    continue
                role = 'you'
                confirmed = False
                if isinstance(meta, dict):
                    role = meta.get('role_hint') or 'you'
                    confirmed = bool(meta.get('connect_to_owner')) or bool(meta.get('role_hint'))
                if confirmed or mention_map.get(pid, 0) > 0:
                    if pid in node_ids and me_pid in node_ids:
                        r_inj = (role or '').strip().lower()
                        inj_rel = None
                        if r_inj in {'mother', 'father', 'parent'}:
                            inj_rel = 'child-of'
                        elif r_inj in {'son', 'daughter', 'child'}:
                            inj_rel = 'parent-of'
                        elif r_inj in {'spouse', 'partner', 'husband', 'wife'}:
                            inj_rel = 'spouse-of'
                        elif r_inj in {'sibling', 'brother', 'sister'}:
                            inj_rel = 'sibling-of'
                        elif r_inj in {'friend', 'neighbor', 'coworker', 'colleague'}:
                            inj_rel = f'{r_inj}-of'
                        if not inj_rel and mention_map.get(pid, 0) > 0:
                            inj_rel = 'friend-of'
                        if inj_rel:
                            edges.append({'src': me_pid, 'dst': pid, 'rel': inj_rel})
                        r = (role or '').strip().lower()
                        if r in {'you', 'self', 'me'}:
                            continue
                        if r in {'mother', 'father', 'parent'}:
                            await _persist_edge_pair(pid, me_pid, 'parent-of')
                        elif r in {'son', 'daughter', 'child'}:
                            await _persist_edge_pair(me_pid, pid, 'parent-of')
                        elif r in {'spouse', 'partner', 'husband', 'wife'}:
                            await _persist_edge_pair(me_pid, pid, 'spouse-of')
                        elif r in {'sibling', 'brother', 'sister'}:
                            await _persist_edge_pair(me_pid, pid, 'sibling-of')
                        elif r in {'friend', 'neighbor', 'coworker', 'colleague'}:
                            base = r if r.endswith('-of') else f'{r}-of'
                            await _persist_edge_pair(me_pid, pid, base)
                        elif not r and mention_map.get(pid, 0) > 0:
                            await _persist_edge_pair(me_pid, pid, 'friend-of')
            try:
                await db.commit()
            except Exception:
                await db.rollback()
    except Exception:
        pass
    return {'nodes': nodes, 'edges': edges}

@router.get('/api/roles')
async def api_roles_whitelist() -> dict:
    roles: list[dict] = []
    try:
        items = WHITELIST or []
        for it in items:
            if isinstance(it, str):
                if it.startswith('role:') or it.startswith('relationship:'):
                    base = it.split(':', 1)[1]
                    label = base.replace('-', ' ').title()
                    roles.append({'slug': f'role:{base}', 'label': label})
            else:
                slug = (it.get('value') or '').strip()
                if slug.startswith('role:') or slug.startswith('relationship:'):
                    base = slug.split(':', 1)[1]
                    label = it.get('label') or base.replace('-', ' ').title()
                    roles.append({'slug': f'role:{base}', 'label': label})
    except Exception:
        pass
    seen = set()
    out = []
    for r in roles:
        if r['slug'] in seen:
            continue
        seen.add(r['slug'])
        out.append(r if r.get('label') else {'slug': r['slug'], 'label': r['slug'].split(':', 1)[1].replace('-', ' ').title()})
    out.sort(key=lambda r: r['label'].lower())
    return {'roles': out}

@router.post('/api/roles/reload')
async def api_roles_reload():
    try:
        n = reload_whitelist()
        items = list(WHITELIST)
        count_roles = len([x for x in items if isinstance(x, str) and (x.startswith('role:') or x.startswith('relationship:'))])
        return {'ok': True, 'loaded': n, 'role_like': count_roles}
    except Exception:
        return {'ok': False}

class PersonPatchReq(BaseModel):
    display_name: Optional[str] = None
    role_hint: Optional[str] = None
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    birth_date: Optional[str] = None
    bio: Optional[str] = None
    dot_color: Optional[str] = None
    deceased: Optional[bool] = None
    connect_to_owner: Optional[bool] = None
    hidden: Optional[bool] = None
    inferred: Optional[bool] = None

@router.get('/api/people/{person_id}')
async def api_people_detail(person_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p: Person | None = await db.get(Person, person_id)
    if not p:
        raise HTTPException(404, 'Person not found')
    is_owner = p.owner_user_id == user.id
    is_shared = False
    if not is_owner:
        is_shared = bool(await db.scalar(select(PersonShare.id).join(KinMembership, KinMembership.group_id == PersonShare.group_id).where(PersonShare.person_id == person_id).where(KinMembership.user_id == user.id).limit(1)))
        if not is_shared:
            raise HTTPException(404, 'Person not found')
    aliases = (await db.execute(select(ResponsePerson.alias_used).where(ResponsePerson.person_id == person_id))).scalars().all()
    edges_out = (await db.execute(select(RelationshipEdge).where(RelationshipEdge.src_id == person_id))).scalars().all()
    edges_in = (await db.execute(select(RelationshipEdge).where(RelationshipEdge.dst_id == person_id))).scalars().all()
    edges = list(edges_out) + list(edges_in)
    neighbor_ids = set()
    for e in edges:
        neighbor_ids.add(e.src_id if e.dst_id == person_id else e.dst_id)
    neighbors = {}
    if neighbor_ids:
        rows = (await db.execute(select(Person.id, Person.display_name).where(Person.id.in_(neighbor_ids)))).all()
        neighbors = {pid: name for pid, name in rows}
    meta_val = getattr(p, 'meta', None)
    role_hint = (meta_val or {}).get('role_hint') if isinstance(meta_val, dict) else None
    dot_color = (meta_val or {}).get('dot_color') if isinstance(meta_val, dict) else None
    deceased = bool((meta_val or {}).get('deceased')) if isinstance(meta_val, dict) else False
    connect_to_owner = bool((meta_val or {}).get('connect_to_owner')) if isinstance(meta_val, dict) else False
    hidden = bool((meta_val or {}).get('hidden')) if isinstance(meta_val, dict) else False
    inferred = bool((meta_val or {}).get('inferred')) if isinstance(meta_val, dict) else False
    photo_rel = getattr(p, 'photo_url', None)
    if photo_rel:
        rel = (photo_rel or '').strip().lstrip('/').replace('\\', '/')
        if not rel.startswith('uploads/'):
            rel = f'uploads/{rel}'
        photo_abs = f'/static/{rel}'
    else:
        photo_abs = None

    def _canon_rel(rt: str) -> str:
        r = (rt or '').strip().lower()
        mapping = {'mother-of': 'parent-of', 'father-of': 'parent-of', 'son-of': 'child-of', 'daughter-of': 'child-of'}
        return mapping.get(r, r)

    def _invert(rt: str) -> str:
        rt = (rt or '').strip().lower()
        mapping = {'mother-of': 'child-of', 'father-of': 'child-of', 'parent-of': 'child-of', 'step-parent-of': 'child-of', 'adoptive-parent-of': 'child-of', 'child-of': 'parent-of', 'son-of': 'parent-of', 'daughter-of': 'parent-of', 'grandparent-of': 'grandchild-of', 'grandchild-of': 'grandparent-of', 'sibling-of': 'sibling-of', 'half-sibling-of': 'half-sibling-of', 'step-sibling-of': 'step-sibling-of', 'aunt-of': 'niece-of', 'uncle-of': 'nephew-of', 'niece-of': 'aunt-of', 'nephew-of': 'uncle-of', 'cousin-of': 'cousin-of', 'aunt-in-law-of': 'niece-in-law-of', 'niece-in-law-of': 'aunt-in-law-of', 'cousin-in-law-of': 'cousin-in-law-of', 'spouse-of': 'spouse-of', 'partner-of': 'partner-of', 'ex-partner-of': 'ex-partner-of', 'friend-of': 'friend-of', 'mentor-of': 'student-of', 'student-of': 'teacher-of', 'teacher-of': 'student-of', 'coworker-of': 'coworker-of', 'neighbor-of': 'neighbor-of', 'coach-of': 'student-of'}
        return mapping.get(rt, rt)
    conns: list[dict] = []

    def _gen_from_meta(e) -> int | None:
        try:
            m = getattr(e, 'meta', None) or {}
            if isinstance(m, dict):
                g = m.get('generation')
                if g is None:
                    return None
                return int(g)
        except Exception:
            return None
        return None
    for e in edges_out:
        conns.append({'edge_id': e.id, 'person_id': e.dst_id, 'name': neighbors.get(e.dst_id, 'Unknown'), 'rel_type': e.rel_type, 'direction': 'out', 'generation': _gen_from_meta(e)})
    for e in edges_in:
        inv = _invert(e.rel_type)
        conns.append({'edge_id': e.id, 'person_id': e.src_id, 'name': neighbors.get(e.src_id, 'Unknown'), 'rel_type': inv, 'direction': 'out', 'generation': _gen_from_meta(e)})
    seen_keys = set()
    deduped: list[dict] = []
    for c in conns:
        key = (int(c.get('person_id')), _canon_rel(c.get('rel_type', '')))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(c)
    return {'id': p.id, 'display_name': p.display_name, 'photo_url': photo_abs, 'role_hint': role_hint, 'dot_color': dot_color, 'birth_year': getattr(p, 'birth_year', None), 'death_year': getattr(p, 'death_year', None), 'deceased': deceased or getattr(p, 'death_year', None) is not None, 'birth_date': (meta_val or {}).get('birth_date') if isinstance(meta_val, dict) else None, 'connect_to_owner': connect_to_owner, 'hidden': hidden, 'inferred': inferred, 'bio': getattr(p, 'notes', None), 'aliases': sorted({a for a in aliases or [] if a}), 'editable': bool(is_owner), 'connections': deduped}

class InferredEdgeDTO(BaseModel):
    src_id: int
    dst_id: int
    rel_type: str
    confidence: float
    source: str
    explain: str | None = None

@router.get('/api/people/{person_id}/infer/preview')
async def api_people_infer_preview(person_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p = await db.get(Person, person_id)
    if not p or p.owner_user_id != user.id:
        raise HTTPException(404, 'Person not found')
    cands = await infer_edges_for_person(db, user_id=user.id, person_id=person_id)
    out = []
    try:
        from app.services.kinship import classify_kinship
    except Exception:
        classify_kinship = None
    for c in cands:
        label = None
        if classify_kinship is not None:
            try:
                res = await classify_kinship(db, user_id=user.id, ego_id=c.src_id, alter_id=c.dst_id)
                if res and res.label_neutral and (str(res.label_neutral).lower() != 'related'):
                    lname = res.label_neutral
                    if res.label_gendered and (not ('aunt/uncle' in lname or 'niece/nephew' in lname)):
                        label = res.label_gendered
                    else:
                        label = lname
            except Exception:
                label = None
        out.append({'src_id': c.src_id, 'dst_id': c.dst_id, 'rel_type': c.rel_type, 'confidence': c.confidence, 'source': c.source, 'explain': c.explain, 'label': label})
    return {'person_id': person_id, 'count': len(out), 'edges': out}

@router.post('/api/people/{person_id}/infer/commit')
async def api_people_infer_commit(person_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p = await db.get(Person, person_id)
    if not p or p.owner_user_id != user.id:
        raise HTTPException(404, 'Person not found')
    cands = await infer_edges_for_person(db, user_id=user.id, person_id=person_id)
    n = await commit_inferred_edges(db, user_id=user.id, candidates=cands)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise HTTPException(500, 'Failed to persist inferred edges')
    return {'person_id': person_id, 'attempted': len(cands), 'inserted': n}

@router.get('/api/people/kinship')
async def api_people_kinship(ego_id: str=Query(...), alter_id: str=Query(...), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    """Return a precise kinship label between two people for the current user.

    Response shape:
      {
        label_neutral, label_gendered, cousin_degree, removed,
        ancestor_steps, descendant_steps, is_half, is_step, is_adoptive, mrca_id
      }
    """
    try:
        e = int(str(ego_id).strip())
        a = int(str(alter_id).strip())
        if e < 1 or a < 1:
            raise ValueError('ids must be >= 1')
    except Exception:
        raise HTTPException(400, 'Invalid person ids')
    res = await classify_kinship(db, user_id=user.id, ego_id=e, alter_id=a)
    return {'ego_id': res.ego_id, 'alter_id': res.alter_id, 'label_neutral': res.label_neutral, 'label_gendered': res.label_gendered, 'cousin_degree': res.cousin_degree, 'removed': res.removed, 'ancestor_steps': res.ancestor_steps, 'descendant_steps': res.descendant_steps, 'is_half': res.is_half, 'is_step': res.is_step, 'is_adoptive': res.is_adoptive, 'mrca_id': res.mrca_id}

@router.get('/api/people/kinship_to_me')
async def api_people_kinship_to_me(person_id: str=Query(...), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    """
    Best-effort kinship label between the current user ("You") and the given person.
    Strategy:
      1) If a Person representing the user exists (meta.connect_to_owner and role_hint in {you,self,me}),
         use full kinship classifier between that Person and target person.
      2) Else, if the target person has a direct role_hint to owner (mother/father/parent, son/daughter/child, spouse/partner),
         return the corresponding simple label.
      3) Else return {label_neutral: null}.
    """
    try:
        pid = int(str(person_id).strip())
        if pid < 1:
            raise ValueError
    except Exception:
        return {'label_neutral': None}
    rows = (await db.execute(select(Person).where(Person.owner_user_id == user.id))).scalars().all()
    self_candidates: list[Person] = []
    for p in rows:
        m = getattr(p, 'meta', None)
        if not isinstance(m, dict):
            continue
        if m.get('connect_to_owner') and str(m.get('role_hint', '')).strip().lower() in {'you', 'self', 'me'}:
            self_candidates.append(p)
    if self_candidates:
        ego = self_candidates[0]
        try:
            res = await classify_kinship(db, user_id=user.id, ego_id=ego.id, alter_id=pid)
            return {'label_neutral': res.label_gendered or res.label_neutral}
        except Exception:
            pass
    anchors = []
    for p in rows:
        m = getattr(p, 'meta', None)
        if not isinstance(m, dict):
            continue
        if not (m.get('connect_to_owner') or m.get('role_hint')):
            continue
        base = str(m.get('role_hint', '')).strip().lower()
        if base in {'mother', 'father', 'parent'}:
            anchors.append(('parent', p))
        elif base in {'spouse', 'partner', 'husband', 'wife'}:
            anchors.append(('spouse', p))
        elif base in {'brother', 'sister', 'sibling'}:
            anchors.append(('sibling', p))

    def _plus_great(term: str) -> str:
        if term.endswith('grandparent'):
            return 'great-' + term
        return term
    for kind, anchor in sorted(anchors, key=lambda t: {'parent': 0, 'spouse': 1, 'sibling': 2}.get(t[0], 9)):
        try:
            b = await classify_kinship(db, user_id=user.id, ego_id=anchor.id, alter_id=int(person_id))
        except Exception:
            continue
        L = (b.label_neutral or '').lower()
        if kind == 'parent':
            if L == 'parent':
                return {'label_neutral': 'grandparent'}
            if L.endswith('grandparent'):
                return {'label_neutral': _plus_great(b.label_neutral)}
            if L == 'child':
                return {'label_neutral': 'sibling'}
            if L.endswith('grandchild'):
                if L == 'grandchild':
                    return {'label_neutral': 'niece/nephew'}
                return {'label_neutral': b.label_neutral.replace('grandchild', 'grandniece/nephew')}
            if L == 'sibling':
                return {'label_neutral': 'aunt/uncle'}
            if 'grandaunt/uncle' in L:
                return {'label_neutral': 'great-' + b.label_neutral}
            if 'aunt/uncle' in L:
                return {'label_neutral': 'grandaunt/uncle'}
            if 'niece/nephew' in L:
                removed = 0
                if L.startswith('great-'):
                    removed = L.count('great-')
                base = 'first cousin'
                if removed == 0:
                    return {'label_neutral': base}
                elif removed == 1:
                    return {'label_neutral': base + ' once removed'}
                else:
                    return {'label_neutral': base + f' {removed} times removed'}
            if 'cousin' in L:
                try:
                    deg_word = L.split(' cousin', 1)[0].split()[-1]
                    removed = 0
                    if 'removed' in L:
                        if 'once' in L:
                            removed = 1
                        elif 'twice' in L:
                            removed = 2
                        else:
                            try:
                                removed = int(L.split(' removed')[0].split()[-1])
                            except Exception:
                                removed = 0
                    new_removed = removed + 1
                    tail = ' once removed' if new_removed == 1 else ' twice removed' if new_removed == 2 else f' {new_removed} times removed'
                    return {'label_neutral': f'{deg_word} cousin{tail}'}
                except Exception:
                    return {'label_neutral': 'cousin once removed'}
        elif kind == 'spouse':
            if L:
                return {'label_neutral': b.label_neutral + '-in-law'}
        elif kind == 'sibling':
            if L == 'child':
                return {'label_neutral': 'niece/nephew'}
            if L.endswith('grandchild'):
                if L == 'grandchild':
                    return {'label_neutral': 'grandniece/nephew'}
                return {'label_neutral': b.label_neutral.replace('grandchild', 'grandniece/nephew')}
            if L in {'spouse', 'partner'}:
                return {'label_neutral': 'sibling-in-law'}
    p: Person | None = await db.get(Person, pid)
    if not p or p.owner_user_id != user.id:
        raise HTTPException(404, 'Person not found')
    role = None
    m = getattr(p, 'meta', None)
    if isinstance(m, dict):
        if m.get('connect_to_owner') or m.get('role_hint'):
            base = str(m.get('role_hint', '')).strip().lower()
            if base in {'mother', 'father', 'parent'}:
                role = 'parent'
            elif base in {'son', 'daughter', 'child'}:
                role = 'child'
            elif base in {'spouse', 'partner', 'husband', 'wife'}:
                role = 'spouse'
            elif base:
                role = base
    return {'label_neutral': role}

@router.get('/api/people/kinship2/{ego_id}/{alter_id}')
async def api_people_kinship_path(ego_id: str, alter_id: str, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    try:
        e = int(str(ego_id).strip())
        a = int(str(alter_id).strip())
        if e < 1 or a < 1:
            raise ValueError
    except Exception:
        return JSONResponse({'error': 'invalid person ids'}, status_code=200)
    res = await classify_kinship(db, user_id=user.id, ego_id=e, alter_id=a)
    return {'ego_id': res.ego_id, 'alter_id': res.alter_id, 'label_neutral': res.label_neutral, 'label_gendered': res.label_gendered, 'cousin_degree': res.cousin_degree, 'removed': res.removed, 'ancestor_steps': res.ancestor_steps, 'descendant_steps': res.descendant_steps, 'is_half': res.is_half, 'is_step': res.is_step, 'is_adoptive': res.is_adoptive, 'mrca_id': res.mrca_id}

@router.get('/api/people/you/{person_id}')
async def api_people_kinship_to_me_path(person_id: str, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    try:
        pid = int(str(person_id).strip())
        if pid < 1:
            raise ValueError
    except Exception:
        return {'label_neutral': None}
    rows = (await db.execute(select(Person).where(Person.owner_user_id == user.id))).scalars().all()
    self_candidates: list[Person] = []
    for p in rows:
        m = getattr(p, 'meta', None)
        if isinstance(m, dict) and m.get('connect_to_owner') and (str(m.get('role_hint', '')).strip().lower() in {'you', 'self', 'me'}):
            self_candidates.append(p)
    if self_candidates:
        try:
            res = await classify_kinship(db, user_id=user.id, ego_id=self_candidates[0].id, alter_id=pid)
            return {'label_neutral': res.label_gendered or res.label_neutral}
        except Exception:
            pass
    anchors = []
    for p in rows:
        m = getattr(p, 'meta', None)
        if not isinstance(m, dict):
            continue
        if not (m.get('connect_to_owner') or m.get('role_hint')):
            continue
        base = str(m.get('role_hint', '')).strip().lower()
        if base in {'mother', 'father', 'parent'}:
            anchors.append(('parent', p))
        elif base in {'spouse', 'partner', 'husband', 'wife'}:
            anchors.append(('spouse', p))
        elif base in {'brother', 'sister', 'sibling'}:
            anchors.append(('sibling', p))

    def _plus_great(term: str) -> str:
        if term.endswith('grandparent'):
            return 'great-' + term
        return term
    for kind, anchor in sorted(anchors, key=lambda t: {'parent': 0, 'spouse': 1, 'sibling': 2}.get(t[0], 9)):
        try:
            b = await classify_kinship(db, user_id=user.id, ego_id=anchor.id, alter_id=pid)
        except Exception:
            continue
        L = (b.label_neutral or '').lower()
        if kind == 'parent':
            if L == 'parent':
                return {'label_neutral': 'grandparent'}
            if L.endswith('grandparent'):
                return {'label_neutral': _plus_great(b.label_neutral)}
            if L == 'child':
                return {'label_neutral': 'sibling'}
            if L.endswith('grandchild'):
                if L == 'grandchild':
                    return {'label_neutral': 'niece/nephew'}
                return {'label_neutral': b.label_neutral.replace('grandchild', 'grandniece/nephew')}
            if L == 'sibling':
                return {'label_neutral': 'aunt/uncle'}
            if 'grandaunt/uncle' in L:
                return {'label_neutral': 'great-' + b.label_neutral}
            if 'aunt/uncle' in L:
                return {'label_neutral': 'grandaunt/uncle'}
            if 'niece/nephew' in L:
                removed = 0
                if L.startswith('great-'):
                    removed = L.count('great-')
                base = 'first cousin'
                if removed == 0:
                    return {'label_neutral': base}
                elif removed == 1:
                    return {'label_neutral': base + ' once removed'}
                else:
                    return {'label_neutral': base + f' {removed} times removed'}
            if 'cousin' in L:
                try:
                    removed = 0
                    if 'removed' in L:
                        if 'once' in L:
                            removed = 1
                        elif 'twice' in L:
                            removed = 2
                        else:
                            try:
                                removed = int(L.split(' removed')[0].split()[-1])
                            except Exception:
                                removed = 0
                    new_removed = removed + 1
                    tail = ' once removed' if new_removed == 1 else ' twice removed' if new_removed == 2 else f' {new_removed} times removed'
                    return {'label_neutral': 'cousin' + tail}
                except Exception:
                    return {'label_neutral': 'cousin once removed'}
        elif kind == 'spouse':
            if L:
                return {'label_neutral': b.label_neutral + '-in-law'}
        elif kind == 'sibling':
            if L == 'child':
                return {'label_neutral': 'niece/nephew'}
            if L.endswith('grandchild'):
                if L == 'grandchild':
                    return {'label_neutral': 'grandniece/nephew'}
                return {'label_neutral': b.label_neutral.replace('grandchild', 'grandniece/nephew')}
            if L in {'spouse', 'partner'}:
                return {'label_neutral': 'sibling-in-law'}
    tp: Person | None = await db.get(Person, pid)
    if tp and tp.owner_user_id == user.id:
        m = getattr(tp, 'meta', None)
        if isinstance(m, dict) and (m.get('connect_to_owner') or m.get('role_hint')):
            base = str(m.get('role_hint', '')).strip().lower()
            if base in {'mother', 'father', 'parent'}:
                return {'label_neutral': 'parent'}
            if base in {'son', 'daughter', 'child'}:
                return {'label_neutral': 'child'}
            if base in {'spouse', 'partner', 'husband', 'wife'}:
                return {'label_neutral': 'spouse'}
            if base:
                return {'label_neutral': base}
    return {'label_neutral': None}

@router.patch('/api/people/{person_id}')
async def api_people_patch(person_id: int, payload: PersonPatchReq, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p: Person | None = await db.get(Person, person_id)
    if not p or p.owner_user_id != user.id:
        raise HTTPException(404, 'Person not found')
    changed = False
    if payload.display_name is not None:
        nm = (payload.display_name or '').strip()
        if nm and nm != p.display_name:
            p.display_name = nm
            changed = True
    if payload.role_hint is not None:
        m = getattr(p, 'meta', None) or {}
        if not isinstance(m, dict):
            m = {}
        m['role_hint'] = (payload.role_hint or '').strip()
        p.meta = m
        try:
            flag_modified(p, 'meta')
        except Exception:
            pass
        changed = True
    if payload.dot_color is not None:
        m = getattr(p, 'meta', None) or {}
        if not isinstance(m, dict):
            m = {}
        clr = (payload.dot_color or '').strip()
        if clr:
            m['dot_color'] = clr
        else:
            try:
                m.pop('dot_color', None)
            except Exception:
                pass
        p.meta = m
        try:
            flag_modified(p, 'meta')
        except Exception:
            pass
        changed = True
    if payload.deceased is not None:
        m = getattr(p, 'meta', None) or {}
        if not isinstance(m, dict):
            m = {}
        m['deceased'] = bool(payload.deceased)
        p.meta = m
        try:
            flag_modified(p, 'meta')
        except Exception:
            pass
        changed = True
    if payload.connect_to_owner is not None:
        m = getattr(p, 'meta', None) or {}
        if not isinstance(m, dict):
            m = {}
        m['connect_to_owner'] = bool(payload.connect_to_owner)
        p.meta = m
        try:
            flag_modified(p, 'meta')
        except Exception:
            pass
        changed = True
    if payload.hidden is not None:
        m = getattr(p, 'meta', None) or {}
        if not isinstance(m, dict):
            m = {}
        m['hidden'] = bool(payload.hidden)
        p.meta = m
        try:
            flag_modified(p, 'meta')
        except Exception:
            pass
        changed = True
    if payload.inferred is not None:
        m = getattr(p, 'meta', None) or {}
        if not isinstance(m, dict):
            m = {}
        m['inferred'] = bool(payload.inferred)
        p.meta = m
        try:
            flag_modified(p, 'meta')
        except Exception:
            pass
        changed = True
    if payload.birth_year is not None:
        try:
            p.birth_year = int(payload.birth_year) if payload.birth_year else None
            changed = True
        except Exception:
            pass
    if payload.death_year is not None:
        try:
            p.death_year = int(payload.death_year) if payload.death_year else None
            changed = True
        except Exception:
            pass
    if payload.bio is not None:
        p.notes = (payload.bio or '').strip()
        changed = True
    if payload.birth_date is not None:
        m = getattr(p, 'meta', None) or {}
        if not isinstance(m, dict):
            m = {}
        bd = (payload.birth_date or '').strip()
        if bd:
            m['birth_date'] = bd
        else:
            try:
                m.pop('birth_date', None)
            except Exception:
                pass
        p.meta = m
        try:
            flag_modified(p, 'meta')
        except Exception:
            pass
        changed = True
    if changed:
        await db.commit()
    return {'ok': True}

@router.delete('/api/people/edges/{edge_id}')
async def api_people_edge_delete(edge_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    e = await db.get(RelationshipEdge, edge_id)
    if not e:
        raise HTTPException(404, 'Edge not found')
    src_p = await db.get(Person, e.src_id)
    dst_p = await db.get(Person, e.dst_id)
    if not src_p or not dst_p:
        raise HTTPException(404, 'Edge endpoints not found')
    if src_p.owner_user_id != user.id and dst_p.owner_user_id != user.id:
        raise HTTPException(403, 'Not allowed to edit this connection')
    await db.delete(e)

    def _inverse_rel(rt: str) -> str:
        rt = (rt or '').strip().lower()
        mapping = {'mother-of': 'child-of', 'father-of': 'child-of', 'parent-of': 'child-of', 'child-of': 'parent-of', 'son-of': 'parent-of', 'daughter-of': 'parent-of', 'grandparent-of': 'grandchild-of', 'grandchild-of': 'grandparent-of', 'sibling-of': 'sibling-of', 'half-sibling-of': 'half-sibling-of', 'step-sibling-of': 'step-sibling-of', 'aunt-of': 'niece-of', 'uncle-of': 'nephew-of', 'niece-of': 'aunt-of', 'nephew-of': 'uncle-of', 'cousin-of': 'cousin-of', 'mother-in-law-of': 'parent-in-law-of', 'father-in-law-of': 'parent-in-law-of', 'parent-in-law-of': 'parent-in-law-of', 'sister-in-law-of': 'sister-in-law-of', 'brother-in-law-of': 'brother-in-law-of', 'spouse-of': 'spouse-of', 'partner-of': 'partner-of', 'ex-partner-of': 'ex-partner-of', 'friend-of': 'friend-of', 'mentor-of': 'student-of', 'student-of': 'teacher-of', 'teacher-of': 'student-of', 'coworker-of': 'coworker-of', 'neighbor-of': 'neighbor-of', 'coach-of': 'student-of'}
        return mapping.get(rt, rt)
    try:
        inv_rel = _inverse_rel(e.rel_type)
        inv_edge = (await db.execute(select(RelationshipEdge).where(RelationshipEdge.src_id == e.dst_id, RelationshipEdge.dst_id == e.src_id, RelationshipEdge.rel_type == inv_rel))).scalars().first()
        if inv_edge:
            await db.delete(inv_edge)
    except Exception:
        pass
    await db.commit()
    return {'ok': True}

@router.get('/api/people/inferred/list')
async def api_people_inferred(user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    rows = (await db.execute(select(Person))).scalars().all()
    mine = [p for p in rows if getattr(p, 'owner_user_id', None) == user.id]
    inferred = []
    pids = []
    for p in mine:
        meta = getattr(p, 'meta', {}) or {}
        if isinstance(meta, dict) and bool(meta.get('inferred')) and (not bool(meta.get('hidden'))):
            inferred.append(p)
            pids.append(p.id)
    counts: dict[int, int] = {}
    if pids:
        cnt_rows = (await db.execute(select(ResponsePerson.person_id, func.count(ResponsePerson.id)).join(Response, Response.id == ResponsePerson.response_id).where(Response.user_id == user.id, ResponsePerson.person_id.in_(pids)).group_by(ResponsePerson.person_id))).all()
        counts = {pid: int(c) for pid, c in cnt_rows}
    data = [{'id': p.id, 'name': p.display_name, 'mentions': counts.get(p.id, 0)} for p in inferred]
    data.sort(key=lambda x: (-x['mentions'], x['name'].lower()))
    return {'people': data}

@router.post('/api/skip_prompt')
async def api_skip_prompt(user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    next_id = await skip_current_prompt(db, user.id)
    return {'next_id': next_id}

@router.get('/api/prompt/{prompt_id}')
async def api_prompt(prompt_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p = await db.get(Prompt, prompt_id)
    if not p:
        raise HTTPException(status_code=404, detail='Prompt not found')
    return {'id': p.id, 'text': p.text, 'chapter': p.chapter}

@router.get('/login', response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse('login.html', {'request': request, 'user': None})

@router.get('/settings', response_class=HTMLResponse)
async def settings_page(request: Request, user=Depends(require_authenticated_html_user), db: AsyncSession=Depends(get_db)):
    from app.models import UserProfile
    profile = (await db.execute(select(UserProfile).where(UserProfile.user_id == user.id))).scalars().first()
    roles: list[str] = []
    if profile and isinstance(profile.tag_weights, dict):
        tw = dict(profile.tag_weights or {}).get('tagWeights') or {}
        try:
            roles = [k.split(':', 1)[1] for k, v in tw.items() if str(k).startswith('role:') and (v or 0) > 0]
        except Exception:
            roles = []
    ob = (getattr(profile, 'privacy_prefs', None) or {}).get('onboarding') if profile else None
    rel_roles = []
    if profile and profile.relation_roles:
        rel_roles = list(profile.relation_roles or [])
    elif roles:
        rel_roles = roles
    roles_for_form = ', '.join(rel_roles)
    interests_for_form = ', '.join(profile.interests or [] if profile and profile.interests else [])
    places_list: list[str] = []
    if profile and isinstance(profile.tag_weights, dict):
        tw2 = dict(profile.tag_weights or {}).get('tagWeights') or {}
        try:
            places_list = [k.split(':', 1)[1].replace('-', ' ') for k, v in tw2.items() if str(k).startswith('place:') and (v or 0) > 0]
        except Exception:
            places_list = []
    places_for_form = ', '.join(places_list)
    gender = None
    if profile and isinstance(profile.privacy_prefs, dict):
        gender = (profile.privacy_prefs or {}).get('user_meta', {}).get('gender')
    watch_rows = await db.execute(
        select(ResponseNotificationTarget, User)
        .join(User, ResponseNotificationTarget.watcher_user_id == User.id)
        .where(ResponseNotificationTarget.owner_user_id == user.id)
        .order_by(func.lower(User.username).nulls_last(), func.lower(User.email))
    )
    watchers = []
    watcher_ids: set[int] = set()
    for rel, watcher in watch_rows.all():
        display = ((watcher.username or '').strip() or (watcher.email or '').strip() or f'User #{watcher.id}')
        watchers.append({
            'id': watcher.id,
            'display': display,
            'email': (watcher.email or '').strip(),
            'is_admin': bool(getattr(watcher, 'is_superuser', False) or getattr(watcher, 'super_admin', False)),
            'notify_enabled': bool(getattr(watcher, 'notify_new_responses', False)),
        })
        watcher_ids.add(watcher.id)

    all_user_rows = await db.execute(
        select(User)
        .where(User.is_active.is_(True))
        .where(User.id != user.id)
        .order_by(func.lower(User.username).nulls_last(), func.lower(User.email))
    )
    notification_options = []
    for candidate in all_user_rows.scalars().all():
        label = ((candidate.username or '').strip() or (candidate.email or '').strip() or f'User #{candidate.id}')
        notification_options.append({
            'id': candidate.id,
            'label': label,
            'email': (candidate.email or '').strip(),
            'selected': candidate.id in watcher_ids,
            'notify_enabled': bool(getattr(candidate, 'notify_new_responses', False)),
            'is_admin': bool(getattr(candidate, 'is_superuser', False) or getattr(candidate, 'super_admin', False)),
        })
    ctx = {
        'request': request,
        'user': user,
        'profile': profile,
        'onboarding_roles': roles,
        'onboarding_meta': ob,
        'roles_for_form': roles_for_form,
        'interests_for_form': interests_for_form,
        'gender': gender,
        'places_for_form': places_for_form,
        'notify_new_responses_enabled': bool(getattr(user, 'notify_new_responses', False)),
        'notification_watchers': watchers,
        'notification_options': notification_options,
    }
    return templates.TemplateResponse('settings.html', ctx)

@router.post('/settings/profile')
async def settings_profile_update(request: Request, display_name: Optional[str]=Form(None), birth_year: Optional[int]=Form(None), location: Optional[str]=Form(None), relation_roles: Optional[str]=Form(None), interests: Optional[str]=Form(None), bio: Optional[str]=Form(None), places: Optional[str]=Form(None), gender: Optional[str]=Form(None), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    from app.models import UserProfile
    prof = (await db.execute(select(UserProfile).where(UserProfile.user_id == user.id))).scalars().first()
    if not prof:
        prof = UserProfile(user_id=user.id)
        db.add(prof)
    prof.display_name = (display_name or '').strip() or None
    prof.birth_year = birth_year if birth_year else None
    prof.location = (location or '').strip() or None

    def _csv(s: Optional[str]):
        return [t.strip() for t in (s or '').split(',') if t.strip()]
    prof.relation_roles = _csv(relation_roles) or None
    prof.interests = _csv(interests) or None
    prof.bio = (bio or '').strip() or None
    tw = dict(prof.tag_weights or {'tagWeights': {}})
    weights = tw.setdefault('tagWeights', {})
    for r in prof.relation_roles or []:
        key = f'role:{r}'
        try:
            weights[key] = max(float(weights.get(key, 0.0) or 0.0), 0.7)
        except Exception:
            weights[key] = 0.7
    for p in _csv(places):
        base = p
        from .utils import slugify as _slugify_local
        slug = f'place:{_slugify_local(base)}'
        try:
            weights[slug] = max(float(weights.get(slug, 0.0) or 0.0), 0.5)
        except Exception:
            weights[slug] = 0.5
    prof.tag_weights = tw
    pp = dict(prof.privacy_prefs or {})
    if gender is not None:
        um = dict(pp.get('user_meta') or {})
        um['gender'] = (gender or '').strip() or None
        pp['user_meta'] = um
    prof.privacy_prefs = pp
    try:
        await build_pool_for_user(db, user.id)
    except Exception:
        pass
    await db.commit()
    return RedirectResponse(url='/settings?notice=Saved', status_code=303)


@router.post('/settings/notifications')
async def settings_notifications_update(notify_new_responses: bool=Form(False), watchers: List[int]=Form([]), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    db_user = (await db.execute(select(User).where(User.id == user.id))).scalars().first()
    if not db_user:
        raise HTTPException(status_code=404, detail='User not found')
    db_user.notify_new_responses = bool(notify_new_responses)
    await db.execute(delete(ResponseNotificationTarget).where(ResponseNotificationTarget.owner_user_id == user.id))
    watcher_ids = []
    for raw in watchers or []:
        try:
            wid = int(raw)
        except (TypeError, ValueError):
            continue
        if wid == user.id:
            continue
        watcher_ids.append(wid)
    unique_ids = sorted(set(watcher_ids))
    for wid in unique_ids:
        db.add(ResponseNotificationTarget(owner_user_id=user.id, watcher_user_id=wid))
    await db.commit()
    msg = 'Notifications+enabled' if db_user.notify_new_responses else 'Notifications+disabled'
    return RedirectResponse(url=f'/settings?notice={msg}', status_code=303)


@router.post('/settings/password')
async def settings_password_update(current_password: str=Form(...), new_password: str=Form(...), confirm_password: str=Form(...), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    cur = (current_password or '').strip()
    new = (new_password or '').strip()
    conf = (confirm_password or '').strip()
    if new != conf:
        return RedirectResponse(url='/settings?notice=Passwords+do+not+match&error=1', status_code=303)
    if len(new) < 8:
        return RedirectResponse(url='/settings?notice=Password+must+be+at+least+8+characters&error=1', status_code=303)
    from passlib.hash import bcrypt as _bcrypt
    try:
        ok = _bcrypt.verify(cur, user.hashed_password or '')
    except Exception:
        ok = False
    if not ok:
        return RedirectResponse(url='/settings?notice=Current+password+is+incorrect&error=1', status_code=303)
    user.hashed_password = _bcrypt.hash(new)
    await db.commit()
    return RedirectResponse(url='/settings?notice=Password+updated', status_code=303)

@router.post('/settings/avatar')
async def settings_avatar_upload(avatar: UploadFile=File(...), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    from app.models import UserProfile
    _, ext = os.path.splitext(avatar.filename or '')
    ext = (ext or '.jpg').lower()
    safe_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    if ext not in safe_exts:
        ext = '.jpg'
    user_dir = (user.username or str(user.id)).replace('/', '_').replace('\\', '_')
    rel_dir = os.path.join('uploads', 'users', user_dir, 'profile')
    abs_dir = os.path.join(STATIC_DIR, rel_dir)
    os.makedirs(abs_dir, exist_ok=True)
    rel_path = os.path.join(rel_dir, f'avatar{ext}').replace('\\', '/')
    abs_path = os.path.join(STATIC_DIR, rel_path)
    with open(abs_path, 'wb') as w:
        shutil.copyfileobj(avatar.file, w)
    prof = (await db.execute(select(UserProfile).where(UserProfile.user_id == user.id))).scalars().first()
    if not prof:
        prof = UserProfile(user_id=user.id)
        db.add(prof)
    pp = dict(prof.privacy_prefs or {})
    pp['avatar_url'] = rel_path
    prof.privacy_prefs = pp
    await db.commit()
    return RedirectResponse(url='/settings?notice=Photo+updated', status_code=303)

@router.get('/user_dashboard', response_class=HTMLResponse)
async def user_dashboard(request: Request, q: str | None=Query(None), prompt_id: int | None=Query(None), ofs: int=Query(0, alias='offset'), user=Depends(require_authenticated_html_user), db: AsyncSession=Depends(get_db)):
    if not user or (not user.super_admin and (not user.is_active)):
        raise HTTPException(status_code=403, detail='Unauthorized')
    stmt = select(Response).join(Prompt, Prompt.id == Response.prompt_id).outerjoin(Tag, Prompt.tags).options(selectinload(Response.prompt)).where(Response.user_id == user.id).order_by(Response.created_at.desc())
    if q:
        like = f'%{q}%'
        stmt = stmt.where(or_(Response.response_text.ilike(like), Response.transcription.ilike(like), Prompt.text.ilike(like), Tag.name.ilike(like)))
    responses = (await db.execute(stmt)).unique().scalars().all()
    chap_rows = await db.execute(select(Prompt.chapter).distinct())
    all_chapters = [row[0] for row in chap_rows.all() if row[0]]
    meta_rows = (await db.execute(select(ChapterMeta))).scalars().all()
    meta_by = {m.name: m for m in meta_rows}
    ordered = sorted(all_chapters, key=lambda nm: (getattr(meta_by.get(nm), 'order', 1000000), nm.lower()))
    base_color = '#e5e7eb'

    def alpha_for_index(i: int) -> float:
        a = 0.04 + i * 0.03
        return min(a, 0.28)
    chapter_styles = {}
    for i, nm in enumerate(ordered):
        m = meta_by.get(nm)
        color = m.tint or base_color if m else base_color
        chapter_styles[nm] = {'color': color, 'alpha': f'{alpha_for_index(i):.2f}'}
    current_prompt = None
    if prompt_id:
        current_prompt = (await db.execute(select(Prompt).options(selectinload(Prompt.tags)).where(Prompt.id == prompt_id))).scalars().first()
    else:
        weekly = await ensure_weekly_prompt(db, user.id)
        if weekly and getattr(weekly, 'prompt_id', None):
            current_prompt = (await db.execute(select(Prompt).options(selectinload(Prompt.tags)).where(Prompt.id == weekly.prompt_id))).scalars().first()
    ctx = {'request': request, 'user': user, 'current_prompt': current_prompt, 'responses': responses, 'chapter_styles': chapter_styles}

    def _iso_year_week():
        iso = date.today().isocalendar()
        return (iso.year, iso.week)
    y, w = _iso_year_week()
    skipped_ids = (await db.execute(select(UserWeeklySkip.prompt_id).where((UserWeeklySkip.user_id == user.id) & (UserWeeklySkip.year == y) & (UserWeeklySkip.week == w)))).scalars().all()
    skipped_prompts = []
    if skipped_ids:
        skipped_prompts = (await db.execute(select(Prompt).options(selectinload(Prompt.tags)).where(Prompt.id.in_(skipped_ids)))).unique().scalars().all()
    ctx['skipped_prompts'] = skipped_prompts
    return templates.TemplateResponse('user_dashboard.html', ctx)

@router.get('/user_record', response_class=HTMLResponse, name='user_record_latest')
async def user_record(request: Request, user=Depends(require_authenticated_html_user), db: AsyncSession=Depends(get_db)):
    prompt = (await db.execute(select(Prompt).options(selectinload(Prompt.media)).order_by(Prompt.created_at.desc()).limit(1))).scalars().first()
    return templates.TemplateResponse('user_record.html', {'request': request, 'user': user, 'prompt': prompt, 'prompt_media': list(prompt.media) if prompt else []})

@router.get('/user_record/freeform', response_class=HTMLResponse, name='user_record_freeform')
async def user_record_freeform(request: Request, user=Depends(require_authenticated_html_user), db: AsyncSession=Depends(get_db)):
    chapters_res = await db.execute(select(Prompt.chapter).distinct().order_by(Prompt.chapter))
    chapters = [row[0] for row in chapters_res.all() if row[0]]
    return templates.TemplateResponse('user_record.html', {'request': request, 'user': user, 'prompt': None, 'prompt_media': [], 'chapters': chapters})

@router.get('/user_record/{prompt_id}', response_class=HTMLResponse, name='user_record')
async def user_record_with_prompt(prompt_id: int, request: Request, token: Optional[str]=Query(None), user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    if not user and token:
        try:
            tok = await mark_clicked(db, token)
            await db.commit()
        except Exception:
            tok = None
        if not tok or tok.prompt_id != prompt_id:
            next_rel = request.url.path + (f'?token={token}' if token else '')
            return RedirectResponse(url=f'/login?next={next_rel}', status_code=303)
    prompt = (await db.execute(select(Prompt).options(selectinload(Prompt.media)).where(Prompt.id == prompt_id))).scalars().first()
    if not prompt:
        raise HTTPException(status_code=404, detail='Prompt not found')
    viewer_id = None
    if user:
        viewer_id = user.id
    elif token:
        try:
            tok2 = await mark_clicked(db, token)
            await db.commit()
            if tok2:
                viewer_id = tok2.user_id
        except Exception:
            viewer_id = None
    media = list(prompt.media or [])

    def _can_view(m, uid):
        assigned_one = getattr(m, 'assignee_user_id', None)
        assigned_many = [getattr(u, 'id', None) for u in getattr(m, 'assignees', []) or []]
        has_any = bool(assigned_one) or bool(assigned_many)
        if uid is None:
            return not has_any
        if not has_any:
            return True
        if assigned_one and assigned_one == uid:
            return True
        return uid in assigned_many
    media = [m for m in media if _can_view(m, viewer_id)]
    return templates.TemplateResponse('user_record.html', {'request': request, 'user': user, 'prompt': prompt, 'prompt_media': media, 'is_token_link': bool(token)})

@router.post('/responses/')
async def create_response(prompt_id: int | None=Form(None), title: str | None=Form(None), chapter: str | None=Form(None), response_text: str | None=Form(None), primary_media: UploadFile | None=File(None), supporting_media: list[UploadFile] | None=File(None), weekly_token: str | None=Form(None), request: Request=None, user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
    acting_user = user
    if weekly_token:
        tok = await mark_clicked(db, weekly_token)
        await db.flush()
        if not tok:
            tok = (await db.execute(select(WeeklyToken).where(WeeklyToken.token == weekly_token))).scalars().first()
            if not tok or (tok.expires_at and tok.expires_at < _now()) or tok.status in (WeeklyTokenStatus.expired, WeeklyTokenStatus.used):
                raise HTTPException(status_code=401, detail='Token invalid or expired')
        acting_user = await db.get(User, tok.user_id)
        if not acting_user:
            raise HTTPException(status_code=401, detail='Token user not found')
        prompt_id = tok.prompt_id
    if not acting_user:
        raise HTTPException(status_code=401, detail='Not authenticated')
    if not prompt_id:
        t = (title or '').strip()
        ch = (chapter or '').strip()
        if not t or not ch:
            raise HTTPException(status_code=422, detail='Title and chapter are required for a freeform story.')
        free_prompt = Prompt(text=t, chapter=ch)
        db.add(free_prompt)
        await db.flush()
        try:
            tag = await _get_or_create_tag(db, 'freeform')
            if tag:
                free_prompt.tags = (free_prompt.tags or []) + [tag]
        except Exception:
            pass
        prompt_id = free_prompt.id
    prompt_obj = None
    if prompt_id:
        prompt_obj = (await db.execute(select(Prompt).options(selectinload(Prompt.tags)).where(Prompt.id == prompt_id))).scalars().first()
    new_response = Response(prompt_id=prompt_id, user_id=acting_user.id, response_text=response_text)
    if hasattr(new_response, 'title'):
        new_response.title = (title or '').strip() or None
    if prompt_obj and prompt_obj.tags:
        new_response.tags = list(prompt_obj.tags)
    db.add(new_response)
    await db.flush()

    def _save_temp(upload) -> FSPath:
        tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(upload.filename or '').suffix}"
        with open(tmp, 'wb') as w:
            shutil.copyfileobj(upload.file, w)
        return tmp
    if primary_media and primary_media.filename:
        tmp = _save_temp(primary_media)
        art = PIPELINE.process_upload(temp_path=tmp, logical='response', role='primary', user_slug_or_id=acting_user.username or str(acting_user.id), prompt_id=None, response_id=new_response.id, media_id=None, original_filename=primary_media.filename, content_type=primary_media.content_type)
        playable_rel = art.playable_rel
        new_response.primary_media_url = playable_rel[len('uploads/'):] if playable_rel.startswith('uploads/') else playable_rel
        new_response.primary_thumbnail_path = art.thumb_rel
        new_response.primary_mime_type = art.mime_type
        new_response.primary_duration_sec = int(art.duration_sec or 0)
        new_response.primary_sample_rate = art.sample_rate
        new_response.primary_channels = art.channels
        new_response.primary_width = art.width
        new_response.primary_height = art.height
        new_response.primary_size_bytes = art.size_bytes
        new_response.primary_codec_audio = art.codec_a
        new_response.primary_codec_video = art.codec_v
        new_response.primary_wav_path = art.wav_rel
        exist = await db.execute(select(ResponseSegment.id).where(ResponseSegment.response_id == new_response.id))
        if not exist.first():
            primary_rel = (new_response.primary_media_url or '').lstrip('/').replace('\\', '/')
            if primary_rel.startswith('uploads/'):
                primary_rel = primary_rel[len('uploads/'):]
            is_composite = primary_rel.startswith(f'responses/{new_response.id}/') and primary_rel.split('/')[-1].startswith('composite-')
            if primary_rel and (not is_composite):
                seg0 = ResponseSegment(response_id=new_response.id, order_index=0, media_path=primary_rel, media_mime=new_response.primary_mime_type or None, transcript=new_response.transcription or '')
                db.add(seg0)
    if supporting_media:
        for f in supporting_media:
            if not f or not f.filename:
                continue
            media = SupportingMedia(response_id=new_response.id, file_path='', media_type=f.content_type.split('/', 1)[0] if f.content_type else 'file')
            db.add(media)
            await db.flush()
            tmp = _save_temp(f)
            art = PIPELINE.process_upload(temp_path=tmp, logical='response', role='supporting', user_slug_or_id=acting_user.username or str(acting_user.id), prompt_id=None, response_id=new_response.id, media_id=media.id, original_filename=f.filename, content_type=f.content_type)
            playable_rel = art.playable_rel
            media.file_path = playable_rel[len('uploads/'):] if playable_rel.startswith('uploads/') else playable_rel
            media.thumbnail_url = art.thumb_rel
            media.mime_type = art.mime_type
            media.duration_sec = int(art.duration_sec or 0)
            media.sample_rate = art.sample_rate
            media.channels = art.channels
            media.width = art.width
            media.height = art.height
            media.size_bytes = art.size_bytes
            media.codec_audio = art.codec_a
            media.codec_video = art.codec_v
            media.wav_path = art.wav_rel
    await db.refresh(new_response, attribute_names=['prompt', 'tags'])
    text_for_tagging = _text_for_tagging(new_response)
    if text_for_tagging.strip():
        draft = suggest_tags_rule_based(text_for_tagging, word_count=len(text_for_tagging.split()))
        existing_ids = {t.id for t in new_response.tags or []}
        for slug, _ in draft:
            tag = await _get_or_create_tag(db, slug)
            if tag and tag.id not in existing_ids:
                new_response.tags.append(tag)
                existing_ids.add(tag.id)
    await db.execute(update(UserPrompt).where((UserPrompt.user_id == acting_user.id) & (UserPrompt.prompt_id == new_response.prompt_id)).values(status='answered'))
    await db.commit()
    try:
        if (new_response.response_text or '').strip():
            await enrich_after_transcription(db, new_response)
    except Exception:
        pass
    spawn(transcribe_and_update(new_response.id, new_response.primary_media_url, bool(weekly_token)), name='transcribe_response_primary')
    try:
        if prompt_id:
            y, w = _iso_year_week()
            weekly = (await db.execute(select(UserWeeklyPrompt).where(UserWeeklyPrompt.user_id == acting_user.id, UserWeeklyPrompt.year == y, UserWeeklyPrompt.week == w))).scalars().first()
            if weekly and weekly.prompt_id == prompt_id:
                weekly.status = 'answered'
                await db.commit()
                await _rotate_to_next_unanswered(db, acting_user.id)
    except Exception:
        pass
    spawn(notify_new_response(new_response.id), name='notify_new_response')
    if weekly_token:
        await mark_completed_and_close(db, weekly_token)
        try:
            await ensure_weekly_prompt(db, acting_user.id)
        except Exception as e:
            logger.exception('weekly rotate after response failed: %s', e)
        await db.commit()
        html = '\n        <section class="fixed inset-0 grid place-items-center bg-black/70">\n          <div class="bg-white/95 rounded-2xl p-6 max-w-md text-center">\n            <h2 class="text-xl font-semibold mb-2">Thanks for sharing your story!</h2>\n            <p class="mb-4 text-slate-700">We saved it for you. If you’d like to read or edit it later, please log in.</p>\n            <div class="flex justify-center gap-2">\n              <a href="/login" class="btn">Go to Login</a>\n              <a href="/" class="btn btn-ghost">Close</a>\n            </div>\n          </div>\n        </section>\n        '
        return templates.TemplateResponse('thank_you.html', {'request': request, 'user': None})
    return RedirectResponse(url=f'/response/{new_response.id}/edit', status_code=303)

@router.get('/response/{response_id}', response_class=HTMLResponse, name='response_view')
async def response_view(response_id: int, request: Request, user=Depends(require_authenticated_html_user), db: AsyncSession=Depends(get_db)):
    q = select(Response).options(selectinload(Response.prompt).selectinload(Prompt.media), selectinload(Response.tags), selectinload(Response.supporting_media), selectinload(Response.segments)).where(Response.id == response_id, Response.user_id == user.id)
    resp = (await db.execute(q)).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    prompt_media = list(resp.prompt.media) if resp.prompt and resp.prompt.media else []
    supporting_media = list(resp.supporting_media or [])
    segments = list(resp.segments or [])
    ctx = {'request': request, 'user': user, 'response': resp, 'prompt_media': prompt_media, 'supporting_media': supporting_media, 'segments': segments, 'is_token_link': False}
    return templates.TemplateResponse('response_view.html', ctx)

@router.get('/response/{response_id}/edit', response_class=HTMLResponse, name='edit_response_page')
async def edit_response_page(response_id: int, request: Request, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    response = (await db.execute(select(Response).options(selectinload(Response.prompt), selectinload(Response.tags)).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    supporting_media = (await db.execute(select(SupportingMedia).where(SupportingMedia.response_id == response_id))).scalars().all()
    pm_res = await db.execute(select(PromptMedia).where(PromptMedia.prompt_id == response.prompt_id))
    prompt_media = pm_res.scalars().all()
    return templates.TemplateResponse('response_edit.html', {'request': request, 'user': user, 'response': response, 'supporting_media': supporting_media, 'prompt_media': prompt_media})

@router.post('/response/{response_id}/edit')
async def save_transcription(response_id: int, transcription: str=Form(...), title: str | None=Form(None), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    response = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    response.transcription = transcription
    if title is not None and hasattr(response, 'title'):
        response.title = (title or '').strip() or None
    await db.commit()
    text_for_tagging = _text_for_tagging(response)
    if text_for_tagging.strip():
        draft = suggest_tags_rule_based(text_for_tagging, word_count=len(text_for_tagging.split()))
        for slug, _ in draft:
            tag = await _get_or_create_tag(db, slug)
            if tag and tag not in response.tags:
                response.tags.append(tag)
    try:
        await enrich_after_transcription(db, response)
    except Exception:
        pass
    await db.commit()
    return RedirectResponse(url=f'/response/{response_id}', status_code=303)

@router.get('/response/{response_id}/versions')
async def user_list_versions(response_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    exists = (await db.execute(select(Response.id).where(Response.id == response_id, Response.user_id == user.id))).scalar_one_or_none()
    if not exists:
        raise HTTPException(status_code=404, detail='Response not found')
    rows = (await db.execute(select(ResponseVersion).where(ResponseVersion.response_id == response_id).order_by(ResponseVersion.created_at.desc(), ResponseVersion.id.desc()))).scalars().all()
    out = []
    for v in rows:
        out.append({'id': v.id, 'created_at': v.created_at.isoformat() if v.created_at else None, 'edited_by_admin_id': v.edited_by_admin_id, 'title': (v.title or '')[:120], 'has_transcription': bool((v.transcription or '').strip()), 'tags': (v.tags_json or {}).get('tags') if isinstance(v.tags_json, dict) else None})
    return {'versions': out}

@router.post('/response/{response_id}/versions/{version_id}/restore')
async def user_restore_version(response_id: int, version_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    resp = await db.get(Response, response_id)
    if not resp or resp.user_id != user.id:
        raise HTTPException(status_code=404, detail='Response not found')
    ver = await db.get(ResponseVersion, version_id)
    if not ver or ver.response_id != response_id:
        raise HTTPException(status_code=404, detail='Version not found')
    if hasattr(resp, 'title'):
        resp.title = ver.title
    resp.transcription = ver.transcription
    try:
        tags = []
        slugs = []
        if isinstance(ver.tags_json, dict):
            slugs = [s for s in (ver.tags_json or {}).get('tags', []) if s]
        for s in slugs:
            t = await _get_or_create_tag(db, s)
            if t:
                tags.append(t)
        assoc_tbl = Response.tags.property.secondary
        await db.execute(delete(assoc_tbl).where(assoc_tbl.c.response_id == response_id))
        if tags:
            stmt = pg_insert(assoc_tbl).on_conflict_do_nothing(index_elements=[assoc_tbl.c.response_id, assoc_tbl.c.tag_id])
            await db.execute(stmt, [{'response_id': response_id, 'tag_id': t.id} for t in tags])
    except Exception:
        pass
    await db.commit()
    return {'ok': True}

def _gen_share_token() -> str:
    return secrets.token_urlsafe(22)

@router.post('/response/{response_id}/share')
async def create_response_share(response_id: int, permanent: bool=Form(True), days: int | None=Form(None), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    resp = await db.get(Response, response_id)
    if not resp or resp.user_id != user.id:
        raise HTTPException(status_code=404, detail='Response not found')
    expires_at = None
    if not permanent and days and (days > 0):
        expires_at = datetime.now(timezone.utc) + timedelta(days=days)
    tok = ResponseShare(token=_gen_share_token(), response_id=response_id, user_id=user.id, permanent=bool(permanent), expires_at=expires_at)
    db.add(tok)
    await db.commit()
    return {'ok': True, 'link': f'/share/r/{tok.token}'}

@router.get('/share/r/{token}', response_class=HTMLResponse)
async def share_response_view(token: str, request: Request, db: AsyncSession=Depends(get_db)):
    share = (await db.execute(select(ResponseShare).where(ResponseShare.token == token))).scalars().first()
    if not share or share.revoked or (share.expires_at and share.expires_at < datetime.now(timezone.utc)):
        raise HTTPException(status_code=404, detail='Link not found or expired')
    resp = (await db.execute(select(Response).options(selectinload(Response.prompt).selectinload(Prompt.media), selectinload(Response.tags), selectinload(Response.supporting_media), selectinload(Response.segments)).where(Response.id == share.response_id))).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    ctx = {'request': request, 'user': None, 'response': resp, 'prompt_media': list(resp.prompt.media) if resp.prompt and resp.prompt.media else [], 'supporting_media': list(resp.supporting_media or []), 'segments': list(resp.segments or []), 'is_token_link': True, 'share_token': token}
    return templates.TemplateResponse('response_view.html', ctx)

def _response_id_from_uploads_path(rel: str) -> int | None:
    p = rel.lstrip('/').replace('\\', '/')
    if p.startswith('static/'):
        p = p[len('static/'):]
    if p.startswith('uploads/'):
        p = p[len('uploads/'):]
    parts = p.split('/')
    try:
        if 'responses' in parts:
            i = parts.index('responses')
            return int(parts[i + 1])
    except Exception:
        return None
    return None

@router.get('/media/share/{token}/{path:path}')
async def media_share_stream(token: str, path: str, db: AsyncSession=Depends(get_db)):
    share = (await db.execute(select(ResponseShare).where(ResponseShare.token == token))).scalars().first()
    if not share or share.revoked or (share.expires_at and share.expires_at < datetime.now(timezone.utc)):
        raise HTTPException(status_code=404, detail='Link not found or expired')
    rid = _response_id_from_uploads_path(path)
    if rid != share.response_id:
        raise HTTPException(status_code=403, detail='Not allowed')
    uploads = STATIC_DIR / path.lstrip('/')
    abspath = (STATIC_DIR / path.lstrip('/').replace('\\', '/')).resolve()
    uploads_root = (STATIC_DIR / 'uploads').resolve()
    if uploads_root not in abspath.parents and abspath != uploads_root:
        raise HTTPException(status_code=400, detail='invalid path')
    if not abspath.exists() or abspath.is_dir():
        raise HTTPException(status_code=404, detail='file not found')
    return FileResponse(str(abspath))

@router.get('/media/auth/{path:path}')
async def media_auth_stream(path: str, user=Depends(require_authenticated_user)):
    rid = _response_id_from_uploads_path(path)
    if rid is None:
        raise HTTPException(status_code=400, detail='invalid path')
    if not getattr(user, 'is_superuser', False):
        async with async_session_maker() as s:
            r = await s.get(Response, rid)
            if not r or r.user_id != user.id:
                raise HTTPException(status_code=404, detail='Not found')
    abspath = (STATIC_DIR / path.lstrip('/').replace('\\', '/')).resolve()
    uploads_root = (STATIC_DIR / 'uploads').resolve()
    if uploads_root not in abspath.parents and abspath != uploads_root:
        raise HTTPException(status_code=400, detail='invalid path')
    if not abspath.exists() or abspath.is_dir():
        raise HTTPException(status_code=404, detail='file not found')
    return FileResponse(str(abspath))

@router.post('/response/{response_id}/delete')
async def delete_response(response_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    response = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    prompt_id = response.prompt_id
    PIPELINE.delete_artifacts(_to_uploads_rel_for_playable(response.primary_media_url) if response.primary_media_url else None, response.primary_wav_path, response.primary_thumbnail_path)
    supp_all = (await db.execute(select(SupportingMedia).where(SupportingMedia.response_id == response_id))).scalars().all()
    for m in supp_all:
        playable_uploads_rel = _to_uploads_rel_for_playable(m.file_path)
        PIPELINE.delete_artifacts(playable_uploads_rel, m.wav_path or None, m.thumbnail_url or None)
        await db.delete(m)
    await db.delete(response)
    await db.flush()
    if prompt_id:
        remaining = (await db.execute(select(func.count(Response.id)).where(Response.user_id == user.id, Response.prompt_id == prompt_id))).scalar_one()
        if remaining == 0:
            up = (await db.execute(select(UserPrompt).where(UserPrompt.user_id == user.id, UserPrompt.prompt_id == prompt_id))).scalars().first()
            if up:
                up.status = 'queued'
    await db.commit()
    return RedirectResponse(url='/user_dashboard', status_code=303)

@router.get('/api/chapters_progress')
async def api_chapters_progress(user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    """
    Per-chapter progress for the current user.
    'Pool' = UserPrompt assignments for this user.
    'Completed' = those assignments with at least one Response by this user.
    """
    pool_rows = await db.execute(select(Prompt.chapter, func.count(Prompt.id)).select_from(UserPrompt).join(Prompt, Prompt.id == UserPrompt.prompt_id).where(UserPrompt.user_id == user.id).group_by(Prompt.chapter))
    totals_by = {row[0] or 'Misc': int(row[1] or 0) for row in pool_rows.all()}
    done_rows = await db.execute(select(Prompt.chapter, func.count(func.distinct(Prompt.id))).select_from(UserPrompt).join(Prompt, Prompt.id == UserPrompt.prompt_id).join(Response, and_(Response.prompt_id == Prompt.id, Response.user_id == user.id)).where(UserPrompt.user_id == user.id).group_by(Prompt.chapter))
    done_by = {row[0] or 'Misc': int(row[1] or 0) for row in done_rows.all()}
    meta_rows = (await db.execute(select(ChapterMeta))).scalars().all()
    meta_map = {m.name: m for m in meta_rows}
    payload = []
    for name, total in totals_by.items():
        m = meta_map.get(name)
        payload.append({'name': name, 'slug': name, 'display_name': (m.display_name if m else None) or (name or 'Misc'), 'tint': m.tint if m else None, 'total': total, 'completed': int(done_by.get(name, 0))})

    def order_of(n):
        mm = meta_map.get(n)
        return getattr(mm, 'order', 999999)
    payload.sort(key=lambda d: (order_of(d['name']), d['display_name'].lower()))
    return payload

@router.post('/response/{response_id}/media')
async def add_supporting_media(response_id: int, media_files: list[UploadFile]=File(...), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    result = await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))
    resp = result.scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    for f in media_files or []:
        if not f or not f.filename:
            continue
        media = SupportingMedia(response_id=resp.id, file_path='', media_type=f.content_type.split('/', 1)[0] if f.content_type else 'file')
        db.add(media)
        await db.flush()
        tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(f.filename or '').suffix}"
        with open(tmp, 'wb') as w:
            shutil.copyfileobj(f.file, w)
        art = PIPELINE.process_upload(temp_path=tmp, logical='response', role='supporting', user_slug_or_id=user.username or str(user.id), prompt_id=None, response_id=resp.id, media_id=media.id, original_filename=f.filename, content_type=f.content_type)
        playable_rel = art.playable_rel
        media.file_path = playable_rel[len('uploads/'):] if playable_rel.startswith('uploads/') else playable_rel
        media.thumbnail_url = art.thumb_rel
        media.mime_type = art.mime_type
        media.duration_sec = int(art.duration_sec or 0)
        media.sample_rate = art.sample_rate
        media.channels = art.channels
        media.width = art.width
        media.height = art.height
        media.size_bytes = art.size_bytes
        media.codec_audio = art.codec_a
        media.codec_video = art.codec_v
        media.wav_path = art.wav_rel
    await db.commit()
    return RedirectResponse(url=f'/response/{resp.id}/edit', status_code=303)

@router.get('/response/{response_id}/next', response_class=HTMLResponse)
async def response_next(response_id: int, request: Request, user=Depends(require_authenticated_html_user), db: AsyncSession=Depends(get_db)):
    cur = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not cur:
        raise HTTPException(status_code=404, detail='Response not found')
    next_q = select(Response).options(selectinload(Response.prompt).selectinload(Prompt.media), selectinload(Response.tags), selectinload(Response.supporting_media), selectinload(Response.segments)).where(Response.user_id == user.id).where(or_(Response.created_at < cur.created_at, and_(Response.created_at == cur.created_at, Response.id < cur.id))).order_by(Response.created_at.desc(), Response.id.desc()).limit(1)
    next_resp = (await db.execute(next_q)).scalars().first()
    if not next_resp:
        return HTMLResponse(status_code=204)
    prompt_media = list(next_resp.prompt.media) if next_resp.prompt and next_resp.prompt.media else []
    supporting_media = list(next_resp.supporting_media or [])
    segments = list(next_resp.segments or [])
    return templates.TemplateResponse('response_view__article_partial.html', {'request': request, 'user': user, 'response': next_resp, 'prompt_media': prompt_media, 'supporting_media': supporting_media, 'segments': segments, 'is_token_link': False})

@router.delete('/response/{response_id}/media/{media_id}')
async def delete_supporting_media(response_id: int, media_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    response = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    media = (await db.execute(select(SupportingMedia).where(SupportingMedia.id == media_id, SupportingMedia.response_id == response_id))).scalars().first()
    if not media:
        raise HTTPException(status_code=404, detail='Media not found')
    playable_uploads_rel = _to_uploads_rel_for_playable(media.file_path)
    wav_uploads_rel = media.wav_path or None
    thumb_uploads_rel = media.thumbnail_url or None
    PIPELINE.delete_artifacts(playable_uploads_rel, wav_uploads_rel, thumb_uploads_rel)
    await db.delete(media)
    await db.commit()
    return {'success': True}

async def transcribe_and_update(response_id: int, media_filename: str, auto_polish: bool=False):
    try:
        async with async_session_maker() as session:
            resp = await session.get(Response, response_id)
            uid = getattr(resp, 'user_id', None) if resp else None
            transcript = await transcribe_file(media_filename, db=session, user_id=uid)
            if resp:
                resp.transcription = transcript
                from sqlalchemy import select
                pm_rel = (resp.primary_media_url or '').lstrip('/').replace('\\', '/')
                if pm_rel.startswith('uploads/'):
                    pm_rel = pm_rel[len('uploads/'):]
                seg_row = (await session.execute(select(ResponseSegment).where(ResponseSegment.response_id == response_id).order_by(ResponseSegment.order_index.asc(), ResponseSegment.id.asc()))).scalars().first()
                if seg_row:
                    if (seg_row.media_path or '') == pm_rel or seg_row.order_index in (0, 1):
                        if not seg_row.transcript:
                            seg_row.transcript = transcript or ''
                await session.commit()
                try:
                    if auto_polish and (resp.transcription or '').strip():
                        cleaned = await polish_text(resp.transcription)
                        resp.transcription = cleaned or resp.transcription
                        await session.commit()
                except Exception as e:
                    logging.getLogger(__name__).warning('Auto-polish failed for response %s: %s', response_id, e)
                from app.transcription import enrich_after_transcription
                await enrich_after_transcription(session, resp)
    except Exception as e:
        print(f'❌ Transcription failed for response {response_id}: {e}')

async def transcribe_segment_and_update(segment_id: int, uploads_rel_path: str):
    """
    Run transcription for a segment (file already stored under 'uploads/...'),
    then persist transcript to the segment.
    """
    try:
        rel = uploads_rel_path
        if rel.startswith('uploads/'):
            rel = rel[len('uploads/'):]
        async with async_session_maker() as s:
            uid = None
            try:
                seg = await s.get(ResponseSegment, segment_id)
                if seg:
                    resp = await s.get(Response, getattr(seg, 'response_id', None))
                    uid = getattr(resp, 'user_id', None)
            except Exception:
                uid = None
            text = await transcribe_file(rel, db=s, user_id=uid)
            seg = await s.get(ResponseSegment, segment_id)
            if seg:
                seg.transcript = text or ''
                await s.commit()
    except Exception:
        try:
            async with async_session_maker() as s:
                seg = await s.get(ResponseSegment, segment_id)
                if seg and (not seg.transcript):
                    seg.transcript = '[Transcription failed]'
                    await s.commit()
        except:
            pass

@router.get('/admin_dashboard')
async def admin_dashboard(request: Request, user: User=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    """
    Admin dashboard context:
      - chapters: { chapter_key: [Prompt, ...] } (prompts eager-loaded with media & tags)
      - tags_map: { prompt_id: [{id, name, slug}], ... } (for chips)
      - media_map: { prompt_id: [{id, file_url, thumb_url}], ... } (for thumbs)
      - users_meta: [{id, email, username, display_name}]
      - invites: existing invites (kept as before)
      - tag_whitelist_json: raw JSON string from /data/tag_whitelist.json
    """
    q = select(Prompt).options(selectinload(Prompt.media), selectinload(Prompt.tags)).order_by(Prompt.chapter, Prompt.id)
    prompts = (await db.execute(q)).scalars().all()
    chapters: dict[str, list[Prompt]] = {}
    for p in prompts:
        chapters.setdefault(p.chapter or 'uncategorized', []).append(p)
    tags_map: dict[int, list[dict]] = {}
    media_map: dict[int, list[dict]] = {}
    for p in prompts:
        if getattr(p, 'tags', None):
            tags_map[p.id] = [{'id': t.id, 'name': getattr(t, 'name', None) or getattr(t, 'slug', None) or '', 'slug': getattr(t, 'slug', None) or getattr(t, 'name', None) or ''} for t in p.tags]
        else:
            tags_map[p.id] = []
        if getattr(p, 'media', None):
            media_map[p.id] = []
            for m in p.media:
                file_url = f'/static/uploads/{m.file_path}' if m.file_path else ''
                thumb_url = f'/static/{m.thumbnail_url}' if m.thumbnail_url else ''
                media_map[p.id].append({'id': m.id, 'file_url': file_url, 'thumb_url': thumb_url or file_url})
        else:
            media_map[p.id] = []
    totals = (await db.execute(select(UserPrompt.user_id, func.count().label('total')).group_by(UserPrompt.user_id))).all()
    answered = (await db.execute(select(UserPrompt.user_id, func.count(func.distinct(Response.id)).label('answered')).join(Response, (Response.user_id == UserPrompt.user_id) & (Response.prompt_id == UserPrompt.prompt_id), isouter=False).group_by(UserPrompt.user_id))).all()
    totals_map = {uid: total for uid, total in totals}
    answered_map = {uid: ans for uid, ans in answered}

    def pct(uid: int) -> int:
        t = totals_map.get(uid, 0)
        a = answered_map.get(uid, 0)
        return int(round(100 * a / t)) if t else 0
    users = (await db.execute(select(User))).scalars().all()
    profiles = (await db.execute(select(UserProfile).where(UserProfile.user_id.in_([u.id for u in users])))).scalars().all()
    pmap = {p.user_id: p for p in profiles}
    users_meta = [{'id': u.id, 'email': u.email, 'username': u.username, 'display_name': (pmap.get(u.id).display_name if pmap.get(u.id) else None) or (u.username or None) or u.email, 'answered_pct': pct(u.id)} for u in users]
    try:
        invites = (await db.execute(select(Invite).order_by(Invite.created_at.desc()))).scalars().all()
    except Exception:
        invites = []
    wl_path = FSPath(__file__).resolve().parents[1] / 'data' / 'tag_whitelist.json'
    try:
        tag_whitelist_json = wl_path.read_text(encoding='utf-8')
    except Exception:
        tag_whitelist_json = '[]'
    rows = (await db.execute(select(UserPrompt.user_id, UserPrompt.prompt_id))).all()
    assignments_by_prompt: dict[int, list[dict]] = {}
    if rows:
        uids = list({uid for uid, _ in rows})
        users = (await db.execute(select(User).where(User.id.in_(uids)))).scalars().all()
        profiles = (await db.execute(select(UserProfile).where(UserProfile.user_id.in_(uids)))).scalars().all()
        pmap = {p.user_id: p for p in profiles}

        def _name(u: User) -> str:
            return (pmap.get(u.id).display_name if pmap.get(u.id) else None) or (u.username or u.email or '')
        umap = {u.id: {'id': u.id, 'name': _name(u), 'email': u.email} for u in users}
        for uid, pid in rows:
            if uid in umap:
                assignments_by_prompt.setdefault(pid, []).append(umap[uid])
    return templates.TemplateResponse('admin_dashboard.html', {'request': request, 'user': user, 'chapters': chapters, 'tags_map': tags_map, 'media_map': media_map, 'users_meta': users_meta, 'invites': invites, 'tag_whitelist_json': tag_whitelist_json, 'assignments_by_prompt': assignments_by_prompt})

@router.post('/admin/users/{user_id}/force_password')
async def admin_force_password(user_id: int, new_password: str=Form(...), user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    target = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if not target:
        raise HTTPException(status_code=404, detail='User not found')
    if not new_password or len(new_password) < 8:
        raise HTTPException(status_code=400, detail='New password too short')
    target.hashed_password = bcrypt.hash(new_password)
    target.must_change_password = True
    await db.commit()
    return RedirectResponse(url='/admin_dashboard?notice=Password+updated', status_code=303)

async def on_prompt_created(db: AsyncSession, prompt_id: int) -> None:
    """
    Called after a Prompt is created (and tags are attached).
    It pre-assigns the prompt to every eligible user, UNLESS the prompt is private_only/only_assigned.
    """
    try:
        from app.models import Prompt, UserProfile, UserPrompt
        from app.services.assignment import _eligible, _user_role_like_slugs
        prompt = await db.get(Prompt, prompt_id)
        if not prompt:
            logging.warning('[prompt-fanout] Prompt %s not found', prompt_id)
            return
        private_flags = [getattr(prompt, 'private_only', False), getattr(prompt, 'only_assigned', False)]
        if any((bool(x) for x in private_flags)):
            logging.info('[prompt-fanout] Prompt %s is private_only/only_assigned; skipping auto-assign', prompt_id)
            return
        tag_vals = set()
        for t in getattr(prompt, 'tags', None) or []:
            if getattr(t, 'slug', None):
                tag_vals.add(t.slug.lower())
            if getattr(t, 'name', None):
                tag_vals.add(t.name.lower())
        if 'scope:private' in tag_vals or 'private' in tag_vals:
            logging.info('[prompt-fanout] Prompt %s has private tag; skipping auto-assign', prompt_id)
            return
        profiles = list((await db.execute(select(UserProfile))).scalars().all())
        to_assign = []
        for prof in profiles:
            user_slugs = _user_role_like_slugs(prof)
            if _eligible(prompt, user_slugs):
                to_assign.append(prof.user_id)
        if not to_assign:
            logging.info('[prompt-fanout] Prompt %s matched 0 users', prompt_id)
            return
        rows = [{'user_id': uid, 'prompt_id': prompt.id} for uid in to_assign]
        stmt = pg_insert(UserPrompt.__table__).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=['user_id', 'prompt_id'])
        await db.execute(stmt)
        await db.commit()
        logging.info('[prompt-fanout] Prompt %s queued for %d user(s)', prompt_id, len(to_assign))
    except Exception:
        logging.exception('[prompt-fanout] failed for %s', prompt_id)
    except Exception:
        logging.exception('[prompt-fanout] on_prompt_created failed for %s', prompt_id)

@router.post('/admin_create_prompt')
async def admin_create_prompt(request: Request, prompt_text: str=Form(...), chapter: str=Form(...), tags: str=Form('[]'), media_files: list[UploadFile] | None=File(None), only_assigned: int | bool=Form(0), user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    """
    Create a prompt (with tags + optional media), then fan-out to eligible users
    via on_prompt_created in the background.
    """
    prompt = Prompt(text=prompt_text, chapter=chapter)

    def _parse_tag_input(raw: str) -> list[str]:
        try:
            data = json.loads(raw) if raw else []
            if not isinstance(data, list):
                data = []
        except Exception:
            data = [p.strip() for p in (raw or '').split(',') if p.strip()]
        out = []
        for item in data:
            if isinstance(item, str):
                out.append(item.strip())
            elif isinstance(item, dict):
                v = (item.get('value') or item.get('slug') or item.get('text') or '').strip()
                if v:
                    out.append(v)
        return list(dict.fromkeys(out))
    tag_names = _parse_tag_input(tags)
    if only_assigned and 'scope:private' not in tag_names:
        tag_names.append('scope:private')
    resolved: list[Tag] = []
    for nm in tag_names:
        tag = await _get_or_create_tag(db, nm)
        if tag:
            resolved.append(tag)
    prompt.tags = resolved
    db.add(prompt)
    await db.flush()
    if media_files:
        for file in media_files:
            if not file or not file.filename:
                continue
            new_media = PromptMedia(prompt_id=prompt.id, file_path='', media_type=file.content_type.split('/', 1)[0] if file.content_type else 'file')
            db.add(new_media)
            await db.flush()
            tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(file.filename or '').suffix}"
            with open(tmp, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
            art = PIPELINE.process_upload(temp_path=tmp, logical='prompt', role='prompt', user_slug_or_id=None, prompt_id=prompt.id, response_id=None, media_id=new_media.id, original_filename=file.filename, content_type=file.content_type)
            playable_rel = art.playable_rel
            new_media.file_path = playable_rel[len('uploads/'):] if playable_rel.startswith('uploads/') else playable_rel
            new_media.thumbnail_url = art.thumb_rel
            new_media.mime_type = art.mime_type
            new_media.duration_sec = int(art.duration_sec or 0)
            new_media.sample_rate = art.sample_rate
            new_media.channels = art.channels
            new_media.width = art.width
            new_media.height = art.height
            new_media.size_bytes = art.size_bytes
            new_media.codec_audio = art.codec_a
            new_media.codec_video = art.codec_v
            new_media.wav_path = art.wav_rel
    await db.commit()
    if not only_assigned:

        async def _fanout(pid: int):
            async with async_session_maker() as s:
                try:
                    n = await on_prompt_created(s, pid)
                    logging.info(f'[prompt-fanout] Prompt {pid} queued for {n} user(s).')
                except Exception:
                    logging.exception(f'[prompt-fanout] on_prompt_created failed for {pid}')
        spawn(_fanout(prompt.id), name='prompt_fanout')
    else:
        logging.info(f'[prompt-fanout] skipped for private-only prompt {prompt.id}')

    async def _fanout(pid: int):
        async with async_session_maker() as s:
            try:
                n = await on_prompt_created(s, pid)
                logging.info(f'[prompt-fanout] Prompt {pid} queued for {n} user(s).')
            except Exception:
                logging.exception(f'[prompt-fanout] on_prompt_created failed for {pid}')
    if request.query_params.get('ajax') == '1':
        return JSONResponse({'ok': True, 'id': prompt.id})
    return RedirectResponse(url='/admin_dashboard', status_code=303)

@router.post('/admin_update_prompt/{prompt_id}')
async def admin_update_prompt(prompt_id: int, request: Request, prompt_text: str=Form(...), chapter: str=Form(...), tags: str=Form('[]'), media_files: list[UploadFile]=File(None), user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    res0 = await db.execute(select(Prompt).where(Prompt.id == prompt_id))
    prompt = res0.scalars().first()
    if not prompt:
        raise HTTPException(status_code=404, detail='Prompt not found')
    prompt.text = prompt_text
    prompt.chapter = chapter

    def _parse_tag_input(raw: str) -> list[str]:
        try:
            data = json.loads(raw) if raw else []
            if not isinstance(data, list):
                data = []
        except Exception:
            data = [p.strip() for p in (raw or '').split(',') if p.strip()]
        out = []
        for item in data:
            if isinstance(item, str):
                out.append(item.strip())
            elif isinstance(item, dict):
                v = (item.get('value') or item.get('slug') or item.get('text') or '').strip()
                if v:
                    out.append(v)
        return list(dict.fromkeys(out))
    tag_names = _parse_tag_input(tags)
    resolved: list[Tag] = []
    for nm in tag_names:
        t = await _get_or_create_tag(db, nm)
        if t:
            resolved.append(t)
    res1 = await db.execute(select(Prompt).options(selectinload(Prompt.tags)).where(Prompt.id == prompt_id))
    prompt = res1.scalars().first()
    if not prompt:
        raise HTTPException(status_code=404, detail='Prompt not found')
    prompt.tags[:] = resolved
    if media_files:
        for file in media_files:
            if not file or not file.filename:
                continue
            new_media = PromptMedia(prompt_id=prompt.id, file_path='', media_type=file.content_type.split('/', 1)[0] if file.content_type else 'file')
            db.add(new_media)
            await db.flush()
            tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(file.filename or '').suffix}"
            with open(tmp, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
            art = PIPELINE.process_upload(temp_path=tmp, logical='prompt', role='prompt', user_slug_or_id=None, prompt_id=prompt.id, response_id=None, media_id=new_media.id, original_filename=file.filename, content_type=file.content_type)
            playable_rel = art.playable_rel
            new_media.file_path = playable_rel[len('uploads/'):] if playable_rel.startswith('uploads/') else playable_rel
            new_media.thumbnail_url = art.thumb_rel
            new_media.mime_type = art.mime_type
            new_media.duration_sec = int(art.duration_sec or 0)
            new_media.sample_rate = art.sample_rate
            new_media.channels = art.channels
            new_media.width = art.width
            new_media.height = art.height
            new_media.size_bytes = art.size_bytes
            new_media.codec_audio = art.codec_a
            new_media.codec_video = art.codec_v
            new_media.wav_path = art.wav_rel
    await db.commit()
    wants_json = request.query_params.get('ajax') == '1' or 'application/json' in (request.headers.get('accept') or '') or request.headers.get('x-requested-with') == 'XMLHttpRequest'
    if wants_json:
        return {'ok': True, 'prompt_id': prompt_id}
    return RedirectResponse(f'/admin_dashboard#prompts?updated_prompt={prompt_id}', status_code=303)

@router.delete('/admin_delete_prompt/{prompt_id}')
async def admin_delete_prompt(prompt_id: int, user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    prompt = (await db.execute(select(Prompt).where(Prompt.id == prompt_id))).scalars().first()
    if not prompt:
        raise HTTPException(status_code=404, detail='Prompt not found')
    media_files = (await db.execute(select(PromptMedia).where(PromptMedia.prompt_id == prompt_id))).scalars().all()
    for media in media_files:
        playable_uploads_rel = _to_uploads_rel_for_playable(media.file_path)
        wav_uploads_rel = media.wav_path or None
        thumb_uploads_rel = media.thumbnail_url or None
        PIPELINE.delete_artifacts(playable_uploads_rel, wav_uploads_rel, thumb_uploads_rel)
        await db.delete(media)
    await db.delete(prompt)
    await db.commit()
    return JSONResponse({'success': True})

@router.get('/admin_edit_prompt/{prompt_id}', response_class=HTMLResponse)
async def admin_edit_prompt_page(prompt_id: int, request: Request, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    res = await db.execute(select(Prompt).options(selectinload(Prompt.tags), selectinload(Prompt.media)).where(Prompt.id == prompt_id))
    p = res.scalars().first()
    if not p:
        raise HTTPException(status_code=404, detail='Prompt not found')
    prompt = {'id': p.id, 'text': p.text or '', 'chapter': p.chapter or ''}
    tag_list = [{'id': t.id, 'slug': getattr(t, 'slug', None) or getattr(t, 'name', '') or '', 'name': getattr(t, 'name', None) or getattr(t, 'slug', '') or '', 'color': getattr(t, 'color', None)} for t in p.tags or []]
    media_list = []
    for m in p.media or []:
        file_url = f'/static/uploads/{m.file_path}' if getattr(m, 'file_path', None) else ''
        thumb_url = f'/static/{m.thumbnail_url}' if getattr(m, 'thumbnail_url', None) else ''
        m_users = []
        try:
            for u in getattr(m, 'assignees', []) or []:
                m_users.append({'id': u.id, 'name': u.username or u.email, 'email': u.email})
        except Exception:
            m_users = []
        media_list.append({'id': m.id, 'file_url': file_url, 'thumbnail_url': thumb_url or file_url, 'mime_type': getattr(m, 'mime_type', None), 'duration_sec': int(getattr(m, 'duration_sec', 0) or 0), 'width': getattr(m, 'width', None), 'height': getattr(m, 'height', None), 'assignees': m_users})
    ups = (await db.execute(select(UserPrompt.user_id).where(UserPrompt.prompt_id == prompt_id))).scalars().all() or []
    assigned_users = []
    if ups:
        users = (await db.execute(select(User).where(User.id.in_(ups)))).scalars().all()
        profs = (await db.execute(select(UserProfile).where(UserProfile.user_id.in_(ups)))).scalars().all()
        pmap = {pr.user_id: pr for pr in profs}
        answered_user_ids = set((uid for uid, in (await db.execute(select(Response.user_id).where(Response.prompt_id == prompt_id, Response.user_id.in_(ups)))).all()))

        def _name(u: User) -> str:
            return (pmap.get(u.id).display_name if pmap.get(u.id) else None) or (u.username or u.email)
        assigned_users = [{'id': u.id, 'name': _name(u), 'email': u.email, 'answered': u.id in answered_user_ids} for u in users]
    ctx = {'request': request, 'user': admin, 'prompt': prompt, 'tag_list': tag_list, 'media_list': media_list, 'assigned_users': assigned_users, 'partial': request.query_params.get('partial') == '1'}
    if request.query_params.get('partial') == '1':
        return templates.TemplateResponse('admin_edit_prompt_partial.html', ctx)
    return templates.TemplateResponse('admin_edit_prompt.html', ctx)

class MediaAssignReq(BaseModel):
    user_id: int

@router.post('/admin/prompt_media/{media_id}/assignees/add')
async def admin_media_assign_add(media_id: int, payload: MediaAssignReq, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    pm = await db.get(PromptMedia, media_id)
    u = await db.get(User, payload.user_id)
    if not pm or not u:
        raise HTTPException(status_code=404, detail='Media or user not found')
    if not any((getattr(x, 'id', None) == u.id for x in pm.assignees or [])):
        pm.assignees.append(u)
    await db.commit()
    return {'ok': True}

@router.post('/admin/prompt_media/{media_id}/assignees/remove')
async def admin_media_assign_remove(media_id: int, payload: MediaAssignReq, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    pm = await db.get(PromptMedia, media_id)
    if not pm:
        raise HTTPException(status_code=404, detail='Media not found')
    pm.assignees = [x for x in pm.assignees or [] if getattr(x, 'id', None) != payload.user_id]
    await db.commit()
    return {'ok': True}

@router.get('/api/admin/prompt/{prompt_id}/media')
async def api_admin_prompt_media(prompt_id: int, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    """Return prompt media with current assignees for building UI."""
    res = await db.execute(select(PromptMedia).options(selectinload(PromptMedia.assignees)).where(PromptMedia.prompt_id == prompt_id))
    items = res.unique().scalars().all() or []
    media = []
    for m in items:
        file_url = f'/static/uploads/{m.file_path}' if getattr(m, 'file_path', None) else ''
        thumb_url = f'/static/{m.thumbnail_url}' if getattr(m, 'thumbnail_url', None) else ''
        m_users = []
        try:
            for u in getattr(m, 'assignees', []) or []:
                m_users.append({'id': u.id, 'name': getattr(u, 'username', None) or getattr(u, 'email', None), 'email': u.email})
        except Exception:
            m_users = []
        media.append({'id': m.id, 'file_url': file_url, 'thumbnail_url': thumb_url or file_url, 'assignees': m_users})
    return {'media': media}

@router.get('/admin_dashboard_partial')
async def admin_dashboard_partial(request: Request, user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    res = await db.execute(select(Prompt).options(selectinload(Prompt.media), selectinload(Prompt.tags)).order_by(Prompt.chapter, Prompt.created_at.desc()))
    prompts = res.scalars().all()
    chapters, tags_map, media_map = ({}, {}, {})
    for p in prompts:
        chapters.setdefault(p.chapter, []).append(p)
        tags_map[p.id] = [{'id': t.id, 'name': t.name, 'slug': t.slug, 'color': t.color} for t in p.tags or []]
        media_map[p.id] = [{'id': m.id, 'file_path': m.file_path, 'media_type': m.media_type, 'thumbnail_url': m.thumbnail_url} for m in p.media or []]
    return templates.TemplateResponse('prompt_list.html', {'request': request, 'user': user, 'chapters': chapters, 'tags_map': tags_map, 'media_map': media_map})

@router.delete('/admin_delete_prompt_media/{media_id}')
async def admin_delete_prompt_media(media_id: int, user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    media = (await db.execute(select(PromptMedia).where(PromptMedia.id == media_id))).scalars().first()
    if not media:
        raise HTTPException(status_code=404, detail='Media not found')
    playable_uploads_rel = _to_uploads_rel_for_playable(media.file_path)
    wav_uploads_rel = media.wav_path or None
    thumb_uploads_rel = media.thumbnail_url or None
    PIPELINE.delete_artifacts(playable_uploads_rel, wav_uploads_rel, thumb_uploads_rel)
    await db.delete(media)
    await db.commit()
    return {'success': True}

@router.post('/admin/prompts/{prompt_id}/media')
async def admin_add_prompt_media(prompt_id: int, media_files: list[UploadFile]=File(...), admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    prompt = await db.get(Prompt, prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail='Prompt not found')
    created = []
    for file in media_files or []:
        if not file or not file.filename:
            continue
        new_media = PromptMedia(prompt_id=prompt.id, file_path='', media_type=file.content_type.split('/', 1)[0] if file.content_type else 'file')
        db.add(new_media)
        await db.flush()
        tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(file.filename or '').suffix}"
        with open(tmp, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        art = PIPELINE.process_upload(temp_path=tmp, logical='prompt', role='prompt', user_slug_or_id=None, prompt_id=prompt.id, response_id=None, media_id=new_media.id, original_filename=file.filename, content_type=file.content_type)
        playable_rel = art.playable_rel
        new_media.file_path = playable_rel[len('uploads/'):] if playable_rel.startswith('uploads/') else playable_rel
        new_media.thumbnail_url = art.thumb_rel
        new_media.mime_type = art.mime_type
        new_media.duration_sec = int(art.duration_sec or 0)
        new_media.sample_rate = art.sample_rate
        new_media.channels = art.channels
        new_media.width = art.width
        new_media.height = art.height
        new_media.size_bytes = art.size_bytes
        new_media.codec_audio = art.codec_a
        new_media.codec_video = art.codec_v
        new_media.wav_path = art.wav_rel
        file_url = f'/static/uploads/{new_media.file_path}' if new_media.file_path else ''
        thumb_url = f'/static/{new_media.thumbnail_url}' if new_media.thumbnail_url else file_url
        created.append({'id': new_media.id, 'file_url': file_url, 'thumbnail_url': thumb_url, 'mime_type': new_media.mime_type, 'width': new_media.width, 'height': new_media.height, 'duration_sec': new_media.duration_sec})
    await db.commit()
    return {'ok': True, 'media': created}

@router.get('/admin/chapters_json')
async def admin_list_chapters_json(user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    rows = await db.execute(select(Prompt.chapter, func.count(Prompt.id)).group_by(Prompt.chapter))
    counts = {r[0] or 'Misc': r[1] for r in rows.all()}
    metas = (await db.execute(select(ChapterMeta))).scalars().all()
    meta_by = {m.name: m for m in metas}
    payload = []
    for name, cnt in counts.items():
        m = meta_by.get(name)
        payload.append({'name': name, 'display_name': m.display_name if m else name or 'Misc', 'order': m.order if m else 999999, 'tint': m.tint if m else None, 'count': cnt, 'description': m.description if m else None, 'keywords': m.keywords if m else None, 'llm_guidance': m.llm_guidance if m else None})
    payload.sort(key=lambda d: (d['order'], d['display_name'].lower()))
    return payload

@router.post('/admin/chapters/reorder')
async def admin_reorder_chapters(items: list[dict]=Body(...), user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    """
    Upsert ChapterMeta rows and persist ordering + visual tint + LLM metadata.
    The UI sends an ordered list; index is the order if 'order' is omitted.
    """

    def _norm_str(val, default=None):
        if val is None:
            return default
        s = str(val).strip()
        return s if s != '' else default
    for idx, it in enumerate(items):
        name = _norm_str(it.get('name'))
        if not name:
            continue
        meta = (await db.execute(select(ChapterMeta).where(ChapterMeta.name == name))).scalars().first()
        if not meta:
            meta = ChapterMeta(name=name, display_name=_norm_str(it.get('display_name'), default=name))
            db.add(meta)
            await db.flush()
        meta.display_name = _norm_str(it.get('display_name'), default=meta.display_name or name)
        meta.order = int(it.get('order') if it.get('order') is not None else idx)
        tint_in = _norm_str(it.get('tint'))
        if tint_in:
            meta.tint = tint_in
        desc_in = it.get('description')
        kw_in = it.get('keywords')
        guide_in = it.get('llm_guidance')
        if desc_in is not None:
            meta.description = _norm_str(desc_in, default=None)
        if kw_in is not None:
            meta.keywords = _norm_str(kw_in, default=None)
        if guide_in is not None:
            meta.llm_guidance = _norm_str(guide_in, default=None)
    await db.commit()
    return {'ok': True}

@router.post('/admin/chapters/rename')
async def admin_rename_chapter(old_name: str=Form(...), new_name: str=Form(...), tint: str | None=Form(None), user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    old_name = old_name.strip()
    new_name = new_name.strip()
    if not old_name or not new_name:
        raise HTTPException(status_code=422, detail='Both old and new chapter names are required.')
    await db.execute(update(Prompt).where(Prompt.chapter == old_name).values(chapter=new_name))
    meta_old = (await db.execute(select(ChapterMeta).where(ChapterMeta.name == old_name))).scalars().first()
    meta_new = (await db.execute(select(ChapterMeta).where(ChapterMeta.name == new_name))).scalars().first()
    if not meta_new:
        meta_new = ChapterMeta(name=new_name, display_name=meta_old.display_name if meta_old and meta_old.display_name else new_name, order=meta_old.order if meta_old else 0, tint=tint or (meta_old.tint if meta_old else None), description=meta_old.description if meta_old else None, keywords=meta_old.keywords if meta_old else None, llm_guidance=meta_old.llm_guidance if meta_old else None)
        db.add(meta_new)
    elif tint:
        meta_new.tint = tint
    if meta_old and meta_old.name != meta_new.name:
        await db.delete(meta_old)
    await db.commit()
    return RedirectResponse(url='/admin_dashboard', status_code=303)

@router.get('/_email/health')
async def email_health():
    import smtplib, ssl, os
    host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
    port = int(os.getenv('SMTP_PORT', '587'))
    user = os.getenv('GMAIL_USER')
    pwd = os.getenv('GMAIL_PASS')
    ctx = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=10) as s:
        s.ehlo()
        s.starttls(context=ctx)
        s.ehlo()
        s.login(user, pwd)
    return {'ok': True}

async def _ensure_profile(db: AsyncSession, user_id: int) -> UserProfile:
    prof = await db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    if not prof:
        prof = UserProfile(user_id=user_id, tag_weights={'tagWeights': {}}, privacy_prefs={'onboarding': {'step': 'you', 'done': False}})
        db.add(prof)
        await db.flush()
        return prof
    if not prof.tag_weights:
        prof.tag_weights = {'tagWeights': {}}
    if not prof.privacy_prefs or not isinstance(prof.privacy_prefs, dict):
        prof.privacy_prefs = {'onboarding': {'step': 'you', 'done': False}}
    else:
        prof.privacy_prefs.setdefault('onboarding', {'step': 'you', 'done': False})
    return prof
@router.get('/api/admin/tags/whitelist')
async def get_tag_whitelist(admin=Depends(require_admin_user)):
    try:
        raw = TAG_WL_PATH.read_text(encoding='utf-8')
        data = json.loads(raw)
    except Exception:
        data = []
    return JSONResponse(data)

@router.post('/api/admin/tags/whitelist')
async def save_tag_whitelist(payload: Any=Body(...), admin=Depends(require_admin_user)):
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            raise HTTPException(status_code=400, detail='Invalid JSON string body')
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail='Whitelist must be a JSON array')
    try:
        TAG_WL_PATH.parent.mkdir(parents=True, exist_ok=True)
        TAG_WL_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to write whitelist: {e}')
    try:
        from app.services.auto_tag import reload_whitelist as _reload
        _reload()
    except Exception:
        try:
            from app.services.auto_tag import _load_whitelist, WHITELIST as WL
            WL.clear()
            WL.update(_load_whitelist())
        except Exception:
            pass
    return {'ok': True, 'count': len(payload)}

def _role_options_from_whitelist() -> list[str]:
    roles = []
    for t in WHITELIST:
        if t.startswith('relationship:'):
            roles.append(t.split(':', 1)[1])
    roles = sorted(set(roles))
    return roles

def _push_graylist(privacy_prefs: dict, slugs: list[str]) -> None:
    if not slugs:
        return
    gp = privacy_prefs.setdefault('graylist_tags', [])
    existing = set(gp)
    for s in slugs:
        if s not in existing:
            gp.append(s)
            existing.add(s)

def _load_role_options_from_whitelist() -> list[str]:
    """
    Read /app/data/tag_whitelist.json and return a de-duplicated, sorted list of role names,
    derived from entries whose 'value' starts with 'relationship:'.

    Example item in JSON:
      {"value":"relationship:grandmother","label":"Relationship · Grandmother"}

    Returned list example:
      ["adopted-child","adopted-daughter","adopted-father", ... , "uncle"]
    """
    wl_path = Path(__file__).resolve().parents[1] / 'data' / 'tag_whitelist.json'
    try:
        data = json.loads(wl_path.read_text(encoding='utf-8'))
    except Exception:
        return ['father', 'mother', 'grandfather', 'grandmother', 'uncle', 'aunt', 'cousin', 'stepfather', 'stepmother', 'sibling', 'friend']
    roles: set[str] = set()
    if isinstance(data, list):
        for item in data:
            val = None
            if isinstance(item, dict):
                val = item.get('value')
            elif isinstance(item, str):
                val = item
            if not isinstance(val, str):
                continue
            val = val.strip()
            if not val:
                continue
            if val.startswith('relationship:'):
                roles.add(val.split(':', 1)[1])
    out = sorted(roles)
    if 'friend' not in out:
        out.append('friend')
    return out

def _normalize_role_input(value: str) -> str:
    """
    Accepts 'relationship:mother', 'role:mother', or 'mother' and returns the standardized
    tag slug 'role:mother' (lowercased, hyphenated preserved).
    """
    from app.utils import slug_role, slugify
    v = (value or '').strip().lower()
    if not v:
        return ''
    if v.startswith('relationship:'):
        v = v.split(':', 1)[1]
    elif v.startswith('role:'):
        v = v.split(':', 1)[1]
    v = slugify(v)
    return slug_role(v)

def _coerce_roles_payload(payload: object) -> list[str]:
    """
    Accepts a Tagify array or {"roles":[...]} (items can be strings or objects),
    and returns a list of standardized 'role:*' slugs.
    """
    items = payload if isinstance(payload, list) else payload.get('roles') or []
    out: list[str] = []
    for it in items:
        if isinstance(it, dict):
            raw = it.get('value') or it.get('slug') or it.get('text') or it.get('name') or ''
        else:
            raw = str(it or '')
        tag = _normalize_role_input(raw)
        if tag:
            out.append(tag)
    return list(dict.fromkeys(out))

@router.get('/onboarding', response_class=HTMLResponse)
async def onboarding_home(request: Request, db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_html_user)):
    prof = await _ensure_profile(db, user.id)
    prof = await db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
    ob = (prof.privacy_prefs or {}).get('onboarding') or {}
    step = ob.get('step') or 'you'
    print(f'[onboarding] user={user.id} step={step}')
    if ob.get('done'):
        return RedirectResponse(url='/onboarding/done', status_code=303)
    role_options = _role_options_from_whitelist()
    return templates.TemplateResponse('onboarding_steps.html', {'request': request, 'user': user, 'step': step, 'role_options': role_options, 'profile': prof})
from sqlalchemy import update, select
from sqlalchemy.orm.attributes import flag_modified

@router.post('/onboarding/you')
async def onboarding_you(payload: dict=Body(...), db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    prof = await _ensure_profile(db, user.id)
    display_name = (payload.get('display_name') or '').strip()
    gender = (payload.get('gender') or '').strip()
    birthdate = (payload.get('birthdate') or '').strip()
    location = (payload.get('location') or '').strip()
    if display_name:
        prof.display_name = display_name
    if location:
        prof.location = location
    if birthdate and len(birthdate) >= 4 and birthdate[:4].isdigit():
        prof.birth_year = int(birthdate[:4])
    pp = dict(prof.privacy_prefs or {})
    user_meta = dict(pp.get('user_meta') or {})
    if gender:
        user_meta['gender'] = gender
    pp['user_meta'] = user_meta
    ob = dict(pp.get('onboarding') or {'step': 'you', 'done': False})
    ob['step'] = 'roles'
    pp['onboarding'] = ob
    prof.privacy_prefs = pp
    flag_modified(prof, 'privacy_prefs')
    await db.commit()
    return RedirectResponse(url='/onboarding', status_code=303)

@router.post('/onboarding/roles')
async def onboarding_roles(payload: dict=Body(...), db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    prof = await _ensure_profile(db, user.id)
    tw = dict(prof.tag_weights or {'tagWeights': {}})
    weights = tw.setdefault('tagWeights', {})
    incoming = payload if isinstance(payload, list) else payload.get('roles') or []
    cleaned_roles = []
    off_whitelist = []
    wl_roles = set(_role_options_from_whitelist())
    for r in incoming:
        v = (r.get('value') if isinstance(r, dict) else str(r or '')).strip()
        if not v:
            continue
        if v.startswith('relationship:'):
            v = v.split(':', 1)[1]
        if v not in wl_roles:
            off_whitelist.append(f'role:{slugify(v)}')
        cleaned_roles.append(v)
    role_slugs = [slug_role(r) for r in cleaned_roles]
    for rs in role_slugs:
        weights[rs] = max(weights.get(rs, 0.0), 0.7)
        await _get_or_create_tag(db, rs)
    prof.tag_weights = tw
    pp = dict(prof.privacy_prefs or {'onboarding': {'step': 'you', 'done': False}})
    _push_graylist(pp, off_whitelist)
    ob = pp.setdefault('onboarding', {'step': 'you', 'done': False})
    ob['step'] = 'family'
    pp = dict(prof.privacy_prefs or {})
    pp['onboarding'] = dict(pp.get('onboarding') or {})
    pp['onboarding']['step'] = 'family'
    prof.privacy_prefs = pp
    flag_modified(prof, 'privacy_prefs')
    await db.commit()
    return {'ok': True, 'next': 'family', 'roles': role_slugs}

@router.post('/onboarding/family')
async def onboarding_family(payload: dict | list=Body(...), db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    from app.services.people import upsert_person_for_user
    prof = await _ensure_profile(db, user.id)
    tw = dict(prof.tag_weights or {'tagWeights': {}})
    weights = tw.setdefault('tagWeights', {})
    items = payload if isinstance(payload, list) else payload.get('family') or []
    wl_roles = set(_role_options_from_whitelist())
    created = []
    gray_add = []
    for row in items:
        name = (row.get('name') or '').strip()
        role = (row.get('role') or '').strip()
        if not name:
            continue
        person = await upsert_person_for_user(db, owner_user_id=user.id, display_name=name, role_hint=role or None)
        p_tag = slug_person(name)
        weights[p_tag] = max(weights.get(p_tag, 0.0), 0.9)
        await _get_or_create_tag(db, p_tag)
        if role:
            base_role = role.split(':', 1)[1] if role.startswith(('relationship:', 'role:')) else role
            r_tag = slug_role(base_role)
            weights[r_tag] = max(weights.get(r_tag, 0.0), 0.7)
            await _get_or_create_tag(db, r_tag)
            if base_role not in wl_roles:
                gray_add.append(r_tag)
        created.append({'person_id': person.id, 'name': name, 'role': role})
    prof.tag_weights = tw
    pp = dict(prof.privacy_prefs or {'onboarding': {'step': 'you', 'done': False}})
    _push_graylist(pp, gray_add)
    ob = pp.setdefault('onboarding', {'step': 'you', 'done': False})
    ob['step'] = 'places'
    pp = dict(prof.privacy_prefs or {})
    pp['onboarding'] = dict(pp.get('onboarding') or {})
    pp['onboarding']['step'] = 'places'
    prof.privacy_prefs = pp
    flag_modified(prof, 'privacy_prefs')
    await db.commit()
    return {'ok': True, 'next': 'places', 'created': created}

@router.post('/onboarding/places')
async def onboarding_places(payload: dict | list=Body(...), db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    prof = await _ensure_profile(db, user.id)
    tw = dict(prof.tag_weights or {'tagWeights': {}})
    weights = tw.setdefault('tagWeights', {})
    items = payload if isinstance(payload, list) else payload.get('places') or []
    place_slugs = []
    gray_add = []
    for p in items:
        s = str(p or '').strip()
        if not s:
            continue
        ps = slug_place(s)
        place_slugs.append(ps)
        weights[ps] = max(weights.get(ps, 0.0), 0.5)
        await _get_or_create_tag(db, ps)
        gray_add.append(ps)
    prof.tag_weights = tw
    pp = dict(prof.privacy_prefs or {'onboarding': {'step': 'you', 'done': False}})
    _push_graylist(pp, gray_add)
    ob = pp.setdefault('onboarding', {'step': 'you', 'done': False})
    ob['step'] = 'interests'
    pp = dict(prof.privacy_prefs or {})
    pp['onboarding'] = dict(pp.get('onboarding') or {})
    pp['onboarding']['step'] = 'interests'
    prof.privacy_prefs = pp
    flag_modified(prof, 'privacy_prefs')
    await db.commit()
    return {'ok': True, 'next': 'interests', 'places': place_slugs}

@router.post('/onboarding/interests')
async def onboarding_interests(payload: dict | list=Body(...), db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    prof = await _ensure_profile(db, user.id)
    tw = dict(prof.tag_weights or {'tagWeights': {}})
    weights = tw.setdefault('tagWeights', {})
    items = payload if isinstance(payload, list) else payload.get('interests') or []
    interest_slugs = []
    gray_add = []
    for raw in items:
        s = (raw.get('value') if isinstance(raw, dict) else str(raw or '')).strip()
        if not s:
            continue
        ts = s if ':' in s else slugify(s)
        interest_slugs.append(ts)
        weights[ts] = max(weights.get(ts, 0.0), 0.6)
        await _get_or_create_tag(db, ts)
        gray_add.append(ts)
    prof.tag_weights = tw
    pp = dict(prof.privacy_prefs or {'onboarding': {'step': 'you', 'done': False}})
    _push_graylist(pp, gray_add)
    ob = pp.setdefault('onboarding', {'step': 'you', 'done': False})
    ob['step'] = 'preview'
    pp = dict(prof.privacy_prefs or {})
    pp['onboarding'] = dict(pp.get('onboarding') or {})
    pp['onboarding']['step'] = 'preview'
    prof.privacy_prefs = pp
    flag_modified(prof, 'privacy_prefs')
    await db.commit()
    return {'ok': True, 'next': 'preview', 'interests': interest_slugs}

@router.post('/onboarding/commit')
async def onboarding_commit(db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    prof = await _ensure_profile(db, user.id)
    try:
        me_person = (await db.execute(select(Person).where(Person.owner_user_id == user.id))).scalars().all()
        have_self = False
        for p in me_person:
            m = getattr(p, 'meta', None)
            if isinstance(m, dict) and m.get('connect_to_owner') and (str(m.get('role_hint', '')).strip().lower() in {'you', 'self', 'me'}):
                have_self = True
                break
        if not have_self:
            disp = getattr(prof, 'display_name', None) or (user.username or user.email or f'User {user.id}')
            g = None
            try:
                g = ((prof.privacy_prefs or {}).get('user_meta') or {}).get('gender')
            except Exception:
                pass
            meta = {'connect_to_owner': True, 'role_hint': 'you'}
            if g:
                meta['gender'] = g
            person = Person(owner_user_id=user.id, display_name=disp, meta=meta)
            try:
                if getattr(prof, 'birth_year', None):
                    person.birth_year = int(prof.birth_year)
            except Exception:
                pass
            db.add(person)
            try:
                await db.flush()
            except Exception:
                await db.rollback()
    except Exception:
        pass
    await build_pool_for_user(db, user.id)
    await ensure_weekly_prompt(db, user.id)
    ob = prof.privacy_prefs.setdefault('onboarding', {'step': 'you', 'done': False})
    ob['done'] = True
    ob['step'] = 'done'
    await db.commit()
    return RedirectResponse(url='/user_dashboard', status_code=303)

@router.get('/admin/suggestions', response_class=HTMLResponse)
async def admin_suggestions(request: Request, q: str | None=None, user_id: int | None=None, tag: str | None=None, status: str | None=None, user=Depends(require_authenticated_html_user), db: AsyncSession=Depends(get_db)):
    stmt = select(PromptSuggestion).order_by(PromptSuggestion.created_at.desc())
    rows = (await db.execute(stmt)).scalars().all()
    return templates.TemplateResponse('admin_suggestions.html', {'request': request, 'user': user, 'rows': rows})

@router.post('/admin/suggestions/{sid}/approve')
async def admin_suggestion_approve(sid: int, user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    s = await db.get(PromptSuggestion, sid)
    if not s:
        raise HTTPException(404, 'Not found')
    s.status = 'approved'
    await db.commit()
    return {'ok': True}

@router.post('/admin/suggestions/{sid}/reject')
async def admin_suggestion_reject(sid: int, user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    s = await db.get(PromptSuggestion, sid)
    if not s:
        raise HTTPException(404, 'Not found')
    s.status = 'rejected'
    await db.commit()
    return {'ok': True}

@router.post('/admin/users/{user_id}/delete')
async def admin_delete_user(user_id: int, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    target = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if not target:
        raise HTTPException(status_code=404, detail='User not found')
    if getattr(admin, 'id', None) == target.id:
        raise HTTPException(status_code=400, detail='You cannot delete your own account from admin.')
    if getattr(target, 'is_superuser', False) and (not getattr(admin, 'is_superuser', False)):
        raise HTTPException(status_code=403, detail='Only a super admin can delete a superuser.')
    await db.delete(target)
    await db.commit()
    return RedirectResponse(url='/admin_dashboard?notice=User+deleted', status_code=303)

@router.post('/admin/users/{user_id}/anonymize')
async def admin_anonymize_user(user_id: int, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    target = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if not target:
        raise HTTPException(status_code=404, detail='User not found')
    if getattr(admin, 'id', None) == target.id:
        raise HTTPException(status_code=400, detail='You cannot anonymize yourself from admin.')
    if getattr(target, 'is_superuser', False) and (not getattr(admin, 'is_superuser', False)):
        raise HTTPException(status_code=403, detail='Only a super admin can anonymize a superuser.')
    import secrets
    suffix = secrets.token_hex(4)
    target.username = f'deleted_user_{target.id}_{suffix}'
    target.email = f'deleted+{target.id}.{suffix}@example.invalid'
    if hasattr(target, 'hashed_password'):
        target.hashed_password = ''
    if hasattr(target, 'is_active'):
        target.is_active = False
    await db.commit()
    return RedirectResponse(url='/admin_dashboard?notice=User+anonymized', status_code=303)

@router.get('/api/chapters_meta')
async def api_chapters_meta(db: AsyncSession=Depends(get_db)):
    metas = (await db.execute(select(ChapterMeta))).scalars().all()
    counts_rows = await db.execute(select(Prompt.chapter, func.count(Prompt.id)).group_by(Prompt.chapter))
    counts = {r[0] or 'Misc': r[1] for r in counts_rows.all()}
    out = []
    for m in metas:
        out.append({'name': m.name, 'display_name': m.display_name, 'order': m.order, 'tint': m.tint, 'count': counts.get(m.name, 0), 'description': m.description, 'keywords': m.keywords, 'llm_guidance': m.llm_guidance})
    out.sort(key=lambda d: (d['order'], d['display_name'].lower()))
    return out

@router.post('/api/polish-transcript')
async def api_polish_transcript(request: Request):
    body = await request.json()
    text = (body or {}).get('text', '')
    style = (body or {}).get('style', 'clean')
    if not text.strip():
        return JSONResponse({'text': ''}, status_code=200)
    try:
        cleaned = await polish_text(text, style=style)
        return {'text': cleaned}
    except OllamaError as e:
        return JSONResponse({'error': str(e)}, status_code=502)

@router.get('/api/response/{response_id}/transcript')
async def api_get_transcript(response_id: int=Path(...), user=Depends(current_active_user)):
    async with async_session_maker() as session:
        result = await session.execute(select(Response).where(Response.id == response_id))
        resp = result.unique().scalar_one_or_none()
        if not resp:
            return JSONResponse({'error': 'Not found'}, status_code=404)
        if not getattr(user, 'is_superuser', False) and resp.user_id != user.id:
            return JSONResponse({'error': 'Forbidden'}, status_code=403)
        return {'text': (resp.transcription or '').strip()}

@router.post('/response/{response_id}/primary')
async def replace_primary(response_id: int, request: Request, primary_media: UploadFile=File(...), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    result = await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))
    resp = result.scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    PIPELINE.delete_artifacts(_to_uploads_rel_for_playable(resp.primary_media_url) if resp.primary_media_url else None, resp.primary_wav_path, resp.primary_thumbnail_path)
    tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(primary_media.filename or '').suffix}"
    with open(tmp, 'wb') as w:
        shutil.copyfileobj(primary_media.file, w)
    spawn(_process_primary_async(resp.id, tmp, primary_media.filename or 'primary', primary_media.content_type, user.username or str(user.id)), name='reprocess_primary_media')
    return JSONResponse({'response': {'id': resp.id}, 'queued': True})

@router.get('/admin/tags')
async def admin_list_tags(q: str | None=None, limit: int=20, user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    stmt = select(Tag).order_by(Tag.name)
    if q:
        ql = f'%{q.lower()}%'
        stmt = stmt.where(or_(Tag.name.ilike(ql), Tag.slug.ilike(ql)))
    tags = (await db.execute(stmt.limit(limit))).scalars().all()
    return [{'id': t.id, 'name': t.name, 'slug': t.slug, 'color': t.color} for t in tags]

@router.post('/admin/tags')
async def admin_create_tag(name: str=Form(...), color: str | None=Form(None), user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    name = name.strip()
    slug = _slugify(name)
    existing = (await db.execute(select(Tag).where(or_(Tag.slug == slug, Tag.name.ilike(name))))).scalar_one_or_none()
    if existing:
        return {'id': existing.id, 'name': existing.name, 'slug': existing.slug, 'color': existing.color}
    t = Tag(name=name, slug=slug, color=color)
    db.add(t)
    await db.commit()
    await db.refresh(t)
    return {'id': t.id, 'name': t.name, 'slug': t.slug, 'color': t.color}

@router.post('/admin/prompts/{prompt_id}/tags')
async def admin_set_prompt_tags(prompt_id: int, payload: Any=Body(...), user=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    tag_names: List[str] = payload if isinstance(payload, list) else payload.get('tags') or []
    tag_names = [(s or '').strip() for s in tag_names if (s or '').strip()]
    tags: list[Tag] = []
    for nm in tag_names:
        t = await _get_or_create_tag(db, nm)
        if t:
            tags.append(t)
    prompt = (await db.execute(select(Prompt).options(selectinload(Prompt.tags)).where(Prompt.id == prompt_id))).scalars().first()
    if not prompt:
        raise HTTPException(404, 'Prompt not found')
    prompt.tags[:] = tags
    await db.commit()
    return {'ok': True, 'tags': [{'id': t.id, 'name': t.name, 'slug': t.slug, 'color': t.color} for t in tags]}

@router.post('/response/{response_id}/tags')
async def set_response_tags(response_id: int, payload: Any=Body(...), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    resp = (await db.execute(select(Response).options(selectinload(Response.tags)).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    try:
        cur_tags = [t.slug for t in resp.tags or []]
        ver = ResponseVersion(response_id=resp.id, user_id=user.id, title=getattr(resp, 'title', None), transcription=resp.transcription, tags_json={'tags': cur_tags}, edited_by_admin_id=None)
        db.add(ver)
    except Exception:
        pass
    raw = payload if isinstance(payload, list) else payload.get('tags') if isinstance(payload, dict) else []
    tag_names = [(s or '').strip() for s in raw or [] if (s or '').strip()]
    tags: list[Tag] = []
    for nm in tag_names:
        t = await _get_or_create_tag(db, nm)
        if t:
            tags.append(t)
    assoc_tbl = Response.tags.property.secondary
    await db.execute(delete(assoc_tbl).where(assoc_tbl.c.response_id == response_id))
    if tags:
        ids = sorted({t.id for t in tags})
        stmt = pg_insert(assoc_tbl).on_conflict_do_nothing(index_elements=[assoc_tbl.c.response_id, assoc_tbl.c.tag_id])
        await db.execute(stmt, [{'response_id': response_id, 'tag_id': tid} for tid in ids])
    await db.commit()
    return {'ok': True, 'tags': [{'id': t.id, 'name': t.name, 'slug': t.slug, 'color': t.color} for t in tags]}

@router.get('/response/{response_id}/tags')
async def set_response_tags(response_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    response = (await db.execute(select(Response).options(selectinload(Response.tags)).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    return {'tags': [{'id': t.id, 'name': t.name, 'slug': t.slug, 'color': t.color} for t in response.tags or []]}

@router.get('/response/{response_id}/segments')
async def list_response_segments(response_id: int, db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    await _ensure_response_owned(db, response_id, user.id)
    rows = (await db.execute(select(ResponseSegment).where(ResponseSegment.response_id == response_id).order_by(ResponseSegment.order_index.asc(), ResponseSegment.id.asc()))).scalars().all()
    out = []
    for s in rows:
        out.append({'id': s.id, 'order_index': getattr(s, 'order_index', None), 'transcript': getattr(s, 'transcript', '') or '', 'media_path': getattr(s, 'media_path', '') or '', 'media_mime': getattr(s, 'media_mime', '') or '', 'thumbnail_path': getattr(s, 'thumbnail_path', None) or getattr(s, 'thumbnail_url', None) or ''})
    return out

@router.post('/response/{response_id}/segments')
async def add_response_segment(response_id: int, file: UploadFile=File(...), note: str | None=Form(None), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    """
    Upload a *new segment* for this response.
    - Stores the media under users/<user>/responses/<id>/supporting/<segment_id>/
    - DOES NOT touch resp.primary_media_url (primary changes only on merge)
    - Returns JSON { ok, segment:{...} } so the overlay JS can poll for transcript
    """
    resp = await _ensure_response_owned(db, response_id, user.id)
    order = await _max_order_index(db, response_id)
    seg = ResponseSegment(response_id=response_id, order_index=order + 1, transcript='')
    db.add(seg)
    await db.flush()
    tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(file.filename or '').suffix}"
    with open(tmp, 'wb') as w:
        shutil.copyfileobj(file.file, w)
    art = PIPELINE.process_upload(temp_path=tmp, logical='response', role='supporting', user_slug_or_id=user.username or str(user.id), prompt_id=None, response_id=response_id, media_id=seg.id, original_filename=file.filename or 'segment', content_type=file.content_type)
    playable_rel = (art.playable_rel or '').lstrip('/').replace('\\', '/')
    if playable_rel.startswith('uploads/'):
        playable_rel = playable_rel[len('uploads/'):]
    seg.media_path = playable_rel
    seg.media_mime = art.mime_type or None
    if not (resp.primary_media_url or '').strip():
        resp.primary_media_url = playable_rel
        resp.primary_mime_type = seg.media_mime or 'audio/mp4'
        resp.primary_thumbnail_path = None
        resp.primary_duration_sec = None
    await db.commit()
    try:
        uploads_rel_for_tx = 'uploads/' + seg.media_path
        spawn(transcribe_segment_and_update(seg.id, uploads_rel_for_tx), name='transcribe_segment_new')
    except Exception:
        pass
    return {'segment': {'id': seg.id}}

@router.patch('/response/{response_id}/segments/reorder')
async def reorder_response_segments(response_id: int, payload: ReorderSegmentsRequest=Body(...), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    await _ensure_response_owned(db, response_id, user.id)
    ids = payload.order or []
    if not ids:
        return {'ok': True}
    found = (await db.execute(select(ResponseSegment.id).where(ResponseSegment.response_id == response_id, ResponseSegment.id.in_(ids)))).scalars().all()
    if set(found) != set(ids):
        raise HTTPException(status_code=400, detail='Invalid segment ids for this response.')
    order_map = {sid: idx + 1 for idx, sid in enumerate(ids)}
    rows = (await db.execute(select(ResponseSegment).where(ResponseSegment.id.in_(ids)))).scalars().all()
    for r in rows:
        r.order_index = order_map.get(r.id, r.order_index)
    await db.commit()
    return {'ok': True}

@router.delete('/response/{response_id}/segments/{segment_id}')
async def delete_response_segment(response_id: int, segment_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    await _ensure_response_owned(db, response_id, user.id)
    seg = await db.get(ResponseSegment, segment_id)
    if not seg or seg.response_id != response_id:
        raise HTTPException(status_code=404, detail='Segment not found')
    playable_uploads_rel = seg.media_path
    if playable_uploads_rel:
        playable_uploads_rel = _to_uploads_rel_for_playable(playable_uploads_rel)
    try:
        PIPELINE.delete_artifacts(playable_uploads_rel, None, None)
    except Exception:
        pass
    await db.delete(seg)
    await db.commit()
    return {'ok': True}

@router.post('/response/{response_id}/segments/bootstrap')
async def bootstrap_first_segment(response_id: int, db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    resp = await db.get(Response, response_id)
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    rows = await db.execute(select(ResponseSegment.id).where(ResponseSegment.response_id == response_id))
    if rows.first():
        return JSONResponse({'ok': True, 'created': False})
    primary_rel = (resp.primary_media_url or '').lstrip('/').replace('\\', '/')
    if not primary_rel:
        return JSONResponse({'ok': True, 'created': False})
    if primary_rel.startswith('static/'):
        primary_rel = primary_rel[len('static/'):]
    if primary_rel.startswith('uploads/'):
        primary_rel = primary_rel[len('uploads/'):]
    if primary_rel.startswith(f'responses/{response_id}/') and primary_rel.split('/')[-1].startswith('composite-'):
        return JSONResponse({'ok': True, 'created': False})
    seg = ResponseSegment(response_id=response_id, order_index=0, media_path=primary_rel, media_mime=resp.primary_mime_type or None, transcript='')
    db.add(seg)
    await db.commit()
    try:
        uploads_rel = primary_rel if primary_rel.startswith('uploads/') else f'uploads/{primary_rel}'
        spawn(transcribe_segment_and_update(seg.id, uploads_rel), name='transcribe_segment_bootstrap')
    except Exception:
        pass
    return JSONResponse({'ok': True, 'created': True, 'segment_id': seg.id})

@router.post('/response/{response_id}/segments/merge-audio')
async def merge_audio_to_primary(response_id: int, db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    resp = await db.get(Response, response_id)
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    rows = await db.execute(select(ResponseSegment).where(ResponseSegment.response_id == response_id).order_by(ResponseSegment.order_index.asc()))
    segs = [s for s, in rows]
    sources: list[str] = []
    for s in segs:
        rel = (s.media_path or '').lstrip('/').replace('\\', '/')
        if not rel:
            continue
        is_composite = rel.startswith(f'responses/{response_id}/') and rel.split('/')[-1].startswith('composite-')
        if is_composite:
            continue
        if rel.startswith('uploads/'):
            rel = rel[len('uploads/'):]
        sources.append(rel)
    if not sources:
        return JSONResponse({'ok': False, 'error': 'no segments to merge'}, status_code=400)
    uploads_root = PIPELINE.static_root / 'uploads'
    bad = []
    for rel in sources:
        p = (uploads_root / rel).resolve()
        if not p.exists() or p.stat().st_size == 0:
            bad.append(rel)
    if bad:
        raise HTTPException(status_code=400, detail={'error': 'missing_or_empty', 'files': bad})
    try:
        out = await PIPELINE.concat_audio_async(sources_rel=sources, response_id=response_id)
    except Exception as e:
        import traceback, logging
        logging.exception('concat_audio_async crashed')
        raise HTTPException(status_code=500, detail='merge crashed')
    if not out.get('ok'):
        raise HTTPException(status_code=500, detail=out.get('error', 'merge failed'))
    dest_rel = out['dest_rel']
    resp.primary_media_url = dest_rel
    resp.primary_mime_type = 'audio/mp4'
    resp.primary_thumbnail_path = None
    resp.primary_duration_sec = None
    await db.commit()
    try:
        if old_primary_rel and old_primary_rel not in sources and (old_primary_rel != resp.primary_media_url):
            await asyncio.to_thread(PIPELINE.delete_artifacts, old_primary_rel)
    except Exception:
        pass
    return JSONResponse({'ok': True, 'primary': resp.primary_media_url})

class AutoTagReq(BaseModel):
    text: str
    word_count: int | None = None
    language_code: str | None = None
    mode: str | None = None

class AutoTagResp(BaseModel):
    tags: list[dict]

@router.get('/api/admin/tags/suggest')
async def api_admin_tags_suggest(prompt_text: str=Query(..., min_length=3), chapter: str | None=Query(None), k: int=Query(12, ge=1, le=50), db: AsyncSession=Depends(get_db)):
    """
    Returns suggested tags for a new Prompt. Format:
    [
      {"label": "grandparents", "slug": "grandparents", "confidence": 0.86, "source": "llm"},
      {"label": "small-town",  "slug": "small-town",  "confidence": 0.71, "source": "kw"}
    ]
    """
    items = await suggest_prompt_tags(prompt_text=prompt_text, chapter=chapter, k=k, db=db)
    return {'items': items}

@router.post('/api/auto_tag/preview', response_model=AutoTagResp)
async def auto_tag_preview(payload: AutoTagReq) -> AutoTagResp:
    """
    PREVIEW-ONLY endpoint to suggest tags for the editor UI.
    ❗️Does NOT persist or mutate Prompt/Tag relations.
    Your approval happens when the editor saves the prompt with the chosen tags.
    """
    text = (payload.text or '').strip()
    if not text:
        raise HTTPException(status_code=400, detail='text is required')
    is_prompt = (payload.mode or '').lower() == 'prompt'
    if not is_prompt:
        is_prompt = _is_prompt_like(text)
    if is_prompt:
        pairs = await suggest_tags_for_prompt(text, word_count=payload.word_count, language_code=payload.language_code)
    else:
        pairs = suggest_tags_rule_based(text, word_count=payload.word_count, language_code=payload.language_code)
    return AutoTagResp(tags=[{'value': slug, 'score': round(score, 2)} for slug, score in pairs])

class AssignPreviewReq(BaseModel):
    user_id: int
    limit: int = 5
    must: list[str] = []
    nice: list[str] = []

class AssignItem(BaseModel):
    prompt_id: int
    chapter: str | None = None
    score: float
    tags: list[str]
    reasons: list[str] = []

class AssignPreviewResp(BaseModel):
    user_id: int
    candidates: list[AssignItem]

@router.post('/api/assignments/preview', response_model=AssignPreviewResp)
async def assignments_preview(payload: AssignPreviewReq, db: AsyncSession=Depends(get_db)):
    profile = {'tagWeights': {}, 'recentHistory': [], 'targets': {}}
    base = select(Prompt).join(prompt_tags).join(Tag)
    if payload.must:
        sub = select(prompt_tags.c.prompt_id).join(Tag, Tag.id == prompt_tags.c.tag_id).where(Tag.slug.in_(payload.must)).group_by(prompt_tags.c.prompt_id).having(func.count(func.distinct(Tag.slug)) == len(payload.must))
        base = base.where(Prompt.id.in_(sub))
    if payload.nice:
        base = base.where(Prompt.id.in_(select(prompt_tags.c.prompt_id).join(Tag).where(Tag.slug.in_(payload.nice))))
    prompts = (await db.execute(base)).scalars().unique().all()
    scored = []
    for p in prompts:
        s = score_prompt(p, profile.get('tagWeights', {}), profile.get('recentHistory', []), profile.get('targets', {}))
        scored.append((p, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    out = []
    for p, s in scored[:payload.limit]:
        out.append({'prompt_id': p.id, 'chapter': p.chapter, 'score': round(s, 3), 'tags': [t.slug for t in p.tags], 'reasons': []})
    return {'user_id': payload.user_id, 'candidates': out}

class AssignCommitReq(BaseModel):
    user_id: int
    prompt_ids: List[int]
    period: str = 'week'

@router.post('/api/assignments/commit')
async def assignments_commit(payload: AssignCommitReq, db: AsyncSession=Depends(get_db)):
    return {'ok': True, 'locked': {'period': payload.period, 'prompt_ids': payload.prompt_ids}}

class ResolvePersonReq(BaseModel):
    display_name: str
    role_hint: str | None = None

@router.post('/api/people/resolve')
async def api_people_resolve(payload: ResolvePersonReq, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p = await resolve_person(db, user.id, payload.display_name)
    await db.commit()
    return {'person_id': p.id, 'created': True}

class PersonCreateReq(BaseModel):
    display_name: str
    role_hint: str | None = None

class AssignMeReq(BaseModel):
    person_id: int

@router.post('/api/people/me/assign')
async def api_people_assign_me(payload: AssignMeReq, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    """Assign the given person as the current user's 'You' node.
    Clears the flag on any other Person records owned by the user.
    """
    pid = int(payload.person_id)
    p: Person | None = await db.get(Person, pid)
    if not p or p.owner_user_id != user.id:
        raise HTTPException(404, 'Person not found')
    rows = (await db.execute(select(Person).where(Person.owner_user_id == user.id))).scalars().all()
    for row in rows:
        try:
            m = dict(row.meta or {})
        except Exception:
            m = {}
        if row.id == pid:
            m['connect_to_owner'] = True
            m['role_hint'] = 'you'
        elif m.get('connect_to_owner'):
            m['connect_to_owner'] = False
        row.meta = m
        try:
            flag_modified(row, 'meta')
        except Exception:
            pass
    await db.commit()
    return {'ok': True, 'me_person_id': pid}

@router.post('/api/people/add')
async def api_people_add(payload: PersonCreateReq, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    """
    Create or upsert a confirmed Person for the current user.
    Ensures aliases are recorded and marks as connected to the owner so it shows on the graph immediately.
    """
    name = (payload.display_name or '').strip()
    if not name:
        raise HTTPException(422, 'display_name is required')
    p = await upsert_person_for_user(db, owner_user_id=user.id, display_name=name, role_hint=payload.role_hint or None)
    try:
        meta = dict(p.meta or {})
    except Exception:
        meta = {}
    meta['inferred'] = False
    meta['hidden'] = False
    meta['connect_to_owner'] = True
    if payload.role_hint:
        meta['role_hint'] = payload.role_hint
    p.meta = meta
    try:
        flag_modified(p, 'meta')
    except Exception:
        pass
    try:
        me_pid = None
        rows = (await db.execute(select(Person.id, Person.meta, Person.display_name).where(Person.owner_user_id == user.id))).all()
        for pid, meta, dname in rows:
            if isinstance(meta, dict) and meta.get('connect_to_owner'):
                rh = str(meta.get('role_hint', '')).strip().lower()
                if rh in {'you', 'self', 'me'}:
                    me_pid = pid
                    break
        if me_pid is None:
            uname = (user.username or '').strip().lower()
            uemail = (user.email or '').strip().lower()
            for pid, meta, dname in rows:
                dn = (dname or '').strip().lower()
                if dn and (dn == uname or dn == uemail or dn == 'you'):
                    me_pid = pid
                    break
        if me_pid and p.id != me_pid:
            r = (payload.role_hint or '').strip().lower() or 'friend'

            def _map_role_to_rel(role: str) -> str:
                if role in {'mother', 'father', 'parent'}:
                    return 'parent-of'
                if role in {'son', 'daughter', 'child'}:
                    return 'parent-of'
                if role in {'spouse', 'partner', 'husband', 'wife'}:
                    return 'spouse-of'
                if role in {'sibling', 'brother', 'sister'}:
                    return 'sibling-of'
                if role in {'friend', 'neighbor', 'coworker', 'colleague'}:
                    return f'{role}-of'
                return 'friend-of'
            rel = _map_role_to_rel(r)
            if r in {'mother', 'father', 'parent'}:
                stmt_f = pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=p.id, dst_id=me_pid, rel_type='parent-of', confidence=0.95).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type'])
                await db.execute(stmt_f)
                stmt_i = pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=me_pid, dst_id=p.id, rel_type='child-of', confidence=0.95).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type'])
                await db.execute(stmt_i)
            elif r in {'son', 'daughter', 'child'}:
                stmt_f = pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=me_pid, dst_id=p.id, rel_type='parent-of', confidence=0.95).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type'])
                await db.execute(stmt_f)
                stmt_i = pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=p.id, dst_id=me_pid, rel_type='child-of', confidence=0.95).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type'])
                await db.execute(stmt_i)
            else:
                stmt_f = pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=me_pid, dst_id=p.id, rel_type=rel, confidence=0.9).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type'])
                await db.execute(stmt_f)
                stmt_i = pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=p.id, dst_id=me_pid, rel_type=rel, confidence=0.9).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type'])
                await db.execute(stmt_i)
    except Exception:
        pass
    await db.commit()
    return {'person_id': p.id}

@router.post('/api/people/{person_id}/photo')
async def api_people_photo_upload(person_id: int, file: UploadFile=File(...), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p: Person | None = await db.get(Person, person_id)
    if not p or p.owner_user_id != user.id:
        raise HTTPException(404, 'Person not found')
    if not file or not file.filename:
        raise HTTPException(422, 'No file uploaded')
    ctype = (file.content_type or '').lower()
    if not ctype.startswith('image/'):
        raise HTTPException(415, 'Only image uploads are supported')
    bucket = _user_bucket_name(user)
    ext = os.path.splitext(file.filename or '')[1].lower() or mimetypes.guess_extension(ctype) or '.jpg'
    if ext in ('.jpeg', '.jpe'):
        ext = '.jpg'
    if ext not in ('.jpg', '.jpeg', '.png', '.webp', '.gif'):
        ext = '.jpg'
    rel_dir = os.path.join('uploads', 'users', bucket, 'people', str(person_id))
    abs_dir = STATIC_DIR / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    rel_path = os.path.join(rel_dir, f'photo{ext}')
    abs_path = STATIC_DIR / rel_path
    with open(abs_path, 'wb') as w:
        shutil.copyfileobj(file.file, w)
    old = (p.photo_url or '').strip()
    if old:
        old_rel = old.lstrip('/').replace('\\', '/')
        if not old_rel.startswith('uploads/'):
            old_rel = f'uploads/{old_rel}'
        if old_rel != rel_path.replace('\\', '/'):
            try:
                (STATIC_DIR / old_rel).unlink(missing_ok=True)
            except Exception:
                pass
    p.photo_url = rel_path.replace('\\', '/')
    await db.commit()
    return {'photo_url': f'/static/{p.photo_url}'}

@router.delete('/api/people/{person_id}/photo')
async def api_people_photo_delete(person_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p: Person | None = await db.get(Person, person_id)
    if not p or p.owner_user_id != user.id:
        raise HTTPException(404, 'Person not found')
    rel = (p.photo_url or '').strip()
    if rel:
        reln = rel.lstrip('/').replace('\\', '/')
        if not reln.startswith('uploads/'):
            reln = f'uploads/{reln}'
        try:
            (STATIC_DIR / reln).unlink(missing_ok=True)
        except Exception:
            pass
        p.photo_url = None
        await db.commit()
    return {'ok': True}

@router.delete('/api/people/{person_id}')
async def api_people_delete(person_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p: Person | None = await db.get(Person, person_id)
    if not p or p.owner_user_id != user.id:
        raise HTTPException(404, 'Person not found')
    rel = (p.photo_url or '').strip()
    if rel:
        reln = rel.lstrip('/').replace('\\', '/')
        if not reln.startswith('uploads/'):
            reln = f'uploads/{reln}'
        try:
            (STATIC_DIR / reln).unlink(missing_ok=True)
        except Exception:
            pass
    await db.delete(p)
    await db.commit()
    return {'ok': True}

class EdgeReq(BaseModel):
    src_person_id: int
    dst_person_id: int
    rel_type: str
    confidence: float = 0.9

@router.post('/api/people/edges')
async def api_people_edge(payload: EdgeReq, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):

    def _canonical_rel(rt: str) -> str:
        rt = (rt or '').strip().lower()
        rt = rt.replace('relationship:', '').replace('role:', '')
        if rt in ('wife-of', 'husband-of'):
            rt = 'spouse-of'
        parent_like = {'mother-of', 'father-of', 'parent-of'}
        child_like = {'son-of', 'daughter-of', 'child-of'}
        if rt in parent_like:
            return 'parent-of'
        if rt in child_like:
            return 'child-of'
        if rt in {'brother-of', 'sister-of', 'sibling-of'}:
            return 'sibling-of'
        if rt in {'half-sibling', 'half-sibling-of'}:
            return 'half-sibling-of'
        if rt in {'step-sibling', 'step-sibling-of'}:
            return 'step-sibling-of'
        if rt in {'step-parent-of', 'adoptive-parent-of', 'step-sibling-of', 'half-sibling-of'}:
            return rt
        for base in ('friend', 'neighbor', 'coworker', 'mentor', 'teacher', 'student', 'partner', 'spouse', 'ex-partner'):
            if rt == base:
                rt = f'{base}-of'
        return rt

    def _inverse_rel(rt: str) -> str:
        rt = (rt or '').strip().lower()
        mapping = {'mother-of': 'child-of', 'father-of': 'child-of', 'parent-of': 'child-of', 'child-of': 'parent-of', 'son-of': 'parent-of', 'daughter-of': 'parent-of', 'grandparent-of': 'grandchild-of', 'grandchild-of': 'grandparent-of', 'sibling-of': 'sibling-of', 'half-sibling-of': 'half-sibling-of', 'step-sibling-of': 'step-sibling-of', 'aunt-of': 'niece-of', 'uncle-of': 'nephew-of', 'niece-of': 'aunt-of', 'nephew-of': 'uncle-of', 'cousin-of': 'cousin-of', 'aunt-in-law-of': 'niece-in-law-of', 'niece-in-law-of': 'aunt-in-law-of', 'cousin-in-law-of': 'cousin-in-law-of', 'mother-in-law-of': 'parent-in-law-of', 'father-in-law-of': 'parent-in-law-of', 'parent-in-law-of': 'parent-in-law-of', 'sister-in-law-of': 'sister-in-law-of', 'brother-in-law-of': 'brother-in-law-of', 'spouse-of': 'spouse-of', 'partner-of': 'partner-of', 'ex-partner-of': 'ex-partner-of', 'friend-of': 'friend-of', 'mentor-of': 'student-of', 'student-of': 'teacher-of', 'teacher-of': 'student-of', 'coworker-of': 'coworker-of', 'neighbor-of': 'neighbor-of', 'coach-of': 'student-of'}
        return mapping.get(rt, rt)
    rel_in = _canonical_rel(payload.rel_type)
    fwd = RelationshipEdge(user_id=user.id, src_id=payload.src_person_id, dst_id=payload.dst_person_id, rel_type=rel_in, confidence=payload.confidence, meta={'role_tag': f"role:{(payload.rel_type or '').strip().lower().replace('-of', '')}"})
    inv_rt = _inverse_rel(rel_in)
    inv = RelationshipEdge(user_id=user.id, src_id=payload.dst_person_id, dst_id=payload.src_person_id, rel_type=inv_rt, confidence=payload.confidence, meta={'role_tag': f"role:{inv_rt.replace('-of', '')}"})
    db.add(fwd)
    try:
        await db.flush()
    except Exception:
        await db.rollback()
        try:
            fwd = (await db.execute(select(RelationshipEdge).where(RelationshipEdge.user_id == user.id, RelationshipEdge.src_id == payload.src_person_id, RelationshipEdge.dst_id == payload.dst_person_id, RelationshipEdge.rel_type == payload.rel_type))).scalars().first() or fwd
        except Exception:
            pass
    db.add(inv)
    try:
        await db.flush()
    except Exception:
        await db.rollback()
    await db.commit()
    return {'ok': True, 'edge_id': getattr(fwd, 'id', None)}

@router.get('/api/people/tree')
async def api_people_tree(user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    own_q = select(Person.id, Person.display_name).where(Person.owner_user_id == user.id)
    shared_exists = exists(select(PersonShare.id).join(KinMembership, KinMembership.group_id == PersonShare.group_id).where(PersonShare.person_id == Person.id).where(KinMembership.user_id == user.id))
    shared_q = select(Person.id, Person.display_name).where((Person.visibility == 'groups') & shared_exists)
    rows = (await db.execute(own_q.union(shared_q))).all()
    nodes = [{'id': rid, 'name': name} for rid, name in rows]
    return {'nodes': nodes, 'edges': []}

class GroupCreateReq(BaseModel):
    name: str
    kind: str = 'family'

@router.post('/api/groups')
async def api_create_group(payload: GroupCreateReq, db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    name = (payload.name or '').strip() or 'Family Group'
    kind = (payload.kind or 'family').strip()
    g = KinGroup(name=name, kind=kind, created_by=user.id, join_code=_join_code(8))
    db.add(g)
    await db.flush()
    db.add(KinMembership(group_id=g.id, user_id=user.id, role='admin'))
    await db.commit()
    return {'id': g.id, 'name': g.name, 'kind': g.kind, 'join_code': g.join_code}

class GroupJoinReq(BaseModel):
    code: str

@router.post('/api/groups/join')
async def api_join_group(payload: GroupJoinReq, db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    code = (payload.code or '').strip().upper()
    g = (await db.execute(select(KinGroup).where(KinGroup.join_code == code))).scalars().first()
    if not g:
        raise HTTPException(404, 'Group not found for that code.')
    exists_row = (await db.execute(select(KinMembership).where(KinMembership.group_id == g.id, KinMembership.user_id == user.id))).scalars().first()
    if exists_row:
        return {'ok': True, 'group_id': g.id, 'already_member': True}
    db.add(KinMembership(group_id=g.id, user_id=user.id, role='member'))
    await db.commit()
    return {'ok': True, 'group_id': g.id}

@router.get('/api/groups/my')
async def api_list_my_groups(db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    gids = (await db.execute(select(KinMembership.group_id).where(KinMembership.user_id == user.id))).scalars().all()
    if not gids:
        return {'groups': []}
    rows = (await db.execute(select(KinGroup).where(KinGroup.id.in_(gids)))).scalars().all()
    return {'groups': [{'id': g.id, 'name': g.name, 'kind': g.kind} for g in rows]}

@router.get('/api/groups/{group_id}')
async def api_group_detail(group_id: int, db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    g = await db.get(KinGroup, group_id)
    if not g:
        raise HTTPException(404, 'Group not found.')
    m = (await db.execute(select(KinMembership).where(KinMembership.group_id == group_id, KinMembership.user_id == user.id))).scalars().first()
    if not m:
        raise HTTPException(403, 'Not a member of this group.')
    members = (await db.execute(select(KinMembership).where(KinMembership.group_id == group_id))).scalars().all()
    return {'id': g.id, 'name': g.name, 'kind': g.kind, 'members': [{'user_id': mm.user_id, 'role': mm.role} for mm in members], 'join_code': g.join_code if m.role == 'admin' else None}

@router.post('/admin/tags/import_whitelist')
async def import_tag_whitelist(db: AsyncSession=Depends(get_db)):
    default_path = os.path.join(os.path.dirname(__file__), 'data', 'tag_whitelist.json')
    path = os.getenv('TAG_WHITELIST_PATH', default_path)
    with open(path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    inserted = updated = skipped = 0
    for it in items:
        slug = (it.get('value') or '').strip().lower()
        name = (it.get('label') or slug.replace('-', ' ').title()).strip()
        if not slug:
            continue
        existing = (await db.execute(select(Tag).where(Tag.slug == slug))).scalar_one_or_none()
        if existing:
            if (existing.name or '').strip() != name:
                existing.name = name
                updated += 1
            else:
                skipped += 1
            continue
        db.add(Tag(name=name, slug=slug))
        try:
            await db.flush()
            inserted += 1
        except IntegrityError:
            await db.rollback()
            skipped += 1
    await db.commit()
    return {'ok': True, 'path': path, 'inserted': inserted, 'updated': updated, 'skipped': skipped}

@router.get('/api/admin/assignments/by-user')
async def admin_assignments_by_user(user_id: int=Query(...), db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    u = await db.get(User, user_id)
    if not u:
        raise HTTPException(404, 'User not found')
    prof = (await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))).scalars().first()
    ups = (await db.execute(select(UserPrompt).options(selectinload(UserPrompt.prompt).selectinload(Prompt.tags)).where(UserPrompt.user_id == user_id))).scalars().all()
    resp_rows = (await db.execute(select(Response).options(selectinload(Response.tags)).where(Response.user_id == user_id))).scalars().all()
    rmap = {r.prompt_id: r for r in resp_rows}
    items = []
    for up in ups:
        p = up.prompt
        resp = rmap.get(p.id)
        items.append({'prompt_id': p.id, 'chapter': p.chapter, 'text': (p.text or '')[:200], 'tags': [t.slug for t in p.tags or []], 'answered': bool(resp), 'response_excerpt': (resp.transcription or resp.response_text or '')[:160] if resp else None, 'response': {'id': resp.id, 'transcription': resp.transcription, 'tags': [t.slug for t in resp.tags or []]} if resp else None})
    assigned_count = len(ups)
    answered_count = sum((1 for it in items if it['answered']))
    pct = int(round(100.0 * answered_count / assigned_count)) if assigned_count else 0
    return {'user': {'id': u.id, 'name': (prof.display_name if prof else None) or (u.username or u.email), 'email': u.email}, 'answered_pct': pct, 'items': items}

@router.get('/api/admin/assignments/by-prompt')
async def admin_assignments_by_prompt(prompt_id: int=Query(...), db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    p = await db.get(Prompt, prompt_id)
    if not p:
        raise HTTPException(404, 'Prompt not found')
    ups = (await db.execute(select(UserPrompt.user_id).where(UserPrompt.prompt_id == prompt_id))).scalars().all() or []
    if not ups:
        return {'prompt_id': p.id, 'users': [], 'answered_pct': 0}
    users = (await db.execute(select(User).where(User.id.in_(ups)))).scalars().all()
    profs = (await db.execute(select(UserProfile).where(UserProfile.user_id.in_(ups)))).scalars().all()
    pmap = {pr.user_id: pr for pr in profs}
    answered_user_ids = set((uid for uid, in (await db.execute(select(Response.user_id).where(Response.prompt_id == prompt_id, Response.user_id.in_(ups)))).all()))
    users_out = [{'id': u.id, 'name': _display_name_or_email(u, pmap.get(u.id)), 'email': u.email, 'answered': u.id in answered_user_ids} for u in users]
    pct = int(round(100.0 * len(answered_user_ids) / max(1, len(ups))))
    return {'prompt_id': p.id, 'users': users_out, 'answered_pct': pct}

@router.post('/api/admin/pool/add')
async def admin_pool_add(payload: dict=Body(...), db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    user_id = int(payload.get('user_id'))
    prompt_id = int(payload.get('prompt_id'))
    exists = (await db.execute(select(UserPrompt).where(UserPrompt.user_id == user_id, UserPrompt.prompt_id == prompt_id))).scalars().first()
    if not exists:
        db.add(UserPrompt(user_id=user_id, prompt_id=prompt_id))
        await db.commit()
    return {'ok': True}

@router.post('/api/admin/pool/remove')
async def admin_pool_remove(payload: dict=Body(...), db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    user_id = int(payload.get('user_id'))
    prompt_id = int(payload.get('prompt_id'))
    await db.execute(delete(UserPrompt).where(UserPrompt.user_id == user_id, UserPrompt.prompt_id == prompt_id))
    await db.commit()
    return {'ok': True}

@router.get('/api/admin/pool/stats')
async def admin_pool_stats(scope: str=Query(..., pattern='^(prompt|user)$'), id: int=Query(...), db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    if scope == 'prompt':
        assigned = (await db.execute(select(func.count()).select_from(UserPrompt).where(UserPrompt.prompt_id == id))).scalar() or 0
        ups = (await db.execute(select(UserPrompt.user_id).where(UserPrompt.prompt_id == id))).scalars().all()
        answered = 0
        if ups:
            answered = (await db.execute(select(func.count(func.distinct(Response.user_id))).where(Response.prompt_id == id, Response.user_id.in_(ups)))).scalar() or 0
        pct = int(round(100.0 * answered / max(1, assigned)))
        return {'scope': 'prompt', 'id': id, 'assigned_count': assigned, 'answered_count': answered, 'answered_pct': pct}
    if scope == 'user':
        assigned = (await db.execute(select(func.count()).select_from(UserPrompt).where(UserPrompt.user_id == id))).scalar() or 0
        answered = (await db.execute(select(func.count()).select_from(Response).where(Response.user_id == id))).scalar() or 0
        pct = int(round(100.0 * answered / max(1, assigned)))
        return {'scope': 'user', 'id': id, 'assigned_count': assigned, 'answered_count': answered, 'answered_pct': pct}

@router.get('/admin/users/search')
async def admin_user_search(q: str=Query('', min_length=0), limit: int=Query(10, ge=1, le=50), exclude_prompt_id: int | None=Query(None), db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    q = (q or '').strip().lower()
    like = f'%{q}%'
    stmt = select(User).join(UserProfile, UserProfile.user_id == User.id, isouter=True).limit(limit)
    if q:
        stmt = stmt.where(func.lower(User.email).like(like) | func.lower(User.username).like(like) | func.lower(func.coalesce(UserProfile.display_name, '')).like(like))
    if exclude_prompt_id is not None:
        assigned_uids = (await db.execute(select(UserPrompt.user_id).where(UserPrompt.prompt_id == exclude_prompt_id))).scalars().all() or []
        if assigned_uids:
            stmt = stmt.where(~User.id.in_(assigned_uids))
    users = (await db.execute(stmt)).scalars().all()
    profs = (await db.execute(select(UserProfile).where(UserProfile.user_id.in_([u.id for u in users])))).scalars().all()
    pmap = {p.user_id: p for p in profs}

    def _disp(u: User) -> str:
        p = pmap.get(u.id)
        return (p.display_name if p and p.display_name else None) or (u.username or u.email)
    return {'users': [{'id': u.id, 'name': _disp(u), 'email': u.email} for u in users]}

@router.get('/api/admin/pool/by-prompt')
async def pool_by_prompt(prompt_id: int, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    """
    Return users currently assigned to this prompt.
    Shape: [{id, name, email}]
    """
    from app.models import User, PromptPool
    res = await db.execute(select(User.id, User.username, User.email).join(PromptPool, PromptPool.user_id == User.id).where(PromptPool.prompt_id == prompt_id).options(noload('*')))
    rows = res.all()
    return [{'id': r.id, 'name': r.username, 'email': r.email} for r in rows]

@router.get('/admin/users/{user_id}/responses/{response_id}', response_class=HTMLResponse)
async def admin_response_view(user_id: int, response_id: int, request: Request, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    resp = (await db.execute(select(Response).options(selectinload(Response.prompt).selectinload(Prompt.media), selectinload(Response.tags), selectinload(Response.supporting_media), selectinload(Response.segments)).where(Response.id == response_id, Response.user_id == user_id))).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    return templates.TemplateResponse('response_view.html', {'request': request, 'user': admin, 'response': resp, 'prompt_media': list(resp.prompt.media) if resp.prompt and resp.prompt.media else [], 'supporting_media': list(resp.supporting_media or []), 'segments': list(resp.segments or []), 'is_token_link': False, 'is_admin_view': True})

@router.get('/admin/users/{user_id}/responses/{response_id}/edit', response_class=HTMLResponse)
async def admin_edit_response_page(user_id: int, response_id: int, request: Request, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    response = (await db.execute(select(Response).options(selectinload(Response.prompt), selectinload(Response.tags)).where(Response.id == response_id, Response.user_id == user_id))).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    supporting_media = (await db.execute(select(SupportingMedia).where(SupportingMedia.response_id == response_id))).scalars().all()
    prompt_media = (await db.execute(select(PromptMedia).where(PromptMedia.prompt_id == response.prompt_id))).scalars().all()
    admin_api_base = f'/admin/users/{user_id}/responses/{response_id}'
    return templates.TemplateResponse('response_edit.html', {'request': request, 'user': admin, 'response': response, 'supporting_media': supporting_media, 'prompt_media': prompt_media, 'is_admin_view': True, 'admin_api_base': admin_api_base, 'acting_as_user': await db.get(User, user_id)})

@router.post('/admin/users/{user_id}/responses/{response_id}/edit')
async def admin_save_response_edit(user_id: int, response_id: int, transcription: str=Form(...), title: str | None=Form(None), admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    response = (await db.execute(select(Response).options(selectinload(Response.tags)).where(Response.id == response_id, Response.user_id == user_id))).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    try:
        cur_tags = [t.slug for t in response.tags or []]
        ver = ResponseVersion(response_id=response.id, user_id=user_id, title=getattr(response, 'title', None), transcription=response.transcription, tags_json={'tags': cur_tags}, edited_by_admin_id=getattr(admin, 'id', None))
        db.add(ver)
    except Exception:
        pass
    try:
        cur_tags = [t.slug for t in getattr(response, 'tags', []) or []]
        db.add(ResponseVersion(response_id=response.id, user_id=user.id, title=getattr(response, 'title', None), transcription=response.transcription, tags_json={'tags': cur_tags}, edited_by_admin_id=None))
    except Exception:
        pass
    response.transcription = transcription
    if title is not None and hasattr(response, 'title'):
        response.title = (title or '').strip() or None
    await db.commit()
    try:
        db.add(AdminEditLog(admin_user_id=getattr(admin, 'id', None), target_user_id=user_id, response_id=response.id, action='edit_text', payload={'title': title, 'len_transcription': len(transcription or '')}))
        await db.commit()
    except Exception:
        pass
    try:
        text_for_tagging = _text_for_tagging(response)
        if text_for_tagging.strip():
            draft = suggest_tags_rule_based(text_for_tagging, word_count=len(text_for_tagging.split()))
            for slug, _ in draft:
                tag = await _get_or_create_tag(db, slug)
                if tag and tag not in response.tags:
                    response.tags.append(tag)
        await db.commit()
    except Exception:
        pass
    try:
        await enrich_after_transcription(db, response)
    except Exception:
        pass
    await db.commit()
    return RedirectResponse(url=f'/admin/users/{user_id}/responses/{response_id}/edit', status_code=303)

@router.post('/admin/users/{user_id}/responses/{response_id}/media')
async def admin_add_supporting_media(user_id: int, response_id: int, media_files: list[UploadFile]=File(...), admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    resp = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user_id))).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    for f in media_files or []:
        if not f or not f.filename:
            continue
        media = SupportingMedia(response_id=resp.id, file_path='', media_type=f.content_type.split('/', 1)[0] if f.content_type else 'file')
        db.add(media)
        await db.flush()
        tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(f.filename or '').suffix}"
        with open(tmp, 'wb') as w:
            shutil.copyfileobj(f.file, w)
        art = PIPELINE.process_upload(temp_path=tmp, logical='response', role='supporting', user_slug_or_id=(await db.get(User, user_id)).username or str(user_id), prompt_id=None, response_id=resp.id, media_id=media.id, original_filename=f.filename, content_type=f.content_type)
        playable_rel = art.playable_rel
        media.file_path = playable_rel[len('uploads/'):] if playable_rel.startswith('uploads/') else playable_rel
        media.thumbnail_url = art.thumb_rel
        media.mime_type = art.mime_type
        media.duration_sec = int(art.duration_sec or 0)
        media.sample_rate = art.sample_rate
        media.channels = art.channels
        media.width = art.width
        media.height = art.height
        media.size_bytes = art.size_bytes
        media.codec_audio = art.codec_a
        media.codec_video = art.codec_v
        media.wav_path = art.wav_rel
    await db.commit()
    try:
        db.add(AdminEditLog(admin_user_id=getattr(admin, 'id', None), target_user_id=user_id, response_id=resp.id, action='add_media', payload={'count': len(media_files or [])}))
        await db.commit()
    except Exception:
        pass
    return RedirectResponse(url=f'/admin/users/{user_id}/responses/{response_id}/edit', status_code=303)

@router.delete('/admin/users/{user_id}/responses/{response_id}/media/{media_id}')
async def admin_delete_supporting_media(user_id: int, response_id: int, media_id: int, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    resp = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user_id))).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    media = await db.get(SupportingMedia, media_id)
    if not media or media.response_id != response_id:
        raise HTTPException(status_code=404, detail='Media not found')
    playable_uploads_rel = _to_uploads_rel_for_playable(media.file_path)
    try:
        PIPELINE.delete_artifacts(playable_uploads_rel, media.wav_path or None, media.thumbnail_url or None)
    except Exception:
        pass
    await db.delete(media)
    await db.commit()
    try:
        db.add(AdminEditLog(admin_user_id=getattr(admin, 'id', None), target_user_id=user_id, response_id=response_id, action='delete_media', payload={'media_id': media_id}))
        await db.commit()
    except Exception:
        pass
    return JSONResponse({'ok': True})

@router.get('/admin/users/{user_id}/responses/{response_id}/tags')
async def admin_get_response_tags(user_id: int, response_id: int, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    response = (await db.execute(select(Response).options(selectinload(Response.tags)).where(Response.id == response_id, Response.user_id == user_id))).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    return {'tags': [{'id': t.id, 'name': t.name, 'slug': t.slug, 'color': t.color} for t in response.tags or []]}

@router.post('/admin/users/{user_id}/responses/{response_id}/tags')
async def admin_set_response_tags(user_id: int, response_id: int, payload: Any=Body(...), admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    exists = (await db.execute(select(Response.id).where(Response.id == response_id, Response.user_id == user_id))).scalar_one_or_none()
    if not exists:
        raise HTTPException(status_code=404, detail='Response not found')
    try:
        resp = (await db.execute(select(Response).options(selectinload(Response.tags)).where(Response.id == response_id, Response.user_id == user_id))).scalars().first()
        cur_tags = [t.slug for t in resp.tags or []] if resp else []
        db.add(ResponseVersion(response_id=response_id, user_id=user_id, title=getattr(resp, 'title', None) if resp else None, transcription=getattr(resp, 'transcription', None) if resp else None, tags_json={'tags': cur_tags}, edited_by_admin_id=getattr(admin, 'id', None)))
    except Exception:
        pass
    raw = payload if isinstance(payload, list) else payload.get('tags') if isinstance(payload, dict) else []
    tag_names = [(s or '').strip() for s in raw or [] if (s or '').strip()]
    tags: list[Tag] = []
    for nm in tag_names:
        t = await _get_or_create_tag(db, nm)
        if t:
            tags.append(t)
    assoc_tbl = Response.tags.property.secondary
    await db.execute(delete(assoc_tbl).where(assoc_tbl.c.response_id == response_id))
    if tags:
        ids = sorted({t.id for t in tags})
        stmt = pg_insert(assoc_tbl).on_conflict_do_nothing(index_elements=[assoc_tbl.c.response_id, assoc_tbl.c.tag_id])
        await db.execute(stmt, [{'response_id': response_id, 'tag_id': tid} for tid in ids])
    await db.commit()
    try:
        db.add(AdminEditLog(admin_user_id=getattr(admin, 'id', None), target_user_id=user_id, response_id=response_id, action='set_tags', payload={'count': len(tags)}))
        await db.commit()
    except Exception:
        pass
    return {'ok': True, 'tags': [{'id': t.id, 'name': t.name, 'slug': t.slug, 'color': t.color} for t in tags]}

@router.get('/admin/users/{user_id}/responses/{response_id}/segments')
async def admin_list_response_segments(user_id: int, response_id: int, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    exists = (await db.execute(select(Response.id).where(Response.id == response_id, Response.user_id == user_id))).scalar_one_or_none()
    if not exists:
        raise HTTPException(status_code=404, detail='Response not found')
    rows = (await db.execute(select(ResponseSegment).where(ResponseSegment.response_id == response_id).order_by(ResponseSegment.order_index.asc(), ResponseSegment.id.asc()))).scalars().all()
    out = []
    for s in rows:
        out.append({'id': s.id, 'order_index': getattr(s, 'order_index', None), 'transcript': getattr(s, 'transcript', '') or '', 'media_path': getattr(s, 'media_path', '') or '', 'media_mime': getattr(s, 'media_mime', '') or '', 'thumbnail_path': getattr(s, 'thumbnail_path', None) or getattr(s, 'thumbnail_url', None) or ''})
    return out

@router.post('/admin/users/{user_id}/responses/{response_id}/segments')
async def admin_add_response_segment(user_id: int, response_id: int, file: UploadFile=File(...), note: str | None=Form(None), admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    resp = await db.get(Response, response_id)
    if not resp or resp.user_id != user_id:
        raise HTTPException(status_code=404, detail='Response not found')
    order = await _max_order_index(db, response_id)
    seg = ResponseSegment(response_id=response_id, order_index=order + 1, transcript='')
    db.add(seg)
    await db.flush()
    tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(file.filename or '').suffix}"
    with open(tmp, 'wb') as w:
        shutil.copyfileobj(file.file, w)
    target_user = await db.get(User, user_id)
    art = PIPELINE.process_upload(temp_path=tmp, logical='response', role='supporting', user_slug_or_id=target_user.username or str(user_id) if target_user else str(user_id), prompt_id=None, response_id=response_id, media_id=seg.id, original_filename=file.filename or 'segment', content_type=file.content_type)
    playable_rel = (art.playable_rel or '').lstrip('/').replace('\\', '/')
    if playable_rel.startswith('uploads/'):
        playable_rel = playable_rel[len('uploads/'):]
    seg.media_path = playable_rel
    seg.media_mime = art.mime_type or None
    if not (resp.primary_media_url or '').strip():
        resp.primary_media_url = playable_rel
        resp.primary_mime_type = seg.media_mime or 'audio/mp4'
        resp.primary_thumbnail_path = None
        resp.primary_duration_sec = None
    await db.commit()
    try:
        uploads_rel_for_tx = 'uploads/' + seg.media_path
        spawn(transcribe_segment_and_update(seg.id, uploads_rel_for_tx), name='transcribe_segment_admin_upload')
    except Exception:
        pass
    return {'segment': {'id': seg.id}}

@router.patch('/admin/users/{user_id}/responses/{response_id}/segments/reorder')
async def admin_reorder_response_segments(user_id: int, response_id: int, payload: ReorderSegmentsRequest=Body(...), admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    exists = (await db.execute(select(Response.id).where(Response.id == response_id, Response.user_id == user_id))).scalar_one_or_none()
    if not exists:
        raise HTTPException(status_code=404, detail='Response not found')
    ids = payload.order or []
    if not ids:
        return {'ok': True}
    found = (await db.execute(select(ResponseSegment.id).where(ResponseSegment.response_id == response_id, ResponseSegment.id.in_(ids)))).scalars().all()
    if set(found) != set(ids):
        raise HTTPException(status_code=400, detail='Invalid segment ids for this response.')
    order_map = {sid: idx + 1 for idx, sid in enumerate(ids)}
    rows = (await db.execute(select(ResponseSegment).where(ResponseSegment.id.in_(ids)))).scalars().all()
    for r in rows:
        r.order_index = order_map.get(r.id, r.order_index)
    await db.commit()
    return {'ok': True}

@router.delete('/admin/users/{user_id}/responses/{response_id}/segments/{segment_id}')
async def admin_delete_response_segment(user_id: int, response_id: int, segment_id: int, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    resp = await db.get(Response, response_id)
    if not resp or resp.user_id != user_id:
        raise HTTPException(status_code=404, detail='Response not found')
    seg = await db.get(ResponseSegment, segment_id)
    if not seg or seg.response_id != response_id:
        raise HTTPException(status_code=404, detail='Segment not found')
    playable_uploads_rel = seg.media_path
    if playable_uploads_rel:
        playable_uploads_rel = _to_uploads_rel_for_playable(playable_uploads_rel)
    try:
        PIPELINE.delete_artifacts(playable_uploads_rel, None, None)
    except Exception:
        pass
    await db.delete(seg)
    await db.commit()
    return {'ok': True}

@router.post('/admin/users/{user_id}/responses/{response_id}/segments/bootstrap')
async def admin_bootstrap_first_segment(user_id: int, response_id: int, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    resp = await db.get(Response, response_id)
    if not resp or resp.user_id != user_id:
        raise HTTPException(status_code=404, detail='Response not found')
    rows = await db.execute(select(ResponseSegment.id).where(ResponseSegment.response_id == response_id))
    if rows.first():
        return JSONResponse({'ok': True, 'created': False})
    primary_rel = (resp.primary_media_url or '').lstrip('/').replace('\\', '/')
    if not primary_rel:
        return JSONResponse({'ok': True, 'created': False})
    if primary_rel.startswith('static/'):
        primary_rel = primary_rel[len('static/'):]
    if primary_rel.startswith('uploads/'):
        primary_rel = primary_rel[len('uploads/'):]
    if primary_rel.startswith(f'responses/{response_id}/') and primary_rel.split('/')[-1].startswith('composite-'):
        return JSONResponse({'ok': True, 'created': False})
    seg = ResponseSegment(response_id=response_id, order_index=0, media_path=primary_rel, media_mime=resp.primary_mime_type or None, transcript='')
    db.add(seg)
    await db.commit()
    try:
        uploads_rel = primary_rel if primary_rel.startswith('uploads/') else f'uploads/{primary_rel}'
        spawn(transcribe_segment_and_update(seg.id, uploads_rel), name='transcribe_segment_admin_bootstrap')
    except Exception:
        pass
    return JSONResponse({'ok': True, 'created': True, 'segment_id': seg.id})

@router.post('/admin/users/{user_id}/responses/{response_id}/segments/merge-audio')
async def admin_merge_audio_to_primary(user_id: int, response_id: int, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    resp = await db.get(Response, response_id)
    if not resp or resp.user_id != user_id:
        raise HTTPException(status_code=404, detail='Response not found')
    rows = await db.execute(select(ResponseSegment).where(ResponseSegment.response_id == response_id).order_by(ResponseSegment.order_index.asc()))
    segs = [s for s, in rows]
    sources: list[str] = []
    for s in segs:
        rel = (s.media_path or '').lstrip('/').replace('\\', '/')
        if not rel:
            continue
        is_composite = rel.startswith(f'responses/{response_id}/') and rel.split('/')[-1].startswith('composite-')
        if is_composite:
            continue
        if rel.startswith('uploads/'):
            rel = rel[len('uploads/'):]
        sources.append(rel)
    if not sources:
        return JSONResponse({'ok': False, 'error': 'no segments to merge'}, status_code=400)
    uploads_root = PIPELINE.static_root / 'uploads'
    bad = []
    for rel in sources:
        p = (uploads_root / rel).resolve()
        if not p.exists() or p.stat().st_size == 0:
            bad.append(rel)
    if bad:
        raise HTTPException(status_code=400, detail={'error': 'missing_or_empty', 'files': bad})
    try:
        out = await PIPELINE.concat_audio_async(sources_rel=sources, response_id=response_id)
    except Exception:
        import logging
        logging.exception('admin concat_audio_async crashed')
        raise HTTPException(status_code=500, detail='merge crashed')
    if not out.get('ok'):
        raise HTTPException(status_code=500, detail=out.get('error', 'merge failed'))
    dest_rel = out['dest_rel']
    resp.primary_media_url = dest_rel
    resp.primary_mime_type = 'audio/mp4'
    resp.primary_thumbnail_path = None
    resp.primary_duration_sec = None
    await db.commit()
    return JSONResponse({'ok': True, 'primary': resp.primary_media_url})

@router.get('/admin/users/{user_id}/responses/{response_id}/versions')
async def admin_list_versions(user_id: int, response_id: int, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    exists = (await db.execute(select(Response.id).where(Response.id == response_id, Response.user_id == user_id))).scalar_one_or_none()
    if not exists:
        raise HTTPException(status_code=404, detail='Response not found')
    rows = (await db.execute(select(ResponseVersion).where(ResponseVersion.response_id == response_id).order_by(ResponseVersion.created_at.desc(), ResponseVersion.id.desc()))).scalars().all()
    out = []
    for v in rows:
        out.append({'id': v.id, 'created_at': v.created_at.isoformat() if v.created_at else None, 'edited_by_admin_id': v.edited_by_admin_id, 'title': (v.title or '')[:120], 'has_transcription': bool((v.transcription or '').strip()), 'tags': (v.tags_json or {}).get('tags') if isinstance(v.tags_json, dict) else None})
    return {'versions': out}

@router.post('/admin/users/{user_id}/responses/{response_id}/versions/{version_id}/restore')
async def admin_restore_version(user_id: int, response_id: int, version_id: int, admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    resp = await db.get(Response, response_id)
    if not resp or resp.user_id != user_id:
        raise HTTPException(status_code=404, detail='Response not found')
    ver = await db.get(ResponseVersion, version_id)
    if not ver or ver.response_id != response_id:
        raise HTTPException(status_code=404, detail='Version not found')
    if hasattr(resp, 'title'):
        resp.title = ver.title
    resp.transcription = ver.transcription
    try:
        tags = []
        slugs = []
        if isinstance(ver.tags_json, dict):
            slugs = [s for s in (ver.tags_json or {}).get('tags', []) if s]
        for s in slugs:
            t = await _get_or_create_tag(db, s)
            if t:
                tags.append(t)
        assoc_tbl = Response.tags.property.secondary
        await db.execute(delete(assoc_tbl).where(assoc_tbl.c.response_id == response_id))
        if tags:
            stmt = pg_insert(assoc_tbl).on_conflict_do_nothing(index_elements=[assoc_tbl.c.response_id, assoc_tbl.c.tag_id])
            await db.execute(stmt, [{'response_id': response_id, 'tag_id': t.id} for t in tags])
    except Exception:
        pass
    await db.commit()
    try:
        db.add(AdminEditLog(admin_user_id=getattr(admin, 'id', None), target_user_id=user_id, response_id=response_id, action='restore_version', payload={'version_id': version_id}))
        await db.commit()
    except Exception:
        pass
    return {'ok': True}

def _imp_sign(data: str) -> str:
    return hmac.new(SECRET.encode('utf-8'), ('imp:' + data).encode('utf-8'), hashlib.sha256).hexdigest()

def _pack_imp_cookie(admin_id: int, ttl_sec: int=1800) -> str:
    ts = int(time.time())
    payload = f'{admin_id}.{ts}.{ttl_sec}'
    sig = _imp_sign(payload)
    return f'{payload}.{sig}'

def _unpack_imp_cookie(raw: str | None) -> int | None:
    if not raw or raw.count('.') < 3:
        return None
    admin_id_s, ts_s, ttl_s, sig = raw.split('.', 3)
    exp_ok = False
    try:
        ts = int(ts_s)
        ttl = int(ttl_s)
        exp_ok = time.time() - ts <= ttl
    except Exception:
        return None
    if not exp_ok:
        return None
    payload = f'{admin_id_s}.{ts_s}.{ttl_s}'
    if hmac.compare_digest(_imp_sign(payload), sig):
        try:
            return int(admin_id_s)
        except Exception:
            return None
    return None

def _is_adminish(u) -> bool:
    return bool(getattr(u, 'is_admin', False) or getattr(u, 'is_superuser', False) or getattr(u, 'super_admin', False))

@router.get('/admin/impersonate/stop')
async def admin_impersonate_stop_disabled():
    raise HTTPException(status_code=404, detail='Impersonation is disabled')

@router.get('/admin/impersonate/{target_user_id:int}')
async def admin_impersonate_disabled(target_user_id: int):
    raise HTTPException(status_code=404, detail='Impersonation is disabled')

class WeeklyRow(BaseModel):
    user_id: int
    display_name: str
    email: str
    current_prompt: Optional[dict] = None
    on_deck_prompt: Optional[dict] = None
    on_deck_candidates: Optional[List[dict]] = None
    state: str
    queued_at: Optional[str] = None
    sent_at: Optional[str] = None
    opened_at: Optional[str] = None
    clicked_at: Optional[str] = None
    used_at: Optional[str] = None
    completed_at: Optional[str] = None
    skipped_at: Optional[str] = None
    expires_at: Optional[str] = None
    token_status: Optional[str] = None
    token_link: Optional[str] = None

class WeeklyListResp(BaseModel):
    rows: List[WeeklyRow]
    total: int

def _user_dn(u: User) -> str:
    return (u.username or u.email or '').strip() or f'User {u.id}'

@router.get('/api/admin/weekly', response_model=WeeklyListResp)
async def admin_weekly_list(page: int=Query(1, ge=1), q: Optional[str]=Query(None), status: Optional[str]=Query(None), db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    PAGE = 25
    ofs = (page - 1) * PAGE
    qstmt = select(User).where(User.is_active == True)
    if q:
        like = f'%{q}%'
        qstmt = qstmt.where(or_(User.email.ilike(like), User.username.ilike(like)))
    if status:
        try:
            st = WeeklyState(status)
            qstmt = qstmt.where(User.weekly_state == st)
        except Exception:
            if status == 'expired':
                qstmt = qstmt.where(User.weekly_state == WeeklyState.expired)
    total = (await db.execute(select(func.count()).select_from(qstmt.subquery()))).scalar_one()
    users = (await db.execute(qstmt.order_by(User.id.asc()).offset(ofs).limit(PAGE))).scalars().all()
    rows = []
    for u in users:
        cur = await db.get(Prompt, u.weekly_current_prompt_id) if u.weekly_current_prompt_id else None
        od = await db.get(Prompt, u.weekly_on_deck_prompt_id) if u.weekly_on_deck_prompt_id else None
        tok = None
        if cur:
            tok = (await db.execute(select(WeeklyToken).where(WeeklyToken.user_id == u.id, WeeklyToken.prompt_id == cur.id))).scalars().first()
        token_link = None
        token_status = None
        expires_iso = None
        if tok:
            token_status = tok.status.value
            expires_iso = tok.expires_at.isoformat() if tok.expires_at else None
            token_link = f'/weekly/t/{tok.token}'
        cand_ids: list[int] = []
        try:
            cand_ids = await get_on_deck_candidates(db, u.id, k=5)
        except Exception:
            cand_ids = []
        cands: list[dict] = []
        if cand_ids:
            prs = (await db.execute(select(Prompt).where(Prompt.id.in_(cand_ids)))).unique().scalars().all()
            pmap = {p.id: p for p in prs}
            for pid in cand_ids:
                p = pmap.get(pid)
                if p:
                    cands.append({'id': p.id, 'title': p.text})
        rows.append(WeeklyRow(user_id=u.id, display_name=_user_dn(u), email=u.email, current_prompt={'id': cur.id, 'title': cur.text, 'tags': [t.slug for t in cur.tags or []]} if cur else None, on_deck_prompt={'id': od.id, 'title': od.text} if od else None, on_deck_candidates=cands, state=u.weekly_state.value, queued_at=u.weekly_queued_at.isoformat() if u.weekly_queued_at else None, sent_at=u.weekly_sent_at.isoformat() if u.weekly_sent_at else None, opened_at=u.weekly_opened_at.isoformat() if u.weekly_opened_at else None, clicked_at=u.weekly_clicked_at.isoformat() if u.weekly_clicked_at else None, used_at=u.weekly_used_at.isoformat() if u.weekly_used_at else None, completed_at=u.weekly_completed_at.isoformat() if u.weekly_completed_at else None, skipped_at=u.weekly_skipped_at.isoformat() if u.weekly_skipped_at else None, expires_at=expires_iso, token_status=token_status, token_link=token_link))
    return WeeklyListResp(rows=rows, total=total)

class Ids(BaseModel):
    user_ids: Optional[List[int]] = None
    user_id: Optional[int] = None

@router.post('/api/admin/weekly/send')
async def admin_weekly_send(payload: Ids, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    ids = payload.user_ids or ([payload.user_id] if payload.user_id else [])
    sent = 0
    for uid in ids:
        u = await db.get(User, uid)
        if not u or not u.weekly_current_prompt_id:
            continue
        tok = await get_or_refresh_active_token(db, uid, u.weekly_current_prompt_id)
        await db.commit()
        provider_id = await send_weekly_email(db, u, tok)
        u.weekly_state = WeeklyState.sent
        u.weekly_sent_at = _now()
        u.weekly_email_provider_id = provider_id
        sent += 1
    await db.commit()
    return {'ok': True, 'sent': sent}

class ScheduleReq(BaseModel):
    user_ids: List[int]
    when: datetime

@router.post('/api/admin/weekly/schedule')
async def admin_weekly_schedule(payload: ScheduleReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    await schedule_bulk_send(payload.user_ids, payload.when)
    return {'ok': True, 'scheduled': len(payload.user_ids)}

class RecurringReq(BaseModel):
    user_ids: List[int]
    days: List[int]
    hour: int
    minute: int
    weeks: int = 12

@router.post('/api/admin/weekly/schedule_recurring')
async def admin_weekly_schedule_recurring(payload: RecurringReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    tz_name = os.getenv('APP_TZ') or os.getenv('TZ') or 'UTC'
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = None
    today = datetime.now(tz).date() if tz else datetime.utcnow().date()
    monday = today - timedelta(days=today.weekday())
    total = 0
    for w in range(max(1, min(payload.weeks, 52))):
        week_start = monday + timedelta(days=7 * w)
        for d in sorted(set((int(x) for x in payload.days if 1 <= int(x) <= 7))):
            target_date = week_start + timedelta(days=d - 1)
            dt_local = datetime.combine(target_date, datetime.min.time()).replace(hour=payload.hour, minute=payload.minute)
            dt = dt_local.replace(tzinfo=tz) if tz else dt_local
            now_cmp = datetime.now(tz) if tz else datetime.utcnow()
            if dt <= now_cmp:
                continue
            await schedule_bulk_send(payload.user_ids, dt)
            total += 1
    return {'ok': True, 'scheduled_windows': total}

class SwapReq(BaseModel):
    user_id: int
    make_on_deck_current: bool = True

@router.post('/api/admin/weekly/swap')
async def admin_weekly_swap(payload: SwapReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    u = await db.get(User, payload.user_id)
    if not u:
        raise HTTPException(404, 'User not found')
    if not u.weekly_on_deck_prompt_id:
        return {'ok': False, 'error': 'No on-deck prompt'}
    prev = u.weekly_current_prompt_id
    u.weekly_current_prompt_id, u.weekly_on_deck_prompt_id = (u.weekly_on_deck_prompt_id, u.weekly_current_prompt_id)
    u.weekly_state = WeeklyState.queued
    u.weekly_queued_at = _now()
    if prev:
        await expire_active_tokens(db, u.id, prev)
    await db.commit()
    return {'ok': True}

class ChooseReq(BaseModel):
    user_id: int
    prompt_id: int
    push_previous_to_on_deck: bool = True

@router.post('/api/admin/weekly/choose')
async def admin_weekly_choose(payload: ChooseReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    u = await db.get(User, payload.user_id)
    p = await db.get(Prompt, payload.prompt_id)
    if not u or not p:
        raise HTTPException(404, 'User or Prompt not found')
    prev = u.weekly_current_prompt_id
    u.weekly_current_prompt_id = p.id
    u.weekly_state = WeeklyState.queued
    u.weekly_queued_at = _now()
    if payload.push_previous_to_on_deck:
        u.weekly_on_deck_prompt_id = prev
    if prev:
        await expire_active_tokens(db, u.id, prev)
    await db.commit()
    return {'ok': True}

class CronReq(BaseModel):
    days: List[int]
    hour: int
    minute: int
    tz: Optional[str] = None

@router.post('/api/admin/weekly/cron')
async def admin_weekly_update_cron(payload: CronReq, admin=Depends(require_admin_user)):
    set_weekly_cron(payload.days, payload.hour, payload.minute, payload.tz)
    return {'ok': True}

class QueueReq(BaseModel):
    user_id: int
    prompt_id: int

@router.post('/api/admin/weekly/queue')
async def admin_weekly_queue(payload: QueueReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    u = await db.get(User, payload.user_id)
    p = await db.get(Prompt, payload.prompt_id)
    if not u or not p:
        raise HTTPException(404, 'User or Prompt not found')
    u.weekly_on_deck_prompt_id = p.id
    if u.weekly_state == WeeklyState.not_sent:
        u.weekly_state = WeeklyState.queued
        u.weekly_queued_at = _now()
    await db.commit()
    return {'ok': True}

class SkipReq(BaseModel):
    user_id: int

@router.post('/api/admin/weekly/skip')
async def admin_weekly_skip(payload: SkipReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    await _skip(db, payload.user_id)
    await db.commit()
    return {'ok': True}

class FollowupReq(BaseModel):
    user_id: int
    response_id: int
    style: str = 'gentle'
    max_tokens: int = 300

@router.post('/api/admin/weekly/followup')
async def admin_weekly_followup(payload: FollowupReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    prompt_id = await make_llm_followup_prompt(db, payload.user_id, payload.response_id, payload.style, payload.max_tokens)
    u = await db.get(User, payload.user_id)
    u.weekly_on_deck_prompt_id = prompt_id
    await db.commit()
    return {'ok': True, 'on_deck_prompt_id': prompt_id}

@router.get('/api/admin/weekly/candidates')
async def admin_weekly_candidates(user_id: int=Query(...), k: int=Query(10, ge=1, le=50), db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    ids = await get_on_deck_candidates(db, user_id, k=k)
    if not ids:
        return {'items': []}
    prs = (await db.execute(select(Prompt).where(Prompt.id.in_(ids)))).unique().scalars().all()
    pmap = {p.id: p for p in prs}
    out = []
    for pid in ids:
        p = pmap.get(pid)
        if p:
            out.append({'id': p.id, 'text': p.text})
    return {'items': out}

@router.get('/api/admin/weekly/context')
async def admin_weekly_context(user_id: int=Query(...), db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    u = await db.get(User, user_id)
    if not u:
        raise HTTPException(404, 'User not found')
    cur = await db.get(Prompt, u.weekly_current_prompt_id) if u.weekly_current_prompt_id else None
    lr = (await db.execute(select(Response).where(Response.user_id == user_id).order_by(Response.created_at.desc()).limit(1))).unique().scalars().first()
    return {'current_prompt': {'id': cur.id, 'text': cur.text} if cur else None, 'last_response': {'id': getattr(lr, 'id', None), 'text': getattr(lr, 'response_text', None) or getattr(lr, 'transcription', None)} if lr else None}

@router.post('/api/admin/weekly/copy-link')
async def admin_weekly_copy_link(payload: Ids, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    if not payload.user_id:
        raise HTTPException(400, 'user_id required')
    u = await db.get(User, payload.user_id)
    if not u or not u.weekly_current_prompt_id:
        raise HTTPException(404, 'No current prompt for user')
    tok = await get_or_refresh_active_token(db, u.id, u.weekly_current_prompt_id)
    await db.commit()
    return {'ok': True, 'link': f'/weekly/t/{tok.token}'}

def _wants_json(request: Request) -> bool:
    qp = request.query_params
    if qp.get('format') == 'json' or qp.get('json') == '1':
        return True
    accept = (request.headers.get('accept') or '').lower()
    return 'application/json' in accept and 'text/html' not in accept

@router.get('/weekly/t/{token}', include_in_schema=False)
async def weekly_token_click(token: str, request: Request, db: AsyncSession=Depends(get_db)):
    if request.method == 'HEAD':
        return FastResponse(status_code=204)
    tok = await mark_clicked(db, token)
    await db.commit()
    if not tok:
        if _wants_json(request):
            return JSONResponse({'ok': False, 'error': 'invalid_or_expired'}, status_code=400)
        return RedirectResponse(url='/login?notice=This+weekly+link+is+invalid+or+expired', status_code=303)
    return RedirectResponse(url=f'/user_record/{tok.prompt_id}?token={token}', status_code=303)

@router.get('/weekly/t/{token}.png')
async def weekly_token_pixel(token: str, db: AsyncSession=Depends(get_db)):
    await mark_opened(db, token)
    await db.commit()
    png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc``\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82'
    return FastResponse(content=png, media_type='image/png', headers={'Cache-Control': 'no-store, max-age=0', 'Pragma': 'no-cache', 'Expires': '0'})

@router.post('/weekly/token/use')
async def weekly_token_use(token: str=Form(...), db: AsyncSession=Depends(get_db)):
    tok = await mark_clicked(db, token)
    if not tok:
        await mark_opened(db, token)
        tok = (await db.execute(select(WeeklyToken).where(WeeklyToken.token == token))).scalars().first()
    await db.commit()
    if not tok:
        raise HTTPException(400, 'Token invalid or expired')
    return {'ok': True, 'user_id': tok.user_id, 'prompt_id': tok.prompt_id}

class HideInferredReq(BaseModel):
    dst_person_id: int
    rel_type: str

@router.get('/api/weekly/{user_id}/on_deck')
async def api_weekly_on_deck(user_id: int, k: int=Query(5, ge=1, le=25), admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    from app.services.assignment import get_on_deck_candidates
    ids = await get_on_deck_candidates(db, user_id, k=k)
    if not ids:
        return []
    rows = (await db.execute(select(Prompt).where(Prompt.id.in_(ids)))).unique().scalars().all()
    by_id = {p.id: p for p in rows}
    out = []
    for pid in ids:
        p = by_id.get(pid)
        if p:
            out.append({'id': p.id, 'text': p.text, 'chapter': p.chapter})
    return out

@router.get('/chapter/{chapter_id}', response_class=HTMLResponse)
async def chapter_view(chapter_id: str, request: Request, user=Depends(require_authenticated_html_user), db: AsyncSession=Depends(get_db)):
    stat = await chapter_status(db, chapter_id, user.id)
    ctx = {'request': request, 'user': user, 'chapter_key': stat.chapter, 'display_name': stat.display_name, 'ready': stat.ready, 'missing_prompts': stat.missing_prompts, 'latest_compilation': stat.latest_compilation.dict() if stat.latest_compilation else None}
    return templates.TemplateResponse('chapter_view.html', ctx)

@router.get('/api/chapter/{chapter_id}/status', response_model=ChapterStatusDTO)
async def api_chapter_status(chapter_id: str, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    return await chapter_status(db, chapter_id, user.id)

@router.get('/api/chapter/{chapter_id}/gaps', response_model=list[dict])
async def api_chapter_gaps(chapter_id: str, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    stat = await chapter_status(db, chapter_id, user.id)
    latest = stat.latest_compilation
    return [g.dict() for g in (latest.gap_questions if latest else [])]

@router.post('/api/chapter/{chapter_id}/compile', response_model=ChapterCompilationDTO)
async def api_chapter_compile(chapter_id: str, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    stat = await chapter_status(db, chapter_id, user.id)
    if not stat.ready:
        raise HTTPException(status_code=400, detail='Chapter is not ready: complete all assigned prompts.')
    dto = await compile_chapter(db, chapter_id, user.id, model='gpt-X')
    return dto

@router.post('/api/chapter/{chapter_id}/publish')
async def api_chapter_publish(chapter_id: str, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    from sqlalchemy import update, desc
    from app.models import ChapterCompilation
    latest = (await db.execute(select(ChapterCompilation).where((ChapterCompilation.user_id == user.id) & (ChapterCompilation.chapter == chapter_id)).order_by(ChapterCompilation.version.desc(), ChapterCompilation.created_at.desc()).limit(1))).scalars().first()
    if not latest:
        raise HTTPException(status_code=404, detail='No compilation to publish.')
    latest.status = 'published'
    await db.commit()
    return {'ok': True, 'id': latest.id, 'version': latest.version}
