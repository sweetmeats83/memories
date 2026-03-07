"""
People, kinship, and groups routes.
Extracted from routes.py.
"""
from __future__ import annotations

import os, re, mimetypes, shutil, string, secrets
from pathlib import Path as FSPath
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Body, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from sqlalchemy import select, exists, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.database import get_db
from app.models import (
    Person, RelationshipEdge, ResponsePerson,
    KinGroup, KinMembership, PersonShare,
    UserProfile, Response,
)
from app.utils import require_authenticated_user, require_authenticated_html_user, slugify
from app.services.people import upsert_person_for_user, resolve_person
from app.services.auto_tag import WHITELIST, reload_whitelist
from app.services.kinship import classify_kinship
from app.services.infer import infer_edges_for_person, commit_inferred_edges
from app.media_pipeline import MediaPipeline, UserBucketsStrategy

router = APIRouter()

templates = Jinja2Templates(directory='templates')

# Path constants (same calculation as routes.py)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_UPLOAD_DIR = os.path.join(_BASE_DIR, 'static', 'uploads')
_STATIC_DIR = FSPath(_BASE_DIR) / 'static'
_PIPELINE = MediaPipeline(static_root=_STATIC_DIR, path_strategy=UserBucketsStrategy())


def _join_code(n: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(n))


def _user_bucket_name(u) -> str:
    try:
        base = (u.username or '').strip()
        return slugify(base) if base else str(u.id)
    except Exception:
        return str(getattr(u, 'id', 'user'))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

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


class InferredEdgeDTO(BaseModel):
    src_id: int
    dst_id: int
    rel_type: str
    confidence: float
    source: str
    explain: str | None = None


class ResolvePersonReq(BaseModel):
    display_name: str
    role_hint: str | None = None


class PersonCreateReq(BaseModel):
    display_name: str
    role_hint: str | None = None


class AssignMeReq(BaseModel):
    person_id: int


class EdgeReq(BaseModel):
    src_person_id: int
    dst_person_id: int
    rel_type: str
    confidence: float = 0.9


class GroupCreateReq(BaseModel):
    name: str
    kind: str = 'family'


class GroupJoinReq(BaseModel):
    code: str


# ---------------------------------------------------------------------------
# People graph page
# ---------------------------------------------------------------------------

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
    # Everyone in the same family group shares their person nodes automatically
    my_groups_sq = select(KinMembership.group_id).where(KinMembership.user_id == user.id).scalar_subquery()
    cogroup_users_sq = select(KinMembership.user_id).where(KinMembership.group_id.in_(my_groups_sq)).scalar_subquery()
    shared_q = select(Person.id, Person.display_name).where(Person.owner_user_id.in_(cogroup_users_sq))
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
            mention_map_local: dict[int, int] = {}
            if owned_ids_only:
                mrows = (await db.execute(select(ResponsePerson.person_id, func.count(ResponsePerson.id)).join(Response, Response.id == ResponsePerson.response_id).where(Response.user_id == user.id, ResponsePerson.person_id.in_(owned_ids_only)).group_by(ResponsePerson.person_id))).all()
                mention_map_local = {pid: int(cnt) for pid, cnt in mrows or []}

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
                if confirmed or mention_map_local.get(pid, 0) > 0:
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
                        if not inj_rel and mention_map_local.get(pid, 0) > 0:
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
                        elif not r and mention_map_local.get(pid, 0) > 0:
                            await _persist_edge_pair(me_pid, pid, 'friend-of')
            try:
                await db.commit()
            except Exception:
                await db.rollback()
    except Exception:
        pass
    return {'nodes': nodes, 'edges': edges}


# ---------------------------------------------------------------------------
# Roles whitelist
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Person detail, patch, infer
# ---------------------------------------------------------------------------

@router.get('/api/people/{person_id}')
async def api_people_detail(person_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p: Person | None = await db.get(Person, person_id)
    if not p:
        raise HTTPException(404, 'Person not found')
    is_owner = p.owner_user_id == user.id
    is_shared = False
    if not is_owner:
        # Visible if the person's owner is in the same family group as the current user
        my_groups_sq = select(KinMembership.group_id).where(KinMembership.user_id == user.id).scalar_subquery()
        is_shared = bool(await db.scalar(
            select(KinMembership.id).where(
                KinMembership.user_id == p.owner_user_id,
                KinMembership.group_id.in_(my_groups_sq),
            ).limit(1)
        ))
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


@router.get('/api/people/{person_id}/infer/preview')
async def api_people_infer_preview(person_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p = await db.get(Person, person_id)
    if not p or p.owner_user_id != user.id:
        raise HTTPException(404, 'Person not found')
    cands = await infer_edges_for_person(db, user_id=user.id, person_id=person_id)
    out = []
    for c in cands:
        label = None
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


@router.post('/api/people/{person_id}/infer/auto')
async def api_people_infer_auto(person_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    """Auto-commit only high-confidence (>=0.80) inferred edges.
    Safe to call silently after saving a person — will not create speculative links."""
    p = await db.get(Person, person_id)
    if not p or p.owner_user_id != user.id:
        raise HTTPException(404, 'Person not found')
    cands = await infer_edges_for_person(db, user_id=user.id, person_id=person_id)
    high_conf = [c for c in cands if c.confidence >= 0.80]
    n = await commit_inferred_edges(db, user_id=user.id, candidates=high_conf)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
    return {'person_id': person_id, 'inserted': n}


# ---------------------------------------------------------------------------
# Kinship classification
# ---------------------------------------------------------------------------

@router.get('/api/people/kinship')
async def api_people_kinship(ego_id: str=Query(...), alter_id: str=Query(...), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
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
    p_obj: Person | None = await db.get(Person, pid)
    if not p_obj or p_obj.owner_user_id != user.id:
        raise HTTPException(404, 'Person not found')
    role = None
    m = getattr(p_obj, 'meta', None)
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


# ---------------------------------------------------------------------------
# Person PATCH / DELETE / inferred list / edge delete
# ---------------------------------------------------------------------------

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
    for bool_field in ('role_hint', 'dot_color', 'deceased', 'connect_to_owner', 'hidden', 'inferred', 'birth_date'):
        val = getattr(payload, bool_field, None)
        if val is None:
            continue
        m = getattr(p, 'meta', None) or {}
        if not isinstance(m, dict):
            m = {}
        if bool_field in ('deceased', 'connect_to_owner', 'hidden', 'inferred'):
            m[bool_field] = bool(val)
        elif bool_field == 'dot_color':
            clr = (val or '').strip()
            if clr:
                m['dot_color'] = clr
            else:
                m.pop('dot_color', None)
        elif bool_field == 'role_hint':
            m['role_hint'] = (val or '').strip()
        elif bool_field == 'birth_date':
            bd = (val or '').strip()
            if bd:
                m['birth_date'] = bd
            else:
                m.pop('birth_date', None)
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


# ---------------------------------------------------------------------------
# People resolve / add / assign-me / photo / delete / edges / tree
# ---------------------------------------------------------------------------

@router.post('/api/people/resolve')
async def api_people_resolve(payload: ResolvePersonReq, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p = await resolve_person(db, user.id, payload.display_name)
    if p is None:
        raise HTTPException(status_code=409, detail="Ambiguous name matches multiple people; provide a full name.")
    await db.commit()
    return {'person_id': p.id, 'created': True}


@router.post('/api/people/me/assign')
async def api_people_assign_me(payload: AssignMeReq, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
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
        for rpid, rmeta, dname in rows:
            if isinstance(rmeta, dict) and rmeta.get('connect_to_owner'):
                rh = str(rmeta.get('role_hint', '')).strip().lower()
                if rh in {'you', 'self', 'me'}:
                    me_pid = rpid
                    break
        if me_pid is None:
            uname = (user.username or '').strip().lower()
            uemail = (user.email or '').strip().lower()
            for rpid, rmeta, dname in rows:
                dn = (dname or '').strip().lower()
                if dn and (dn == uname or dn == uemail or dn == 'you'):
                    me_pid = rpid
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
                await db.execute(pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=p.id, dst_id=me_pid, rel_type='parent-of', confidence=0.95).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type']))
                await db.execute(pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=me_pid, dst_id=p.id, rel_type='child-of', confidence=0.95).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type']))
            elif r in {'son', 'daughter', 'child'}:
                await db.execute(pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=me_pid, dst_id=p.id, rel_type='parent-of', confidence=0.95).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type']))
                await db.execute(pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=p.id, dst_id=me_pid, rel_type='child-of', confidence=0.95).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type']))
            else:
                await db.execute(pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=me_pid, dst_id=p.id, rel_type=rel, confidence=0.9).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type']))
                await db.execute(pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=p.id, dst_id=me_pid, rel_type=rel, confidence=0.9).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type']))
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
    abs_dir = _STATIC_DIR / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    rel_path = os.path.join(rel_dir, f'photo{ext}')
    abs_path = _STATIC_DIR / rel_path
    with open(abs_path, 'wb') as w:
        shutil.copyfileobj(file.file, w)
    old = (p.photo_url or '').strip()
    if old:
        old_rel = old.lstrip('/').replace('\\', '/')
        if not old_rel.startswith('uploads/'):
            old_rel = f'uploads/{old_rel}'
        if old_rel != rel_path.replace('\\', '/'):
            try:
                (_STATIC_DIR / old_rel).unlink(missing_ok=True)
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
            (_STATIC_DIR / reln).unlink(missing_ok=True)
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
            (_STATIC_DIR / reln).unlink(missing_ok=True)
        except Exception:
            pass
    await db.delete(p)
    await db.commit()
    return {'ok': True}


@router.post('/api/people/edges')
async def api_people_edge(payload: EdgeReq, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):

    def _canonical_rel(rt: str) -> str:
        rt = (rt or '').strip().lower()
        rt = rt.replace('relationship:', '').replace('role:', '')
        if rt in ('wife-of', 'husband-of'):
            rt = 'spouse-of'
        if rt in {'mother-of', 'father-of', 'parent-of'}:
            return 'parent-of'
        if rt in {'son-of', 'daughter-of', 'child-of'}:
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
    my_groups_sq = select(KinMembership.group_id).where(KinMembership.user_id == user.id).scalar_subquery()
    cogroup_users_sq = select(KinMembership.user_id).where(KinMembership.group_id.in_(my_groups_sq)).scalar_subquery()
    shared_q = select(Person.id, Person.display_name).where(Person.owner_user_id.in_(cogroup_users_sq))
    rows = (await db.execute(own_q.union(shared_q))).all()
    nodes = [{'id': rid, 'name': name} for rid, name in rows]
    return {'nodes': nodes, 'edges': []}


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

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
