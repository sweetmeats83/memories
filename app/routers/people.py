"""
People, kinship, and groups routes.
Extracted from routes.py.
"""
from __future__ import annotations

import logging
import os, re, mimetypes, shutil, string, secrets
from pathlib import Path as FSPath
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Body, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from sqlalchemy import select, exists, func, update as sa_update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.database import get_db
from app.models import (
    Person, PersonAlias, RelationshipEdge, ResponsePerson,
    KinGroup, KinMembership, PersonShare,
    UserProfile, Response,
)
from app.utils import require_authenticated_user, require_authenticated_html_user, slugify
from app.services.people import upsert_person_for_user, resolve_person

# In-memory set of user_ids currently running infer/run-all — prevents concurrent DB-heavy runs
_inferring: set[int] = set()
from app.services.auto_tag import WHITELIST, reload_whitelist
from app.services.kinship import classify_kinship
from app.services.infer import infer_edges_for_person, commit_inferred_edges, infer_all_for_user, FAMILY_EDGE_TYPES
from app.media_pipeline import MediaPipeline, UserBucketsStrategy

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level edge inverse mapping (shared by edge injection + detail view)
# ---------------------------------------------------------------------------
_INVERT_REL: dict[str, str] = {
    'mother-of': 'child-of', 'father-of': 'child-of',
    'parent-of': 'child-of', 'step-parent-of': 'child-of', 'adoptive-parent-of': 'child-of',
    'child-of': 'parent-of', 'son-of': 'parent-of', 'daughter-of': 'parent-of',
    'grandparent-of': 'grandchild-of', 'grandchild-of': 'grandparent-of',
    'sibling-of': 'sibling-of', 'half-sibling-of': 'half-sibling-of', 'step-sibling-of': 'step-sibling-of',
    'aunt-of': 'niece-of', 'uncle-of': 'nephew-of', 'niece-of': 'aunt-of', 'nephew-of': 'uncle-of',
    'cousin-of': 'cousin-of',
    'aunt-in-law-of': 'niece-in-law-of', 'niece-in-law-of': 'aunt-in-law-of',
    'cousin-in-law-of': 'cousin-in-law-of',
    'spouse-of': 'spouse-of', 'partner-of': 'partner-of',
    'ex-partner-of': 'ex-partner-of', 'ex-spouse-of': 'ex-spouse-of',
    'friend-of': 'friend-of', 'neighbor-of': 'neighbor-of', 'coworker-of': 'coworker-of',
    'mentor-of': 'student-of', 'teacher-of': 'student-of', 'student-of': 'teacher-of',
    'coach-of': 'student-of',
}

def _invert_rel(rt: str) -> str:
    return _INVERT_REL.get((rt or '').strip().lower(), rt)


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
# Self-person helpers
# ---------------------------------------------------------------------------

async def _get_self_person_id(
    db: AsyncSession,
    user,
    group_id: int | None = None,
    *,
    save: bool = True,
) -> int | None:
    """
    Find the person node that represents the logged-in user (their "you" dot).

    Resolution order:
    1. privacy_prefs["self_person_id"] — authoritative stored value
    2. Any person owned by this user with connect_to_owner=True and role_hint='you'
    3. Any person in the visible set (group + private) whose display_name matches
       the user's display name, username, or email
    4. Any private person with role_hint='you'

    When found via fallback, saves to privacy_prefs so future lookups are O(1).
    """
    from sqlalchemy import or_

    prof = await db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))

    # 1) Stored reference
    stored_id: int | None = None
    if prof:
        stored_id = (prof.privacy_prefs or {}).get("self_person_id")
    if stored_id:
        # Verify the person still exists and is visible
        exists = await db.get(Person, stored_id)
        if exists:
            return stored_id
        # Stale — fall through to detection below
        stored_id = None

    # 2) Search among all owned persons first (connect_to_owner + you)
    owned_rows = (await db.execute(
        select(Person.id, Person.meta, Person.display_name)
        .where(Person.owner_user_id == user.id)
    )).all()
    me_pid: int | None = None
    me_name: str | None = None
    for pid, meta, dname in owned_rows:
        if isinstance(meta, dict) and meta.get('connect_to_owner'):
            rh = str(meta.get('role_hint', '')).strip().lower()
            if rh in {'you', 'self', 'me'}:
                me_pid = pid
                me_name = dname
                break

    # 3) Name-match across ALL visible persons (catches migrated/merged self-person)
    if me_pid is None:
        uname = (getattr(user, 'username', '') or '').strip().lower()
        uemail = (getattr(user, 'email', '') or '').strip().lower()
        disp_name = (getattr(prof, 'display_name', '') or '').strip().lower() if prof else ''
        name_candidates = {n for n in [uname, uemail, disp_name] if n}

        if name_candidates:
            if group_id is not None:
                all_rows = (await db.execute(
                    select(Person.id, Person.meta, Person.display_name)
                    .where(or_(Person.group_id == group_id,
                               (Person.owner_user_id == user.id) & Person.group_id.is_(None)))
                )).all()
            else:
                all_rows = owned_rows
            for pid, meta, dname in all_rows:
                dn = (dname or '').strip().lower()
                if dn and dn in name_candidates:
                    me_pid = pid
                    me_name = dname
                    break

    # 4) Any private person with role_hint='you' (even without connect_to_owner)
    if me_pid is None:
        for pid, meta, dname in owned_rows:
            if isinstance(meta, dict):
                rh = str(meta.get('role_hint', '')).strip().lower()
                if rh in {'you', 'self', 'me'}:
                    me_pid = pid
                    me_name = dname
                    break

    # Save for future lookups
    if me_pid is not None and save and prof and (prof.privacy_prefs or {}).get('self_person_id') != me_pid:
        prefs = prof.privacy_prefs or {}
        prefs['self_person_id'] = me_pid
        prof.privacy_prefs = prefs
        from sqlalchemy.orm.attributes import flag_modified as _fm
        _fm(prof, 'privacy_prefs')
        try:
            await db.flush()
        except Exception:
            pass

    return me_pid


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
    _pg_memberships = (await db.execute(select(KinMembership.group_id).where(KinMembership.user_id == user.id))).scalars().all()
    _pg_group_id: int | None = _pg_memberships[0] if len(_pg_memberships) == 1 else None
    try:
        me_person_id = await _get_self_person_id(db, user, group_id=_pg_group_id)
        await db.commit()
    except Exception:
        me_person_id = None
        try:
            await db.rollback()
        except Exception:
            pass
    return templates.TemplateResponse(request, 'people_graph.html', {'request': request, 'user': user, 'me_person_id': me_person_id})


@router.get('/api/people/graph/version')
async def api_people_graph_version(user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    """Return a lightweight version token (person count) the JS can poll to detect graph changes."""
    _gv_memberships = (await db.execute(select(KinMembership.group_id).where(KinMembership.user_id == user.id))).scalars().all()
    _gv_group_id: int | None = _gv_memberships[0] if len(_gv_memberships) == 1 else None
    from sqlalchemy import or_
    if _gv_group_id is not None:
        count = await db.scalar(select(func.count(Person.id)).where(
            or_(Person.group_id == _gv_group_id, (Person.owner_user_id == user.id) & Person.group_id.is_(None))
        ))
    else:
        count = await db.scalar(select(func.count(Person.id)).where(Person.owner_user_id == user.id, Person.group_id.is_(None)))
    return {'version': count or 0}


@router.post('/api/people/claim-self')
async def api_people_claim_self(
    payload: dict = Body(...),
    user=Depends(require_authenticated_user),
    db: AsyncSession=Depends(get_db),
):
    """Set a person as the logged-in user's self/center node."""
    person_id = payload.get('person_id')
    if not person_id:
        raise HTTPException(400, 'person_id required')
    p = await db.get(Person, int(person_id))
    if not p:
        raise HTTPException(404, 'Person not found')
    # Tag the person as connect_to_owner + role_hint='you'
    meta = dict(p.meta or {})
    meta['connect_to_owner'] = True
    meta['role_hint'] = 'you'
    p.meta = meta
    flag_modified(p, 'meta')
    # Save to UserProfile.privacy_prefs
    prof = await db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
    if prof:
        prefs = prof.privacy_prefs or {}
        prefs['self_person_id'] = p.id
        prof.privacy_prefs = prefs
        flag_modified(prof, 'privacy_prefs')
    await db.commit()
    return {'ok': True, 'self_person_id': p.id}


@router.get('/api/people/graph')
async def api_people_graph(user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    # Determine user's single group (if any) for shared-scope queries
    _graph_memberships = (await db.execute(select(KinMembership.group_id).where(KinMembership.user_id == user.id))).scalars().all()
    _graph_group_id: int | None = _graph_memberships[0] if len(_graph_memberships) == 1 else None

    # Persons: group-scoped (shared) + user's private (no group)
    private_q = select(Person.id, Person.display_name).where(Person.owner_user_id == user.id, Person.group_id.is_(None))
    if _graph_group_id is not None:
        group_q = select(Person.id, Person.display_name).where(Person.group_id == _graph_group_id)
        prows = (await db.execute(private_q.union(group_q))).all()
    else:
        prows = (await db.execute(private_q)).all()
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
                if _graph_group_id is not None:
                    prows = (await db.execute(private_q.union(group_q))).all()
                else:
                    prows = (await db.execute(private_q)).all()
        except Exception:
            pass
    pmap = {rid: name for rid, name in prows}
    node_ids = set(pmap.keys())
    # Resolve self/center node and ensure it's in the graph even if filtered
    me_pid: int | None = None
    try:
        me_pid = await _get_self_person_id(db, user, group_id=_graph_group_id)
        if me_pid is not None and me_pid not in node_ids:
            me_name_fallback = await db.scalar(select(Person.display_name).where(Person.id == me_pid))
            prows = list(prows) + [(me_pid, me_name_fallback or 'You')]
            pmap[me_pid] = me_name_fallback or 'You'
            node_ids.add(me_pid)
        # Commit the self_person_id write from _get_self_person_id (if any)
        await db.commit()
    except Exception:
        try:
            await db.rollback()
        except Exception:
            pass
    # Load per-user hidden set from privacy_prefs
    _graph_prof = await db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
    _graph_prefs = (_graph_prof.privacy_prefs or {}) if _graph_prof else {}
    user_hidden_ids: set[int] = set(int(x) for x in (_graph_prefs.get('hidden_person_ids') or []))

    dead_map: dict[int, bool] = {}
    color_map: dict[int, str | None] = {}
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
        if rid in user_hidden_ids:
            continue
        if rid in own_ids and inferred_map.get(rid) and (mention_map.get(rid, 0) == 0) and (not (rolehint_map.get(rid) or connect_map.get(rid))):
            continue
        nodes.append({'id': rid, 'name': name, 'kind': 'person', 'dead': bool(dead_map.get(rid)), 'color': color_map.get(rid), 'mentions': mention_map.get(rid, 0), 'role_hint': rolehint_map.get(rid), 'connect_to_owner': connect_map.get(rid, False)})
    # Edges: group-scoped family edges + user's private social edges
    from sqlalchemy import or_
    if _graph_group_id is not None:
        ers = (await db.execute(select(RelationshipEdge).where(
            or_(
                (RelationshipEdge.user_id == user.id) & RelationshipEdge.group_id.is_(None),
                RelationshipEdge.group_id == _graph_group_id,
            )
        ))).scalars().all()
    else:
        ers = (await db.execute(select(RelationshipEdge).where(RelationshipEdge.user_id == user.id, RelationshipEdge.group_id.is_(None)))).scalars().all()
    edges = [{'id': e.id, 'src': e.src_id, 'dst': e.dst_id, 'rel': e.rel_type, 'confidence': getattr(e, 'confidence', None), 'generation': int((e.meta or {}).get('generation')) if isinstance(getattr(e, 'meta', None), dict) and (e.meta or {}).get('generation') is not None else None, 'hide_line': bool((e.meta or {}).get('hide_line', False))} for e in ers if e.src_id in node_ids and e.dst_id in node_ids]
    try:
        # me_pid was resolved above via _get_self_person_id
        if me_pid is not None:
            # Fetch owned persons for edge-injection (role_hint → infer initial edges)
            rows = (await db.execute(select(Person.id, Person.meta).where(Person.owner_user_id == user.id))).all()
            owned_ids_only = [pid for pid, _m in rows] or []
            mention_map_local: dict[int, int] = {}
            if owned_ids_only:
                mrows = (await db.execute(select(ResponsePerson.person_id, func.count(ResponsePerson.id)).join(Response, Response.id == ResponsePerson.response_id).where(Response.user_id == user.id, ResponsePerson.person_id.in_(owned_ids_only)).group_by(ResponsePerson.person_id))).all()
                mention_map_local = {pid: int(cnt) for pid, cnt in mrows or []}

            # Resolve user's single group for family-edge scoping
            _ep_memberships = (await db.execute(select(KinMembership.group_id).where(KinMembership.user_id == user.id))).scalars().all()
            _ep_group_id: int | None = _ep_memberships[0] if len(_ep_memberships) == 1 else None

            async def _persist_edge_pair(f_src: int, f_dst: int, f_rel: str):
                f_rel = (f_rel or '').strip().lower()
                if not f_rel:
                    return
                inv = _invert_rel(f_rel)
                if f_rel in FAMILY_EDGE_TYPES and _ep_group_id is not None:
                    stmt_fwd = pg_insert(RelationshipEdge.__table__).values(group_id=_ep_group_id, user_id=None, src_id=f_src, dst_id=f_dst, rel_type=f_rel, confidence=0.9).on_conflict_do_nothing()
                    await db.execute(stmt_fwd)
                    stmt_inv = pg_insert(RelationshipEdge.__table__).values(group_id=_ep_group_id, user_id=None, src_id=f_dst, dst_id=f_src, rel_type=inv, confidence=0.9).on_conflict_do_nothing()
                    await db.execute(stmt_inv)
                else:
                    stmt_fwd = pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=f_src, dst_id=f_dst, rel_type=f_rel, confidence=0.9).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type'])
                    await db.execute(stmt_fwd)
                    stmt_inv = pg_insert(RelationshipEdge.__table__).values(user_id=user.id, src_id=f_dst, dst_id=f_src, rel_type=inv, confidence=0.9).on_conflict_do_nothing(index_elements=['user_id', 'src_id', 'dst_id', 'rel_type'])
                    await db.execute(stmt_inv)

            for pid, meta in rows:
                if pid == me_pid:
                    continue
                role = 'you'
                confirmed = False
                if isinstance(meta, dict):
                    role = meta.get('role_hint') or 'you'
                    confirmed = bool(meta.get('connect_to_owner')) and bool(meta.get('role_hint'))
                # Only inject edges for explicitly confirmed people with a real role.
                # Mention-only people are shown in the graph but NOT auto-connected.
                if confirmed and pid in node_ids and me_pid in node_ids:
                    r = (role or '').strip().lower()
                    if r in {'you', 'self', 'me'}:
                        continue
                    inj_rel = None
                    if r in {'mother', 'father', 'parent'}:
                        inj_rel = 'child-of'
                        await _persist_edge_pair(pid, me_pid, 'parent-of')
                    elif r in {'son', 'daughter', 'child'}:
                        inj_rel = 'parent-of'
                        await _persist_edge_pair(me_pid, pid, 'parent-of')
                    elif r in {'spouse', 'partner', 'husband', 'wife'}:
                        inj_rel = 'spouse-of'
                        await _persist_edge_pair(me_pid, pid, 'spouse-of')
                    elif r in {'sibling', 'brother', 'sister'}:
                        inj_rel = 'sibling-of'
                        await _persist_edge_pair(me_pid, pid, 'sibling-of')
                    elif r in {'friend', 'neighbor', 'coworker', 'colleague'}:
                        inj_rel = f'{r}-of'
                        await _persist_edge_pair(me_pid, pid, inj_rel)
                    if inj_rel:
                        edges.append({'src': me_pid, 'dst': pid, 'rel': inj_rel})
            try:
                await db.commit()
            except Exception as _commit_err:
                logger.warning("Edge injection commit failed: %s", _commit_err)
                await db.rollback()
    except Exception as _inject_err:
        logger.warning("Edge injection failed: %s", _inject_err)

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
# Static-path people endpoints (must come before /{person_id} wildcard)
# ---------------------------------------------------------------------------

@router.get('/api/people-cleanup/inferred')
async def api_people_inferred(user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    _cu_memberships = (await db.execute(select(KinMembership.group_id).where(KinMembership.user_id == user.id))).scalars().all()
    _cu_group_id: int | None = _cu_memberships[0] if len(_cu_memberships) == 1 else None
    from sqlalchemy import or_
    if _cu_group_id is not None:
        rows = (await db.execute(select(Person).where(or_(Person.group_id == _cu_group_id, (Person.owner_user_id == user.id) & Person.group_id.is_(None))))).scalars().all()
    else:
        rows = (await db.execute(select(Person).where(Person.owner_user_id == user.id))).scalars().all()
    _inf_prof = await db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
    _inf_prefs = (_inf_prof.privacy_prefs or {}) if _inf_prof else {}
    _inf_hidden_ids: set[int] = set(int(x) for x in (_inf_prefs.get('hidden_person_ids') or []))
    inferred = []
    pids = []
    for p in rows:
        if p.id in _inf_hidden_ids:
            continue
        meta = getattr(p, 'meta', {}) or {}
        if isinstance(meta, dict) and bool(meta.get('inferred')):
            inferred.append(p)
            pids.append(p.id)
    counts: dict[int, int] = {}
    if pids:
        cnt_rows = (await db.execute(select(ResponsePerson.person_id, func.count(ResponsePerson.id)).join(Response, Response.id == ResponsePerson.response_id).where(Response.user_id == user.id, ResponsePerson.person_id.in_(pids)).group_by(ResponsePerson.person_id))).all()
        counts = {pid: int(c) for pid, c in cnt_rows}
    data = [{'id': p.id, 'name': p.display_name, 'mentions': counts.get(p.id, 0)} for p in inferred]
    data.sort(key=lambda x: (-x['mentions'], x['name'].lower()))
    return {'people': data}


@router.get('/api/people-cleanup/all')
async def api_people_all(user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    """Flat list of all visible people (group-scoped + private) including hidden — for the cleanup panel."""
    _ca_memberships = (await db.execute(select(KinMembership.group_id).where(KinMembership.user_id == user.id))).scalars().all()
    _ca_group_id: int | None = _ca_memberships[0] if len(_ca_memberships) == 1 else None
    # Per-user hidden set
    _ca_prof = await db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
    _ca_prefs = (_ca_prof.privacy_prefs or {}) if _ca_prof else {}
    _ca_hidden_ids: set[int] = set(int(x) for x in (_ca_prefs.get('hidden_person_ids') or []))
    from sqlalchemy import or_
    if _ca_group_id is not None:
        rows = (await db.execute(select(Person).where(or_(Person.group_id == _ca_group_id, (Person.owner_user_id == user.id) & Person.group_id.is_(None))))).scalars().all()
    else:
        rows = (await db.execute(select(Person).where(Person.owner_user_id == user.id))).scalars().all()
    pids = [p.id for p in rows]
    counts: dict[int, int] = {}
    if pids:
        cnt_rows = (await db.execute(
            select(ResponsePerson.person_id, func.count(ResponsePerson.id))
            .join(Response, Response.id == ResponsePerson.response_id)
            .where(Response.user_id == user.id, ResponsePerson.person_id.in_(pids))
            .group_by(ResponsePerson.person_id)
        )).all()
        counts = {pid: int(c) for pid, c in cnt_rows}
    edge_counts: dict[int, int] = {}
    if pids:
        _edge_scope_cond = or_(RelationshipEdge.user_id == user.id, RelationshipEdge.group_id == _ca_group_id) if _ca_group_id else (RelationshipEdge.user_id == user.id)
        e_rows = (await db.execute(
            select(RelationshipEdge.src_id, func.count(RelationshipEdge.id))
            .where(_edge_scope_cond, RelationshipEdge.src_id.in_(pids))
            .group_by(RelationshipEdge.src_id)
        )).all()
        for pid, cnt in e_rows:
            edge_counts[pid] = edge_counts.get(pid, 0) + int(cnt)
        e_rows2 = (await db.execute(
            select(RelationshipEdge.dst_id, func.count(RelationshipEdge.id))
            .where(_edge_scope_cond, RelationshipEdge.dst_id.in_(pids))
            .group_by(RelationshipEdge.dst_id)
        )).all()
        for pid, cnt in e_rows2:
            edge_counts[pid] = edge_counts.get(pid, 0) + int(cnt)
    data = []
    for p in rows:
        meta = getattr(p, 'meta', {}) or {}
        data.append({
            'id':      p.id,
            'name':    p.display_name or '',
            'hidden':  p.id in _ca_hidden_ids,
            'role_hint': meta.get('role_hint'),
            'connect_to_owner': bool(meta.get('connect_to_owner')),
            'mentions': counts.get(p.id, 0),
            'edges':   edge_counts.get(p.id, 0),
        })
    data.sort(key=lambda x: (x['mentions'], x['name'].lower()))
    return {'people': data}


def _name_key(name: str) -> str:
    """Normalise a display name for fuzzy duplicate detection."""
    return re.sub(r'[^a-z0-9]', '', (name or '').lower().strip())


def _names_similar(a: str, b: str) -> str | None:
    """Return a reason string if two normalised name keys are likely the same person, else None.
    Checks: prefix match (one starts with the other, min 4 chars),
            then fuzzy ratio >= 0.82 via SequenceMatcher (catches transcription misspellings).
    """
    from difflib import SequenceMatcher
    if not a or not b:
        return None
    if a == b:
        return 'Same name'
    # Prefix match — shorter must be at least 4 chars
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 4 and longer.startswith(shorter):
        return 'Name prefix match'
    # Fuzzy match for transcription misspellings — both must be at least 4 chars
    if len(a) >= 4 and len(b) >= 4:
        ratio = SequenceMatcher(None, a, b).ratio()
        if ratio >= 0.82:
            return f'Similar name ({int(ratio * 100)}% match)'
    return None


@router.get('/api/people-cleanup/duplicates')
async def api_people_duplicates(user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    """Return groups of people whose normalised names are identical or very similar."""
    _cd_memberships = (await db.execute(select(KinMembership.group_id).where(KinMembership.user_id == user.id))).scalars().all()
    _cd_group_id: int | None = _cd_memberships[0] if len(_cd_memberships) == 1 else None
    from sqlalchemy import or_
    if _cd_group_id is not None:
        rows = (await db.execute(
            select(Person.id, Person.display_name, Person.meta)
            .where(or_(Person.group_id == _cd_group_id, (Person.owner_user_id == user.id) & Person.group_id.is_(None)))
        )).all()
    else:
        rows = (await db.execute(
            select(Person.id, Person.display_name, Person.meta)
            .where(Person.owner_user_id == user.id)
        )).all()

    pids = [r[0] for r in rows]
    alias_rows: list[tuple[int, str]] = []
    if pids:
        alias_rows = (await db.execute(
            select(PersonAlias.person_id, PersonAlias.alias).where(PersonAlias.person_id.in_(pids))
        )).all()

    # Per-user hidden set (reuse from cleanup all if possible; re-query here for safety)
    _dup_prof = await db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
    _dup_prefs = (_dup_prof.privacy_prefs or {}) if _dup_prof else {}
    _dup_hidden_ids: set[int] = set(int(x) for x in (_dup_prefs.get('hidden_person_ids') or []))

    # Build a node lookup
    node_by_pid: dict[int, dict] = {}
    for pid, dname, meta in rows:
        node_by_pid[pid] = {'id': pid, 'name': dname or '', 'hidden': pid in _dup_hidden_ids}

    # All normalised keys per person: display_name + all aliases
    all_keys_for: dict[int, set[str]] = {}  # pid → set of norm keys
    for pid, dname, _ in rows:
        k = _name_key(dname)
        if k:
            all_keys_for.setdefault(pid, set()).add(k)
    for pid, alias in alias_rows:
        k = _name_key(alias)
        if k and pid in node_by_pid:
            all_keys_for.setdefault(pid, set()).add(k)

    # Compare every pair once
    seen_pairs: set[tuple[int, int]] = set()
    exact_groups: dict[str, list[dict]] = {}   # norm_key → persons with that exact key
    fuzzy_pairs: list[dict] = []

    person_list = list(node_by_pid.items())
    for i, (pid_a, node_a) in enumerate(person_list):
        keys_a = all_keys_for.get(pid_a, set())
        for pid_b, node_b in person_list[i + 1:]:
            pair = (pid_a, pid_b)
            if pair in seen_pairs:
                continue
            keys_b = all_keys_for.get(pid_b, set())
            reason = ''
            # Exact match on any key
            common = keys_a & keys_b
            if common:
                reason = 'Same name'
            # Prefix/fuzzy match across all key pairs
            if not reason:
                for ka in keys_a:
                    for kb in keys_b:
                        if ka != kb:
                            r = _names_similar(ka, kb)
                            if r:
                                reason = r
                                break
                    if reason:
                        break
            if reason:
                seen_pairs.add(pair)
                fuzzy_pairs.append({'a': node_a, 'b': node_b, 'reason': reason})

    # Also build exact-match groups (same display_name key) for the old `groups` field
    by_key: dict[str, list[dict]] = {}
    for pid, dname, _ in rows:
        k = _name_key(dname)
        if k:
            by_key.setdefault(k, []).append(node_by_pid[pid])
    exact_groups_list = [v for v in by_key.values() if len(v) > 1]

    return {
        'groups': exact_groups_list,
        'alias_pairs': fuzzy_pairs,
        '_debug': {
            'group_id': _cd_group_id,
            'person_count': len(rows),
            'alias_count': len(alias_rows),
            'keys_sample': {str(pid): list(keys) for pid, keys in list(all_keys_for.items())[:10]},
        },
    }


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
        # Visible if the person is scoped to a group the current user belongs to
        if p.group_id is not None:
            is_shared = bool(await db.scalar(
                select(KinMembership.id).where(
                    KinMembership.user_id == user.id,
                    KinMembership.group_id == p.group_id,
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

    _invert = _invert_rel

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
        conns.append({'edge_id': e.id, 'person_id': e.dst_id, 'name': neighbors.get(e.dst_id, 'Unknown'), 'rel_type': e.rel_type, 'direction': 'out', 'generation': _gen_from_meta(e), 'hide_line': bool((e.meta or {}).get('hide_line', False))})
    for e in edges_in:
        inv = _invert(e.rel_type)
        conns.append({'edge_id': e.id, 'person_id': e.src_id, 'name': neighbors.get(e.src_id, 'Unknown'), 'rel_type': inv, 'direction': 'out', 'generation': _gen_from_meta(e), 'hide_line': bool((e.meta or {}).get('hide_line', False))})
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
    if not p or not await _can_edit_person(p, user.id, db):
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
    if not p or not await _can_edit_person(p, user.id, db):
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
    if not p or not await _can_edit_person(p, user.id, db):
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
# Kinship helpers
# ---------------------------------------------------------------------------

def _kin_plus_great(term: str) -> str:
    return 'great-' + term if term.endswith('grandparent') else term


async def _kinship_label_from_anchors(
    db: AsyncSession, *, user_id: int, pid: int
) -> str | None:
    """
    Derive a kinship label using known-role persons (parent / spouse / sibling) as anchors.
    Used as fallback when no authoritative self-person node is available.
    Returns the label string, or None when undetermined.
    """
    rows = (await db.execute(select(Person).where(Person.owner_user_id == user_id))).scalars().all()
    anchors: list[tuple[str, Person]] = []
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

    for kind, anchor in sorted(anchors, key=lambda t: {'parent': 0, 'spouse': 1, 'sibling': 2}.get(t[0], 9)):
        try:
            b = await classify_kinship(db, user_id=user_id, ego_id=anchor.id, alter_id=pid)
        except Exception:
            continue
        L = (b.label_neutral or '').lower()
        if kind == 'parent':
            if L == 'parent':
                return 'grandparent'
            if L.endswith('grandparent'):
                return _kin_plus_great(b.label_neutral)
            if L == 'child':
                return 'sibling'
            if L.endswith('grandchild'):
                if L == 'grandchild':
                    return 'niece/nephew'
                return b.label_neutral.replace('grandchild', 'grandniece/nephew')
            if L == 'sibling':
                return 'aunt/uncle'
            if 'grandaunt/uncle' in L:
                return 'great-' + b.label_neutral
            if 'aunt/uncle' in L:
                return 'grandaunt/uncle'
            if 'niece/nephew' in L:
                removed = L.count('great-') if L.startswith('great-') else 0
                base_lbl = 'first cousin'
                if removed == 0:
                    return base_lbl
                elif removed == 1:
                    return base_lbl + ' once removed'
                else:
                    return base_lbl + f' {removed} times removed'
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
                    return f'{deg_word} cousin{tail}'
                except Exception:
                    return 'cousin once removed'
        elif kind == 'spouse':
            if L == 'self':
                return 'spouse'
            if L.endswith('-in-law'):
                return b.label_neutral[:-len('-in-law')]
            if L and L not in ('spouse', 'partner', 'ex-spouse', 'ex-partner', 'related'):
                return b.label_neutral + '-in-law'
        elif kind == 'sibling':
            if L == 'child':
                return 'niece/nephew'
            if L.endswith('grandchild'):
                if L == 'grandchild':
                    return 'grandniece/nephew'
                return b.label_neutral.replace('grandchild', 'grandniece/nephew')
            if L in {'spouse', 'partner'}:
                return 'sibling-in-law'

    # Role-hint final fallback: trust the target person's own meta
    p_obj: Person | None = await db.get(Person, pid)
    if p_obj and p_obj.owner_user_id == user_id:
        m = getattr(p_obj, 'meta', None)
        if isinstance(m, dict) and (m.get('connect_to_owner') or m.get('role_hint')):
            base = str(m.get('role_hint', '')).strip().lower()
            if base in {'mother', 'father', 'parent'}:
                return 'parent'
            if base in {'son', 'daughter', 'child'}:
                return 'child'
            if base in {'spouse', 'partner', 'husband', 'wife'}:
                return 'spouse'
            if base:
                return base
    return None


# ---------------------------------------------------------------------------
# Kinship classification
# ---------------------------------------------------------------------------


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
    # Use the authoritative self-person lookup (privacy_prefs → fallbacks)
    self_pid: int | None = None
    try:
        _you_group_memberships = (await db.execute(select(KinMembership.group_id).where(KinMembership.user_id == user.id))).scalars().all()
        _you_group_id = _you_group_memberships[0] if len(_you_group_memberships) == 1 else None
        self_pid = await _get_self_person_id(db, user, group_id=_you_group_id)
        await db.commit()
    except Exception:
        try: await db.rollback()
        except Exception: pass

    if self_pid is not None:
        if pid == self_pid:
            return {'label_neutral': 'you'}
        try:
            res = await classify_kinship(db, user_id=user.id, ego_id=self_pid, alter_id=pid)
            lbl = res.label_gendered or res.label_neutral
            if lbl and lbl != 'related':
                return {'label_neutral': lbl}
        except Exception:
            pass
        return {'label_neutral': None}

    lbl = await _kinship_label_from_anchors(db, user_id=user.id, pid=pid)
    return {'label_neutral': lbl}


# ---------------------------------------------------------------------------
# Person PATCH / DELETE / inferred list / edge delete
# ---------------------------------------------------------------------------

async def _can_edit_person(p: Person, user_id: int, db: AsyncSession) -> bool:
    """Owner can always edit. Group members can edit group-scoped persons."""
    if p.owner_user_id == user_id:
        return True
    if p.group_id is not None:
        return bool(await db.scalar(
            select(KinMembership.id).where(
                KinMembership.user_id == user_id,
                KinMembership.group_id == p.group_id,
            ).limit(1)
        ))
    return False


@router.patch('/api/people/{person_id}')
async def api_people_patch(person_id: int, payload: PersonPatchReq, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    p: Person | None = await db.get(Person, person_id)
    if not p or not await _can_edit_person(p, user.id, db):
        raise HTTPException(404, 'Person not found')
    changed = False
    if payload.display_name is not None:
        nm = (payload.display_name or '').strip()
        if nm and nm != p.display_name:
            p.display_name = nm
            changed = True
    if payload.hidden is not None:
        # Per-user hide: stored in UserProfile.privacy_prefs, not on the shared Person record
        prof = await db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
        if prof:
            prefs = prof.privacy_prefs or {}
            hidden_ids: list = list(prefs.get('hidden_person_ids') or [])
            if payload.hidden:
                if person_id not in hidden_ids:
                    hidden_ids.append(person_id)
            else:
                hidden_ids = [x for x in hidden_ids if x != person_id]
            prefs['hidden_person_ids'] = hidden_ids
            prof.privacy_prefs = prefs
            flag_modified(prof, 'privacy_prefs')
            changed = True

    for bool_field in ('role_hint', 'dot_color', 'deceased', 'connect_to_owner', 'inferred', 'birth_date'):
        val = getattr(payload, bool_field, None)
        if val is None:
            continue
        m = getattr(p, 'meta', None) or {}
        if not isinstance(m, dict):
            m = {}
        if bool_field in ('deceased', 'connect_to_owner', 'inferred'):
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


@router.patch('/api/people/edges/{edge_id}')
async def api_people_edge_patch(edge_id: int, body: dict = Body(...), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    e = await db.get(RelationshipEdge, edge_id)
    if not e:
        raise HTTPException(404, 'Edge not found')
    src_p = await db.get(Person, e.src_id)
    dst_p = await db.get(Person, e.dst_id)
    if not src_p or not dst_p:
        raise HTTPException(404, 'Edge endpoints not found')
    if src_p.owner_user_id != user.id and dst_p.owner_user_id != user.id:
        raise HTTPException(403, 'Not allowed to edit this connection')
    if 'hide_line' in body:
        meta = dict(e.meta or {})
        meta['hide_line'] = bool(body['hide_line'])
        e.meta = meta
        # Mirror onto the inverse edge so both directions agree
        inv_rel = _invert_rel(e.rel_type)
        inv_edge = (await db.execute(
            select(RelationshipEdge).where(
                RelationshipEdge.src_id == e.dst_id,
                RelationshipEdge.dst_id == e.src_id,
                RelationshipEdge.rel_type == inv_rel,
            )
        )).scalars().first()
        if inv_edge:
            inv_meta = dict(inv_edge.meta or {})
            inv_meta['hide_line'] = bool(body['hide_line'])
            inv_edge.meta = inv_meta
    await db.commit()
    return {'ok': True}


class MergePersonReq(BaseModel):
    keep_id: int
    delete_id: int


@router.post('/api/people/merge')
async def api_people_merge(payload: MergePersonReq, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    """Merge delete_id into keep_id: transfer all references then delete."""
    keep = await db.get(Person, payload.keep_id)
    gone = await db.get(Person, payload.delete_id)
    if not keep or not await _can_edit_person(keep, user.id, db):
        raise HTTPException(404, 'Keep person not found')
    if not gone or not await _can_edit_person(gone, user.id, db):
        raise HTTPException(404, 'Delete person not found')
    if keep.id == gone.id:
        raise HTTPException(400, 'Cannot merge a person with themselves')

    # 1. Add deleted person's display name (and their aliases) as aliases on keep
    existing_aliases = {
        a.alias.lower() for a in (await db.execute(
            select(PersonAlias).where(PersonAlias.person_id == keep.id)
        )).scalars().all()
    }
    new_alias_names = {gone.display_name} if gone.display_name else set()
    gone_aliases = (await db.execute(
        select(PersonAlias).where(PersonAlias.person_id == gone.id)
    )).scalars().all()
    for a in gone_aliases:
        new_alias_names.add(a.alias)
    for aname in new_alias_names:
        if aname and aname.lower() not in existing_aliases and aname.lower() != (keep.display_name or '').lower():
            db.add(PersonAlias(person_id=keep.id, alias=aname))

    # 2. Re-point ResponsePerson rows
    await db.execute(
        sa_update(ResponsePerson)
        .where(ResponsePerson.person_id == gone.id)
        .values(person_id=keep.id)
    )

    # 3. Re-point RelationshipEdge rows (src and dst), skip if would create a duplicate
    # Include both user-private and group-scoped edges that reference the gone person
    _merge_group_ids = [keep.group_id, gone.group_id]
    _merge_group_ids = [g for g in _merge_group_ids if g is not None]
    from sqlalchemy import or_
    edge_scope = or_(
        RelationshipEdge.user_id == user.id,
        *([RelationshipEdge.group_id.in_(_merge_group_ids)] if _merge_group_ids else [])
    )
    edge_rows = (await db.execute(
        select(RelationshipEdge).where(
            edge_scope,
            (RelationshipEdge.src_id == gone.id) | (RelationshipEdge.dst_id == gone.id)
        )
    )).scalars().all()
    for e in edge_rows:
        new_src = keep.id if e.src_id == gone.id else e.src_id
        new_dst = keep.id if e.dst_id == gone.id else e.dst_id
        if new_src == new_dst:
            await db.delete(e)
            continue
        # Check for existing identical edge (same scope)
        if e.group_id is not None:
            clash = await db.scalar(select(RelationshipEdge.id).where(
                RelationshipEdge.group_id == e.group_id,
                RelationshipEdge.src_id == new_src,
                RelationshipEdge.dst_id == new_dst,
                RelationshipEdge.rel_type == e.rel_type,
            ))
        else:
            clash = await db.scalar(select(RelationshipEdge.id).where(
                RelationshipEdge.user_id == user.id,
                RelationshipEdge.src_id == new_src,
                RelationshipEdge.dst_id == new_dst,
                RelationshipEdge.rel_type == e.rel_type,
            ))
        if clash:
            await db.delete(e)
        else:
            e.src_id = new_src
            e.dst_id = new_dst

    # 4. Delete the gone person's aliases (already migrated) then the person itself
    for a in gone_aliases:
        await db.delete(a)
    await db.delete(gone)
    await db.commit()
    return {'ok': True, 'kept_id': keep.id}


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
    if payload.role_hint:
        meta['role_hint'] = payload.role_hint
        meta['connect_to_owner'] = True
    # If no role_hint, person is added to graph without auto-connecting to the user
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
        if me_pid and p.id != me_pid and payload.role_hint:
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
    if not p or not await _can_edit_person(p, user.id, db):
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
    if not p or not await _can_edit_person(p, user.id, db):
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
    if not p or not await _can_edit_person(p, user.id, db):
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
    inv_rt = _inverse_rel(rel_in)

    # Use group scope for family edges when user belongs to exactly one group
    _edge_memberships = (await db.execute(select(KinMembership.group_id).where(KinMembership.user_id == user.id))).scalars().all()
    _edge_group_id: int | None = _edge_memberships[0] if len(_edge_memberships) == 1 else None
    use_group = rel_in in FAMILY_EDGE_TYPES and _edge_group_id is not None

    if use_group:
        fwd = RelationshipEdge(group_id=_edge_group_id, user_id=None, src_id=payload.src_person_id, dst_id=payload.dst_person_id, rel_type=rel_in, confidence=payload.confidence, meta={'role_tag': f"role:{(payload.rel_type or '').strip().lower().replace('-of', '')}"})
        inv = RelationshipEdge(group_id=_edge_group_id, user_id=None, src_id=payload.dst_person_id, dst_id=payload.src_person_id, rel_type=inv_rt, confidence=payload.confidence, meta={'role_tag': f"role:{inv_rt.replace('-of', '')}"})
    else:
        fwd = RelationshipEdge(user_id=user.id, src_id=payload.src_person_id, dst_id=payload.dst_person_id, rel_type=rel_in, confidence=payload.confidence, meta={'role_tag': f"role:{(payload.rel_type or '').strip().lower().replace('-of', '')}"})
        inv = RelationshipEdge(user_id=user.id, src_id=payload.dst_person_id, dst_id=payload.src_person_id, rel_type=inv_rt, confidence=payload.confidence, meta={'role_tag': f"role:{inv_rt.replace('-of', '')}"})
    db.add(fwd)
    try:
        await db.flush()
    except Exception:
        await db.rollback()
        try:
            if use_group:
                fwd = (await db.execute(select(RelationshipEdge).where(RelationshipEdge.group_id == _edge_group_id, RelationshipEdge.src_id == payload.src_person_id, RelationshipEdge.dst_id == payload.dst_person_id, RelationshipEdge.rel_type == rel_in))).scalars().first() or fwd
            else:
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


@router.post('/api/people/infer/run-all')
async def api_infer_run_all(db: AsyncSession=Depends(get_db), user=Depends(require_authenticated_user)):
    """Run inference across all of this user's people and commit high-confidence edges."""
    if user.id in _inferring:
        return {'ok': True, 'committed': 0, 'skipped': 'already_running'}
    _inferring.add(user.id)
    try:
        return await _do_infer_run_all(db, user)
    finally:
        _inferring.discard(user.id)


async def _do_infer_run_all(db: AsyncSession, user):
    # Require at least 3 edges before bothering
    edge_count = (await db.execute(
        select(func.count()).select_from(RelationshipEdge).where(RelationshipEdge.user_id == user.id)
    )).scalar_one()
    if edge_count < 3:
        return {'ok': True, 'committed': 0, 'skipped': 'not_enough_edges'}

    all_candidates = await infer_all_for_user(db, user_id=user.id)
    # Flatten candidates, deduplicate by (src_id, dst_id, rel_type), keep highest confidence
    seen: dict[tuple, float] = {}
    flat = []
    for cands in all_candidates.values():
        for c in cands:
            key = (c.src_id, c.dst_id, c.rel_type)
            if seen.get(key, 0) < c.confidence:
                seen[key] = c.confidence
                flat.append(c)

    # Filter to >= 0.65 confidence
    high = [c for c in flat if c.confidence >= 0.65]
    committed = await commit_inferred_edges(db, user_id=user.id, candidates=high)
    return {'ok': True, 'committed': committed}


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
