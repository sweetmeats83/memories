from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from pathlib import Path as FSPath
from typing import Any, List, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .database import get_db, async_session_maker
from .llm_client import polish_text, OllamaError
from .media_pipeline import MediaPipeline, UserBucketsStrategy
from .models import (
    ChapterMeta,
    Invite,
    Prompt,
    PromptMedia,
    Response,
    Tag,
    User,
    UserProfile,
    UserPrompt,
)
from .utils import require_authenticated_user, require_admin_user
from .routes_shared import (
    STATIC_DIR,
    _get_or_create_tag,
    _to_uploads_rel_for_playable,
)
from app.background import spawn
from app.services.assignment import skip_current_prompt

templates = Jinja2Templates(directory='templates')
logger = logging.getLogger(__name__)
router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
PIPELINE = MediaPipeline(static_root=FSPath(BASE_DIR) / 'static', path_strategy=UserBucketsStrategy())


# ---------------------------------------------------------------------------
# Prompt API
# ---------------------------------------------------------------------------

@router.post('/api/skip_prompt')
async def api_skip_prompt(user=Depends(require_authenticated_user), db: AsyncSession = Depends(get_db)):
    next_id = await skip_current_prompt(db, user.id)
    return {'next_id': next_id}


@router.get('/api/prompt/{prompt_id}')
async def api_prompt(prompt_id: int, user=Depends(require_authenticated_user), db: AsyncSession = Depends(get_db)):
    p = await db.get(Prompt, prompt_id)
    if not p:
        raise HTTPException(status_code=404, detail='Prompt not found')
    return {'id': p.id, 'text': p.text, 'chapter': p.chapter}


# ---------------------------------------------------------------------------
# Admin dashboard
# ---------------------------------------------------------------------------

@router.get('/admin_dashboard')
async def admin_dashboard(request: Request, user: User = Depends(require_admin_user), db: AsyncSession = Depends(get_db)):
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
    answered = (await db.execute(
        select(UserPrompt.user_id, func.count(func.distinct(Response.id)).label('answered'))
        .join(Response, (Response.user_id == UserPrompt.user_id) & (Response.prompt_id == UserPrompt.prompt_id), isouter=False)
        .group_by(UserPrompt.user_id)
    )).all()
    totals_map = {uid: total for uid, total in totals}
    answered_map = {uid: ans for uid, ans in answered}

    def pct(uid: int) -> int:
        t = totals_map.get(uid, 0)
        a = answered_map.get(uid, 0)
        return int(round(100 * a / t)) if t else 0

    users = (await db.execute(select(User))).scalars().all()
    profiles = (await db.execute(select(UserProfile).where(UserProfile.user_id.in_([u.id for u in users])))).scalars().all()
    pmap = {p.user_id: p for p in profiles}
    users_meta = [{
        'id': u.id,
        'email': u.email,
        'username': u.username,
        'display_name': (pmap.get(u.id).display_name if pmap.get(u.id) else None) or (u.username or None) or u.email,
        'answered_pct': pct(u.id),
    } for u in users]
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
    return templates.TemplateResponse('admin_dashboard.html', {
        'request': request,
        'user': user,
        'chapters': chapters,
        'tags_map': tags_map,
        'media_map': media_map,
        'users_meta': users_meta,
        'invites': invites,
        'tag_whitelist_json': tag_whitelist_json,
        'assignments_by_prompt': assignments_by_prompt,
    })


# ---------------------------------------------------------------------------
# Prompt fanout helper
# ---------------------------------------------------------------------------

async def on_prompt_created(db: AsyncSession, prompt_id: int) -> None:
    """
    Called after a Prompt is created (and tags are attached).
    Pre-assigns the prompt to every eligible user unless private_only/only_assigned.
    """
    try:
        from app.models import Prompt, UserProfile, UserPrompt
        from app.services.assignment import _eligible, _get_profile_weights
        prompt = await db.get(Prompt, prompt_id)
        if not prompt:
            logging.warning('[prompt-fanout] Prompt %s not found', prompt_id)
            return
        private_flags = [getattr(prompt, 'private_only', False), getattr(prompt, 'only_assigned', False)]
        if any(bool(x) for x in private_flags):
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
            weights = await _get_profile_weights(db, prof.user_id)
            if _eligible(prompt, weights, user_id=prof.user_id):
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


# ---------------------------------------------------------------------------
# Admin prompt CRUD
# ---------------------------------------------------------------------------

@router.post('/admin_create_prompt')
async def admin_create_prompt(
    request: Request,
    prompt_text: str = Form(...),
    chapter: str = Form(...),
    tags: str = Form('[]'),
    media_files: list[UploadFile] | None = File(None),
    only_assigned: int | bool = Form(0),
    assign_user_ids: str = Form('[]'),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
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
    assigned_ids: list[int] = []
    try:
        parsed_assign = json.loads(assign_user_ids) if assign_user_ids else []
        if isinstance(parsed_assign, list):
            for raw in parsed_assign:
                try:
                    val = int(raw)
                    if val > 0:
                        assigned_ids.append(val)
                except (TypeError, ValueError):
                    continue
    except Exception:
        assigned_ids = []
    if media_files:
        for file in media_files:
            if not file or not file.filename:
                continue
            new_media = PromptMedia(
                prompt_id=prompt.id,
                file_path='',
                media_type=file.content_type.split('/', 1)[0] if file.content_type else 'file',
            )
            db.add(new_media)
            await db.flush()
            tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(file.filename or '').suffix}"
            with open(tmp, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
            art = PIPELINE.process_upload(
                temp_path=tmp, logical='prompt', role='prompt', user_slug_or_id=None,
                prompt_id=prompt.id, response_id=None, media_id=new_media.id,
                original_filename=file.filename, content_type=file.content_type,
            )
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
    if assigned_ids:
        for uid in assigned_ids:
            exists = (await db.execute(
                select(UserPrompt).where(UserPrompt.user_id == uid, UserPrompt.prompt_id == prompt.id)
            )).scalars().first()
            if not exists:
                db.add(UserPrompt(user_id=uid, prompt_id=prompt.id))
        await db.commit()
    if not only_assigned:
        async def _fanout(pid: int):
            async with async_session_maker() as s:
                try:
                    await on_prompt_created(s, pid)
                except Exception:
                    logging.exception('[prompt-fanout] on_prompt_created failed for %s', pid)
        spawn(_fanout(prompt.id), name='prompt_fanout')
    else:
        logging.info('[prompt-fanout] skipped for private-only prompt %s', prompt.id)
    if request.query_params.get('ajax') == '1':
        return JSONResponse({'ok': True, 'id': prompt.id})
    return RedirectResponse(url='/admin_dashboard', status_code=303)


@router.post('/admin_update_prompt/{prompt_id}')
async def admin_update_prompt(
    prompt_id: int,
    request: Request,
    prompt_text: str = Form(...),
    chapter: str = Form(...),
    tags: str = Form('[]'),
    media_files: list[UploadFile] = File(None),
    assign_user_ids: str = Form('[]'),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
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
    assigned_ids: list[int] = []
    try:
        parsed_assign = json.loads(assign_user_ids) if assign_user_ids else []
        if isinstance(parsed_assign, list):
            for raw in parsed_assign:
                try:
                    val = int(raw)
                    if val > 0:
                        assigned_ids.append(val)
                except (TypeError, ValueError):
                    continue
    except Exception:
        assigned_ids = []
    if media_files:
        for file in media_files:
            if not file or not file.filename:
                continue
            new_media = PromptMedia(
                prompt_id=prompt.id,
                file_path='',
                media_type=file.content_type.split('/', 1)[0] if file.content_type else 'file',
            )
            db.add(new_media)
            await db.flush()
            tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(file.filename or '').suffix}"
            with open(tmp, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
            art = PIPELINE.process_upload(
                temp_path=tmp, logical='prompt', role='prompt', user_slug_or_id=None,
                prompt_id=prompt.id, response_id=None, media_id=new_media.id,
                original_filename=file.filename, content_type=file.content_type,
            )
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
    existing_assignments = (await db.execute(
        select(UserPrompt).where(UserPrompt.prompt_id == prompt_id)
    )).unique().scalars().all()
    existing_ids = {up.user_id for up in existing_assignments}
    target_ids = set(assigned_ids)
    to_remove = existing_ids - target_ids
    to_add = target_ids - existing_ids
    if to_remove:
        await db.execute(
            delete(UserPrompt)
            .where(UserPrompt.prompt_id == prompt_id)
            .where(UserPrompt.user_id.in_(to_remove))
        )
    for uid in to_add:
        db.add(UserPrompt(user_id=uid, prompt_id=prompt_id))
    if to_remove or to_add:
        await db.commit()

    async def _refanout(pid: int):
        async with async_session_maker() as s:
            try:
                await on_prompt_created(s, pid)
            except Exception:
                logging.exception('[prompt-refanout] on_prompt_created failed for %s', pid)

    spawn(_refanout(prompt_id), name='prompt_refanout')
    wants_json = (
        request.query_params.get('ajax') == '1'
        or 'application/json' in (request.headers.get('accept') or '')
        or request.headers.get('x-requested-with') == 'XMLHttpRequest'
    )
    if wants_json:
        return {'ok': True, 'prompt_id': prompt_id}
    return RedirectResponse(f'/admin_dashboard#prompts?updated_prompt={prompt_id}', status_code=303)


@router.delete('/admin_delete_prompt/{prompt_id}')
async def admin_delete_prompt(prompt_id: int, user=Depends(require_admin_user), db: AsyncSession = Depends(get_db)):
    prompt = (await db.execute(select(Prompt).where(Prompt.id == prompt_id))).scalars().first()
    if not prompt:
        raise HTTPException(status_code=404, detail='Prompt not found')
    media_files = (await db.execute(select(PromptMedia).where(PromptMedia.prompt_id == prompt_id))).scalars().all()
    for media in media_files:
        PIPELINE.delete_artifacts(
            _to_uploads_rel_for_playable(media.file_path),
            media.wav_path or None,
            media.thumbnail_url or None,
        )
        await db.delete(media)
    await db.delete(prompt)
    await db.commit()
    return JSONResponse({'success': True})


@router.get('/admin_edit_prompt/{prompt_id}', response_class=HTMLResponse)
async def admin_edit_prompt_page(
    prompt_id: int,
    request: Request,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(Prompt).options(selectinload(Prompt.tags), selectinload(Prompt.media)).where(Prompt.id == prompt_id)
    )
    p = res.scalars().first()
    if not p:
        raise HTTPException(status_code=404, detail='Prompt not found')
    prompt = {'id': p.id, 'text': p.text or '', 'chapter': p.chapter or ''}
    tag_list = [{
        'id': t.id,
        'slug': getattr(t, 'slug', None) or getattr(t, 'name', '') or '',
        'name': getattr(t, 'name', None) or getattr(t, 'slug', '') or '',
        'color': getattr(t, 'color', None),
    } for t in p.tags or []]
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
        media_list.append({
            'id': m.id,
            'file_url': file_url,
            'thumbnail_url': thumb_url or file_url,
            'mime_type': getattr(m, 'mime_type', None),
            'duration_sec': int(getattr(m, 'duration_sec', 0) or 0),
            'width': getattr(m, 'width', None),
            'height': getattr(m, 'height', None),
            'assignees': m_users,
        })
    ups = (await db.execute(select(UserPrompt.user_id).where(UserPrompt.prompt_id == prompt_id))).scalars().all() or []
    assigned_users = []
    if ups:
        users = (await db.execute(select(User).where(User.id.in_(ups)))).scalars().all()
        profs = (await db.execute(select(UserProfile).where(UserProfile.user_id.in_(ups)))).scalars().all()
        pmap = {pr.user_id: pr for pr in profs}
        answered_user_ids = set(uid for uid, in (await db.execute(
            select(Response.user_id).where(Response.prompt_id == prompt_id, Response.user_id.in_(ups))
        )).all())

        def _name(u: User) -> str:
            return (pmap.get(u.id).display_name if pmap.get(u.id) else None) or (u.username or u.email)

        assigned_users = [{'id': u.id, 'name': _name(u), 'email': u.email, 'answered': u.id in answered_user_ids} for u in users]
    ctx = {
        'request': request,
        'user': admin,
        'prompt': prompt,
        'tag_list': tag_list,
        'media_list': media_list,
        'assigned_users': assigned_users,
        'partial': request.query_params.get('partial') == '1',
    }
    if request.query_params.get('partial') == '1':
        return templates.TemplateResponse('admin_edit_prompt_partial.html', ctx)
    return templates.TemplateResponse('admin_edit_prompt.html', ctx)


@router.get('/admin_dashboard_partial')
async def admin_dashboard_partial(request: Request, user=Depends(require_admin_user), db: AsyncSession = Depends(get_db)):
    res = await db.execute(
        select(Prompt).options(selectinload(Prompt.media), selectinload(Prompt.tags)).order_by(Prompt.chapter, Prompt.created_at.desc())
    )
    prompts = res.scalars().all()
    chapters, tags_map, media_map = {}, {}, {}
    for p in prompts:
        chapters.setdefault(p.chapter, []).append(p)
        tags_map[p.id] = [{'id': t.id, 'name': t.name, 'slug': t.slug, 'color': t.color} for t in p.tags or []]
        media_map[p.id] = [{'id': m.id, 'file_path': m.file_path, 'media_type': m.media_type, 'thumbnail_url': m.thumbnail_url} for m in p.media or []]
    return templates.TemplateResponse('prompt_list.html', {
        'request': request,
        'user': user,
        'chapters': chapters,
        'tags_map': tags_map,
        'media_map': media_map,
    })


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

@router.get('/_email/health')
async def email_health():
    import smtplib
    import ssl
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


@router.get('/api/chapters_meta')
async def api_chapters_meta(db: AsyncSession = Depends(get_db)):
    metas = (await db.execute(select(ChapterMeta))).scalars().all()
    counts_rows = await db.execute(select(Prompt.chapter, func.count(Prompt.id)).group_by(Prompt.chapter))
    counts = {r[0] or 'Misc': r[1] for r in counts_rows.all()}
    out = []
    for m in metas:
        out.append({
            'name': m.name,
            'display_name': m.display_name,
            'order': m.order,
            'tint': m.tint,
            'count': counts.get(m.name, 0),
            'description': m.description,
            'keywords': m.keywords,
            'llm_guidance': m.llm_guidance,
        })
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
