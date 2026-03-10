from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path as FSPath
from typing import Any, List, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Path, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.background import spawn
from app.database import get_db, async_session_maker
from app.llm_client import polish_text
from app.models import (
    ChapterMeta,
    Prompt,
    PromptMedia,
    Response,
    ResponseSegment,
    ResponseShare,
    ResponseVersion,
    SupportingMedia,
    Tag,
    User,
    UserProfile,
    UserPrompt,
    WeeklyToken,
    WeeklyTokenStatus,
)
from app.routes_shared import (
    PIPELINE,
    STATIC_DIR,
    UPLOAD_DIR,
    _get_or_create_tag,
    _max_order_index,
    _text_for_tagging,
    _to_uploads_rel_for_playable,
    templates,
    transcribe_segment_and_update,
)
from app.routers.upload import get_staged_file, release_staged_dir
from app.schemas import ReorderSegmentsRequest
from app.services.assignment import _iso_year_week, ensure_weekly_prompt, rotate_to_next_unanswered as _rotate
from app.services.auto_tag import suggest_tags_rule_based
from app.services.notifications import notify_new_response
from app.services.utils_weekly import _now, mark_clicked, mark_completed_and_close
from app.transcription import enrich_after_transcription, transcribe_file
from app.users import current_active_user
from app.utils import get_current_user, require_authenticated_html_user, require_authenticated_user

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

async def _ensure_response_owned(db: AsyncSession, response_id: int, user_id: int) -> Response:
    resp = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user_id))).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    return resp


async def _process_primary_async(response_id: int, tmp: FSPath, filename: str, content_type: str, user_slug: str, weekly_token_used: bool):
    try:
        async with async_session_maker() as s:
            resp = await s.get(Response, response_id)
            if resp:
                resp.processing_state = 'processing'
                resp.processing_error = None
                await s.commit()
        loop = asyncio.get_running_loop()
        art = await loop.run_in_executor(
            None,
            lambda: PIPELINE.process_upload(
                temp_path=tmp,
                logical='response',
                role='primary',
                user_slug_or_id=user_slug,
                prompt_id=None,
                response_id=response_id,
                media_id=None,
                original_filename=filename,
                content_type=content_type,
            ),
        )
        playable_rel = (art.playable_rel or '').lstrip('/').replace('\\', '/')
        async with async_session_maker() as s:
            resp = await s.get(Response, response_id)
            if not resp:
                return
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
            resp.processing_state = 'ready'
            resp.processing_error = None
            await s.commit()

            exist = await s.execute(select(ResponseSegment.id).where(ResponseSegment.response_id == response_id))
            if not exist.first():
                primary_rel = (resp.primary_media_url or '').lstrip('/').replace('\\', '/')
                if primary_rel.startswith('uploads/'):
                    primary_rel = primary_rel[len('uploads/'):]
                is_composite = primary_rel.startswith(f'responses/{response_id}/') and primary_rel.split('/')[-1].startswith('composite-')
                if primary_rel and not is_composite:
                    seg0 = ResponseSegment(
                        response_id=response_id,
                        order_index=0,
                        media_path=primary_rel,
                        media_mime=resp.primary_mime_type or None,
                        transcript=resp.transcription or '',
                    )
                    s.add(seg0)
                    await s.commit()

        if playable_rel:
            spawn(transcribe_and_update(response_id, playable_rel, weekly_token_used), name='transcribe_response_primary')
        spawn(notify_new_response(response_id), name='notify_new_response')
    except Exception as exc:
        logger.exception('Primary media processing failed for response %s: %s', response_id, exc)
        async with async_session_maker() as s:
            resp = await s.get(Response, response_id)
            if resp:
                resp.processing_state = 'failed'
                resp.processing_error = str(exc)
                await s.commit()
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


async def _process_supporting_async(media_id: int, tmp: FSPath, filename: str, content_type: str, user_slug: str):
    try:
        loop = asyncio.get_running_loop()
        art = await loop.run_in_executor(
            None,
            lambda: PIPELINE.process_upload(
                temp_path=tmp,
                logical='response',
                role='supporting',
                user_slug_or_id=user_slug,
                prompt_id=None,
                response_id=None,
                media_id=media_id,
                original_filename=filename,
                content_type=content_type,
            ),
        )
        async with async_session_maker() as s:
            media = await s.get(SupportingMedia, media_id)
            if not media:
                return
            playable_rel = (art.playable_rel or '').lstrip('/').replace('\\', '/')
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
            await s.commit()
    except Exception as exc:
        logger.exception('Supporting media processing failed for media %s: %s', media_id, exc)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _gen_share_token() -> str:
    return secrets.token_urlsafe(22)


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


async def transcribe_and_update(response_id: int, media_filename: str, auto_polish: bool = False):
    try:
        async with async_session_maker() as session:
            resp = await session.get(Response, response_id)
            uid = getattr(resp, 'user_id', None) if resp else None
            transcript = await transcribe_file(media_filename, db=session, user_id=uid)
            if resp:
                resp.transcription = transcript
                pm_rel = (resp.primary_media_url or '').lstrip('/').replace('\\', '/')
                if pm_rel.startswith('uploads/'):
                    pm_rel = pm_rel[len('uploads/'):]
                seg_row = (await session.execute(
                    select(ResponseSegment)
                    .where(ResponseSegment.response_id == response_id)
                    .order_by(ResponseSegment.order_index.asc(), ResponseSegment.id.asc())
                )).scalars().first()
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
                    logger.warning('Auto-polish failed for response %s: %s', response_id, e)
                await enrich_after_transcription(session, resp)
    except Exception as e:
        logger.error('Transcription failed for response %s: %s', response_id, e)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post('/responses/')
async def create_response(
    prompt_id: int | None = Form(None),
    title: str | None = Form(None),
    chapter: str | None = Form(None),
    response_text: str | None = Form(None),
    primary_media: UploadFile | None = File(None),
    primary_staged_id: str | None = Form(None),
    supporting_media: list[UploadFile] | None = File(None),
    weekly_token: str | None = Form(None),
    request: Request = None,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
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

    pending_primary: tuple[FSPath, str, str | None] | None = None
    if primary_staged_id:
        staged = get_staged_file(primary_staged_id, acting_user.id)
        if not staged:
            raise HTTPException(status_code=400, detail='Staged upload not found or expired')
        new_response.processing_state = 'processing'
        new_response.processing_error = None
        pending_primary = (
            FSPath(staged['assembled_path']),
            staged['filename'],
            staged['content_type'],
        )
        release_staged_dir(primary_staged_id)
    elif primary_media and primary_media.filename:
        tmp = _save_temp(primary_media)
        new_response.processing_state = 'processing'
        new_response.processing_error = None
        pending_primary = (
            tmp,
            primary_media.filename or 'primary',
            primary_media.content_type or 'application/octet-stream',
        )
    else:
        new_response.processing_state = 'ready'
        new_response.processing_error = None

    pending_supporting: list[tuple[int, FSPath, str, str | None]] = []
    if supporting_media:
        for f in supporting_media:
            if not f or not f.filename:
                continue
            media = SupportingMedia(
                response_id=new_response.id,
                file_path='',
                media_type=f.content_type.split('/', 1)[0] if f.content_type else 'file',
            )
            db.add(media)
            await db.flush()
            tmp = _save_temp(f)
            pending_supporting.append((
                media.id,
                tmp,
                f.filename or f'supporting-{media.id}',
                f.content_type or 'application/octet-stream',
            ))

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
    await db.execute(
        update(UserPrompt)
        .where((UserPrompt.user_id == acting_user.id) & (UserPrompt.prompt_id == new_response.prompt_id))
        .values(status='answered')
    )
    await db.commit()
    try:
        if (new_response.response_text or '').strip():
            await enrich_after_transcription(db, new_response)
    except Exception:
        pass
    if pending_primary:
        tmp, fname, ctype = pending_primary
        spawn(
            _process_primary_async(
                new_response.id,
                tmp,
                fname,
                ctype,
                acting_user.username or str(acting_user.id),
                bool(weekly_token),
            ),
            name='process_response_primary',
        )
    else:
        spawn(notify_new_response(new_response.id), name='notify_new_response')
    for media_id, tmp, fname, ctype in pending_supporting:
        spawn(
            _process_supporting_async(
                media_id,
                tmp,
                fname,
                ctype,
                acting_user.username or str(acting_user.id),
            ),
            name=f'process_supporting_media_{media_id}',
        )
    try:
        if prompt_id:
            y, w = _iso_year_week()
            from app.models import UserWeeklyPrompt
            weekly = (await db.execute(
                select(UserWeeklyPrompt).where(UserWeeklyPrompt.user_id == acting_user.id, UserWeeklyPrompt.year == y, UserWeeklyPrompt.week == w)
            )).scalars().first()
            if weekly and weekly.prompt_id == prompt_id:
                weekly.status = 'answered'
                await db.commit()
                await _rotate(db, acting_user.id)
    except Exception:
        pass
    if weekly_token:
        await mark_completed_and_close(db, weekly_token)
        try:
            await ensure_weekly_prompt(db, acting_user.id)
        except Exception as e:
            logger.exception('weekly rotate after response failed: %s', e)
        await db.commit()
        return templates.TemplateResponse('thank_you.html', {'request': request, 'user': None})
    _wants_json = 'application/json' in (request.headers.get('accept') or '') or request.headers.get('x-requested-with') == 'XMLHttpRequest'
    if _wants_json:
        return {'id': new_response.id, 'processing_state': new_response.processing_state}
    if new_response.processing_state == 'ready':
        return RedirectResponse(url=f'/response/{new_response.id}/edit', status_code=303)
    return RedirectResponse(url=f'/response/{new_response.id}/processing', status_code=303)


@router.get('/response/{response_id}', response_class=HTMLResponse, name='response_view')
async def response_view(
    response_id: int,
    request: Request,
    user=Depends(require_authenticated_html_user),
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(Response)
        .options(
            selectinload(Response.prompt).selectinload(Prompt.media),
            selectinload(Response.tags),
            selectinload(Response.supporting_media),
            selectinload(Response.segments),
        )
        .where(Response.id == response_id, Response.user_id == user.id)
    )
    resp = (await db.execute(q)).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    if getattr(resp, 'processing_state', 'ready') != 'ready':
        return RedirectResponse(url=f'/response/{response_id}/processing', status_code=303)
    prompt_media = list(resp.prompt.media) if resp.prompt and resp.prompt.media else []
    supporting_media = list(resp.supporting_media or [])
    segments = list(resp.segments or [])
    ctx = {
        'request': request,
        'user': user,
        'response': resp,
        'prompt_media': prompt_media,
        'supporting_media': supporting_media,
        'segments': segments,
        'is_token_link': False,
    }
    return templates.TemplateResponse('response_view.html', ctx)


@router.get('/response/{response_id}/edit', response_class=HTMLResponse, name='edit_response_page')
async def edit_response_page(
    response_id: int,
    request: Request,
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    response = (await db.execute(
        select(Response)
        .options(selectinload(Response.prompt), selectinload(Response.tags))
        .where(Response.id == response_id, Response.user_id == user.id)
    )).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    if getattr(response, 'processing_state', 'ready') != 'ready':
        return RedirectResponse(url=f'/response/{response_id}/processing', status_code=303)
    supporting_media = (await db.execute(select(SupportingMedia).where(SupportingMedia.response_id == response_id))).scalars().all()
    pm_res = await db.execute(select(PromptMedia).where(PromptMedia.prompt_id == response.prompt_id))
    prompt_media = pm_res.scalars().all()
    # Build profile tag suggestions (high-weight tags not already on this response)
    profile_tag_suggestions: list[str] = []
    try:
        profile = await db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
        if profile:
            weights = (profile.tag_weights or {}).get("tagWeights", {})
            existing_slugs = {t.slug for t in (response.tags or [])}
            profile_tag_suggestions = [
                slug for slug, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)
                if w >= 0.5 and slug not in existing_slugs
            ][:20]
    except Exception:
        pass
    return templates.TemplateResponse('response_edit.html', {
        'request': request,
        'user': user,
        'response': response,
        'supporting_media': supporting_media,
        'prompt_media': prompt_media,
        'profile_tag_suggestions': profile_tag_suggestions,
    })


@router.get('/response/{response_id}/processing', response_class=HTMLResponse, name='response_processing_page')
async def response_processing_page(
    response_id: int,
    request: Request,
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    response = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    state = getattr(response, 'processing_state', 'ready')
    if state == 'ready':
        return RedirectResponse(url=f'/response/{response_id}/edit', status_code=303)
    ctx = {
        'request': request,
        'user': user,
        'response_id': response_id,
        'state': state,
        'error': getattr(response, 'processing_error', None),
        'redirect_url': f'/response/{response_id}/edit',
    }
    return templates.TemplateResponse('response_processing.html', ctx)


@router.get('/api/responses/{response_id}/status')
async def api_response_status(
    response_id: int,
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    response = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    return {
        'id': response_id,
        'state': getattr(response, 'processing_state', 'ready'),
        'error': getattr(response, 'processing_error', None),
        'redirect_url': f'/response/{response_id}/edit',
    }


@router.post('/response/{response_id}/edit')
async def save_transcription(
    response_id: int,
    transcription: str = Form(...),
    title: str | None = Form(None),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    response = (await db.execute(
        select(Response)
        .options(selectinload(Response.tags))
        .where(Response.id == response_id, Response.user_id == user.id)
    )).scalars().first()
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
    # Auto-apply user's high-weight profile tags to this response
    try:
        profile = await db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
        if profile:
            weights = (profile.tag_weights or {}).get("tagWeights", {})
            high_weight = sorted(
                [(slug, w) for slug, w in weights.items() if w >= 0.5],
                key=lambda x: x[1], reverse=True,
            )
            existing_ids = {t.id for t in (response.tags or [])}
            for slug, _ in high_weight[:20]:
                tag = await db.scalar(select(Tag).where(Tag.slug == slug))
                if tag and tag.id not in existing_ids:
                    response.tags.append(tag)
                    existing_ids.add(tag.id)
    except Exception:
        pass
    await db.commit()
    return RedirectResponse(url=f'/response/{response_id}', status_code=303)


@router.get('/response/{response_id}/versions')
async def user_list_versions(
    response_id: int,
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    exists = (await db.execute(select(Response.id).where(Response.id == response_id, Response.user_id == user.id))).scalar_one_or_none()
    if not exists:
        raise HTTPException(status_code=404, detail='Response not found')
    rows = (await db.execute(
        select(ResponseVersion)
        .where(ResponseVersion.response_id == response_id)
        .order_by(ResponseVersion.created_at.desc(), ResponseVersion.id.desc())
    )).scalars().all()
    out = []
    for v in rows:
        out.append({
            'id': v.id,
            'created_at': v.created_at.isoformat() if v.created_at else None,
            'edited_by_admin_id': v.edited_by_admin_id,
            'title': (v.title or '')[:120],
            'has_transcription': bool((v.transcription or '').strip()),
            'tags': (v.tags_json or {}).get('tags') if isinstance(v.tags_json, dict) else None,
        })
    return {'versions': out}


@router.post('/response/{response_id}/versions/{version_id}/restore')
async def user_restore_version(
    response_id: int,
    version_id: int,
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
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


@router.post('/response/{response_id}/share')
async def create_response_share(
    response_id: int,
    permanent: bool = Form(True),
    days: int | None = Form(None),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    resp = await db.get(Response, response_id)
    if not resp or resp.user_id != user.id:
        raise HTTPException(status_code=404, detail='Response not found')
    expires_at = None
    if not permanent and days and days > 0:
        expires_at = datetime.now(timezone.utc) + timedelta(days=days)
    tok = ResponseShare(
        token=_gen_share_token(),
        response_id=response_id,
        user_id=user.id,
        permanent=bool(permanent),
        expires_at=expires_at,
    )
    db.add(tok)
    await db.commit()
    return {'ok': True, 'link': f'/share/r/{tok.token}'}


@router.get('/share/r/{token}', response_class=HTMLResponse)
async def share_response_view(token: str, request: Request, db: AsyncSession = Depends(get_db)):
    share = (await db.execute(select(ResponseShare).where(ResponseShare.token == token))).scalars().first()
    if not share or share.revoked or (share.expires_at and share.expires_at < datetime.now(timezone.utc)):
        raise HTTPException(status_code=404, detail='Link not found or expired')
    resp = (await db.execute(
        select(Response)
        .options(
            selectinload(Response.prompt).selectinload(Prompt.media),
            selectinload(Response.tags),
            selectinload(Response.supporting_media),
            selectinload(Response.segments),
        )
        .where(Response.id == share.response_id)
    )).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    if getattr(resp, 'processing_state', 'ready') != 'ready':
        return templates.TemplateResponse(
            'response_processing_public.html',
            {'request': request, 'message': 'This response is still processing. Please try again in a few minutes.'},
            status_code=202,
        )
    ctx = {
        'request': request,
        'user': None,
        'response': resp,
        'prompt_media': list(resp.prompt.media) if resp.prompt and resp.prompt.media else [],
        'supporting_media': list(resp.supporting_media or []),
        'segments': list(resp.segments or []),
        'is_token_link': True,
        'share_token': token,
    }
    return templates.TemplateResponse('response_view.html', ctx)


@router.get('/media/share/{token}/{path:path}')
async def media_share_stream(token: str, path: str, db: AsyncSession = Depends(get_db)):
    share = (await db.execute(select(ResponseShare).where(ResponseShare.token == token))).scalars().first()
    if not share or share.revoked or (share.expires_at and share.expires_at < datetime.now(timezone.utc)):
        raise HTTPException(status_code=404, detail='Link not found or expired')
    rid = _response_id_from_uploads_path(path)
    if rid != share.response_id:
        raise HTTPException(status_code=403, detail='Not allowed')
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
async def delete_response(
    response_id: int,
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    response = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    prompt_id = response.prompt_id
    PIPELINE.delete_artifacts(
        _to_uploads_rel_for_playable(response.primary_media_url) if response.primary_media_url else None,
        response.primary_wav_path,
        response.primary_thumbnail_path,
    )
    supp_all = (await db.execute(select(SupportingMedia).where(SupportingMedia.response_id == response_id))).scalars().all()
    for m in supp_all:
        PIPELINE.delete_artifacts(_to_uploads_rel_for_playable(m.file_path), m.wav_path or None, m.thumbnail_url or None)
        await db.delete(m)
    await db.delete(response)
    await db.flush()
    if prompt_id:
        remaining = (await db.execute(
            select(func.count(Response.id)).where(Response.user_id == user.id, Response.prompt_id == prompt_id)
        )).scalar_one()
        if remaining == 0:
            up = (await db.execute(
                select(UserPrompt).where(UserPrompt.user_id == user.id, UserPrompt.prompt_id == prompt_id)
            )).scalars().first()
            if up:
                up.status = 'queued'
    await db.commit()
    return RedirectResponse(url='/user_dashboard', status_code=303)


@router.get('/api/chapters_progress')
async def api_chapters_progress(user=Depends(require_authenticated_user), db: AsyncSession = Depends(get_db)):
    pool_rows = await db.execute(
        select(Prompt.chapter, func.count(Prompt.id))
        .select_from(UserPrompt)
        .join(Prompt, Prompt.id == UserPrompt.prompt_id)
        .where(UserPrompt.user_id == user.id)
        .group_by(Prompt.chapter)
    )
    totals_by = {row[0] or 'Misc': int(row[1] or 0) for row in pool_rows.all()}
    done_rows = await db.execute(
        select(Prompt.chapter, func.count(func.distinct(Prompt.id)))
        .select_from(UserPrompt)
        .join(Prompt, Prompt.id == UserPrompt.prompt_id)
        .join(Response, and_(Response.prompt_id == Prompt.id, Response.user_id == user.id))
        .where(UserPrompt.user_id == user.id)
        .group_by(Prompt.chapter)
    )
    done_by = {row[0] or 'Misc': int(row[1] or 0) for row in done_rows.all()}
    meta_rows = (await db.execute(select(ChapterMeta))).scalars().all()
    meta_map = {m.name: m for m in meta_rows}
    payload = []
    for name, total in totals_by.items():
        m = meta_map.get(name)
        payload.append({
            'name': name,
            'slug': name,
            'display_name': (m.display_name if m else None) or (name or 'Misc'),
            'tint': m.tint if m else None,
            'total': total,
            'completed': int(done_by.get(name, 0)),
        })

    def order_of(n):
        mm = meta_map.get(n)
        return getattr(mm, 'order', 999999)
    payload.sort(key=lambda d: (order_of(d['name']), d['display_name'].lower()))
    return payload


@router.post('/response/{response_id}/media')
async def add_supporting_media(
    response_id: int,
    media_files: list[UploadFile] | None = File(None),
    staged_ids: str | None = Form(None),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    resp = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    for f in media_files or []:
        if not f or not f.filename:
            continue
        media = SupportingMedia(
            response_id=resp.id,
            file_path='',
            media_type=f.content_type.split('/', 1)[0] if f.content_type else 'file',
        )
        db.add(media)
        await db.flush()
        tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(f.filename or '').suffix}"
        with open(tmp, 'wb') as w:
            shutil.copyfileobj(f.file, w)
        art = PIPELINE.process_upload(
            temp_path=tmp, logical='response', role='supporting',
            user_slug_or_id=user.username or str(user.id),
            prompt_id=None, response_id=resp.id, media_id=media.id,
            original_filename=f.filename, content_type=f.content_type,
        )
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
    if staged_ids:
        try:
            uid_list = json.loads(staged_ids)
        except Exception:
            uid_list = []
        for uid in uid_list:
            staged = get_staged_file(uid, user.id)
            if not staged:
                continue
            media = SupportingMedia(
                response_id=resp.id,
                file_path='',
                media_type=staged['content_type'].split('/', 1)[0] if staged.get('content_type') else 'file',
            )
            db.add(media)
            await db.flush()
            tmp = FSPath(staged['assembled_path'])
            art = PIPELINE.process_upload(
                temp_path=tmp, logical='response', role='supporting',
                user_slug_or_id=user.username or str(user.id),
                prompt_id=None, response_id=resp.id, media_id=media.id,
                original_filename=staged['filename'], content_type=staged.get('content_type'),
            )
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
            release_staged_dir(uid)
    await db.commit()
    return RedirectResponse(url=f'/response/{resp.id}/edit', status_code=303)


@router.get('/response/{response_id}/next', response_class=HTMLResponse)
async def response_next(
    response_id: int,
    request: Request,
    user=Depends(require_authenticated_html_user),
    db: AsyncSession = Depends(get_db),
):
    cur = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not cur:
        raise HTTPException(status_code=404, detail='Response not found')
    next_q = (
        select(Response)
        .options(
            selectinload(Response.prompt).selectinload(Prompt.media),
            selectinload(Response.tags),
            selectinload(Response.supporting_media),
            selectinload(Response.segments),
        )
        .where(Response.user_id == user.id)
        .where(or_(
            Response.created_at < cur.created_at,
            and_(Response.created_at == cur.created_at, Response.id < cur.id),
        ))
        .order_by(Response.created_at.desc(), Response.id.desc())
        .limit(1)
    )
    next_resp = (await db.execute(next_q)).scalars().first()
    if not next_resp:
        return HTMLResponse(status_code=204)
    prompt_media = list(next_resp.prompt.media) if next_resp.prompt and next_resp.prompt.media else []
    supporting_media = list(next_resp.supporting_media or [])
    segments = list(next_resp.segments or [])
    return templates.TemplateResponse('response_view__article_partial.html', {
        'request': request,
        'user': user,
        'response': next_resp,
        'prompt_media': prompt_media,
        'supporting_media': supporting_media,
        'segments': segments,
        'is_token_link': False,
    })


@router.delete('/response/{response_id}/media/{media_id}')
async def delete_supporting_media(
    response_id: int,
    media_id: int,
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    response = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    media = (await db.execute(select(SupportingMedia).where(SupportingMedia.id == media_id, SupportingMedia.response_id == response_id))).scalars().first()
    if not media:
        raise HTTPException(status_code=404, detail='Media not found')
    PIPELINE.delete_artifacts(
        _to_uploads_rel_for_playable(media.file_path),
        media.wav_path or None,
        media.thumbnail_url or None,
    )
    await db.delete(media)
    await db.commit()
    return {'success': True}


@router.get('/api/response/{response_id}/transcript')
async def api_get_transcript(response_id: int = Path(...), user=Depends(current_active_user)):
    async with async_session_maker() as session:
        result = await session.execute(select(Response).where(Response.id == response_id))
        resp = result.unique().scalar_one_or_none()
        if not resp:
            return JSONResponse({'error': 'Not found'}, status_code=404)
        if not getattr(user, 'is_superuser', False) and resp.user_id != user.id:
            return JSONResponse({'error': 'Forbidden'}, status_code=403)
        return {'text': (resp.transcription or '').strip()}


@router.post('/response/{response_id}/primary')
async def replace_primary(
    response_id: int,
    request: Request,
    primary_media: UploadFile = File(...),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    resp = (await db.execute(select(Response).where(Response.id == response_id, Response.user_id == user.id))).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    PIPELINE.delete_artifacts(
        _to_uploads_rel_for_playable(resp.primary_media_url) if resp.primary_media_url else None,
        resp.primary_wav_path,
        resp.primary_thumbnail_path,
    )
    tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(primary_media.filename or '').suffix}"
    with open(tmp, 'wb') as w:
        shutil.copyfileobj(primary_media.file, w)
    spawn(
        _process_primary_async(resp.id, tmp, primary_media.filename or 'primary', primary_media.content_type, user.username or str(user.id), False),
        name='reprocess_primary_media',
    )
    return JSONResponse({'response': {'id': resp.id}, 'queued': True})


@router.post('/response/{response_id}/tags')
async def set_response_tags(
    response_id: int,
    payload: Any = Body(...),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    resp = (await db.execute(
        select(Response).options(selectinload(Response.tags)).where(Response.id == response_id, Response.user_id == user.id)
    )).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    try:
        cur_tags = [t.slug for t in resp.tags or []]
        ver = ResponseVersion(
            response_id=resp.id,
            user_id=user.id,
            title=getattr(resp, 'title', None),
            transcription=resp.transcription,
            tags_json={'tags': cur_tags},
            edited_by_admin_id=None,
        )
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
async def get_response_tags(
    response_id: int,
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    response = (await db.execute(
        select(Response).options(selectinload(Response.tags)).where(Response.id == response_id, Response.user_id == user.id)
    )).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    return {'tags': [{'id': t.id, 'name': t.name, 'slug': t.slug, 'color': t.color} for t in response.tags or []]}


@router.get('/response/{response_id}/segments')
async def list_response_segments(
    response_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_authenticated_user),
):
    await _ensure_response_owned(db, response_id, user.id)
    rows = (await db.execute(
        select(ResponseSegment)
        .where(ResponseSegment.response_id == response_id)
        .order_by(ResponseSegment.order_index.asc(), ResponseSegment.id.asc())
    )).scalars().all()
    out = []
    for s in rows:
        out.append({
            'id': s.id,
            'order_index': getattr(s, 'order_index', None),
            'transcript': getattr(s, 'transcript', '') or '',
            'media_path': getattr(s, 'media_path', '') or '',
            'media_mime': getattr(s, 'media_mime', '') or '',
            'thumbnail_path': getattr(s, 'thumbnail_path', None) or getattr(s, 'thumbnail_url', None) or '',
        })
    return out


@router.post('/response/{response_id}/segments')
async def add_response_segment(
    response_id: int,
    file: UploadFile | None = File(None),
    staged_id: str | None = Form(None),
    note: str | None = Form(None),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    if staged_id:
        staged = get_staged_file(staged_id, user.id)
        if not staged:
            raise HTTPException(status_code=400, detail='Staged upload not found or expired')
        original_filename = staged['filename']
        content_type = staged.get('content_type')
    elif file and file.filename:
        staged = None
        original_filename = file.filename
        content_type = file.content_type
    else:
        raise HTTPException(status_code=400, detail='No file or staged_id provided')
    resp = await _ensure_response_owned(db, response_id, user.id)
    order = await _max_order_index(db, response_id)
    seg = ResponseSegment(response_id=response_id, order_index=order + 1, transcript='')
    db.add(seg)
    await db.flush()
    if staged:
        tmp = FSPath(staged['assembled_path'])
        release_staged_dir(staged_id)
    else:
        tmp = FSPath(UPLOAD_DIR) / f"_tmp_{uuid.uuid4().hex}{FSPath(file.filename or '').suffix}"
        with open(tmp, 'wb') as w:
            shutil.copyfileobj(file.file, w)
    art = PIPELINE.process_upload(
        temp_path=tmp, logical='response', role='supporting',
        user_slug_or_id=user.username or str(user.id),
        prompt_id=None, response_id=response_id, media_id=seg.id,
        original_filename=original_filename or 'segment', content_type=content_type,
    )
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
async def reorder_response_segments(
    response_id: int,
    payload: ReorderSegmentsRequest = Body(...),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    await _ensure_response_owned(db, response_id, user.id)
    ids = payload.order or []
    if not ids:
        return {'ok': True}
    found = (await db.execute(
        select(ResponseSegment.id).where(ResponseSegment.response_id == response_id, ResponseSegment.id.in_(ids))
    )).scalars().all()
    if set(found) != set(ids):
        raise HTTPException(status_code=400, detail='Invalid segment ids for this response.')
    order_map = {sid: idx + 1 for idx, sid in enumerate(ids)}
    rows = (await db.execute(select(ResponseSegment).where(ResponseSegment.id.in_(ids)))).scalars().all()
    for r in rows:
        r.order_index = order_map.get(r.id, r.order_index)
    await db.commit()
    return {'ok': True}


@router.delete('/response/{response_id}/segments/{segment_id}')
async def delete_response_segment(
    response_id: int,
    segment_id: int,
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
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
async def bootstrap_first_segment(
    response_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_authenticated_user),
):
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
    seg = ResponseSegment(
        response_id=response_id,
        order_index=0,
        media_path=primary_rel,
        media_mime=resp.primary_mime_type or None,
        transcript='',
    )
    db.add(seg)
    await db.commit()
    try:
        uploads_rel = primary_rel if primary_rel.startswith('uploads/') else f'uploads/{primary_rel}'
        spawn(transcribe_segment_and_update(seg.id, uploads_rel), name='transcribe_segment_bootstrap')
    except Exception:
        pass
    return JSONResponse({'ok': True, 'created': True, 'segment_id': seg.id})


@router.post('/response/{response_id}/segments/merge-audio')
async def merge_audio_to_primary(
    response_id: int,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_authenticated_user),
):
    resp = await db.get(Response, response_id)
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    rows = await db.execute(
        select(ResponseSegment)
        .where(ResponseSegment.response_id == response_id)
        .order_by(ResponseSegment.order_index.asc())
    )
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
        logger.exception('concat_audio_async crashed')
        raise HTTPException(status_code=500, detail='merge crashed')
    if not out.get('ok'):
        raise HTTPException(status_code=500, detail=out.get('error', 'merge failed'))
    dest_rel = out['dest_rel']
    old_primary_rel = resp.primary_media_url
    resp.primary_media_url = dest_rel
    resp.primary_mime_type = 'audio/mp4'
    resp.primary_thumbnail_path = None
    resp.primary_duration_sec = None
    await db.commit()
    try:
        if old_primary_rel and old_primary_rel not in sources and old_primary_rel != resp.primary_media_url:
            await asyncio.to_thread(PIPELINE.delete_artifacts, old_primary_rel)
    except Exception:
        pass
    return JSONResponse({'ok': True, 'primary': resp.primary_media_url})


__all__ = ['router']
