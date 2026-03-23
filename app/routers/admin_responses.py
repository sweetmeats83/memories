"""Admin routes for user and response management."""
from __future__ import annotations

import secrets
import uuid
import shutil
import logging

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from pathlib import Path as FSPath
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from typing import Any

from app.database import get_db
from app.models import (
    AdminEditLog,
    Response,
    ResponseSegment,
    ResponseVersion,
    SupportingMedia,
    PromptMedia,
    Prompt,
    Tag,
    User,
)
from app.routes_shared import (
    PIPELINE,
    UPLOAD_DIR,
    _get_or_create_tag,
    _max_order_index,
    _text_for_tagging,
    _to_uploads_rel_for_playable,
    transcribe_segment_and_update,
    templates,
)
from app.schemas import ReorderSegmentsRequest
from app.services.auto_tag import suggest_tags_rule_based
from app.transcription import enrich_after_transcription
from app.utils import require_admin_user, slug_role
from app.models import UserProfile
from app.background import spawn
import bcrypt as _bcrypt_lib

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

@router.post('/admin/users/{user_id}/force_password')
async def admin_force_password(
    user_id: int,
    new_password: str = Form(...),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    target = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if not target:
        raise HTTPException(status_code=404, detail='User not found')
    if not new_password or len(new_password) < 8:
        raise HTTPException(status_code=400, detail='New password too short')
    target.hashed_password = _bcrypt_lib.hashpw(new_password.encode(), _bcrypt_lib.gensalt()).decode()
    target.must_change_password = True
    await db.commit()
    return RedirectResponse(url='/admin_dashboard?notice=Password+updated', status_code=303)


@router.post('/admin/users/{user_id}/delete')
async def admin_delete_user(
    user_id: int,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
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


# ---------------------------------------------------------------------------
# User roles management
# ---------------------------------------------------------------------------

# Curated role presets shown in the admin UI (value → display label)
_ROLE_PRESETS = [
    ("parent",      "Parent"),
    ("grandparent", "Grandparent"),
    ("spouse",      "Spouse / Partner"),
    ("sibling",     "Sibling"),
    ("aunt-uncle",  "Aunt / Uncle"),
    ("child",       "Child (has living parents)"),
    ("grandchild",  "Grandchild"),
]

_GENDER_OPTIONS = [
    ("male",   "Male"),
    ("female", "Female"),
]


@router.get('/api/admin/users/{user_id}/roles')
async def admin_get_user_roles(
    user_id: int,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    prof = (await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))).scalars().first()
    if not prof:
        return {"roles": [], "gender": None}
    tw = (prof.tag_weights or {}).get("tagWeights", {}) or {}
    roles = [k for k in tw if k.startswith("role:") and float(tw[k] or 0) >= 0.3]
    gender = ((prof.privacy_prefs or {}).get("user_meta") or {}).get("gender")
    return {"roles": sorted(roles), "gender": gender}


@router.post('/api/admin/users/{user_id}/roles')
async def admin_set_user_roles(
    user_id: int,
    payload: dict = Body(...),
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Replace the user's role:* weights and gender setting.
    payload: {"roles": ["role:parent", "role:spouse"], "gender": "female" | null}
    Rebuilds the prompt pool automatically.
    """
    from app.services.assignment import build_pool_for_user
    from sqlalchemy.orm.attributes import flag_modified

    prof = (await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))).scalars().first()
    if not prof:
        raise HTTPException(404, "User profile not found")

    incoming_roles: list[str] = [
        slug_role(str(r)) for r in (payload.get("roles") or []) if r
    ]
    gender = (payload.get("gender") or "").strip().lower() or None

    # Update tag_weights: remove old role:* keys, add new ones at weight 0.7
    tw = dict(prof.tag_weights or {"tagWeights": {}})
    weights: dict = tw.setdefault("tagWeights", {})
    for k in list(weights.keys()):
        if k.startswith("role:"):
            del weights[k]
    for rs in incoming_roles:
        weights[rs] = 0.7

    prof.tag_weights = tw
    flag_modified(prof, "tag_weights")

    # Update gender in privacy_prefs
    pp = dict(prof.privacy_prefs or {})
    user_meta = dict(pp.get("user_meta") or {})
    if gender:
        user_meta["gender"] = gender
    else:
        user_meta.pop("gender", None)
    pp["user_meta"] = user_meta
    prof.privacy_prefs = pp
    flag_modified(prof, "privacy_prefs")

    await db.commit()

    # Rebuild prompt pool so new roles take effect immediately
    added = await build_pool_for_user(db, user_id)
    return {"ok": True, "roles": incoming_roles, "gender": gender, "pool_added": added}


@router.post('/admin/users/{user_id}/anonymize')
async def admin_anonymize_user(
    user_id: int,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    target = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if not target:
        raise HTTPException(status_code=404, detail='User not found')
    if getattr(admin, 'id', None) == target.id:
        raise HTTPException(status_code=400, detail='You cannot anonymize yourself from admin.')
    if getattr(target, 'is_superuser', False) and (not getattr(admin, 'is_superuser', False)):
        raise HTTPException(status_code=403, detail='Only a super admin can anonymize a superuser.')
    suffix = secrets.token_hex(4)
    target.username = f'deleted_user_{target.id}_{suffix}'
    target.email = f'deleted+{target.id}.{suffix}@example.invalid'
    if hasattr(target, 'hashed_password'):
        target.hashed_password = ''
    if hasattr(target, 'is_active'):
        target.is_active = False
    await db.commit()
    return RedirectResponse(url='/admin_dashboard?notice=User+anonymized', status_code=303)


# ---------------------------------------------------------------------------
# Admin response view/edit
# ---------------------------------------------------------------------------

@router.get('/admin/users/{user_id}/responses/{response_id}')
async def admin_response_view(
    user_id: int,
    response_id: int,
    request: Request,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    resp = (await db.execute(
        select(Response).options(
            selectinload(Response.prompt).selectinload(Prompt.media),
            selectinload(Response.tags),
            selectinload(Response.supporting_media),
            selectinload(Response.segments),
        ).where(Response.id == response_id, Response.user_id == user_id)
    )).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    return templates.TemplateResponse(request, 'response_view.html', {
        'request': request,
        'user': admin,
        'response': resp,
        'prompt_media': list(resp.prompt.media) if resp.prompt and resp.prompt.media else [],
        'supporting_media': list(resp.supporting_media or []),
        'segments': list(resp.segments or []),
        'is_token_link': False,
        'is_admin_view': True,
    })


@router.get('/admin/users/{user_id}/responses/{response_id}/edit')
async def admin_edit_response_page(
    user_id: int,
    response_id: int,
    request: Request,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    response = (await db.execute(
        select(Response).options(
            selectinload(Response.prompt),
            selectinload(Response.tags),
        ).where(Response.id == response_id, Response.user_id == user_id)
    )).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    supporting_media = (await db.execute(
        select(SupportingMedia).where(SupportingMedia.response_id == response_id)
    )).scalars().all()
    prompt_media = (await db.execute(
        select(PromptMedia).where(PromptMedia.prompt_id == response.prompt_id)
    )).scalars().all()
    admin_api_base = f'/admin/users/{user_id}/responses/{response_id}'
    return templates.TemplateResponse(request, 'response_edit.html', {
        'request': request,
        'user': admin,
        'response': response,
        'supporting_media': supporting_media,
        'prompt_media': prompt_media,
        'is_admin_view': True,
        'admin_api_base': admin_api_base,
        'acting_as_user': await db.get(User, user_id),
    })


@router.post('/admin/users/{user_id}/responses/{response_id}/edit')
async def admin_save_response_edit(
    user_id: int,
    response_id: int,
    transcription: str = Form(...),
    title: str | None = Form(None),
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    response = (await db.execute(
        select(Response).options(selectinload(Response.tags))
        .where(Response.id == response_id, Response.user_id == user_id)
    )).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    try:
        cur_tags = [t.slug for t in response.tags or []]
        ver = ResponseVersion(
            response_id=response.id,
            user_id=user_id,
            title=getattr(response, 'title', None),
            transcription=response.transcription,
            tags_json={'tags': cur_tags},
            edited_by_admin_id=getattr(admin, 'id', None),
        )
        db.add(ver)
    except Exception:
        pass
    response.transcription = transcription
    if title is not None and hasattr(response, 'title'):
        response.title = (title or '').strip() or None
    await db.commit()
    try:
        db.add(AdminEditLog(
            admin_user_id=getattr(admin, 'id', None),
            target_user_id=user_id,
            response_id=response.id,
            action='edit_text',
            payload={'title': title, 'len_transcription': len(transcription or '')},
        ))
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


# ---------------------------------------------------------------------------
# Admin supporting media
# ---------------------------------------------------------------------------

@router.post('/admin/users/{user_id}/responses/{response_id}/media')
async def admin_add_supporting_media(
    user_id: int,
    response_id: int,
    media_files: list[UploadFile] = File(...),
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    resp = (await db.execute(
        select(Response).where(Response.id == response_id, Response.user_id == user_id)
    )).scalars().first()
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
        target_user = await db.get(User, user_id)
        art = PIPELINE.process_upload(
            temp_path=tmp,
            logical='response',
            role='supporting',
            user_slug_or_id=target_user.username or str(user_id) if target_user else str(user_id),
            prompt_id=None,
            response_id=resp.id,
            media_id=media.id,
            original_filename=f.filename,
            content_type=f.content_type,
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
    await db.commit()
    try:
        db.add(AdminEditLog(
            admin_user_id=getattr(admin, 'id', None),
            target_user_id=user_id,
            response_id=resp.id,
            action='add_media',
            payload={'count': len(media_files or [])},
        ))
        await db.commit()
    except Exception:
        pass
    return RedirectResponse(url=f'/admin/users/{user_id}/responses/{response_id}/edit', status_code=303)


@router.delete('/admin/users/{user_id}/responses/{response_id}/media/{media_id}')
async def admin_delete_supporting_media(
    user_id: int,
    response_id: int,
    media_id: int,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    resp = (await db.execute(
        select(Response).where(Response.id == response_id, Response.user_id == user_id)
    )).scalars().first()
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
        db.add(AdminEditLog(
            admin_user_id=getattr(admin, 'id', None),
            target_user_id=user_id,
            response_id=response_id,
            action='delete_media',
            payload={'media_id': media_id},
        ))
        await db.commit()
    except Exception:
        pass
    return JSONResponse({'ok': True})


# ---------------------------------------------------------------------------
# Admin response tags
# ---------------------------------------------------------------------------

@router.get('/admin/users/{user_id}/responses/{response_id}/tags')
async def admin_get_response_tags(
    user_id: int,
    response_id: int,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    response = (await db.execute(
        select(Response).options(selectinload(Response.tags))
        .where(Response.id == response_id, Response.user_id == user_id)
    )).scalars().first()
    if not response:
        raise HTTPException(status_code=404, detail='Response not found')
    return {'tags': [{'id': t.id, 'name': t.name, 'slug': t.slug, 'color': t.color} for t in response.tags or []]}


@router.post('/admin/users/{user_id}/responses/{response_id}/tags')
async def admin_set_response_tags(
    user_id: int,
    response_id: int,
    payload: Any = Body(...),
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    exists = (await db.execute(
        select(Response.id).where(Response.id == response_id, Response.user_id == user_id)
    )).scalar_one_or_none()
    if not exists:
        raise HTTPException(status_code=404, detail='Response not found')
    try:
        resp = (await db.execute(
            select(Response).options(selectinload(Response.tags))
            .where(Response.id == response_id, Response.user_id == user_id)
        )).scalars().first()
        cur_tags = [t.slug for t in resp.tags or []] if resp else []
        db.add(ResponseVersion(
            response_id=response_id,
            user_id=user_id,
            title=getattr(resp, 'title', None) if resp else None,
            transcription=getattr(resp, 'transcription', None) if resp else None,
            tags_json={'tags': cur_tags},
            edited_by_admin_id=getattr(admin, 'id', None),
        ))
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
        db.add(AdminEditLog(
            admin_user_id=getattr(admin, 'id', None),
            target_user_id=user_id,
            response_id=response_id,
            action='set_tags',
            payload={'count': len(tags)},
        ))
        await db.commit()
    except Exception:
        pass
    return {'ok': True, 'tags': [{'id': t.id, 'name': t.name, 'slug': t.slug, 'color': t.color} for t in tags]}


# ---------------------------------------------------------------------------
# Admin segments
# ---------------------------------------------------------------------------

@router.get('/admin/users/{user_id}/responses/{response_id}/segments')
async def admin_list_response_segments(
    user_id: int,
    response_id: int,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    exists = (await db.execute(
        select(Response.id).where(Response.id == response_id, Response.user_id == user_id)
    )).scalar_one_or_none()
    if not exists:
        raise HTTPException(status_code=404, detail='Response not found')
    rows = (await db.execute(
        select(ResponseSegment)
        .where(ResponseSegment.response_id == response_id)
        .order_by(ResponseSegment.order_index.asc(), ResponseSegment.id.asc())
    )).scalars().all()
    return [
        {
            'id': s.id,
            'order_index': getattr(s, 'order_index', None),
            'transcript': getattr(s, 'transcript', '') or '',
            'media_path': getattr(s, 'media_path', '') or '',
            'media_mime': getattr(s, 'media_mime', '') or '',
            'thumbnail_path': getattr(s, 'thumbnail_path', None) or getattr(s, 'thumbnail_url', None) or '',
        }
        for s in rows
    ]


@router.post('/admin/users/{user_id}/responses/{response_id}/segments')
async def admin_add_response_segment(
    user_id: int,
    response_id: int,
    file: UploadFile = File(...),
    note: str | None = Form(None),
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
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
    art = PIPELINE.process_upload(
        temp_path=tmp,
        logical='response',
        role='supporting',
        user_slug_or_id=target_user.username or str(user_id) if target_user else str(user_id),
        prompt_id=None,
        response_id=response_id,
        media_id=seg.id,
        original_filename=file.filename or 'segment',
        content_type=file.content_type,
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
        spawn(transcribe_segment_and_update(seg.id, uploads_rel_for_tx), name='transcribe_segment_admin_upload')
    except Exception:
        pass
    return {'segment': {'id': seg.id}}


@router.patch('/admin/users/{user_id}/responses/{response_id}/segments/reorder')
async def admin_reorder_response_segments(
    user_id: int,
    response_id: int,
    payload: ReorderSegmentsRequest = Body(...),
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    exists = (await db.execute(
        select(Response.id).where(Response.id == response_id, Response.user_id == user_id)
    )).scalar_one_or_none()
    if not exists:
        raise HTTPException(status_code=404, detail='Response not found')
    ids = payload.order or []
    if not ids:
        return {'ok': True}
    found = (await db.execute(
        select(ResponseSegment.id)
        .where(ResponseSegment.response_id == response_id, ResponseSegment.id.in_(ids))
    )).scalars().all()
    if set(found) != set(ids):
        raise HTTPException(status_code=400, detail='Invalid segment ids for this response.')
    order_map = {sid: idx + 1 for idx, sid in enumerate(ids)}
    rows = (await db.execute(
        select(ResponseSegment).where(ResponseSegment.id.in_(ids))
    )).scalars().all()
    for r in rows:
        r.order_index = order_map.get(r.id, r.order_index)
    await db.commit()
    return {'ok': True}


@router.delete('/admin/users/{user_id}/responses/{response_id}/segments/{segment_id}')
async def admin_delete_response_segment(
    user_id: int,
    response_id: int,
    segment_id: int,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
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
async def admin_bootstrap_first_segment(
    user_id: int,
    response_id: int,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
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
        spawn(transcribe_segment_and_update(seg.id, uploads_rel), name='transcribe_segment_admin_bootstrap')
    except Exception:
        pass
    return JSONResponse({'ok': True, 'created': True, 'segment_id': seg.id})


@router.post('/admin/users/{user_id}/responses/{response_id}/segments/merge-audio')
async def admin_merge_audio_to_primary(
    user_id: int,
    response_id: int,
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    resp = await db.get(Response, response_id)
    if not resp or resp.user_id != user_id:
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
    except Exception:
        logger.exception('admin concat_audio_async crashed')
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


# ---------------------------------------------------------------------------
# Admin versions
# ---------------------------------------------------------------------------

@router.get('/admin/users/{user_id}/responses/{response_id}/versions')
async def admin_list_versions(
    user_id: int,
    response_id: int,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    exists = (await db.execute(
        select(Response.id).where(Response.id == response_id, Response.user_id == user_id)
    )).scalar_one_or_none()
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


@router.post('/admin/users/{user_id}/responses/{response_id}/versions/{version_id}/restore')
async def admin_restore_version(
    user_id: int,
    response_id: int,
    version_id: int,
    admin=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
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
        db.add(AdminEditLog(
            admin_user_id=getattr(admin, 'id', None),
            target_user_id=user_id,
            response_id=response_id,
            action='restore_version',
            payload={'version_id': version_id},
        ))
        await db.commit()
    except Exception:
        pass
    return {'ok': True}


# ---------------------------------------------------------------------------
# Impersonation stubs (disabled)
# ---------------------------------------------------------------------------

@router.get('/admin/impersonate/stop')
async def admin_impersonate_stop_disabled():
    raise HTTPException(status_code=404, detail='Impersonation is disabled')


@router.get('/admin/impersonate/{target_user_id:int}')
async def admin_impersonate_disabled(target_user_id: int):
    raise HTTPException(status_code=404, detail='Impersonation is disabled')
