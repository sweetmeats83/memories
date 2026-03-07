from fastapi import APIRouter, Depends, Request, Form, UploadFile, File, HTTPException, Path, Query, Body
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func, and_, update, delete
from sqlalchemy.orm import selectinload
from typing import List, Any, Optional
import os, re, asyncio, json, uuid, shutil, secrets, logging
from datetime import datetime, timedelta, timezone, date
from pathlib import Path as FSPath
from sqlalchemy.exc import IntegrityError
from .database import get_db, async_session_maker
from .models import (
    Prompt,
    PromptMedia,
    Response,
    SupportingMedia,
    Invite,
    User,
    Tag,
    UserWeeklyPrompt,
    UserWeeklySkip,
    ResponseSegment,
    UserProfile,
    ChapterMeta,
    UserPrompt,
    WeeklyToken,
    WeeklyTokenStatus,
    ResponseVersion,
    ResponseNotificationTarget,
    ResponseShare,
)
from .utils import require_authenticated_user, require_admin_user, require_authenticated_html_user, get_current_user
from .schemas import ReorderSegmentsRequest
from .transcription import transcribe_file, enrich_after_transcription
from .llm_client import polish_text, OllamaError
from .users import current_active_user
from fastapi.templating import Jinja2Templates
from sqlalchemy.dialects.postgresql import insert as pg_insert
from .media_pipeline import MediaPipeline, UserBucketsStrategy
from app.services.notifications import notify_new_response
from pydantic import BaseModel
from app.services.auto_tag import suggest_tags_rule_based
from app.services.assignment import ensure_weekly_prompt, rotate_to_next_unanswered as _rotate, skip_current_prompt, build_pool_for_user, _iso_year_week
from app.services.utils_weekly import _now, mark_clicked, mark_completed_and_close
from app.background import spawn
from app.routers.upload import get_staged_file, release_staged_dir

templates = Jinja2Templates(directory='templates')
logger = logging.getLogger(__name__)
router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
STATIC_DIR = FSPath(BASE_DIR) / 'static'
PIPELINE = MediaPipeline(static_root=STATIC_DIR, path_strategy=UserBucketsStrategy())

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
                if primary_rel and (not is_composite):
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
    except Exception as exc:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
    except Exception as exc:  # noqa: BLE001
        logger.exception('Supporting media processing failed for media %s: %s', media_id, exc)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

# People/kinship/groups routes moved to app/routers/people.py

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
            'is_admin': bool(getattr(watcher, 'is_superuser', False)),
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
            'is_admin': bool(getattr(candidate, 'is_superuser', False)),
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

    def _parse_tag_input(raw: Optional[str]) -> list[str]:
        text = (raw or '').strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            out: list[str] = []
            for item in data:
                if isinstance(item, str):
                    v = item.strip()
                    if v:
                        out.append(v)
                elif isinstance(item, dict):
                    v = (item.get('value') or item.get('slug') or item.get('name') or '').strip()
                    if v:
                        out.append(v)
            return out
        return [t.strip() for t in text.split(',') if t.strip()]

    prof.relation_roles = _parse_tag_input(relation_roles) or None
    prof.interests = _parse_tag_input(interests) or None
    prof.bio = (bio or '').strip() or None
    tw = dict(prof.tag_weights or {'tagWeights': {}})
    weights = tw.setdefault('tagWeights', {})
    for r in prof.relation_roles or []:
        from .utils import slugify as _slugify_local
        slug_val = _slugify_local(r)
        if not slug_val:
            continue
        key = f'role:{slug_val}'
        try:
            weights[key] = max(float(weights.get(key, 0.0) or 0.0), 0.7)
        except Exception:
            weights[key] = 0.7
    for base in _parse_tag_input(places):
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
    if not user or not user.is_active:
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
async def create_response(prompt_id: int | None=Form(None), title: str | None=Form(None), chapter: str | None=Form(None), response_text: str | None=Form(None), primary_media: UploadFile | None=File(None), primary_staged_id: str | None=Form(None), supporting_media: list[UploadFile] | None=File(None), weekly_token: str | None=Form(None), request: Request=None, user=Depends(get_current_user), db: AsyncSession=Depends(get_db)):
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
            pending_supporting.append(
                (
                    media.id,
                    tmp,
                    f.filename or f'supporting-{media.id}',
                    f.content_type or 'application/octet-stream',
                )
            )
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
            weekly = (await db.execute(select(UserWeeklyPrompt).where(UserWeeklyPrompt.user_id == acting_user.id, UserWeeklyPrompt.year == y, UserWeeklyPrompt.week == w))).scalars().first()
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
        html = '\n        <section class="fixed inset-0 grid place-items-center bg-black/70">\n          <div class="bg-white/95 rounded-2xl p-6 max-w-md text-center">\n            <h2 class="text-xl font-semibold mb-2">Thanks for sharing your story!</h2>\n            <p class="mb-4 text-slate-700">We saved it for you. If you’d like to read or edit it later, please log in.</p>\n            <div class="flex justify-center gap-2">\n              <a href="/login" class="btn">Go to Login</a>\n              <a href="/" class="btn btn-ghost">Close</a>\n            </div>\n          </div>\n        </section>\n        '
        return templates.TemplateResponse('thank_you.html', {'request': request, 'user': None})
    _wants_json = 'application/json' in (request.headers.get('accept') or '') or request.headers.get('x-requested-with') == 'XMLHttpRequest'
    if _wants_json:
        return {'id': new_response.id, 'processing_state': new_response.processing_state}
    if new_response.processing_state == 'ready':
        return RedirectResponse(url=f'/response/{new_response.id}/edit', status_code=303)
    return RedirectResponse(url=f'/response/{new_response.id}/processing', status_code=303)

@router.get('/response/{response_id}', response_class=HTMLResponse, name='response_view')
async def response_view(response_id: int, request: Request, user=Depends(require_authenticated_html_user), db: AsyncSession=Depends(get_db)):
    q = select(Response).options(selectinload(Response.prompt).selectinload(Prompt.media), selectinload(Response.tags), selectinload(Response.supporting_media), selectinload(Response.segments)).where(Response.id == response_id, Response.user_id == user.id)
    resp = (await db.execute(q)).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail='Response not found')
    if getattr(resp, 'processing_state', 'ready') != 'ready':
        return RedirectResponse(url=f'/response/{response_id}/processing', status_code=303)
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
    if getattr(response, 'processing_state', 'ready') != 'ready':
        return RedirectResponse(url=f'/response/{response_id}/processing', status_code=303)
    supporting_media = (await db.execute(select(SupportingMedia).where(SupportingMedia.response_id == response_id))).scalars().all()
    pm_res = await db.execute(select(PromptMedia).where(PromptMedia.prompt_id == response.prompt_id))
    prompt_media = pm_res.scalars().all()
    return templates.TemplateResponse('response_edit.html', {'request': request, 'user': user, 'response': response, 'supporting_media': supporting_media, 'prompt_media': prompt_media})


@router.get('/response/{response_id}/processing', response_class=HTMLResponse, name='response_processing_page')
async def response_processing_page(response_id: int, request: Request, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
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
async def api_response_status(response_id: int, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
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
    if getattr(resp, 'processing_state', 'ready') != 'ready':
        return templates.TemplateResponse(
            'response_processing_public.html',
            {
                'request': request,
                'message': 'This response is still processing. Please try again in a few minutes.',
            },
            status_code=202,
        )
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
async def add_supporting_media(response_id: int, media_files: list[UploadFile] | None=File(None), staged_ids: str | None=Form(None), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
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
    if staged_ids:
        import json as _json
        try:
            uid_list = _json.loads(staged_ids)
        except Exception:
            uid_list = []
        for uid in uid_list:
            staged = get_staged_file(uid, user.id)
            if not staged:
                continue
            media = SupportingMedia(response_id=resp.id, file_path='', media_type=staged['content_type'].split('/', 1)[0] if staged.get('content_type') else 'file')
            db.add(media)
            await db.flush()
            tmp = FSPath(staged['assembled_path'])
            art = PIPELINE.process_upload(temp_path=tmp, logical='response', role='supporting', user_slug_or_id=user.username or str(user.id), prompt_id=None, response_id=resp.id, media_id=media.id, original_filename=staged['filename'], content_type=staged.get('content_type'))
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

async def on_prompt_created(db: AsyncSession, prompt_id: int) -> None:
    """
    Called after a Prompt is created (and tags are attached).
    It pre-assigns the prompt to every eligible user, UNLESS the prompt is private_only/only_assigned.
    """
    try:
        from app.models import Prompt, UserProfile, UserPrompt
        from app.services.assignment import _eligible, _get_profile_weights
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
    except Exception:
        logging.exception('[prompt-fanout] on_prompt_created failed for %s', prompt_id)

@router.post('/admin_create_prompt')
async def admin_create_prompt(
    request: Request,
    prompt_text: str=Form(...),
    chapter: str=Form(...),
    tags: str=Form('[]'),
    media_files: list[UploadFile] | None=File(None),
    only_assigned: int | bool=Form(0),
    assign_user_ids: str=Form('[]'),
    user=Depends(require_admin_user),
    db: AsyncSession=Depends(get_db),
):
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

    if assigned_ids:
        for uid in assigned_ids:
            exists = (
                await db.execute(
                    select(UserPrompt).where(
                        UserPrompt.user_id == uid,
                        UserPrompt.prompt_id == prompt.id,
                    )
                )
            ).scalars().first()
            if not exists:
                db.add(UserPrompt(user_id=uid, prompt_id=prompt.id))
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
async def admin_update_prompt(
    prompt_id: int,
    request: Request,
    prompt_text: str=Form(...),
    chapter: str=Form(...),
    tags: str=Form('[]'),
    media_files: list[UploadFile]=File(None),
    assign_user_ids: str=Form('[]'),
    user=Depends(require_admin_user),
    db: AsyncSession=Depends(get_db),
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

    existing_assignments = (
        await db.execute(
            select(UserPrompt).where(UserPrompt.prompt_id == prompt_id)
        )
    ).unique().scalars().all()
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
async def add_response_segment(response_id: int, file: UploadFile | None=File(None), staged_id: str | None=Form(None), note: str | None=Form(None), user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    """
    Upload a *new segment* for this response.
    - Stores the media under users/<user>/responses/<id>/supporting/<segment_id>/
    - DOES NOT touch resp.primary_media_url (primary changes only on merge)
    - Returns JSON { ok, segment:{...} } so the overlay JS can poll for transcript
    """
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
    art = PIPELINE.process_upload(temp_path=tmp, logical='response', role='supporting', user_slug_or_id=user.username or str(user.id), prompt_id=None, response_id=response_id, media_id=seg.id, original_filename=original_filename or 'segment', content_type=content_type)
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
        if old_primary_rel and old_primary_rel not in sources and (old_primary_rel != resp.primary_media_url):
            await asyncio.to_thread(PIPELINE.delete_artifacts, old_primary_rel)
    except Exception:
        pass
    return JSONResponse({'ok': True, 'primary': resp.primary_media_url})

