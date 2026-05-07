from __future__ import annotations

import os
import shutil
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import (
    ChapterMeta,
    Prompt,
    Response,
    ResponseNotificationTarget,
    Tag,
    User,
    UserProfile,
    UserWeeklySkip,
)
from app.routes_shared import STATIC_DIR, templates
from app.services.assignment import build_pool_for_user, ensure_weekly_prompt
from app.services.utils_weekly import mark_clicked
from app.utils import require_authenticated_html_user, require_authenticated_user, get_current_user


router = APIRouter()


@router.get('/login', response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, 'login.html', {'request': request, 'user': None})


@router.get('/settings', response_class=HTMLResponse)
async def settings_page(
    request: Request,
    user=Depends(require_authenticated_html_user),
    db: AsyncSession = Depends(get_db),
):
    profile = (
        await db.execute(select(UserProfile).where(UserProfile.user_id == user.id))
    ).scalars().first()

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

    interests_list: list[str] = []
    if profile and profile.interests:
        interests_list = list(profile.interests)
    elif profile and isinstance(profile.tag_weights, dict):
        tw_i = dict(profile.tag_weights or {}).get('tagWeights') or {}
        _skip = ('role:', 'place:', 'person:')
        interests_list = [k for k, v in tw_i.items() if (v or 0) > 0 and not any(k.startswith(p) for p in _skip)]
    interests_for_form = ', '.join(interests_list)

    places_list: list[str] = []
    if profile and isinstance(profile.tag_weights, dict):
        tw2 = dict(profile.tag_weights or {}).get('tagWeights') or {}
        try:
            places_list = [
                k.split(':', 1)[1].replace('-', ' ')
                for k, v in tw2.items()
                if str(k).startswith('place:') and (v or 0) > 0
            ]
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

    # All profile tags sorted by weight descending, for the settings panel
    all_profile_tags: list[dict] = []
    if profile and isinstance(profile.tag_weights, dict):
        tw_all = dict(profile.tag_weights or {}).get('tagWeights') or {}
        all_profile_tags = sorted(
            [{'slug': k, 'weight': round(v, 2)} for k, v in tw_all.items() if (v or 0) > 0],
            key=lambda x: x['weight'], reverse=True,
        )

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
        'notify_daily_reminders_enabled': bool(getattr(user, 'notify_daily_reminders', False)),
        'notify_new_responses_enabled': bool(getattr(user, 'notify_new_responses', False)),
        'notification_watchers': watchers,
        'notification_options': notification_options,
        'all_profile_tags': all_profile_tags,
    }
    return templates.TemplateResponse(request, 'settings.html', ctx)


@router.delete('/settings/profile-tags/{slug:path}')
async def remove_profile_tag(
    slug: str,
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    profile = (await db.execute(select(UserProfile).where(UserProfile.user_id == user.id))).scalars().first()
    if not profile:
        raise HTTPException(status_code=404, detail='Profile not found')
    tw = dict(profile.tag_weights or {})
    weights = tw.get('tagWeights') or {}
    if slug in weights:
        del weights[slug]
        tw['tagWeights'] = weights
        profile.tag_weights = tw
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(profile, 'tag_weights')
        await db.commit()
    return {'ok': True}


@router.post('/settings/profile')
async def settings_profile_update(
    request: Request,
    display_name: Optional[str] = Form(None),
    birth_year: Optional[int] = Form(None),
    location: Optional[str] = Form(None),
    relation_roles: Optional[str] = Form(None),
    interests: Optional[str] = Form(None),
    bio: Optional[str] = Form(None),
    places: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    prof = (
        await db.execute(select(UserProfile).where(UserProfile.user_id == user.id))
    ).scalars().first()
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
        import json as _json
        try:
            data = _json.loads(text)
        except _json.JSONDecodeError:
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
    from app.utils import slugify as _slugify_local
    for r in prof.relation_roles or []:
        slug_val = _slugify_local(r)
        if not slug_val:
            continue
        key = f'role:{slug_val}'
        try:
            weights[key] = max(float(weights.get(key, 0.0) or 0.0), 0.7)
        except Exception:
            weights[key] = 0.7
    for interest in prof.interests or []:
        slug_val = interest if ':' in interest else _slugify_local(interest)
        if not slug_val:
            continue
        try:
            weights[slug_val] = max(float(weights.get(slug_val, 0.0) or 0.0), 0.6)
        except Exception:
            weights[slug_val] = 0.6
    for base in _parse_tag_input(places):
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

    # Sync display_name change to the "self" Person node in the people graph
    if prof.display_name:
        try:
            from app.models import Person
            self_people = (await db.execute(
                select(Person).where(Person.owner_user_id == user.id)
            )).scalars().all()
            for p in self_people:
                m = p.meta if isinstance(p.meta, dict) else {}
                if m.get('connect_to_owner') and str(m.get('role_hint', '')).strip().lower() in {'you', 'self', 'me'}:
                    p.display_name = prof.display_name
                    break
        except Exception:
            pass

    try:
        await build_pool_for_user(db, user.id)
    except Exception:
        pass

    await db.commit()
    return RedirectResponse(url='/settings?notice=Saved', status_code=303)


@router.post('/settings/notifications')
async def settings_notifications_update(
    notify_daily_reminders: bool = Form(False),
    notify_new_responses: bool = Form(False),
    watchers: List[int] = Form([]),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    db_user = (await db.execute(select(User).where(User.id == user.id))).scalars().first()
    if not db_user:
        raise HTTPException(status_code=404, detail='User not found')
    db_user.notify_daily_reminders = bool(notify_daily_reminders)
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
    for wid in sorted(set(watcher_ids)):
        db.add(ResponseNotificationTarget(owner_user_id=user.id, watcher_user_id=wid))
    await db.commit()
    msg = 'Notifications+enabled' if db_user.notify_new_responses else 'Notifications+disabled'
    return RedirectResponse(url=f'/settings?notice={msg}', status_code=303)


@router.post('/settings/password')
async def settings_password_update(
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    import bcrypt as _bcrypt_lib

    cur = (current_password or '').strip()
    new = (new_password or '').strip()
    conf = (confirm_password or '').strip()
    if new != conf:
        return RedirectResponse(url='/settings?notice=Passwords+do+not+match&error=1', status_code=303)
    if len(new) < 8:
        return RedirectResponse(url='/settings?notice=Password+must+be+at+least+8+characters&error=1', status_code=303)
    try:
        ok = _bcrypt_lib.checkpw(cur.encode(), (user.hashed_password or '').encode())
    except Exception:
        ok = False
    if not ok:
        return RedirectResponse(url='/settings?notice=Current+password+is+incorrect&error=1', status_code=303)
    user.hashed_password = _bcrypt_lib.hashpw(new.encode(), _bcrypt_lib.gensalt()).decode()
    await db.commit()
    return RedirectResponse(url='/settings?notice=Password+updated', status_code=303)


@router.post('/settings/avatar')
async def settings_avatar_upload(
    avatar: UploadFile = File(...),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    _, ext = os.path.splitext(avatar.filename or '')
    ext = (ext or '.jpg').lower()
    safe_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    if ext not in safe_exts:
        ext = '.jpg'

    user_dir = (user.username or str(user.id)).replace('/', '_').replace('\\', '_')
    rel_dir = os.path.join('uploads', 'users', user_dir, 'profile')
    abs_dir = STATIC_DIR / rel_dir
    os.makedirs(abs_dir, exist_ok=True)

    rel_path = os.path.join(rel_dir, f'avatar{ext}').replace('\\', '/')
    abs_path = STATIC_DIR / rel_path

    with open(abs_path, 'wb') as buffer:
        shutil.copyfileobj(avatar.file, buffer)

    prof = (
        await db.execute(select(UserProfile).where(UserProfile.user_id == user.id))
    ).scalars().first()
    if not prof:
        prof = UserProfile(user_id=user.id)
        db.add(prof)

    prefs = dict(prof.privacy_prefs or {})
    prefs['avatar_url'] = rel_path
    prof.privacy_prefs = prefs
    await db.commit()
    return RedirectResponse(url='/settings?notice=Photo+updated', status_code=303)


@router.get('/user_dashboard', response_class=HTMLResponse)
async def user_dashboard(
    request: Request,
    q: str | None = Query(None),
    prompt_id: int | None = Query(None),
    ofs: int = Query(0, alias='offset'),
    user=Depends(require_authenticated_html_user),
    db: AsyncSession = Depends(get_db),
):
    if not user or not user.is_active:
        raise HTTPException(status_code=403, detail='Unauthorized')

    # Ensure user has a self-person node in the graph (no-op if already set)
    try:
        from app.services.people import ensure_self_person
        display = user.username or (user.email.split("@")[0] if user.email else "Me")
        await ensure_self_person(db, user.id, display)
    except Exception:
        pass  # Never block the dashboard for a graph setup issue

    stmt = (
        select(Response)
        .join(Prompt, Prompt.id == Response.prompt_id)
        .outerjoin(Tag, Prompt.tags)
        .options(selectinload(Response.prompt))
        .where(Response.user_id == user.id)
        .order_by(Response.created_at.desc())
    )
    if q:
        like = f'%{q}%'
        stmt = stmt.where(or_(
            Response.response_text.ilike(like),
            Response.transcription.ilike(like),
            Prompt.text.ilike(like),
            Tag.name.ilike(like),
        ))

    responses = (await db.execute(stmt)).unique().scalars().all()

    chap_rows = await db.execute(select(Prompt.chapter).distinct())
    all_chapters = [row[0] for row in chap_rows.all() if row[0]]

    meta_rows = (await db.execute(select(ChapterMeta))).scalars().all()
    meta_by = {m.name: m for m in meta_rows}

    ordered = sorted(all_chapters, key=lambda nm: (getattr(meta_by.get(nm), 'order', 1_000_000), nm.lower()))
    base_color = '#e5e7eb'

    def alpha_for_index(i: int) -> float:
        return min(0.04 + i * 0.03, 0.28)

    chapter_styles = {}
    for i, name in enumerate(ordered):
        meta = meta_by.get(name)
        color = (meta.tint or base_color) if meta else base_color
        chapter_styles[name] = {'color': color, 'alpha': f'{alpha_for_index(i):.2f}'}

    current_prompt = None
    if prompt_id:
        current_prompt = (await db.execute(
            select(Prompt).options(selectinload(Prompt.tags)).where(Prompt.id == prompt_id)
        )).scalars().first()
    else:
        weekly = await ensure_weekly_prompt(db, user.id)
        if weekly and getattr(weekly, 'prompt_id', None):
            current_prompt = (await db.execute(
                select(Prompt).options(selectinload(Prompt.tags)).where(Prompt.id == weekly.prompt_id)
            )).scalars().first()

    iso = date.today().isocalendar()
    y, w = iso.year, iso.week
    skipped_ids = (await db.execute(
        select(UserWeeklySkip.prompt_id).where(
            (UserWeeklySkip.user_id == user.id)
            & (UserWeeklySkip.year == y)
            & (UserWeeklySkip.week == w)
        )
    )).scalars().all()

    skipped_prompts = []
    if skipped_ids:
        skipped_prompts = (await db.execute(
            select(Prompt).options(selectinload(Prompt.tags)).where(Prompt.id.in_(skipped_ids))
        )).unique().scalars().all()

    ctx = {
        'request': request,
        'user': user,
        'current_prompt': current_prompt,
        'responses': responses,
        'chapter_styles': chapter_styles,
        'skipped_prompts': skipped_prompts,
    }
    return templates.TemplateResponse(request, 'user_dashboard.html', ctx)


@router.get('/user_record', response_class=HTMLResponse, name='user_record_latest')
async def user_record(
    request: Request,
    user=Depends(require_authenticated_html_user),
    db: AsyncSession = Depends(get_db),
):
    prompt = (await db.execute(
        select(Prompt)
        .options(selectinload(Prompt.media))
        .order_by(Prompt.created_at.desc())
        .limit(1)
    )).scalars().first()

    return templates.TemplateResponse(
        request,
        'user_record.html',
        {
            'request': request,
            'user': user,
            'prompt': prompt,
            'prompt_media': list(prompt.media) if prompt else [],
        },
    )


@router.get('/user_record/freeform', response_class=HTMLResponse, name='user_record_freeform')
async def user_record_freeform(
    request: Request,
    user=Depends(require_authenticated_html_user),
    db: AsyncSession = Depends(get_db),
):
    chapters_res = await db.execute(select(Prompt.chapter).distinct().order_by(Prompt.chapter))
    chapters = [row[0] for row in chapters_res.all() if row[0]]
    return templates.TemplateResponse(
        request,
        'user_record.html',
        {
            'request': request,
            'user': user,
            'prompt': None,
            'prompt_media': [],
            'chapters': chapters,
        },
    )


@router.get('/user_record/{prompt_id}', response_class=HTMLResponse, name='user_record')
async def user_record_with_prompt(
    prompt_id: int,
    request: Request,
    token: Optional[str] = Query(None),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not user and token:
        try:
            tok = await mark_clicked(db, token)
            await db.commit()
        except Exception:
            tok = None
        if not tok or tok.prompt_id != prompt_id:
            next_rel = request.url.path + (f'?token={token}' if token else '')
            return RedirectResponse(url=f'/login?next={next_rel}', status_code=303)

    prompt = (await db.execute(
        select(Prompt).options(selectinload(Prompt.media)).where(Prompt.id == prompt_id)
    )).scalars().first()
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
    return templates.TemplateResponse(
        request,
        'user_record.html',
        {
            'request': request,
            'user': user,
            'prompt': prompt,
            'prompt_media': media,
            'is_token_link': bool(token),
        },
    )


__all__ = ['router']
