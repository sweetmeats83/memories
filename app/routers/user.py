from __future__ import annotations

import os
import shutil
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import (
    ChapterMeta,
    Prompt,
    Response,
    Tag,
    UserProfile,
    UserWeeklySkip,
)
from app.routes_shared import STATIC_DIR, templates
from app.services.assignment import build_pool_for_user, ensure_weekly_prompt
from app.services.utils_weekly import mark_clicked
from app.utils import require_authenticated_html_user, require_authenticated_user


router = APIRouter()


@router.get("/settings", response_class=HTMLResponse)
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
        tw = dict(profile.tag_weights or {}).get("tagWeights") or {}
        try:
            roles = [k.split(":", 1)[1] for k, v in tw.items() if str(k).startswith("role:") and (v or 0) > 0]
        except Exception:
            roles = []

    ob = (getattr(profile, "privacy_prefs", None) or {}).get("onboarding") if profile else None

    rel_roles = []
    if profile and profile.relation_roles:
        rel_roles = list(profile.relation_roles or [])
    elif roles:
        rel_roles = roles
    roles_for_form = ", ".join(rel_roles)
    interests_for_form = ", ".join((profile.interests or []) if profile and profile.interests else [])

    places_list: list[str] = []
    if profile and isinstance(profile.tag_weights, dict):
        tw2 = dict(profile.tag_weights or {}).get("tagWeights") or {}
        try:
            places_list = [
                k.split(":", 1)[1].replace("-", " ")
                for k, v in tw2.items()
                if str(k).startswith("place:") and (v or 0) > 0
            ]
        except Exception:
            places_list = []
    places_for_form = ", ".join(places_list)

    gender = None
    if profile and isinstance(profile.privacy_prefs, dict):
        gender = (profile.privacy_prefs or {}).get("user_meta", {}).get("gender")

    ctx = {
        "request": request,
        "user": user,
        "profile": profile,
        "onboarding_roles": roles,
        "onboarding_meta": ob,
        "roles_for_form": roles_for_form,
        "interests_for_form": interests_for_form,
        "gender": gender,
        "places_for_form": places_for_form,
    }
    return templates.TemplateResponse("settings.html", ctx)


@router.post("/settings/profile")
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

    prof.display_name = (display_name or "").strip() or None
    prof.birth_year = birth_year if birth_year else None
    prof.location = (location or "").strip() or None

    def _csv(value: Optional[str]):
        return [t.strip() for t in (value or "").split(",") if t.strip()]

    prof.relation_roles = _csv(relation_roles) or None
    prof.interests = _csv(interests) or None
    prof.bio = (bio or "").strip() or None

    tw = dict(prof.tag_weights or {"tagWeights": {}})
    weights = tw.setdefault("tagWeights", {})
    for role in (prof.relation_roles or []):
        key = f"role:{role}"
        try:
            weights[key] = max(float(weights.get(key, 0.0) or 0.0), 0.7)
        except Exception:
            weights[key] = 0.7

    from app.utils import slugify as slugify_local

    for place in _csv(places):
        slug = f"place:{slugify_local(place)}"
        try:
            weights[slug] = max(float(weights.get(slug, 0.0) or 0.0), 0.5)
        except Exception:
            weights[slug] = 0.5
    prof.tag_weights = tw

    prefs = dict(prof.privacy_prefs or {})
    if gender is not None:
        meta = dict(prefs.get("user_meta") or {})
        meta["gender"] = (gender or "").strip() or None
        prefs["user_meta"] = meta
    prof.privacy_prefs = prefs

    try:
        await build_pool_for_user(db, user.id)
    except Exception:
        pass

    await db.commit()
    return RedirectResponse(url="/settings?notice=Saved", status_code=303)


@router.post("/settings/password")
async def settings_password_update(
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    from passlib.hash import bcrypt as _bcrypt

    current = (current_password or "").strip()
    new = (new_password or "").strip()
    confirm = (confirm_password or "").strip()

    if new != confirm:
        return RedirectResponse(url="/settings?notice=Passwords+do+not+match&error=1", status_code=303)
    if len(new) < 8:
        return RedirectResponse(url="/settings?notice=Password+must+be+at+least+8+characters&error=1", status_code=303)

    try:
        ok = _bcrypt.verify(current, user.hashed_password or "")
    except Exception:
        ok = False
    if not ok:
        return RedirectResponse(url="/settings?notice=Current+password+is+incorrect&error=1", status_code=303)

    user.hashed_password = _bcrypt.hash(new)
    await db.commit()
    return RedirectResponse(url="/settings?notice=Password+updated", status_code=303)


@router.post("/settings/avatar")
async def settings_avatar_upload(
    avatar: UploadFile = File(...),
    user=Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    _, ext = os.path.splitext(avatar.filename or "")
    ext = (ext or ".jpg").lower()
    safe_exts = {".jpg", ".jpeg", ".png", ".webp"}
    if ext not in safe_exts:
        ext = ".jpg"

    user_dir = (user.username or str(user.id)).replace("/", "_").replace("\\", "_")
    rel_dir = os.path.join("uploads", "users", user_dir, "profile")
    abs_dir = STATIC_DIR / rel_dir
    os.makedirs(abs_dir, exist_ok=True)

    rel_path = os.path.join(rel_dir, f"avatar{ext}").replace("\\", "/")
    abs_path = STATIC_DIR / rel_path

    with open(abs_path, "wb") as buffer:
        shutil.copyfileobj(avatar.file, buffer)

    prof = (
        await db.execute(select(UserProfile).where(UserProfile.user_id == user.id))
    ).scalars().first()
    if not prof:
        prof = UserProfile(user_id=user.id)
        db.add(prof)

    prefs = dict(prof.privacy_prefs or {})
    prefs["avatar_url"] = rel_path
    prof.privacy_prefs = prefs
    await db.commit()
    return RedirectResponse(url="/settings?notice=Photo+updated", status_code=303)


@router.get("/user_dashboard", response_class=HTMLResponse)
async def user_dashboard(
    request: Request,
    q: str | None = Query(None),
    prompt_id: int | None = Query(None),
    ofs: int = Query(0, alias="offset"),
    user=Depends(require_authenticated_html_user),
    db: AsyncSession = Depends(get_db),
):
    if not user or (not getattr(user, "super_admin", False) and not user.is_active):
        raise HTTPException(status_code=403, detail="Unauthorized")

    stmt = (
        select(Response)
        .join(Prompt, Prompt.id == Response.prompt_id)
        .outerjoin(Tag, Prompt.tags)
        .options(selectinload(Response.prompt))
        .where(Response.user_id == user.id)
        .order_by(Response.created_at.desc())
    )

    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            or_(
                Response.response_text.ilike(like),
                Response.transcription.ilike(like),
                Prompt.text.ilike(like),
                Tag.name.ilike(like),
            )
        )

    responses = (await db.execute(stmt)).unique().scalars().all()

    chap_rows = await db.execute(select(Prompt.chapter).distinct())
    all_chapters = [row[0] for row in chap_rows.all() if row[0]]

    meta_rows = (await db.execute(select(ChapterMeta))).scalars().all()
    meta_by = {m.name: m for m in meta_rows}

    ordered = sorted(
        all_chapters,
        key=lambda nm: (getattr(meta_by.get(nm), "order", 1_000_000), nm.lower()),
    )

    base_color = "#e5e7eb"

    def alpha_for_index(i: int) -> float:
        return min(0.04 + i * 0.03, 0.28)

    chapter_styles = {}
    for i, name in enumerate(ordered):
        meta = meta_by.get(name)
        color = (meta.tint or base_color) if meta else base_color
        chapter_styles[name] = {"color": color, "alpha": f"{alpha_for_index(i):.2f}"}

    current_prompt = None
    if prompt_id:
        current_prompt = (
            await db.execute(
                select(Prompt)
                .options(selectinload(Prompt.tags))
                .where(Prompt.id == prompt_id)
            )
        ).scalars().first()
    else:
        weekly = await ensure_weekly_prompt(db, user.id)
        if weekly and getattr(weekly, "prompt_id", None):
            current_prompt = (
                await db.execute(
                    select(Prompt)
                    .options(selectinload(Prompt.tags))
                    .where(Prompt.id == weekly.prompt_id)
                )
            ).scalars().first()

    def _iso_year_week():
        iso = date.today().isocalendar()
        return iso.year, iso.week

    y, w = _iso_year_week()
    skipped_ids = (
        await db.execute(
            select(UserWeeklySkip.prompt_id).where(
                (UserWeeklySkip.user_id == user.id)
                & (UserWeeklySkip.year == y)
                & (UserWeeklySkip.week == w)
            )
        )
    ).scalars().all()

    skipped_prompts = []
    if skipped_ids:
        skipped_prompts = (
            await db.execute(
                select(Prompt)
                .options(selectinload(Prompt.tags))
                .where(Prompt.id.in_(skipped_ids))
            )
        ).unique().scalars().all()

    ctx = {
        "request": request,
        "user": user,
        "current_prompt": current_prompt,
        "responses": responses,
        "chapter_styles": chapter_styles,
        "skipped_prompts": skipped_prompts,
    }
    return templates.TemplateResponse("user_dashboard.html", ctx)


@router.get("/user_record", response_class=HTMLResponse, name="user_record_latest")
async def user_record(
    request: Request,
    user=Depends(require_authenticated_html_user),
    db: AsyncSession = Depends(get_db),
):
    prompt = (
        await db.execute(
            select(Prompt)
            .options(selectinload(Prompt.media))
            .order_by(Prompt.created_at.desc())
            .limit(1)
        )
    ).scalars().first()

    return templates.TemplateResponse(
        "user_record.html",
        {
            "request": request,
            "user": user,
            "prompt": prompt,
            "prompt_media": list(prompt.media) if prompt else [],
        },
    )


@router.get("/user_record/freeform", response_class=HTMLResponse, name="user_record_freeform")
async def user_record_freeform(
    request: Request,
    user=Depends(require_authenticated_html_user),
    db: AsyncSession = Depends(get_db),
):
    chapters_res = await db.execute(select(Prompt.chapter).distinct().order_by(Prompt.chapter))
    chapters = [row[0] for row in chapters_res.all() if row[0]]
    return templates.TemplateResponse(
        "user_record.html",
        {
            "request": request,
            "user": user,
            "prompt": None,
            "prompt_media": [],
            "chapters": chapters,
        },
    )


@router.get("/user_record/{prompt_id}", response_class=HTMLResponse, name="user_record")
async def user_record_with_prompt(
    prompt_id: int,
    request: Request,
    token: Optional[str] = Query(None),
    user=Depends(require_authenticated_html_user),
    db: AsyncSession = Depends(get_db),
):
    if not user and token:
        try:
            tok = await mark_clicked(db, token)
            await db.commit()
        except Exception:
            tok = None
        if not tok or tok.prompt_id != prompt_id:
            next_rel = request.url.path + (f"?token={token}" if token else "")
            return RedirectResponse(url=f"/login?next={next_rel}", status_code=303)

    prompt = (
        await db.execute(
            select(Prompt)
            .options(selectinload(Prompt.media))
            .where(Prompt.id == prompt_id)
        )
    ).scalars().first()
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    viewer_id = user.id if user else None
    prompt_media = list(prompt.media or [])

    def _can_view(media, uid):
        assigned_one = getattr(media, "assignee_user_id", None)
        assigned_many = [getattr(u, "id", None) for u in (getattr(media, "assignees", []) or [])]
        has_any = bool(assigned_one) or bool(assigned_many)
        if uid is None:
            return not has_any
        if not has_any:
            return True
        if assigned_one and assigned_one == uid:
            return True
        return uid in assigned_many

    visible_media = [m for m in prompt_media if _can_view(m, viewer_id)]

    return templates.TemplateResponse(
        "user_record.html",
        {
            "request": request,
            "user": user,
            "prompt": prompt,
            "prompt_media": visible_media,
            "is_token_link": bool(token),
        },
    )
