"""Onboarding flow routes."""
from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.database import get_db
from app.models import Person, UserProfile
from app.routes_shared import _get_or_create_tag, templates
from app.services.assignment import build_pool_for_user, ensure_weekly_prompt
from app.services.auto_tag import WHITELIST
from app.services.people import upsert_person_for_user
from app.utils import require_authenticated_html_user, require_authenticated_user, slug_person, slug_place, slug_role, slugify

router = APIRouter()


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

async def _ensure_profile(db: AsyncSession, user_id: int) -> UserProfile:
    prof = await db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    if not prof:
        prof = UserProfile(
            user_id=user_id,
            tag_weights={"tagWeights": {}},
            privacy_prefs={"onboarding": {"step": "you", "done": False}},
        )
        db.add(prof)
        await db.flush()
        return prof
    if not prof.tag_weights:
        prof.tag_weights = {"tagWeights": {}}
    if not prof.privacy_prefs or not isinstance(prof.privacy_prefs, dict):
        prof.privacy_prefs = {"onboarding": {"step": "you", "done": False}}
    else:
        prof.privacy_prefs.setdefault("onboarding", {"step": "you", "done": False})
    return prof


def _role_options_from_whitelist() -> list[str]:
    roles = sorted({t.split(":", 1)[1] for t in WHITELIST if t.startswith("relationship:")})
    return roles


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/onboarding")
async def onboarding_home(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user=Depends(require_authenticated_html_user),
):
    prof = await _ensure_profile(db, user.id)
    prof = await db.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
    ob = (prof.privacy_prefs or {}).get("onboarding") or {}
    step = ob.get("step") or "you"
    if ob.get("done"):
        return RedirectResponse(url="/onboarding/done", status_code=303)
    role_options = _role_options_from_whitelist()
    return templates.TemplateResponse(
        request,
        "onboarding_steps.html",
        {"request": request, "user": user, "step": step, "role_options": role_options, "profile": prof},
    )


@router.post("/onboarding/you")
async def onboarding_you(
    payload: dict = Body(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_authenticated_user),
):
    prof = await _ensure_profile(db, user.id)
    display_name = (payload.get("display_name") or "").strip()
    gender = (payload.get("gender") or "").strip()
    birthdate = (payload.get("birthdate") or "").strip()
    location = (payload.get("location") or "").strip()
    if display_name:
        prof.display_name = display_name
    if location:
        prof.location = location
    if birthdate and len(birthdate) >= 4 and birthdate[:4].isdigit():
        prof.birth_year = int(birthdate[:4])
    pp = dict(prof.privacy_prefs or {})
    user_meta = dict(pp.get("user_meta") or {})
    if gender:
        user_meta["gender"] = gender
    pp["user_meta"] = user_meta
    ob = dict(pp.get("onboarding") or {"step": "you", "done": False})
    ob["step"] = "roles"
    pp["onboarding"] = ob
    prof.privacy_prefs = pp
    flag_modified(prof, "privacy_prefs")
    await db.commit()
    return RedirectResponse(url="/onboarding", status_code=303)


@router.post("/onboarding/roles")
async def onboarding_roles(
    payload: dict = Body(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_authenticated_user),
):
    prof = await _ensure_profile(db, user.id)
    tw = dict(prof.tag_weights or {"tagWeights": {}})
    weights = tw.setdefault("tagWeights", {})
    incoming = payload if isinstance(payload, list) else payload.get("roles") or []
    cleaned_roles = []
    for r in incoming:
        v = (r.get("value") if isinstance(r, dict) else str(r or "")).strip()
        if not v:
            continue
        if v.startswith("relationship:"):
            v = v.split(":", 1)[1]
        cleaned_roles.append(v)
    role_slugs = [slug_role(r) for r in cleaned_roles]
    for rs in role_slugs:
        weights[rs] = max(weights.get(rs, 0.0), 0.7)
        await _get_or_create_tag(db, rs)
    prof.tag_weights = tw
    pp = dict(prof.privacy_prefs or {})
    pp["onboarding"] = dict(pp.get("onboarding") or {})
    pp["onboarding"]["step"] = "family"
    prof.privacy_prefs = pp
    flag_modified(prof, "privacy_prefs")
    await db.commit()
    return {"ok": True, "next": "family", "roles": role_slugs}


@router.post("/onboarding/family")
async def onboarding_family(
    payload: dict | list = Body(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_authenticated_user),
):
    prof = await _ensure_profile(db, user.id)
    tw = dict(prof.tag_weights or {"tagWeights": {}})
    weights = tw.setdefault("tagWeights", {})
    items = payload if isinstance(payload, list) else payload.get("family") or []
    created = []
    for row in items:
        name = (row.get("name") or "").strip()
        role = (row.get("role") or "").strip()
        if not name:
            continue
        person = await upsert_person_for_user(db, owner_user_id=user.id, display_name=name, role_hint=role or None)
        p_tag = slug_person(name)
        weights[p_tag] = max(weights.get(p_tag, 0.0), 0.9)
        await _get_or_create_tag(db, p_tag)
        if role:
            base_role = role.split(":", 1)[1] if role.startswith(("relationship:", "role:")) else role
            r_tag = slug_role(base_role)
            weights[r_tag] = max(weights.get(r_tag, 0.0), 0.7)
            await _get_or_create_tag(db, r_tag)
        created.append({"person_id": person.id, "name": name, "role": role})
    prof.tag_weights = tw
    pp = dict(prof.privacy_prefs or {})
    pp["onboarding"] = dict(pp.get("onboarding") or {})
    pp["onboarding"]["step"] = "places"
    prof.privacy_prefs = pp
    flag_modified(prof, "privacy_prefs")
    await db.commit()
    return {"ok": True, "next": "places", "created": created}


@router.post("/onboarding/places")
async def onboarding_places(
    payload: dict | list = Body(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_authenticated_user),
):
    prof = await _ensure_profile(db, user.id)
    tw = dict(prof.tag_weights or {"tagWeights": {}})
    weights = tw.setdefault("tagWeights", {})
    items = payload if isinstance(payload, list) else payload.get("places") or []
    place_slugs = []
    gray_add = []
    for p in items:
        s = str(p or "").strip()
        if not s:
            continue
        ps = slug_place(s)
        place_slugs.append(ps)
        weights[ps] = max(weights.get(ps, 0.0), 0.5)
        await _get_or_create_tag(db, ps)
    prof.tag_weights = tw
    pp = dict(prof.privacy_prefs or {})
    pp["onboarding"] = dict(pp.get("onboarding") or {})
    pp["onboarding"]["step"] = "interests"
    prof.privacy_prefs = pp
    flag_modified(prof, "privacy_prefs")
    await db.commit()
    return {"ok": True, "next": "interests", "places": place_slugs}


@router.post("/onboarding/interests")
async def onboarding_interests(
    payload: dict | list = Body(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(require_authenticated_user),
):
    prof = await _ensure_profile(db, user.id)
    tw = dict(prof.tag_weights or {"tagWeights": {}})
    weights = tw.setdefault("tagWeights", {})
    items = payload if isinstance(payload, list) else payload.get("interests") or []
    interest_slugs = []
    gray_add = []
    for raw in items:
        s = (raw.get("value") if isinstance(raw, dict) else str(raw or "")).strip()
        if not s:
            continue
        ts = s if ":" in s else slugify(s)
        interest_slugs.append(ts)
        weights[ts] = max(weights.get(ts, 0.0), 0.6)
        await _get_or_create_tag(db, ts)
    prof.tag_weights = tw
    prof.interests = interest_slugs or None
    pp = dict(prof.privacy_prefs or {})
    pp["onboarding"] = dict(pp.get("onboarding") or {})
    pp["onboarding"]["step"] = "preview"
    prof.privacy_prefs = pp
    flag_modified(prof, "privacy_prefs")
    await db.commit()
    return {"ok": True, "next": "preview", "interests": interest_slugs}


@router.post("/onboarding/commit")
async def onboarding_commit(
    db: AsyncSession = Depends(get_db),
    user=Depends(require_authenticated_user),
):
    prof = await _ensure_profile(db, user.id)
    try:
        # Find existing self-person (check privacy_prefs first, then meta scan)
        self_pid: int | None = (prof.privacy_prefs or {}).get("self_person_id")
        self_person = None
        if self_pid:
            self_person = await db.get(Person, self_pid)
        if self_person is None:
            me_person = (await db.execute(select(Person).where(Person.owner_user_id == user.id))).scalars().all()
            for p in me_person:
                m = getattr(p, "meta", None) or {}
                if isinstance(m, dict) and m.get("connect_to_owner") and str(m.get("role_hint", "")).strip().lower() in {"you", "self", "me"}:
                    self_person = p
                    break
        if self_person is None:
            disp = getattr(prof, "display_name", None) or (user.username or user.email or f"User {user.id}")
            g = None
            try:
                g = ((prof.privacy_prefs or {}).get("user_meta") or {}).get("gender")
            except Exception:
                pass
            meta = {"connect_to_owner": True, "role_hint": "you"}
            if g:
                meta["gender"] = g
            # Self-persons stay private (group_id=NULL) so they're never deduplicated
            self_person = Person(owner_user_id=user.id, display_name=disp, meta=meta)
            try:
                if getattr(prof, "birth_year", None):
                    self_person.birth_year = int(prof.birth_year)
            except Exception:
                pass
            db.add(self_person)
            try:
                await db.flush()
            except Exception:
                await db.rollback()
        # Pin self_person_id in privacy_prefs for fast, reliable lookup
        if self_person and self_person.id:
            prefs = prof.privacy_prefs or {}
            if prefs.get("self_person_id") != self_person.id:
                prefs["self_person_id"] = self_person.id
                prof.privacy_prefs = prefs
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(prof, "privacy_prefs")
    except Exception:
        pass
    await build_pool_for_user(db, user.id)
    await ensure_weekly_prompt(db, user.id)
    ob = prof.privacy_prefs.setdefault("onboarding", {"step": "you", "done": False})
    ob["done"] = True
    ob["step"] = "done"
    await db.commit()
    return RedirectResponse(url="/user_dashboard", status_code=303)
