"""Onboarding flow routes."""
from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.database import get_db
from app.models import KinGroup, KinMembership, Person, RelationshipEdge, UserProfile
from app.routes_shared import _get_or_create_tag, templates
from app.services.assignment import build_pool_for_user, ensure_weekly_prompt
from app.services.auto_tag import WHITELIST
from app.services.people import upsert_person_for_user
from app.utils import require_authenticated_html_user, require_authenticated_user, slug_person, slug_place, slug_role, slugify

router = APIRouter()


# ---------------------------------------------------------------------------
# Role → gender inference
# ---------------------------------------------------------------------------

_ROLE_TO_GENDER: dict[str, str] = {
    "mother": "female", "mom": "female", "grandmother": "female", "grandma": "female",
    "wife": "female", "sister": "female", "daughter": "female", "aunt": "female",
    "niece": "female", "stepmother": "female", "stepmom": "female",
    "mother-in-law": "female", "sister-in-law": "female",
    "great-grandmother": "female",
    "father": "male", "dad": "male", "grandfather": "male", "grandpa": "male",
    "husband": "male", "brother": "male", "son": "male", "uncle": "male",
    "nephew": "male", "stepfather": "male", "stepdad": "male",
    "father-in-law": "male", "brother-in-law": "male",
    "great-grandfather": "male",
}


# ---------------------------------------------------------------------------
# Role → RelationshipEdge definition
#
# Tuple: (rel_type, src_token, dst_token)
# Tokens are "self" (self-person) or "other" (the family member).
# Edges are designed to produce correct results via _SRC_EDGE_TO_ROLE /
# _DST_EDGE_TO_ROLE in services/people.py build_narrator_graph().
# ---------------------------------------------------------------------------

_ROLE_TO_EDGE_DEF: dict[str, tuple[str, str, str]] = {
    # Parents — other→self with typed rel_type for gender resolution
    "mother":            ("mother-of",       "other", "self"),
    "mom":               ("mother-of",       "other", "self"),
    "father":            ("father-of",       "other", "self"),
    "dad":               ("father-of",       "other", "self"),
    "parent":            ("parent-of",       "other", "self"),
    "stepmother":        ("step-parent-of",  "other", "self"),
    "stepmom":           ("step-parent-of",  "other", "self"),
    "stepfather":        ("step-parent-of",  "other", "self"),
    "stepdad":           ("step-parent-of",  "other", "self"),
    "mother-in-law":     ("parent-of",       "other", "self"),
    "father-in-law":     ("parent-of",       "other", "self"),
    # Grandparents — self-[grandchild-of]→other so _SRC["grandchild-of"]="grandparent" fires
    "grandmother":       ("grandchild-of",   "self",  "other"),
    "grandma":           ("grandchild-of",   "self",  "other"),
    "grandfather":       ("grandchild-of",   "self",  "other"),
    "grandpa":           ("grandchild-of",   "self",  "other"),
    "grandparent":       ("grandchild-of",   "self",  "other"),
    "great-grandmother": ("grandchild-of",   "self",  "other"),
    "great-grandfather": ("grandchild-of",   "self",  "other"),
    # Aunts / Uncles — self-[niece-of/nephew-of]→other so _SRC fires "aunt or uncle"
    "aunt":              ("niece-of",        "self",  "other"),
    "uncle":             ("nephew-of",       "self",  "other"),
    # Children — other→self
    "son":               ("son-of",          "other", "self"),
    "daughter":          ("daughter-of",     "other", "self"),
    "child":             ("child-of",        "other", "self"),
    # Siblings — other→self so _DST["sibling-of"]="sibling" fires
    "sibling":           ("sibling-of",      "other", "self"),
    "brother":           ("sibling-of",      "other", "self"),
    "sister":            ("sibling-of",      "other", "self"),
    "half-brother":      ("half-sibling-of", "self",  "other"),
    "half-sister":       ("half-sibling-of", "self",  "other"),
    "stepbrother":       ("step-sibling-of", "self",  "other"),
    "stepsister":        ("step-sibling-of", "self",  "other"),
    # Spouses / Partners
    "spouse":            ("spouse-of",       "other", "self"),
    "wife":              ("spouse-of",       "other", "self"),
    "husband":           ("spouse-of",       "other", "self"),
    "partner":           ("partner-of",      "other", "self"),
    # Extended / Social
    "cousin":            ("cousin-of",       "self",  "other"),
    "friend":            ("friend-of",       "self",  "other"),
    "neighbor":          ("friend-of",       "self",  "other"),
}


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


def _clean_role(raw: str) -> str:
    """Strip namespace prefixes and lowercase a role string."""
    r = (raw or "").strip()
    if ":" in r:
        r = r.split(":", 1)[1]
    return r.lower().strip()


async def _create_family_edges(
    db: AsyncSession,
    self_pid: int,
    user_id: int,
    family_members: list[dict],
) -> int:
    """
    Create RelationshipEdge records connecting the self-person to each family member.
    Edges are user-scoped (user_id set, group_id NULL) so they're picked up by both
    build_narrator_graph() and the people-graph infer scope filter.
    Returns the number of new edges created.
    """
    count = 0
    for fm in family_members:
        pid = fm.get("person_id")
        role = _clean_role(fm.get("role") or "")
        if not pid or not role or pid == self_pid:
            continue
        edge_def = _ROLE_TO_EDGE_DEF.get(role)
        if not edge_def:
            continue
        rel_type, src_tok, dst_tok = edge_def
        src_id = self_pid if src_tok == "self" else pid
        dst_id = pid    if src_tok == "self" else self_pid

        exists = await db.scalar(
            select(RelationshipEdge.id).where(
                RelationshipEdge.src_id == src_id,
                RelationshipEdge.dst_id == dst_id,
                RelationshipEdge.rel_type == rel_type,
                RelationshipEdge.user_id == user_id,
            ).limit(1)
        )
        if not exists:
            db.add(RelationshipEdge(
                src_id=src_id,
                dst_id=dst_id,
                rel_type=rel_type,
                user_id=user_id,
                confidence=1.0,
            ))
            count += 1
    if count:
        await db.flush()
    return count


async def _build_preview_data(db: AsyncSession, user_id: int, prof: UserProfile) -> dict:
    """Build context dict for the preview step template."""
    ob = (prof.privacy_prefs or {}).get("onboarding", {})
    family_raw: list[dict] = ob.get("family_members") or []

    # KinGroup membership
    kin_group_id = await db.scalar(
        select(KinMembership.group_id).where(KinMembership.user_id == user_id).limit(1)
    )
    group_name: str | None = None
    if kin_group_id:
        g = await db.get(KinGroup, kin_group_id)
        group_name = g.name if g else None

    # Load person records for each onboarding family member
    preview_people: list[dict] = []
    for fm in family_raw:
        pid = fm.get("person_id")
        if not pid:
            continue
        p = await db.get(Person, pid)
        if not p:
            continue
        role = _clean_role(fm.get("role") or (p.meta or {}).get("role_hint") or "")
        preview_people.append({
            "name": p.display_name,
            "role": role,
            "in_group": p.group_id is not None,
        })

    # Role labels from tag_weights
    tw = (prof.tag_weights or {}).get("tagWeights", {})
    role_labels = sorted({
        k.split(":", 1)[1].replace("-", " ").title()
        for k, v in tw.items()
        if k.startswith("role:") and float(v or 0) >= 0.5
    })

    return {
        "people": preview_people,
        "roles": role_labels,
        "group_name": group_name,
        "in_group": bool(kin_group_id),
    }


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
    ob = (prof.privacy_prefs or {}).get("onboarding") or {}
    step = ob.get("step") or "you"
    if ob.get("done"):
        return RedirectResponse(url="/onboarding/done", status_code=303)
    role_options = _role_options_from_whitelist()

    preview_data: dict = {}
    if step == "preview":
        preview_data = await _build_preview_data(db, user.id, prof)

    return templates.TemplateResponse(
        request,
        "onboarding_steps.html",
        {
            "request": request,
            "user": user,
            "step": step,
            "role_options": role_options,
            "profile": prof,
            "preview_data": preview_data,
        },
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

        # Assign gender from role when not already known
        if not person.gender:
            role_key = _clean_role(role)
            inferred_gender = _ROLE_TO_GENDER.get(role_key)
            if inferred_gender:
                person.gender = inferred_gender

        p_tag = slug_person(name)
        weights[p_tag] = max(weights.get(p_tag, 0.0), 0.9)
        await _get_or_create_tag(db, p_tag)
        if role:
            base_role = role.split(":", 1)[1] if role.startswith(("relationship:", "role:")) else role
            r_tag = slug_role(base_role)
            weights[r_tag] = max(weights.get(r_tag, 0.0), 0.7)
            await _get_or_create_tag(db, r_tag)
        created.append({"person_id": person.id, "name": name, "role": _clean_role(role)})

    prof.tag_weights = tw
    pp = dict(prof.privacy_prefs or {})
    ob = dict(pp.get("onboarding") or {})
    ob["step"] = "places"
    # Replace (not append) family_members so re-submissions stay idempotent
    ob["family_members"] = created
    pp["onboarding"] = ob
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
        # Find or create the self-person
        self_pid: int | None = (prof.privacy_prefs or {}).get("self_person_id")
        self_person = None
        if self_pid:
            self_person = await db.get(Person, self_pid)
        if self_person is None:
            me_persons = (await db.execute(select(Person).where(Person.owner_user_id == user.id))).scalars().all()
            for p in me_persons:
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

        # Pin self_person_id
        if self_person and self_person.id:
            prefs = prof.privacy_prefs or {}
            if prefs.get("self_person_id") != self_person.id:
                prefs["self_person_id"] = self_person.id
                prof.privacy_prefs = prefs
                flag_modified(prof, "privacy_prefs")

        # Wire RelationshipEdges from self-person to each onboarding family member
        if self_person and self_person.id:
            family_members = (prof.privacy_prefs or {}).get("onboarding", {}).get("family_members") or []
            if family_members:
                await _create_family_edges(db, self_person.id, user.id, family_members)

    except Exception:
        pass

    await build_pool_for_user(db, user.id)
    await ensure_weekly_prompt(db, user.id)
    ob = prof.privacy_prefs.setdefault("onboarding", {"step": "you", "done": False})
    ob["done"] = True
    ob["step"] = "done"
    await db.commit()
    return RedirectResponse(url="/user_dashboard", status_code=303)
