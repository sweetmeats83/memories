# services/assignment.py
from __future__ import annotations
import hashlib, random
from datetime import date, datetime
from sqlalchemy import select, update, func, insert, case, and_, not_ as NOT_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.models import (
    Prompt, Tag, PromptSuggestion, UserProfile, Response,
    UserWeeklyPrompt, UserWeeklySkip, UserPrompt, User
)
from app.services.assignment_core import score_prompt  # used by tag/bucket scoring



# ---------------------------------------------
# ISO week helper
# ---------------------------------------------
def _iso_year_week(d: date | None = None) -> tuple[int, int]:
    d = d or date.today()
    iso = d.isocalendar()
    return int(iso[0]), int(iso[1])

def _seed_for_week(user_id: int, year: int, week: int) -> int:
    msg = f"{user_id}:{year}:{week}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")

def _shuffle_deterministic(items, seed: int):
    rnd = random.Random(seed)
    items = list(items)
    rnd.shuffle(items)
    return items
# ---------------------------------------------
# Unanswered / fallback candidates
# ---------------------------------------------
async def _candidate_prompt_ids(
    db: AsyncSession,
    user_id: int,
    include_skipped_if_none_left: bool = True,
) -> list[int]:
    # Unanswered first
    unanswered_stmt = (
        select(Prompt.id)
        .where(~Prompt.id.in_(select(Response.prompt_id).where(Response.user_id == user_id)))
        .order_by(Prompt.created_at.desc())
    )
    ids = [row[0] for row in (await db.execute(unanswered_stmt)).all()]
    if ids or not include_skipped_if_none_left:
        return ids

    # Fall back to skipped-this-week if nothing else left
    y, w = _iso_year_week()
    skipped_stmt = (
        select(UserWeeklySkip.prompt_id)
        .where(UserWeeklySkip.user_id == user_id, UserWeeklySkip.year == y, UserWeeklySkip.week == w)
        .order_by(UserWeeklySkip.created_at.desc())
    )
    return [row[0] for row in (await db.execute(skipped_stmt)).all()]

# ---------------------------------------------
# Weekly-prompt selection/rotation API
# ---------------------------------------------



async def ensure_weekly_prompt(db: AsyncSession, user_id: int):
    y, w, _ = date.today().isocalendar()

    # 0) ensure weekly row exists
    upsert = (
        pg_insert(UserWeeklyPrompt.__table__)
        .values(user_id=user_id, year=y, week=w, prompt_id=None, status="active")
        .on_conflict_do_nothing(index_elements=["user_id", "year", "week"])
        .returning(UserWeeklyPrompt.id)
    )
    res = await db.execute(upsert)
    new_id = res.scalar_one_or_none()
    if new_id is not None:
        weekly = await db.get(UserWeeklyPrompt, new_id)
    else:
        weekly = (
            await db.execute(
                select(UserWeeklyPrompt).where(
                    (UserWeeklyPrompt.user_id == user_id)
                    & (UserWeeklyPrompt.year == y)
                    & (UserWeeklyPrompt.week == w)
                )
            )
        ).scalars().first()

    # 1) build eligibility sets
    answered_ids = set(
        (await db.execute(
            select(Response.prompt_id).where(Response.user_id == user_id)
        )).scalars().all()
    )
    # hard exclude prompts skipped THIS ISO week (so "skip" always moves forward)
    skipped_ids = set(
        (await db.execute(
            select(UserWeeklySkip.prompt_id).where(
                (UserWeeklySkip.user_id == user_id)
                & (UserWeeklySkip.year == y)
                & (UserWeeklySkip.week == w)
            )
        )).scalars().all()
    )
    if weekly.prompt_id and weekly.prompt_id in answered_ids:
        weekly.status = "answered"
        weekly.prompt_id = None
        await db.flush()
    # 2) gather pool (queued or skipped) minus answered
    preds = [UserPrompt.user_id == user_id, UserPrompt.status.in_(("queued", "skipped"))]
    if answered_ids:
        preds.append(NOT_(UserPrompt.prompt_id.in_(answered_ids)))

    pool_rows = (await db.execute(select(UserPrompt).where(*preds))).unique().scalars().all()
    if not pool_rows:
        # nothing to pick; mirror cleared state and return
        user = await db.get(User, user_id)
        if user:
            user.weekly_current_prompt_id = None
            user.weekly_on_deck_prompt_id = None
        await db.commit()
        return weekly

    # 3) derive one deterministic weekly order: queued first, then skipped; both randomized
    queued_ids  = [r.prompt_id for r in pool_rows if r.status == "queued" and r.prompt_id not in skipped_ids]
    skipped_qids = [r.prompt_id for r in pool_rows if r.status == "skipped" and r.prompt_id not in skipped_ids]

    seed = _seed_for_week(user_id, y, w)
    q_order = _shuffle_deterministic(queued_ids,  seed ^ 0xA5A5A5A5)
    s_order = _shuffle_deterministic(skipped_qids, seed ^ 0x5A5A5A5A)
    ordered_ids = q_order + s_order

    # If excluding weekly-skips leaves nothing, allow re-use of skipped items (but still exclude answered)
    if not ordered_ids:
        q_fallback = [r.prompt_id for r in pool_rows if r.status == "queued"]
        s_fallback = [r.prompt_id for r in pool_rows if r.status == "skipped"]
        q_order = _shuffle_deterministic(q_fallback,  seed ^ 0x11111111)
        s_order = _shuffle_deterministic(s_fallback, seed ^ 0x22222222)
        ordered_ids = q_order + s_order

    # 4) pin current if none pinned, using the FIRST of this weekly order
    if not weekly.prompt_id and ordered_ids:
        chosen_id = ordered_ids[0]
        # activate the corresponding pool row
        for r in pool_rows:
            if r.prompt_id == chosen_id:
                r.status = "active"
                r.times_sent = (r.times_sent or 0) + 1
                break
        weekly.prompt_id = chosen_id
        weekly.status = "active"
        await db.flush()

    # 5) compute "on deck" as next after current in the SAME order
    user = await db.get(User, user_id)
    next_id = None
    if weekly.prompt_id and weekly.prompt_id in ordered_ids:
        idx = ordered_ids.index(weekly.prompt_id)
        next_id = ordered_ids[idx + 1] if idx + 1 < len(ordered_ids) else None
    elif ordered_ids:
        next_id = ordered_ids[0]

    # 6) mirror to User for scheduler/admin alignment
    if user:
        user.weekly_current_prompt_id = weekly.prompt_id or None
        user.weekly_on_deck_prompt_id = next_id or None
        await db.flush()

    await db.commit()
    return weekly


async def get_on_deck_candidates(db: AsyncSession, user_id: int, k: int = 5) -> list[int]:
    y, w, _ = date.today().isocalendar()

    # answered
    answered_ids = set(
        (await db.execute(select(Response.prompt_id).where(Response.user_id == user_id)))
        .scalars().all()
    )

    # pool
    pool_rows = (await db.execute(
        select(UserPrompt).where(
            (UserPrompt.user_id == user_id) &
            (UserPrompt.status.in_(("queued", "skipped")))
        )
    )).unique().scalars().all()
    pool_ids = [r.prompt_id for r in pool_rows if r.prompt_id not in answered_ids]
    if not pool_ids:
        return []

    seed = _seed_for_week(user_id, y, w)
    ordered_ids = _shuffle_deterministic(pool_ids, seed)

    # if a current is pinned, drop it from the preview
    weekly = (await db.execute(
        select(UserWeeklyPrompt).where(
            (UserWeeklyPrompt.user_id == user_id) &
            (UserWeeklyPrompt.year == y) &
            (UserWeeklyPrompt.week == w)
        )
    )).scalars().first()
    if weekly and weekly.prompt_id in ordered_ids:
        ordered_ids = [pid for pid in ordered_ids if pid != weekly.prompt_id]

    return ordered_ids[:max(0, int(k))]



def _user_role_like_slugs(profile: UserProfile) -> set[str]:
    tw = (profile.tag_weights or {}).get("tagWeights", {}) if profile else {}
    # consider weight threshold if you like; 0.3 is generous
    return {k for k,v in tw.items() if k.startswith(("role:", "life:")) and float(v or 0) >= 0.3}

def _prompt_for_slugs(p: Prompt) -> set[str]:
    return {t.slug for t in (p.tags or []) if t.slug.startswith("for:")}

def _prompt_exclude_slugs(p: Prompt) -> set[str]:
    return {t.slug for t in (p.tags or []) if t.slug.startswith("exclude:")}

def _eligible(prompt, user_tag_weights: dict[str, float], *, user_id: int | None = None) -> bool:
    slugs = {t.slug for t in (prompt.tags or [])}
    gates = {s for s in slugs if s.startswith("for:")}

    if not gates or "for:all" in gates:
        return True

    # User profile tags
    keys = set((user_tag_weights or {}).keys())
    role_keys   = {k.split(":", 1)[1] for k in keys if k.startswith("role:")}
    person_keys = {k.split(":", 1)[1] for k in keys if k.startswith("person:")}

    aliases = set()
    rk = {r.lower() for r in role_keys}

    # spouse
    if {"husband", "wife", "spouse", "partner"} & rk:
        aliases.add("spouse")

    # grandparent
    if {"grandmother", "grandfather", "grandparent"} & rk:
        aliases.add("grandparent")

    # sibling-in-law normalization example (extend as needed)
    if {"brother-in-law", "sister-in-law", "sibling-in-law"} & rk:
        aliases.add("sibling-in-law")

    # gender (lets prompts use for:male / for:female)
    if "male" in rk: aliases.add("male")
    if "female" in rk: aliases.add("female")

    # college
    if {"college-student", "college", "university"} & rk:
        aliases.add("college-student")

    role_keys_norm = rk | aliases
    
    # Role-gated: for:<role>
    simple_roles = {g[4:] for g in gates if ":" not in g[4:]}  # e.g., "for:grandmother" -> "grandmother"
    if simple_roles & role_keys:
        return True

    # Person-gated: for:person:<slug>
    for_person = {g.split(":", 2)[2] for g in gates if g.startswith("for:person:")}
    if for_person & person_keys:
        return True

    # User-gated: for:user:<id-or-slug>
    if user_id is not None:
        for_user = {g.split(":", 2)[2] for g in gates if g.startswith("for:user:")}
        if str(user_id) in for_user:
            return True

    return False


# helper: safely fetch the user's tagWeights (lowercased keys)
async def _get_profile_weights(db: AsyncSession, user_id: int) -> dict[str, float]:
    prof = (
        await db.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
    ).scalars().first()
    if not prof:
        return {}
    tw = (prof.tag_weights or {}).get("tagWeights", {}) or {}
    weights = {str(k).lower(): v for k, v in tw.items()} if isinstance(tw, dict) else {}

    # Inject gender preference (from onboarding/settings) as role:* for gating
    try:
        g = ((prof.privacy_prefs or {}).get("user_meta") or {}).get("gender")
        g = (str(g).strip().lower() or None) if g is not None else None
        if g in {"male", "man"}:
            # Treat as role:male to satisfy for:male prompts
            weights.setdefault("role:male", 0.8)
        elif g in {"female", "woman"}:
            # Treat as role:female to satisfy for:female prompts
            weights.setdefault("role:female", 0.8)
    except Exception:
        pass

    return weights

async def build_pool_for_user(db: AsyncSession, user_id: int) -> int:
    # 1) Load the user's weights once (includes gender alias as role:male/female)
    weights = await _get_profile_weights(db, user_id)

    # 2) Load all prompts with tags (single query)
    rows = await db.execute(
        select(Prompt).options(selectinload(Prompt.tags))
    )
    prompts = rows.scalars().unique().all()

    # 3) Use shared gating logic (_eligible) which supports for:male/female, for:user, etc.
    eligible_ids = [p.id for p in prompts if _eligible(p, weights, user_id=user_id)]
    if not eligible_ids:
        return 0

    # 4) Bulk upsert into user_prompts
    stmt = pg_insert(UserPrompt.__table__).values([
        {"user_id": user_id, "prompt_id": pid, "status": "queued", "score": 0, "times_sent": 0}
        for pid in eligible_ids
    ]).on_conflict_do_nothing(index_elements=["user_id", "prompt_id"])

    await db.execute(stmt)
    await db.commit()
    return len(eligible_ids)




# Duplicate _iso_year_week removed (defined earlier in module)

async def rotate_to_next_unanswered(db, user_id: int) -> int | None:
    y, w = _iso_year_week()

    # Current weekly pointer
    cur_pid = (await db.execute(
        select(UserWeeklyPrompt.prompt_id).where(
            (UserWeeklyPrompt.user_id == user_id) &
            (UserWeeklyPrompt.year == y) &
            (UserWeeklyPrompt.week == w)
        )
    )).scalar_one_or_none()

    # Prefer the next id from this week's deterministic order
    next_ids = await get_on_deck_candidates(db, user_id, k=1)
    next_id = next_ids[0] if next_ids else None

    if next_id is None:
        # No more on-deck (maybe pool changed) -> recompute/ensure weekly current
        wrec = await ensure_weekly_prompt(db, user_id)
        return getattr(wrec, "prompt_id", None)

    # Demote prior active to skipped (cool-down), if it exists and differs
    if cur_pid and cur_pid != next_id:
        await db.execute(
            update(UserPrompt)
            .where((UserPrompt.user_id == user_id) & (UserPrompt.prompt_id == cur_pid))
            .values(status="skipped")
        )

    # Activate the chosen next
    await db.execute(
        update(UserPrompt)
        .where((UserPrompt.user_id == user_id) & (UserPrompt.prompt_id == next_id))
        .values(
            status="active",
            last_sent_at=datetime.utcnow(),
            times_sent=func.coalesce(UserPrompt.times_sent, 0) + 1,
        )
    )

    # Point weekly to new current
    await db.execute(
        update(UserWeeklyPrompt)
        .where(
            (UserWeeklyPrompt.user_id == user_id) &
            (UserWeeklyPrompt.year == y) &
            (UserWeeklyPrompt.week == w)
        )
        .values(prompt_id=next_id, status="active")
    )

    await db.commit()
    return next_id





# ---------------------------------------------
# Tag/bucket helpers + onboarding assignment
# ---------------------------------------------
async def tag_based_prompts(db: AsyncSession, tag_slugs: list[str], limit: int = 10) -> list[Prompt]:
    """Fetch prompts overlapping the given tag slugs."""
    if not tag_slugs:
        return []
    q = (
        select(Prompt)
        .join(Prompt.tags)
        .where(Tag.slug.in_(tag_slugs))
        .limit(limit)
    )
    return list((await db.execute(q)).scalars().unique().all())

async def persist_suggestions(db: AsyncSession, user_id: int, suggestions: list[dict], source: str):
    """Persist AI-curated suggestions (unmapped to Prompt for now)."""
    rows = []
    for s in suggestions:
        rows.append(PromptSuggestion(
            user_id=user_id,
            prompt_id=None,
            source=source,
            title=s.get("title"),
            text=s["text"],
            tags=s.get("tags"),
            status="pending",
            rationale_json={"rationale": s.get("rationale")},
        ))
    db.add_all(rows)
    await db.commit()
    return rows

async def assign_for_onboarding(db: AsyncSession, user_id: int, limit: int = 10) -> list[UserWeeklyPrompt]:
    """
    After onboarding, pick a *set* of prompts tailored to the user by tag buckets and score,
    then seed this week’s UserWeeklyPrompt rows to prime the pump.
    """
    prof = (await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))).scalars().first()
    if not prof:
        return []

    weights = (prof.tag_weights or {}).get("tagWeights", {})
    recent: list[str] = []
    targets: dict[str, float] = {}

    # answered prompts to exclude
    answered_ids = set(
        row[0]
        for row in (await db.execute(select(Response.prompt_id).where(Response.user_id == user_id))).all()
        if row[0] is not None
    )

    # load all prompts + tags
    prompts = list(
        (await db.execute(
            select(Prompt).options(selectinload(Prompt.tags))
        )).scalars().unique().all()
    )

    # buckets by tag namespace
    person_bucket, role_bucket, general_bucket, place_bucket = [], [], [], []
    for p in prompts:
        if p.id in answered_ids:
            continue
        slugs = {t.slug for t in (p.tags or [])}
        is_person = any(s.startswith("person:") for s in slugs)
        is_role   = any(s.startswith("role:")   for s in slugs)
        is_place  = any(s.startswith("place:")  for s in slugs)
        if is_person:
            person_bucket.append(p)
        elif is_role:
            role_bucket.append(p)
        elif not is_person and not is_role and not is_place:
            general_bucket.append(p)
        else:
            place_bucket.append(p)

    def _score(p: Prompt) -> float:
        return score_prompt(p, weights, recent, targets)

    person_bucket.sort(key=_score, reverse=True)
    role_bucket.sort(key=_score, reverse=True)
    general_bucket.sort(key=_score, reverse=True)
    place_bucket.sort(key=_score, reverse=True)

    ordered = person_bucket + role_bucket + general_bucket + place_bucket
    picks = ordered[:max(1, int(limit))]

    # seed rows for this ISO week (use upsert to avoid duplicates)
    y, w, _ = datetime.utcnow().isocalendar()
    rows: list[UserWeeklyPrompt] = []
    for p in picks:
        ins = pg_insert(UserWeeklyPrompt.__table__).values(
            user_id=user_id, year=int(y), week=int(w), prompt_id=p.id, status="active"
        ).on_conflict_do_nothing(index_elements=["user_id", "year", "week", "prompt_id"])
        await db.execute(ins)
        rows.append(UserWeeklyPrompt(
            user_id=user_id, year=int(y), week=int(w), prompt_id=p.id, status="active"
        ))
    await db.commit()
    return rows

async def skip_current_prompt(db: AsyncSession, user_id: int) -> int | None:
    y, w = _iso_year_week()

    # Ensure weekly row exists / load it
    weekly = (await db.execute(
        select(UserWeeklyPrompt).where(
            (UserWeeklyPrompt.user_id == user_id) &
            (UserWeeklyPrompt.year == y) &
            (UserWeeklyPrompt.week == w)
        )
    )).scalars().first()
    if not weekly:
        weekly = await ensure_weekly_prompt(db, user_id)

    # If nothing pinned yet, try to pin one and return it
    if not weekly or not weekly.prompt_id:
        weekly = await ensure_weekly_prompt(db, user_id)
        return weekly.prompt_id if weekly and weekly.prompt_id else None

    pid = weekly.prompt_id

    # 1) Record the skip this week (idempotent)
    await db.execute(
        pg_insert(UserWeeklySkip.__table__)
        .values(user_id=user_id, year=y, week=w, prompt_id=pid)
        .on_conflict_do_nothing(index_elements=["user_id", "year", "week", "prompt_id"])
    )

    # 2) Demote the active pool row → skipped
    up = (await db.execute(
        select(UserPrompt).where(
            (UserPrompt.user_id == user_id) & (UserPrompt.prompt_id == pid)
        )
    )).scalars().first()
    if up:
        up.status = "skipped"
        up.last_sent_at = datetime.utcnow()

    # 3) Clear the weekly pin and mirror to User
    weekly.prompt_id = None
    u = await db.get(User, user_id)
    if u:
        u.weekly_current_prompt_id = None
        # keep weekly_on_deck_prompt_id intact for now; ensure_weekly_prompt will recompute
    await db.flush()

    # 4) Re-pin next eligible prompt (excludes this-week’s skipped/answered)
    nxt = await ensure_weekly_prompt(db, user_id)
    return nxt.prompt_id if nxt and nxt.prompt_id else None
