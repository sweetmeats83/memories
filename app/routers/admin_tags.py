"""Admin routes for tag management, auto-tagging, and prompt assignment."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path as FSPath
from typing import Any, List

from fastapi import APIRouter, Body, Depends, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import noload, selectinload

from app.database import get_db
from app.models import (
    Prompt,
    Response,
    Tag,
    User,
    UserProfile,
    UserPrompt,
    prompt_tags,
)
from app.routes_shared import _display_name_or_email, _get_or_create_tag, _slugify
from app.services.assignment import build_pool_for_user
from app.services.assignment_core import score_prompt
from app.services.auto_tag import (
    WHITELIST,
    _is_prompt_like,
    suggest_tags_for_prompt,
    suggest_tags_rule_based,
)
from app.utils import require_admin_user

logger = logging.getLogger(__name__)

router = APIRouter()

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TAG_WL_PATH = FSPath(_DATA_DIR) / "tag_whitelist.json"


async def suggest_prompt_tags(prompt_text: str, chapter: str | None, k: int, db) -> list[dict]:
    pairs = await suggest_tags_for_prompt(prompt_text, word_count=len(prompt_text.split()))
    return [
        {
            "label": slug.split(":")[-1].replace("-", " ").title(),
            "slug": slug,
            "confidence": round(score, 2),
            "source": "kw",
        }
        for slug, score in pairs[:k]
    ]


# ---------------------------------------------------------------------------
# Tag whitelist admin
# ---------------------------------------------------------------------------

@router.get("/api/admin/tags/whitelist")
async def get_tag_whitelist(admin=Depends(require_admin_user)):
    try:
        data = json.loads(TAG_WL_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = []
    return JSONResponse(data)


@router.post("/api/admin/tags/whitelist")
async def save_tag_whitelist(payload: Any = Body(...), admin=Depends(require_admin_user)):
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON string body")
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="Whitelist must be a JSON array")
    try:
        TAG_WL_PATH.parent.mkdir(parents=True, exist_ok=True)
        TAG_WL_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write whitelist: {e}")
    try:
        from app.services.auto_tag import reload_whitelist as _reload
        _reload()
    except Exception:
        try:
            from app.services.auto_tag import _load_whitelist, WHITELIST as WL
            WL.clear()
            WL.update(_load_whitelist())
        except Exception:
            pass
    return {"ok": True, "count": len(payload)}


# ---------------------------------------------------------------------------
# Tag CRUD
# ---------------------------------------------------------------------------

@router.get("/admin/tags")
async def admin_list_tags(
    q: str | None = None,
    limit: int = 20,
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Tag).order_by(Tag.name)
    if q:
        ql = f"%{q.lower()}%"
        stmt = stmt.where(or_(Tag.name.ilike(ql), Tag.slug.ilike(ql)))
    tags = (await db.execute(stmt.limit(limit))).scalars().all()
    return [{"id": t.id, "name": t.name, "slug": t.slug, "color": t.color} for t in tags]


@router.post("/admin/tags")
async def admin_create_tag(
    name: str = Form(...),
    color: str | None = Form(None),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    name = name.strip()
    slug = _slugify(name)
    existing = (
        await db.execute(select(Tag).where(or_(Tag.slug == slug, Tag.name.ilike(name))))
    ).scalar_one_or_none()
    if existing:
        return {"id": existing.id, "name": existing.name, "slug": existing.slug, "color": existing.color}
    t = Tag(name=name, slug=slug, color=color)
    db.add(t)
    await db.commit()
    await db.refresh(t)
    return {"id": t.id, "name": t.name, "slug": t.slug, "color": t.color}


@router.post("/admin/prompts/{prompt_id}/tags")
async def admin_set_prompt_tags(
    prompt_id: int,
    payload: Any = Body(...),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    tag_names: List[str] = payload if isinstance(payload, list) else payload.get("tags") or []
    tag_names = [(s or "").strip() for s in tag_names if (s or "").strip()]
    tags: list[Tag] = []
    for nm in tag_names:
        t = await _get_or_create_tag(db, nm)
        if t:
            tags.append(t)
    prompt = (
        await db.execute(
            select(Prompt).options(selectinload(Prompt.tags)).where(Prompt.id == prompt_id)
        )
    ).scalars().first()
    if not prompt:
        raise HTTPException(404, "Prompt not found")
    prompt.tags[:] = tags
    await db.commit()
    return {"ok": True, "tags": [{"id": t.id, "name": t.name, "slug": t.slug, "color": t.color} for t in tags]}


@router.post("/admin/tags/import_whitelist")
async def import_tag_whitelist(db: AsyncSession = Depends(get_db)):
    from sqlalchemy.exc import IntegrityError
    default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tag_whitelist.json")
    path = os.getenv("TAG_WHITELIST_PATH", default_path)
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    inserted = updated = skipped = 0
    for it in items:
        slug = (it.get("value") or "").strip().lower()
        name = (it.get("label") or slug.replace("-", " ").title()).strip()
        if not slug:
            continue
        existing = (await db.execute(select(Tag).where(Tag.slug == slug))).scalar_one_or_none()
        if existing:
            if (existing.name or "").strip() != name:
                existing.name = name
                updated += 1
            else:
                skipped += 1
            continue
        db.add(Tag(name=name, slug=slug))
        try:
            await db.flush()
            inserted += 1
        except IntegrityError:
            await db.rollback()
            skipped += 1
    await db.commit()
    return {"ok": True, "path": path, "inserted": inserted, "updated": updated, "skipped": skipped}


# ---------------------------------------------------------------------------
# Auto-tag
# ---------------------------------------------------------------------------

class AutoTagReq(BaseModel):
    text: str
    word_count: int | None = None
    language_code: str | None = None
    mode: str | None = None


class AutoTagResp(BaseModel):
    tags: list[dict]


@router.get("/api/admin/tags/suggest")
async def api_admin_tags_suggest(
    prompt_text: str = Query(..., min_length=3),
    chapter: str | None = Query(None),
    k: int = Query(12, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    items = await suggest_prompt_tags(prompt_text=prompt_text, chapter=chapter, k=k, db=db)
    return {"items": items}


@router.post("/api/auto_tag/preview", response_model=AutoTagResp)
async def auto_tag_preview(payload: AutoTagReq) -> AutoTagResp:
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    is_prompt = (payload.mode or "").lower() == "prompt" or _is_prompt_like(text)
    if is_prompt:
        pairs = await suggest_tags_for_prompt(text, word_count=payload.word_count, language_code=payload.language_code)
    else:
        pairs = suggest_tags_rule_based(text, word_count=payload.word_count, language_code=payload.language_code)
    return AutoTagResp(tags=[{"value": slug, "score": round(score, 2)} for slug, score in pairs])


# ---------------------------------------------------------------------------
# Assignment preview / commit
# ---------------------------------------------------------------------------

class AssignPreviewReq(BaseModel):
    user_id: int
    limit: int = 5
    must: list[str] = []
    nice: list[str] = []


class AssignItem(BaseModel):
    prompt_id: int
    chapter: str | None = None
    score: float
    tags: list[str]
    reasons: list[str] = []


class AssignPreviewResp(BaseModel):
    user_id: int
    candidates: list[AssignItem]


class AssignCommitReq(BaseModel):
    user_id: int
    prompt_ids: List[int]
    period: str = "week"


@router.post("/api/assignments/preview", response_model=AssignPreviewResp)
async def assignments_preview(payload: AssignPreviewReq, db: AsyncSession = Depends(get_db)):
    profile = {"tagWeights": {}, "recentHistory": [], "targets": {}}
    base = select(Prompt).join(prompt_tags).join(Tag)
    if payload.must:
        sub = (
            select(prompt_tags.c.prompt_id)
            .join(Tag, Tag.id == prompt_tags.c.tag_id)
            .where(Tag.slug.in_(payload.must))
            .group_by(prompt_tags.c.prompt_id)
            .having(func.count(func.distinct(Tag.slug)) == len(payload.must))
        )
        base = base.where(Prompt.id.in_(sub))
    if payload.nice:
        base = base.where(
            Prompt.id.in_(select(prompt_tags.c.prompt_id).join(Tag).where(Tag.slug.in_(payload.nice)))
        )
    prompts = (await db.execute(base)).scalars().unique().all()
    scored = sorted(
        ((p, score_prompt(p, profile["tagWeights"], profile["recentHistory"], profile["targets"])) for p in prompts),
        key=lambda x: x[1],
        reverse=True,
    )
    out = [
        {"prompt_id": p.id, "chapter": p.chapter, "score": round(s, 3), "tags": [t.slug for t in p.tags], "reasons": []}
        for p, s in scored[: payload.limit]
    ]
    return {"user_id": payload.user_id, "candidates": out}


@router.post("/api/assignments/commit")
async def assignments_commit(payload: AssignCommitReq, db: AsyncSession = Depends(get_db)):
    return {"ok": True, "locked": {"period": payload.period, "prompt_ids": payload.prompt_ids}}


# ---------------------------------------------------------------------------
# Assignment / pool admin
# ---------------------------------------------------------------------------

@router.get("/api/admin/assignments/by-user")
async def admin_assignments_by_user(
    user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    u = await db.get(User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    prof = (await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))).scalars().first()
    ups = (
        await db.execute(
            select(UserPrompt)
            .options(selectinload(UserPrompt.prompt).selectinload(Prompt.tags))
            .where(UserPrompt.user_id == user_id)
        )
    ).scalars().all()
    resp_rows = (
        await db.execute(select(Response).options(selectinload(Response.tags)).where(Response.user_id == user_id))
    ).scalars().all()
    rmap = {r.prompt_id: r for r in resp_rows}
    items = []
    for up in ups:
        p = up.prompt
        resp = rmap.get(p.id)
        items.append(
            {
                "prompt_id": p.id,
                "chapter": p.chapter,
                "text": (p.text or "")[:200],
                "tags": [t.slug for t in p.tags or []],
                "answered": bool(resp),
                "response_excerpt": (resp.transcription or resp.response_text or "")[:160] if resp else None,
                "response": {"id": resp.id, "transcription": resp.transcription, "tags": [t.slug for t in resp.tags or []]} if resp else None,
            }
        )
    assigned_count = len(ups)
    answered_count = sum(1 for it in items if it["answered"])
    pct = int(round(100.0 * answered_count / assigned_count)) if assigned_count else 0
    return {
        "user": {"id": u.id, "name": (prof.display_name if prof else None) or (u.username or u.email), "email": u.email},
        "answered_pct": pct,
        "items": items,
    }


@router.get("/api/admin/assignments/by-prompt")
async def admin_assignments_by_prompt(
    prompt_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    p = await db.get(Prompt, prompt_id)
    if not p:
        raise HTTPException(404, "Prompt not found")
    ups = (await db.execute(select(UserPrompt.user_id).where(UserPrompt.prompt_id == prompt_id))).scalars().all() or []
    if not ups:
        return {"prompt_id": p.id, "users": [], "answered_pct": 0}
    users = (await db.execute(select(User).where(User.id.in_(ups)))).scalars().all()
    profs = (await db.execute(select(UserProfile).where(UserProfile.user_id.in_(ups)))).scalars().all()
    pmap = {pr.user_id: pr for pr in profs}
    answered_user_ids = set(
        uid for uid, in (
            await db.execute(select(Response.user_id).where(Response.prompt_id == prompt_id, Response.user_id.in_(ups)))
        ).all()
    )
    users_out = [
        {"id": u.id, "name": _display_name_or_email(u, pmap.get(u.id)), "email": u.email, "answered": u.id in answered_user_ids}
        for u in users
    ]
    pct = int(round(100.0 * len(answered_user_ids) / max(1, len(ups))))
    return {"prompt_id": p.id, "users": users_out, "answered_pct": pct}


@router.post("/api/admin/pool/add")
async def admin_pool_add(
    payload: dict = Body(...),
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    user_id = int(payload.get("user_id"))
    prompt_id = int(payload.get("prompt_id"))
    exists = (
        await db.execute(select(UserPrompt).where(UserPrompt.user_id == user_id, UserPrompt.prompt_id == prompt_id))
    ).scalars().first()
    if not exists:
        db.add(UserPrompt(user_id=user_id, prompt_id=prompt_id))
        await db.commit()
    return {"ok": True}


@router.post("/api/admin/pool/remove")
async def admin_pool_remove(
    payload: dict = Body(...),
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    user_id = int(payload.get("user_id"))
    prompt_id = int(payload.get("prompt_id"))
    await db.execute(delete(UserPrompt).where(UserPrompt.user_id == user_id, UserPrompt.prompt_id == prompt_id))
    await db.commit()
    return {"ok": True}


@router.get("/api/admin/pool/stats")
async def admin_pool_stats(
    scope: str = Query(..., pattern="^(prompt|user)$"),
    id: int = Query(...),
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    if scope == "prompt":
        assigned = (await db.execute(select(func.count()).select_from(UserPrompt).where(UserPrompt.prompt_id == id))).scalar() or 0
        ups = (await db.execute(select(UserPrompt.user_id).where(UserPrompt.prompt_id == id))).scalars().all()
        answered = 0
        if ups:
            answered = (
                await db.execute(
                    select(func.count(func.distinct(Response.user_id))).where(
                        Response.prompt_id == id, Response.user_id.in_(ups)
                    )
                )
            ).scalar() or 0
        pct = int(round(100.0 * answered / max(1, assigned)))
        return {"scope": "prompt", "id": id, "assigned_count": assigned, "answered_count": answered, "answered_pct": pct}
    assigned = (await db.execute(select(func.count()).select_from(UserPrompt).where(UserPrompt.user_id == id))).scalar() or 0
    answered = (await db.execute(select(func.count()).select_from(Response).where(Response.user_id == id))).scalar() or 0
    pct = int(round(100.0 * answered / max(1, assigned)))
    return {"scope": "user", "id": id, "assigned_count": assigned, "answered_count": answered, "answered_pct": pct}


@router.get("/admin/users/search")
async def admin_user_search(
    q: str = Query("", min_length=0),
    limit: int = Query(10, ge=1, le=50),
    exclude_prompt_id: int | None = Query(None),
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    q = (q or "").strip().lower()
    like = f"%{q}%"
    stmt = select(User).join(UserProfile, UserProfile.user_id == User.id, isouter=True).limit(limit)
    if q:
        stmt = stmt.where(
            func.lower(User.email).like(like)
            | func.lower(User.username).like(like)
            | func.lower(func.coalesce(UserProfile.display_name, "")).like(like)
        )
    if exclude_prompt_id is not None:
        assigned_uids = (
            await db.execute(select(UserPrompt.user_id).where(UserPrompt.prompt_id == exclude_prompt_id))
        ).scalars().all() or []
        if assigned_uids:
            stmt = stmt.where(~User.id.in_(assigned_uids))
    users = (await db.execute(stmt)).scalars().all()
    profs = (
        await db.execute(select(UserProfile).where(UserProfile.user_id.in_([u.id for u in users])))
    ).scalars().all()
    pmap = {p.user_id: p for p in profs}
    return {"users": [{"id": u.id, "name": _display_name_or_email(u, pmap.get(u.id)), "email": u.email} for u in users]}


@router.get("/api/admin/pool/by-prompt")
async def pool_by_prompt(
    prompt_id: int,
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    res = await db.execute(
        select(User.id, User.username, User.email)
        .join(UserPrompt, UserPrompt.user_id == User.id)
        .where(UserPrompt.prompt_id == prompt_id)
        .options(noload("*"))
    )
    rows = res.all()
    return [{"id": r.id, "name": r.username, "email": r.email} for r in rows]
