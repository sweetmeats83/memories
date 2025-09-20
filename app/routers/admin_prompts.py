from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import get_db
from ..models import Prompt, Tag
from ..utils import require_admin_user

router = APIRouter(prefix="/api/admin/prompts", tags=["admin", "prompts"])


@router.get("/export")
async def admin_prompts_export(
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    rows = (await db.execute(select(Prompt).options(selectinload(Prompt.tags)))).scalars().all()
    by_chapter: dict[str, list] = {}
    total = 0
    for prompt in rows:
        chapter = (prompt.chapter or "").strip() or "general"
        tags: list[str] = []
        try:
            tags = [t.slug for t in prompt.tags or [] if getattr(t, "slug", None)]
        except Exception:
            tags = []
        by_chapter.setdefault(chapter, []).append(
            {
                "id": prompt.id,
                "text": prompt.text,
                "chapter": chapter,
                "tags": tags,
            }
        )
        total += 1
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total_prompts": total,
        "chapters": [{"chapter": key, "prompts": value} for key, value in by_chapter.items()],
    }
    return JSONResponse(payload)


class PromptImportItem(BaseModel):
    id: Optional[int] = None
    text: str
    chapter: Optional[str] = None
    tags: Optional[list[str]] = None


class PromptImportPayload(BaseModel):
    prompts: Optional[list[PromptImportItem]] = None
    chapters: Optional[list[dict]] = None
    patch: Optional[bool] = True


async def _ensure_tag_slug_admin(db: AsyncSession, slug: str) -> Optional[Tag]:
    normalized = (slug or "").strip()
    if not normalized:
        return None
    existing = (await db.execute(select(Tag).where(Tag.slug == normalized))).scalar_one_or_none()
    if existing:
        return existing
    name = normalized.split(":", 1)[-1].replace("-", " ").title()
    tag = Tag(name=name, slug=normalized)
    db.add(tag)
    try:
        await db.flush()
    except Exception:
        await db.rollback()
        tag = (await db.execute(select(Tag).where(Tag.slug == normalized))).scalar_one_or_none()
    return tag


@router.post("/import")
async def admin_prompts_import(
    payload: PromptImportPayload,
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    items: list[PromptImportItem] = []
    if payload.prompts:
        items = payload.prompts
    elif payload.chapters:
        for chapter_blob in payload.chapters:
            chapter_value = (chapter_blob.get("chapter") or chapter_blob.get("name") or "").strip() or None
            for item_blob in chapter_blob.get("prompts") or []:
                if isinstance(item_blob, dict):
                    items.append(
                        PromptImportItem(
                            **{
                                "id": item_blob.get("id"),
                                "text": item_blob.get("text", ""),
                                "chapter": item_blob.get("chapter") or chapter_value,
                                "tags": item_blob.get("tags"),
                            }
                        )
                    )
    if not items:
        raise HTTPException(400, "No prompts provided")
    created = 0
    updated = 0
    for item in items:
        chapter = (item.chapter or "").strip() or "general"
        prompt: Optional[Prompt] = None
        if payload.patch and item.id:
            prompt = await db.get(Prompt, item.id)
        if prompt:
            prompt.text = item.text
            prompt.chapter = chapter
            updated += 1
        else:
            prompt = Prompt(text=item.text, chapter=chapter)
            db.add(prompt)
            await db.flush()
            created += 1
        try:
            if item.tags is not None:
                new_tags = []
                for tag_slug in item.tags:
                    tag = await _ensure_tag_slug_admin(db, tag_slug)
                    if tag:
                        new_tags.append(tag)
                prompt.tags = new_tags
        except Exception:
            pass
    await db.commit()
    return {"ok": True, "created": created, "updated": updated, "total": created + updated}
