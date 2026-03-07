import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

logger = logging.getLogger(__name__)
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


# ---------------------------------------------------------------------------
# Role keyword scanner
# ---------------------------------------------------------------------------

# Keyword → for:* gate tag(s) to apply
_ROLE_KEYWORD_RULES: list[tuple[list[str], str]] = [
    # grandparent first (more specific than parent)
    (["grandchildren", "grandchild", "grandkids", "grandkid", "grandparent", "grandmother", "grandfather", "grandma", "grandpa", "nana", "pop-pop", "popop", "granny"], "for:grandparent"),
    # parent
    (["children", "child", "kids", "kid", "son", "daughter", "your children", "your kids", "your child", "parenting", "raised", "parent", "mother", "father", "mom", "dad"], "for:parent"),
    # spouse / partner
    (["spouse", "husband", "wife", "partner", "married", "wedding", "engagement", "significant other", "better half", "marriage"], "for:spouse"),
    # sibling
    (["sibling", "brother", "sister", "siblings"], "for:sibling"),
]


def _scan_prompt_for_roles(text: str) -> list[str]:
    """Return the list of for:* slugs that match keywords in *text*."""
    lower = text.lower()
    hits: list[str] = []
    for keywords, slug in _ROLE_KEYWORD_RULES:
        if any(kw in lower for kw in keywords):
            hits.append(slug)
    return hits


class ScanRolesOptions(BaseModel):
    commit: bool = False  # dry-run by default
    overwrite: bool = False  # if True, replace existing for:* tags; otherwise only add


@router.post("/scan-roles")
async def admin_prompts_scan_roles(
    options: ScanRolesOptions,
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    """
    Scan all prompt texts for role-indicating keywords and apply `for:*` gate tags.
    By default this is a dry-run (commit=false). Pass commit=true to persist.
    """
    from sqlalchemy.orm import selectinload as _sl

    rows = (
        await db.execute(select(Prompt).options(_sl(Prompt.tags)))
    ).scalars().unique().all()

    results = []
    changed = 0

    for prompt in rows:
        suggested = _scan_prompt_for_roles(prompt.text or "")
        if not suggested:
            continue

        existing_for = {t.slug for t in (prompt.tags or []) if t.slug.startswith("for:")}

        to_add = [s for s in suggested if s not in existing_for]
        if not to_add and not options.overwrite:
            results.append({
                "id": prompt.id,
                "text": (prompt.text or "")[:80],
                "suggested": suggested,
                "existing_for": list(existing_for),
                "action": "skipped (already tagged)",
            })
            continue

        if options.commit:
            if options.overwrite:
                # Replace existing for:* tags; keep all other tags
                other_tags = [t for t in (prompt.tags or []) if not t.slug.startswith("for:")]
                new_for_tags = []
                for slug in suggested:
                    tag = await _ensure_tag_slug_admin(db, slug)
                    if tag:
                        new_for_tags.append(tag)
                prompt.tags = other_tags + new_for_tags
            else:
                # Only add missing for:* tags
                current_tags = list(prompt.tags or [])
                for slug in to_add:
                    tag = await _ensure_tag_slug_admin(db, slug)
                    if tag and tag not in current_tags:
                        current_tags.append(tag)
                prompt.tags = current_tags

        results.append({
            "id": prompt.id,
            "text": (prompt.text or "")[:80],
            "suggested": suggested,
            "existing_for": list(existing_for),
            "added": to_add if options.commit else [],
            "action": "updated" if options.commit else "would add",
        })
        changed += 1

    if options.commit:
        await db.commit()

    return {
        "ok": True,
        "commit": options.commit,
        "scanned": len(rows),
        "changed": changed,
        "results": results,
    }


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
    try:
        # Use a savepoint so a constraint violation only rolls back this
        # one insert, not the entire import transaction.
        async with db.begin_nested():
            db.add(tag)
            await db.flush()
    except Exception:
        # Tag was inserted by another concurrent request or already exists;
        # fetch the real row.
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
    skipped = 0
    for item in items:
        try:
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
            # Only update tags when the import includes valid prefix:value slugs.
            # Raw tags without a colon (e.g. "general", "parent") are skipped so
            # they don't overwrite properly-formatted tags already in the DB.
            valid_slugs = [s for s in (item.tags or []) if ":" in s]
            if valid_slugs:
                new_tags = []
                for tag_slug in valid_slugs:
                    tag = await _ensure_tag_slug_admin(db, tag_slug)
                    if tag:
                        new_tags.append(tag)
                prompt.tags = new_tags
        except Exception as exc:
            skipped += 1
            logger.warning("Import skipped item %s: %s", getattr(item, "id", "?"), exc)
    await db.commit()
    return {"ok": True, "created": created, "updated": updated, "skipped": skipped, "total": created + updated}
