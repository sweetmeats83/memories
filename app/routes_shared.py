"""
Shared constants and helpers used by routes.py and the extracted routers.
Import from here instead of defining duplicates in each file.
"""
from __future__ import annotations

import os
import re
import logging
from pathlib import Path as FSPath

from fastapi.templating import Jinja2Templates
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .media_pipeline import MediaPipeline, UserBucketsStrategy

logger = logging.getLogger(__name__)

BASE_DIR   = FSPath(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = str(STATIC_DIR / "uploads")

templates = Jinja2Templates(directory="templates")
PIPELINE  = MediaPipeline(static_root=STATIC_DIR, path_strategy=UserBucketsStrategy())


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9:/\-]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def _to_uploads_rel_for_playable(file_path_under_uploads: str | None) -> str | None:
    """Return 'uploads/...' relative path suitable for delete_artifacts()."""
    if not file_path_under_uploads:
        return None
    rel = file_path_under_uploads.strip().lstrip("/").replace("\\", "/")
    return rel if rel.startswith("uploads/") else f"uploads/{rel}"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _text_for_tagging(resp) -> str:
    parts = []
    try:
        if getattr(resp, "prompt", None) and getattr(resp.prompt, "text", None):
            parts.append(resp.prompt.text)
    except Exception:
        pass
    for attr in ("title", "response_text", "transcription"):
        if getattr(resp, attr, None):
            parts.append(getattr(resp, attr))
    return " \n".join(p for p in parts if p and str(p).strip())


def _display_name_or_email(u, profile=None) -> str:
    if profile and getattr(profile, "display_name", None):
        return profile.display_name
    return u.username or u.email or f"User {u.id}"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _max_order_index(db: AsyncSession, response_id: int) -> int:
    from .models import ResponseSegment
    res = await db.execute(
        select(func.max(ResponseSegment.order_index)).where(
            ResponseSegment.response_id == response_id
        )
    )
    return int(res.scalar_one_or_none() or 0)


async def _get_or_create_tag(db: AsyncSession, name: str):
    from .models import Tag
    nm = (name or "").strip()
    if not nm:
        return None
    slug = _slugify(nm)
    existing = (
        await db.execute(
            select(Tag).where(or_(Tag.slug == slug, func.lower(Tag.name) == name.strip().lower()))
        )
    ).scalar_one_or_none()
    if existing:
        return existing
    t = Tag(name=nm, slug=slug)
    db.add(t)
    try:
        await db.flush()
        return t
    except IntegrityError:
        await db.rollback()
        return (
            await db.execute(
                select(Tag).where(or_(Tag.slug == slug, func.lower(Tag.name) == nm.lower()))
            )
        ).scalar_one_or_none()


# ---------------------------------------------------------------------------
# Background task: transcribe a segment
# ---------------------------------------------------------------------------

async def transcribe_segment_and_update(segment_id: int, uploads_rel_path: str) -> None:
    from .database import async_session_maker
    from .models import Response, ResponseSegment
    from .transcription import transcribe_file

    try:
        rel = uploads_rel_path
        if rel.startswith("uploads/"):
            rel = rel[len("uploads/"):]
        async with async_session_maker() as s:
            uid = None
            try:
                seg = await s.get(ResponseSegment, segment_id)
                if seg:
                    resp = await s.get(Response, getattr(seg, "response_id", None))
                    uid = getattr(resp, "user_id", None)
            except Exception:
                uid = None
            text = await transcribe_file(rel, db=s, user_id=uid)
            seg = await s.get(ResponseSegment, segment_id)
            if seg:
                seg.transcript = text or ""
                await s.commit()
    except Exception:
        try:
            from .database import async_session_maker as _asm
            async with _asm() as s:
                seg = await s.get(ResponseSegment, segment_id)
                if seg and not seg.transcript:
                    seg.transcript = "[Transcription failed]"
                    await s.commit()
        except Exception:
            pass


__all__ = [
    "BASE_DIR", "STATIC_DIR", "UPLOAD_DIR",
    "templates", "PIPELINE",
    "_slugify", "_to_uploads_rel_for_playable", "_text_for_tagging",
    "_display_name_or_email", "_max_order_index", "_get_or_create_tag",
    "transcribe_segment_and_update",
]
