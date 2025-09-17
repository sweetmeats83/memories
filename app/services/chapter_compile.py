# app/services/chapter_compile.py
from __future__ import annotations
import json, math, asyncio
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from app.models import (
    Prompt, Response, UserPrompt, ChapterMeta, ChapterCompilation
)
from app.schemas import ChapterCompilationDTO, GapQuestion, UsedBlock, ChapterStatusDTO
from app.llm_client import your_chat_completion, OLLAMA_MODEL

# ---------- helpers ----------

async def _resolve_chapter_key(db: AsyncSession, chapter_id: str | int) -> Tuple[str, str]:
    """
    Accept a numeric ChapterMeta.id or the string key (Prompt.chapter / ChapterMeta.name).
    Returns (chapter_key, display_name).
    """
    # If numeric → lookup meta by id
    name, display = None, None
    try:
        cid = int(chapter_id)
        meta = (await db.execute(select(ChapterMeta).where(ChapterMeta.id == cid))).scalars().first()
        if meta:
            return meta.name, (meta.display_name or meta.name)
    except Exception:
        pass

    # Else treat input as the key
    nm = str(chapter_id).strip()
    meta = (await db.execute(select(ChapterMeta).where(ChapterMeta.name == nm))).scalars().first()
    if meta:
        return meta.name, (meta.display_name or meta.name)
    return nm, nm


async def chapter_status(db: AsyncSession, chapter_id: str | int, user_id: int) -> ChapterStatusDTO:
    chapter_key, display = await _resolve_chapter_key(db, chapter_id)

    # All prompts ASSIGNED to this user in this chapter
    assigned_rows = await db.execute(
        select(UserPrompt.prompt_id).join(Prompt, Prompt.id == UserPrompt.prompt_id)
        .where((UserPrompt.user_id == user_id) & (Prompt.chapter == chapter_key))
    )
    assigned_ids = set(assigned_rows.scalars().all())

    # Completed responses
    resp_rows = await db.execute(
        select(Response.prompt_id).where((Response.user_id == user_id) & (Response.prompt_id.in_(assigned_ids)))
    )
    completed_ids = set(resp_rows.scalars().all())

    missing_ids = list(assigned_ids - completed_ids)

    missing_prompts = []
    if missing_ids:
        prs = await db.execute(select(Prompt).where(Prompt.id.in_(missing_ids)))
        # If Prompt has joined eager loads on collections, require unique() before scalars()
        for p in prs.unique().scalars().all():
            missing_prompts.append({"id": p.id, "text": p.text})

    # latest draft for this user/chapter
    latest = (
        await db.execute(
            select(ChapterCompilation)
            .where((ChapterCompilation.user_id == user_id) & (ChapterCompilation.chapter == chapter_key))
            .order_by(ChapterCompilation.version.desc(), ChapterCompilation.created_at.desc())
            .limit(1)
        )
    ).scalars().first()

    latest_dto = None
    if latest:
        latest_dto = ChapterCompilationDTO(
            id=latest.id,
            user_id=latest.user_id,
            chapter=latest.chapter,
            version=latest.version,
            status=latest.status,
            compiled_markdown=latest.compiled_markdown or "",
            gap_questions=[GapQuestion(**g) for g in (latest.gap_questions or [])],
            used_blocks=[UsedBlock(**b) for b in (latest.used_blocks or [])],
            model_name=latest.model_name,
            token_stats={
                "prompt_tokens": latest.prompt_tokens,
                "completion_tokens": latest.completion_tokens,
                "total_tokens": latest.total_tokens,
            },
            created_at=latest.created_at.isoformat() if getattr(latest, "created_at", None) else None,
            updated_at=latest.updated_at.isoformat() if getattr(latest, "updated_at", None) else None,
        )

    return ChapterStatusDTO(
        chapter=chapter_key,
        display_name=display,
        ready=(len(missing_ids) == 0 and len(assigned_ids) > 0),
        missing_prompts=missing_prompts,
        latest_compilation=latest_dto,
    )


# ---------- compile service ----------

SYSTEM_PROMPT = """You are a compassionate memoir editor.
You will receive a CHAPTER meta block and a list of RESPONSES from one person.
Tasks:
1) Weave the chapter into a compelling, readable narrative in MARKDOWN. Keep the speaker’s authentic voice.
2) You MAY reorder, group, and lightly rewrite for flow and clarity. Do not invent facts.
3) Identify important GAPS or unclear areas. Propose 3–8 concise follow-up questions.
4) Keep a traceable ORDER array mapping which responses/excerpts you used and in what order.

Return STRICT JSON with keys:
{
  "chapter_markdown": "...",
  "gap_questions": [{"question":"...", "why":"...", "tags":["optional","slugs"]}, ...],
  "order": [{"prompt_id": 12, "response_id": 34, "title":"optional","used_excerpt":"optional"}]
}
"""

def _responses_payload(items: list[tuple[Prompt, Response]]) -> list[dict]:
    out = []
    for p, r in items:
        out.append({
            "prompt_id": p.id,
            "prompt_text": p.text,
            "response_id": r.id,
            "response_title": getattr(r, "title", None),
            "response_text": (r.response_text or "").strip(),
        })
    return out

async def compile_chapter(
    db: AsyncSession,
    chapter_id: str | int,
    user_id: int,
    *,
    model: str = "gpt-X"  # compatibility knob; we pass through to DTO but we actually call OLLAMA_MODEL
) -> ChapterCompilationDTO:
    chapter_key, display = await _resolve_chapter_key(db, chapter_id)

    # load meta (guidance signals)
    meta = (await db.execute(select(ChapterMeta).where(ChapterMeta.name == chapter_key))).scalars().first()
    guidance = (meta.llm_guidance or "") if meta else None

    # gather ALL responses for this user in this chapter (only from assigned prompts is fine too)
    rows = await db.execute(
        select(Prompt, Response)
        .join(Response, Response.prompt_id == Prompt.id)
        .options(selectinload(Prompt.tags))
        .where((Prompt.chapter == chapter_key) & (Response.user_id == user_id))
        .order_by(Response.created_at.asc())
    )
    pairs = rows.all()  # list[(Prompt, Response)]

    # build user payload
    user_payload = {
        "chapter": {"key": chapter_key, "display_name": display, "llm_guidance": guidance},
        "style": {
            "tone": "warm, first-person, clear",
            "paragraphing": "short paragraphs with natural breaks",
            "avoid": ["inventing facts", "changing meaning"],
        },
        "responses": _responses_payload(pairs),
    }
    user_str = json.dumps(user_payload, ensure_ascii=False)

    # call LLM via existing helper (returns string; we parse JSON)
    # NOTE: using your existing llm_client -> your_chat_completion with strict JSON return
    # (We still tag the DTO with the model name we *intended*, and also record OLLAMA_MODEL used.)
    raw = await your_chat_completion(system=SYSTEM_PROMPT, user=user_str, response_format="json", temperature=0.2)

    try:
        data = json.loads(raw)
    except Exception:
        # defensive fallback schema
        data = {"chapter_markdown": raw.strip() or "# (empty)", "gap_questions": [], "order": []}

    chapter_md = (data.get("chapter_markdown") or "").strip() or "# (empty)"
    gaps = data.get("gap_questions") or []
    order = data.get("order") or []

    # save as a new version
    latest_ver = (
        await db.execute(
            select(func.max(ChapterCompilation.version))
            .where((ChapterCompilation.user_id == user_id) & (ChapterCompilation.chapter == chapter_key))
        )
    ).scalar() or 0
    new_ver = int(latest_ver) + 1

    row = ChapterCompilation(
        user_id=user_id,
        chapter=chapter_key,
        version=new_ver,
        status="draft",
        compiled_markdown=chapter_md,
        gap_questions=gaps,
        used_blocks=order,
        model_name=OLLAMA_MODEL,     # actual model used underneath
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)

    return ChapterCompilationDTO(
        id=row.id,
        user_id=row.user_id,
        chapter=row.chapter,
        version=row.version,
        status=row.status,
        compiled_markdown=row.compiled_markdown,
        gap_questions=[GapQuestion(**g) for g in (row.gap_questions or [])],
        used_blocks=[UsedBlock(**b) for b in (row.used_blocks or [])],
        model_name=row.model_name,
        token_stats={
            "prompt_tokens": row.prompt_tokens,
            "completion_tokens": row.completion_tokens,
            "total_tokens": row.total_tokens,
        },
        created_at=row.created_at.isoformat() if row.created_at else None,
        updated_at=row.updated_at.isoformat() if row.updated_at else None,
    )
