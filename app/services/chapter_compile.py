# app/services/chapter_compile.py
from __future__ import annotations
import json, asyncio
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

import re as _re
from html.parser import HTMLParser

from app.models import (
    Prompt, Response, UserPrompt, ChapterMeta, ChapterCompilation
)
from app.schemas import ChapterCompilationDTO, GapQuestion, UsedBlock, ChapterStatusDTO
from app.llm_client import OLLAMA_BASE_URL, OLLAMA_MODEL


def _strip_html(html: str) -> str:
    """Convert HTML (from Quill editor) to plain text."""
    if not html:
        return ""
    # Replace block-level tags with newlines before stripping
    text = _re.sub(r'<br\s*/?>', '\n', html, flags=_re.IGNORECASE)
    text = _re.sub(r'</(p|div|li|h[1-6]|blockquote)>', '\n', text, flags=_re.IGNORECASE)
    # Strip all remaining tags
    text = _re.sub(r'<[^>]+>', '', text)
    # Decode common HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>') \
               .replace('&nbsp;', ' ').replace('&#39;', "'").replace('&quot;', '"')
    # Collapse excessive blank lines
    text = _re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ---------------------------------------------------------------------------
# Low-level LLM helpers (longer timeout than the global llm_client helpers)
# ---------------------------------------------------------------------------

async def _llm_json(system: str, user: str, timeout: int = 180) -> dict:
    """Call Ollama and parse the response as JSON. Returns {} on failure."""
    import httpx
    prompt = f"{system}\n\nReturn ONLY valid JSON, no commentary.\n\nUSER:\n{user}"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.15, "num_ctx": 8192},
        "think": False,  # disable thinking mode (Qwen3/3.5) to prevent <think> bleed in JSON
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            r.raise_for_status()
            raw = (r.json() or {}).get("response", "").strip()
        return json.loads(raw)
    except Exception:
        return {}


async def _llm_prose(system: str, user: str, timeout: int = 300) -> str:
    """Call Ollama for free-form prose. Returns plain text string."""
    import httpx
    prompt = f"{system}\n\nUSER:\n{user}"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3},
        "think": False,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            r.raise_for_status()
            return (r.json() or {}).get("response", "").strip()
    except Exception as e:
        raise RuntimeError(f"LLM prose call failed: {e}") from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _resolve_chapter_key(db: AsyncSession, chapter_id: str | int) -> Tuple[str, str]:
    name, display = None, None
    try:
        cid = int(chapter_id)
        meta = (await db.execute(select(ChapterMeta).where(ChapterMeta.id == cid))).scalars().first()
        if meta:
            return meta.name, (meta.display_name or meta.name)
    except Exception:
        pass
    nm = str(chapter_id).strip()
    meta = (await db.execute(select(ChapterMeta).where(ChapterMeta.name == nm))).scalars().first()
    if meta:
        return meta.name, (meta.display_name or meta.name)
    return nm, nm


def _format_responses(pairs: list) -> str:
    """Format prompt/response pairs as a numbered list for LLM input."""
    lines = []
    for i, (p, r) in enumerate(pairs, 1):
        title = getattr(r, "title", None) or ""
        # Audio responses store HTML (Quill editor) in transcription; typed responses use response_text
        raw = (r.response_text or r.transcription or "")
        text = _strip_html(raw) if raw else ""
        if not text:
            continue  # skip responses with no text content at all
        lines.append(
            f"[{i}] PROMPT: {p.text}\n"
            + (f"    TITLE: {title}\n" if title else "")
            + f"    RESPONSE:\n{text}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# chapter_status (unchanged logic)
# ---------------------------------------------------------------------------

async def chapter_status(db: AsyncSession, chapter_id: str | int, user_id: int) -> ChapterStatusDTO:
    chapter_key, display = await _resolve_chapter_key(db, chapter_id)

    assigned_rows = await db.execute(
        select(UserPrompt.prompt_id).join(Prompt, Prompt.id == UserPrompt.prompt_id)
        .where((UserPrompt.user_id == user_id) & (Prompt.chapter == chapter_key))
    )
    assigned_ids = set(assigned_rows.scalars().all())

    resp_rows = await db.execute(
        select(Response.prompt_id).where((Response.user_id == user_id) & (Response.prompt_id.in_(assigned_ids)))
    )
    completed_ids = set(resp_rows.scalars().all())
    missing_ids = list(assigned_ids - completed_ids)

    missing_prompts = []
    if missing_ids:
        prs = await db.execute(select(Prompt).where(Prompt.id.in_(missing_ids)))
        for p in prs.unique().scalars().all():
            missing_prompts.append({"id": p.id, "text": p.text})

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


# ---------------------------------------------------------------------------
# Multi-pass compile
# ---------------------------------------------------------------------------

async def compile_chapter(
    db: AsyncSession,
    chapter_id: str | int,
    user_id: int,
) -> ChapterCompilationDTO:
    chapter_key, display = await _resolve_chapter_key(db, chapter_id)

    meta = (await db.execute(select(ChapterMeta).where(ChapterMeta.name == chapter_key))).scalars().first()
    guidance = (meta.llm_guidance or "").strip() if meta else ""
    description = (meta.description or "").strip() if meta else ""

    rows = await db.execute(
        select(Prompt, Response)
        .join(Response, Response.prompt_id == Prompt.id)
        .options(selectinload(Prompt.tags))
        .where((Prompt.chapter == chapter_key) & (Response.user_id == user_id))
        .order_by(Response.created_at.asc())
    )
    pairs = rows.unique().all()

    if not pairs:
        raise RuntimeError("No responses found for this chapter.")

    # Warn loudly if responses have no text — catches the audio-only case early
    empty = [(p.text, r.id) for p, r in pairs if not _strip_html(r.response_text or r.transcription or "")]
    if empty:
        import logging
        logging.getLogger(__name__).warning(
            "compile_chapter: %d response(s) have no text or transcription and will be skipped: %s",
            len(empty), empty
        )

    responses_text = _format_responses(pairs)
    idx_to_ids = {i + 1: (p.id, r.id) for i, (p, r) in enumerate(pairs)}

    # Release the DB connection back to the pool before the long LLM calls.
    # asyncpg returns the connection on commit; subsequent writes re-acquire one.
    await db.commit()

    # -----------------------------------------------------------------------
    # Pass 1 — Thematic grouping
    # -----------------------------------------------------------------------
    p1_system = f"""\
You are organizing memoir content for the chapter "{display}".
{f'Chapter description: {description}' if description else ''}
{f'Editorial guidance: {guidance}' if guidance else ''}

Read all the question-response pairs below (numbered [1], [2], …).
Group them into 3–6 thematic sections that will flow well together in a memoir chapter.
Each section should have a short evocative title and a list of the response numbers that belong.

Return ONLY JSON in this exact shape:
{{
  "sections": [
    {{"title": "section title", "theme": "one sentence describing what ties these together", "response_numbers": [1, 3, 5]}},
    ...
  ]
}}"""

    p1_data = await _llm_json(p1_system, responses_text, timeout=180)
    sections = p1_data.get("sections") or []

    # Fallback: if grouping failed, treat all as one section
    if not sections:
        sections = [{"title": display, "theme": "memoir narrative", "response_numbers": list(range(1, len(pairs) + 1))}]

    # -----------------------------------------------------------------------
    # Pass 2 — Draft each section as flowing prose
    # -----------------------------------------------------------------------
    p2_system = """\
You are a memoir ghostwriter editing a real person's recorded answers into prose.

STRICT RULES — failure to follow these will ruin the memoir:
1. Write ONLY in the FIRST PERSON ("I", "my", "we").
2. Use ONLY the names, places, dates, and events that appear in the responses below.
   DO NOT invent, assume, or add ANY detail not explicitly stated in the responses.
3. If a response is vague, keep the prose vague — do not fill gaps with guesses.
4. Preserve the person's own phrasing and vocabulary as much as possible.
5. Write 2–4 flowing paragraphs with a warm, honest, conversational tone.
6. Output ONLY the prose paragraphs — no headers, no JSON, no commentary, no preamble."""

    section_drafts: list[str] = []
    used_blocks: list[dict] = []

    for sec in sections:
        title = sec.get("title", "")
        theme = sec.get("theme", "")
        nums = [n for n in (sec.get("response_numbers") or []) if 1 <= n <= len(pairs)]
        if not nums:
            continue

        # Build the subset of responses for this section
        subset_pairs = [pairs[n - 1] for n in nums]
        subset_text = _format_responses(subset_pairs)

        p2_user = (
            f'Section title: "{title}"\n'
            f'Theme: {theme}\n\n'
            f"Responses to use:\n{subset_text}"
        )
        draft = await _llm_prose(p2_system, p2_user, timeout=300)
        section_drafts.append(f"### {title}\n\n{draft}")

        for n in nums:
            pid, rid = idx_to_ids.get(n, (None, None))
            if pid:
                used_blocks.append({"prompt_id": pid, "response_id": rid, "section": title})

    # -----------------------------------------------------------------------
    # Pass 3 — Assemble and polish the full chapter
    # -----------------------------------------------------------------------
    combined_draft = "\n\n".join(section_drafts)

    p3_system = f"""\
You are a senior memoir editor. Below are drafted sections of a memoir chapter \
titled "{display}".
{f'Editorial guidance: {guidance}' if guidance else ''}

Your task:
1. Write a compelling OPENING paragraph drawn from the actual content — be specific
   and vivid using only details already present in the drafts.
2. Arrange the sections in the most natural narrative order.
3. Write smooth TRANSITION sentences between sections so the chapter flows as one piece.
4. Write a CLOSING paragraph that brings the chapter to a satisfying close using
   only details already present in the drafts.
5. Preserve the first-person voice throughout.
6. CRITICAL: Do NOT add, invent, or assume any names, places, events, or details
   that are not already in the drafted sections. Light editing only.

Output the complete, polished chapter as clean markdown (use ## for the chapter \
title, ### for section headers if needed). No JSON, no commentary, no preamble."""

    p3_user = f"CHAPTER TITLE: {display}\n\nDRAFTED SECTIONS:\n\n{combined_draft}"
    final_markdown = await _llm_prose(p3_system, p3_user, timeout=360)

    if not final_markdown.strip():
        final_markdown = combined_draft  # fallback to unpolished draft

    # -----------------------------------------------------------------------
    # Pass 4 — Gap analysis (small, fast call)
    # -----------------------------------------------------------------------
    p4_system = f"""\
You have just read a memoir chapter about "{display}".
Identify 3–6 important gaps — things that seem missing, underdeveloped, or \
worth exploring further that would enrich this chapter.
Return ONLY JSON:
{{
  "gap_questions": [
    {{"question": "...", "why": "one sentence on why this would enrich the chapter"}},
    ...
  ]
}}"""

    p4_user = f"CHAPTER CONTENT (summary):\n\n{final_markdown[:3000]}"  # cap to avoid timeout
    p4_data = await _llm_json(p4_system, p4_user, timeout=120)
    gaps = p4_data.get("gap_questions") or []

    # -----------------------------------------------------------------------
    # Save result
    # -----------------------------------------------------------------------
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
        compiled_markdown=final_markdown,
        gap_questions=gaps,
        used_blocks=used_blocks,
        model_name=OLLAMA_MODEL,
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
