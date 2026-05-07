"""
People database cleanup service.

Handles two problems left by the old (narrator-unaware) extraction pipeline:
  1. Ghost persons — Person rows with meta.inferred=True that were auto-created
     when no match was found and now have real counterparts in the tree.
  2. Wrong ResponsePerson links — stories tagged to an inferred ghost instead of
     the actual person.

Cleanup strategy:
  - Delete ResponsePerson rows that point to ghost persons.
  - Re-run extraction on all affected responses using the improved
    narrator-aware pipeline so they get re-linked correctly.
  - Delete ghost persons that have no remaining ResponsePerson rows.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Person, PersonAlias, Prompt, Response, ResponsePerson, User
from app.services.people_acl import visible_group_ids, person_visibility_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

async def audit_people(db: AsyncSession) -> dict:
    """
    Return admin-wide stats about ghost persons and response tagging.
    Safe to call at any time; read-only.
    """
    all_persons = (await db.execute(select(Person))).scalars().all()
    ghost_persons = [p for p in all_persons if isinstance(p.meta, dict) and p.meta.get("inferred")]

    ghost_ids = {p.id for p in ghost_persons}

    ghost_rp_rows = (
        await db.execute(
            select(ResponsePerson).where(ResponsePerson.person_id.in_(ghost_ids))
        )
    ).scalars().all() if ghost_ids else []

    affected_response_ids = {rp.response_id for rp in ghost_rp_rows}

    total_responses = (await db.execute(select(Response.id))).scalars().all()
    total_rp = (await db.execute(select(ResponsePerson.id))).scalars().all()

    # Fetch story metadata for each ghost's linked responses
    ghost_stories: dict[int, list[dict]] = {}
    if ghost_ids:
        story_rows = (
            await db.execute(
                select(ResponsePerson.person_id, Response.id, Response.title, Prompt.text)
                .join(Response, Response.id == ResponsePerson.response_id)
                .outerjoin(Prompt, Prompt.id == Response.prompt_id)
                .where(ResponsePerson.person_id.in_(ghost_ids))
            )
        ).all()
        for person_id, resp_id, resp_title, prompt_text in story_rows:
            ghost_stories.setdefault(person_id, []).append({
                "id": resp_id,
                "title": resp_title or (prompt_text[:80] if prompt_text else f"Story #{resp_id}"),
            })

    # All persons for the merge dropdown (including ghosts, so duplicates can be consolidated)
    all_persons_sorted = sorted(all_persons, key=lambda p: (p.display_name or "").lower())

    return {
        "total_persons": len(all_persons),
        "ghost_persons": len(ghost_persons),
        "ghost_person_list": [
            {
                "id": p.id,
                "display_name": p.display_name,
                "owner_user_id": p.owner_user_id,
                "response_count": sum(1 for rp in ghost_rp_rows if rp.person_id == p.id),
                "stories": ghost_stories.get(p.id, []),
            }
            for p in ghost_persons
        ],
        "all_person_list": [
            {
                "id": p.id,
                "display_name": p.display_name,
                "is_ghost": bool(isinstance(p.meta, dict) and p.meta.get("inferred")),
            }
            for p in all_persons_sorted
        ],
        "ghost_response_person_rows": len(ghost_rp_rows),
        "affected_responses": len(affected_response_ids),
        "total_responses": len(total_responses),
        "total_response_person_rows": len(total_rp),
    }


# ---------------------------------------------------------------------------
# Re-extraction helper (mirrors the people-extraction block in transcription.py)
# ---------------------------------------------------------------------------

async def _reextract_people_for_response(
    db: AsyncSession, response: Response
) -> int:
    """
    Re-run narrator-aware people extraction for a single response.
    Deletes existing ResponsePerson rows first.
    Returns the number of people linked.
    """
    try:
        from app.services.people import (
            extract_name_spans, role_hint_near,
            resolve_person, link_mention,
            llm_extract_people, build_narrator_graph,
        )
    except ImportError:
        logger.warning("people services not available; skipping reextract")
        return 0

    # Delete existing links for this response
    await db.execute(
        delete(ResponsePerson).where(ResponsePerson.response_id == response.id)
    )

    people_text = (
        (getattr(response, "ai_polished", "") or "").strip()
        or (getattr(response, "response_text", "") or "").strip()
        or (getattr(response, "transcription", "") or "").strip()
    )
    if not people_text:
        return 0

    # Build narrator context
    known_names: list[str] = []
    narrator_graph: list[dict] = []
    tree_people: list = []
    try:
        _group_ids = await visible_group_ids(db, response.user_id)
        tree_people = (
            await db.execute(
                select(Person).where(person_visibility_filter(response.user_id, _group_ids))
            )
        ).scalars().all()
        known_names = [p.display_name for p in tree_people if p.display_name]
        narrator_graph = await build_narrator_graph(db, response.user_id)
    except Exception:
        pass

    # LLM extraction → fallback spaCy/regex
    extractions = await llm_extract_people(
        people_text, known_names=known_names, narrator_graph=narrator_graph
    )
    if not extractions:
        try:
            pids = [p.id for p in tree_people]
            alias_rows = (
                await db.execute(
                    select(PersonAlias.alias).where(PersonAlias.person_id.in_(pids))
                )
            ).scalars().all() if pids else []
            aliases = known_names + list(alias_rows)
            spans = extract_name_spans(people_text, aliases=aliases)
            extractions = [
                {"name": surface, "role": role_hint_near(people_text, s, e)}
                for surface, s, e in spans
            ]
        except Exception:
            pass

    seen_ids: set[int] = set()
    for item in extractions:
        surface = (item.get("name") or "").strip()
        role = item.get("role") or None
        if not surface:
            continue
        person = await resolve_person(
            db, response.user_id, surface,
            role_hint=role, context_text=people_text,
            narrator_graph=narrator_graph,
        )
        if person is None or person.id in seen_ids:
            continue
        seen_ids.add(person.id)
        await link_mention(
            db, response.id, person,
            alias_used=surface, confidence=0.85, role_hint=role,
        )

    return len(seen_ids)


# ---------------------------------------------------------------------------
# Main cleanup job (runs as background task)
# ---------------------------------------------------------------------------

async def run_people_cleanup(scope: str = "ghosts") -> dict:
    """
    Background-safe entry point.

    scope="ghosts"  — only reprocess responses that had ghost-person links
    scope="all"     — reprocess every response (thorough but slow)
    """
    from app.database import async_session_maker

    stats = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "responses_processed": 0,
        "people_linked": 0,
        "ghost_persons_deleted": 0,
        "errors": 0,
    }

    async with async_session_maker() as db:
        try:
            # 1. Find ghost persons
            all_persons = (await db.execute(select(Person))).scalars().all()
            ghost_ids = {
                p.id for p in all_persons
                if isinstance(p.meta, dict) and p.meta.get("inferred")
            }

            # 2. Determine which responses to reprocess
            if scope == "all":
                response_ids = (await db.execute(select(Response.id))).scalars().all()
            else:
                if ghost_ids:
                    rp_rows = (
                        await db.execute(
                            select(ResponsePerson.response_id)
                            .where(ResponsePerson.person_id.in_(ghost_ids))
                            .distinct()
                        )
                    ).scalars().all()
                    response_ids = list(rp_rows)
                else:
                    response_ids = []

            logger.info("people_cleanup: reprocessing %d responses (scope=%s)", len(response_ids), scope)

            # 3. Reprocess each response
            for resp_id in response_ids:
                try:
                    async with async_session_maker() as inner_db:
                        resp = await inner_db.get(Response, resp_id)
                        if not resp:
                            continue
                        n = await _reextract_people_for_response(inner_db, resp)
                        await inner_db.commit()
                        stats["responses_processed"] += 1
                        stats["people_linked"] += n
                except Exception:
                    logger.exception("people_cleanup: error reprocessing response %s", resp_id)
                    stats["errors"] += 1

            # 4. Delete ghost persons that now have no ResponsePerson rows
            if ghost_ids:
                remaining_rp = (
                    await db.execute(
                        select(ResponsePerson.person_id)
                        .where(ResponsePerson.person_id.in_(ghost_ids))
                        .distinct()
                    )
                ).scalars().all()
                still_linked = set(remaining_rp)
                to_delete = ghost_ids - still_linked
                for pid in to_delete:
                    p = await db.get(Person, pid)
                    if p:
                        await db.delete(p)
                        stats["ghost_persons_deleted"] += 1
                await db.commit()

        except Exception:
            logger.exception("people_cleanup: job failed")
            stats["error"] = "Job failed — see server logs"

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    logger.info("people_cleanup done: %s", stats)
    return stats
