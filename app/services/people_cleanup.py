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

from app.models import (
    Event, EventPerson, EventPlace,
    Person, PersonAlias,
    Place, PlaceAlias,
    Prompt, Response, ResponseSegment,
    ResponseEvent, ResponsePerson, ResponsePlace,
    User, WikiArticle,
)
from app.services.people_acl import visible_group_ids, person_visibility_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

async def _member_ids_for_group(db: AsyncSession, group_id: int | None) -> set[int] | None:
    """Return user IDs belonging to group_id, or None if no group filter."""
    if not group_id:
        return None
    from app.models import KinMembership
    rows = (await db.execute(
        select(KinMembership.user_id).where(KinMembership.group_id == group_id)
    )).scalars().all()
    return set(rows)


async def audit_people(db: AsyncSession, group_id: int | None = None) -> dict:
    """
    Return stats about ghost persons and response tagging.
    When group_id is provided, stories and persons are scoped to that group's members.
    """
    member_ids = await _member_ids_for_group(db, group_id)

    # Persons: if group-scoped, only show persons belonging to that group
    person_stmt = select(Person)
    if group_id:
        person_stmt = person_stmt.where(Person.group_id == group_id)
    all_persons = (await db.execute(person_stmt)).scalars().all()
    ghost_persons = [p for p in all_persons if isinstance(p.meta, dict) and p.meta.get("inferred")]

    ghost_ids = {p.id for p in ghost_persons}

    # ResponsePerson rows scoped to group members
    rp_stmt = select(ResponsePerson)
    if ghost_ids:
        rp_stmt = rp_stmt.where(ResponsePerson.person_id.in_(ghost_ids))
        if member_ids:
            rp_stmt = rp_stmt.join(Response, Response.id == ResponsePerson.response_id)\
                             .where(Response.user_id.in_(member_ids))
        ghost_rp_rows = (await db.execute(rp_stmt)).scalars().all()
    else:
        ghost_rp_rows = []

    affected_response_ids = {rp.response_id for rp in ghost_rp_rows}

    # Total counts scoped to group members
    resp_stmt = select(Response.id)
    if member_ids:
        resp_stmt = resp_stmt.where(Response.user_id.in_(member_ids))
    total_responses = (await db.execute(resp_stmt)).scalars().all()
    total_rp = (await db.execute(select(ResponsePerson.id))).scalars().all()

    # Story metadata for ghost linked responses — scoped to group members
    ghost_stories: dict[int, list[dict]] = {}
    if ghost_ids:
        story_stmt = (
            select(ResponsePerson.person_id, Response.id, Response.title, Prompt.text)
            .join(Response, Response.id == ResponsePerson.response_id)
            .outerjoin(Prompt, Prompt.id == Response.prompt_id)
            .where(ResponsePerson.person_id.in_(ghost_ids))
        )
        if member_ids:
            story_stmt = story_stmt.where(Response.user_id.in_(member_ids))
        story_rows = (await db.execute(story_stmt)).all()
        for person_id, resp_id, resp_title, prompt_text in story_rows:
            ghost_stories.setdefault(person_id, []).append({
                "id": resp_id,
                "title": resp_title or (prompt_text[:80] if prompt_text else f"Story #{resp_id}"),
            })

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
# Places + Events audit
# ---------------------------------------------------------------------------

async def audit_places(db: AsyncSession, group_id: int | None = None) -> dict:
    """All places in the DB with story counts and wiki status, scoped to group."""
    from sqlalchemy import func as sqlfunc
    stmt = select(Place)
    if group_id:
        stmt = stmt.where(Place.group_id == group_id)
    places = (await db.execute(stmt)).scalars().all()

    # Story counts per place
    count_rows = (await db.execute(
        select(ResponsePlace.place_id, sqlfunc.count(ResponsePlace.id))
        .group_by(ResponsePlace.place_id)
    )).all()
    story_counts = {pid: cnt for pid, cnt in count_rows}

    # Wiki status per place
    wiki_rows = (await db.execute(
        select(WikiArticle.entity_id, WikiArticle.status)
        .where(WikiArticle.entity_type == "place")
    )).all()
    wiki_status = {eid: status for eid, status in wiki_rows}

    places_sorted = sorted(places, key=lambda p: (p.name or "").lower())
    return {
        "total_places": len(places),
        "place_list": [
            {
                "id": p.id,
                "name": p.name,
                "place_type": p.place_type,
                "city": p.city,
                "state": p.state,
                "story_count": story_counts.get(p.id, 0),
                "wiki_status": wiki_status.get(p.id, "none"),
            }
            for p in places_sorted
        ],
        "all_place_list": [
            {"id": p.id, "name": p.name}
            for p in places_sorted
        ],
    }


async def audit_events(db: AsyncSession, group_id: int | None = None) -> dict:
    """All events in the DB with story counts and wiki status, scoped to group."""
    from sqlalchemy import func as sqlfunc
    stmt = select(Event)
    if group_id:
        stmt = stmt.where(Event.group_id == group_id)
    events = (await db.execute(stmt)).scalars().all()

    count_rows = (await db.execute(
        select(ResponseEvent.event_id, sqlfunc.count(ResponseEvent.id))
        .group_by(ResponseEvent.event_id)
    )).all()
    story_counts = {eid: cnt for eid, cnt in count_rows}

    wiki_rows = (await db.execute(
        select(WikiArticle.entity_id, WikiArticle.status)
        .where(WikiArticle.entity_type == "event")
    )).all()
    wiki_status = {eid: status for eid, status in wiki_rows}

    events_sorted = sorted(events, key=lambda e: (e.name or "", e.year or 0))
    return {
        "total_events": len(events),
        "event_list": [
            {
                "id": e.id,
                "name": e.name,
                "year": e.year,
                "display_name": f"{e.name} {e.year}" if e.year else e.name,
                "event_type": e.event_type,
                "story_count": story_counts.get(e.id, 0),
                "wiki_status": wiki_status.get(e.id, "none"),
            }
            for e in events_sorted
        ],
        "all_event_list": [
            {"id": e.id, "display_name": f"{e.name} {e.year}" if e.year else e.name}
            for e in events_sorted
        ],
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
            llm_extract_entities, build_narrator_graph,
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
        segs = (await db.execute(
            select(ResponseSegment.transcript)
            .where(ResponseSegment.response_id == response.id)
            .order_by(ResponseSegment.order_index)
        )).scalars().all()
        people_text = " ".join(s for s in segs if s).strip()
    if not people_text:
        return 0

    # Build narrator context — split so narrator_graph is always attempted
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
    except Exception:
        pass
    try:
        narrator_graph = await build_narrator_graph(db, response.user_id)
    except Exception:
        pass

    # Build exclusion set from existing Place/Event records so role-word ghost persons
    # aren't created for things already known to be places or events.
    _known_place_names: set[str] = set()
    try:
        _group_ids_for_places = await visible_group_ids(db, response.user_id)
        _place_rows = (await db.execute(
            select(Place.name).where(Place.group_id.in_(_group_ids_for_places))
        )).scalars().all()
        _event_rows = (await db.execute(
            select(Event.name).where(Event.group_id.in_(_group_ids_for_places))
        )).scalars().all()
        _known_place_names = {n.strip().lower() for n in (_place_rows + _event_rows) if n}
    except Exception:
        pass

    # LLM extraction → fallback spaCy/regex
    entities = await llm_extract_entities(
        people_text, known_names=known_names, narrator_graph=narrator_graph
    )
    extractions = [
        p for p in (entities.get("people") or [])
        if (p.get("name") or "").strip().lower() not in _known_place_names
    ]
    if not extractions:
        try:
            from app.services.people import _apply_role_fallback
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
            extractions = _apply_role_fallback(people_text, extractions, narrator_graph)
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

async def run_people_cleanup(scope: str = "ghosts", group_id: int | None = None) -> dict:
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
            # Resolve group member IDs for scoping
            member_ids = await _member_ids_for_group(db, group_id)

            # 1. Find ghost persons (scoped to group if provided)
            person_stmt = select(Person)
            if group_id:
                person_stmt = person_stmt.where(Person.group_id == group_id)
            all_persons = (await db.execute(person_stmt)).scalars().all()
            ghost_ids = {
                p.id for p in all_persons
                if isinstance(p.meta, dict) and p.meta.get("inferred")
            }

            # 2. Determine which responses to reprocess (scoped to group members)
            if scope == "all":
                resp_stmt = select(Response.id)
                if member_ids:
                    resp_stmt = resp_stmt.where(Response.user_id.in_(member_ids))
                response_ids = (await db.execute(resp_stmt)).scalars().all()
            else:
                if ghost_ids:
                    rp_stmt = (
                        select(ResponsePerson.response_id)
                        .where(ResponsePerson.person_id.in_(ghost_ids))
                        .distinct()
                    )
                    if member_ids:
                        rp_stmt = rp_stmt.join(
                            Response, Response.id == ResponsePerson.response_id
                        ).where(Response.user_id.in_(member_ids))
                    response_ids = (await db.execute(rp_stmt)).scalars().all()
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


# ---------------------------------------------------------------------------
# Full entity re-extraction (people + places + events) for all responses
# ---------------------------------------------------------------------------

async def _reextract_all_entities_for_response(
    db: AsyncSession, response: Response
) -> dict:
    """
    Re-run the full entity extraction pipeline (people + places + events)
    for a single response. Clears existing ResponsePerson/Place/Event links first.
    Returns counts: {people, places, events}.
    """
    try:
        from app.services.people import (
            extract_name_spans, role_hint_near,
            resolve_person, link_mention,
            llm_extract_entities, llm_extract_places_events, build_narrator_graph,
        )
        from app.services.places_events import (
            upsert_place_for_group, upsert_event_for_group,
            link_place_to_response, link_event_to_response,
            link_event_to_place, link_event_to_person,
        )
        from app.models import KinMembership
    except ImportError as exc:
        logger.warning("entity reextract imports failed: %s", exc)
        return {"people": 0, "places": 0, "events": 0}

    # Determine group_id for the response owner
    group_id = await db.scalar(
        select(KinMembership.group_id).where(KinMembership.user_id == response.user_id).limit(1)
    )

    # Clear existing links
    await db.execute(delete(ResponsePerson).where(ResponsePerson.response_id == response.id))
    await db.execute(delete(ResponsePlace).where(ResponsePlace.response_id == response.id))
    await db.execute(delete(ResponseEvent).where(ResponseEvent.response_id == response.id))

    story_text = (
        (getattr(response, "ai_polished", "") or "").strip()
        or (getattr(response, "response_text", "") or "").strip()
        or (getattr(response, "transcription", "") or "").strip()
    )
    if not story_text:
        segs = (await db.execute(
            select(ResponseSegment.transcript)
            .where(ResponseSegment.response_id == response.id)
            .order_by(ResponseSegment.order_index)
        )).scalars().all()
        story_text = " ".join(s for s in segs if s).strip()
    if not story_text:
        return {"people": 0, "places": 0, "events": 0}

    # Build narrator context — split so narrator_graph is always attempted
    known_names: list[str] = []
    narrator_graph: list[dict] = []
    tree_people: list = []
    try:
        from app.models import PersonAlias as _PA
        _group_ids = await visible_group_ids(db, response.user_id)
        tree_people = (
            await db.execute(
                select(Person).where(person_visibility_filter(response.user_id, _group_ids))
            )
        ).scalars().all()
        known_names = [p.display_name for p in tree_people if p.display_name]
    except Exception:
        pass
    try:
        narrator_graph = await build_narrator_graph(db, response.user_id)
    except Exception:
        pass

    # Two separate LLM calls: people (narrator-aware) and places+events (simple, focused)
    people_result = await llm_extract_entities(
        story_text, known_names=known_names, narrator_graph=narrator_graph
    )
    places_events_result = await llm_extract_places_events(story_text)

    entities = {
        "people": people_result["people"],
        "places": places_events_result["places"],
        "events": places_events_result["events"],
    }

    # --- People ---
    # Build exclusion set from place/event names — don't create Person records for things
    # the LLM already classified as places or events.
    _non_person_names: set[str] = set()
    for _pl in (entities.get("places") or []):
        _n = (_pl.get("name") or "").strip().lower()
        if _n:
            _non_person_names.add(_n)
    for _ev in (entities.get("events") or []):
        _n = (_ev.get("name") or "").strip().lower()
        if _n:
            _non_person_names.add(_n)

    people_count = 0
    people_extractions = [
        p for p in (entities.get("people") or [])
        if (p.get("name") or "").strip().lower() not in _non_person_names
    ]
    if not people_extractions:
        try:
            from app.services.people import _apply_role_fallback
            pids = [p.id for p in tree_people]
            alias_rows = (
                await db.execute(
                    select(PersonAlias.alias).where(PersonAlias.person_id.in_(pids))
                )
            ).scalars().all() if pids else []
            aliases = known_names + list(alias_rows)
            spans = extract_name_spans(story_text, aliases=aliases)
            people_extractions = [
                {"name": surface, "role": role_hint_near(story_text, s, e)}
                for surface, s, e in spans
            ]
            people_extractions = _apply_role_fallback(story_text, people_extractions, narrator_graph)
        except Exception:
            pass

    seen_person_ids: set[int] = set()
    person_name_to_id: dict[str, int] = {}
    for item in people_extractions:
        surface = (item.get("name") or "").strip()
        role = item.get("role") or None
        if not surface:
            continue
        person = await resolve_person(
            db, response.user_id, surface,
            role_hint=role, context_text=story_text,
            narrator_graph=narrator_graph,
        )
        if person is None or person.id in seen_person_ids:
            continue
        seen_person_ids.add(person.id)
        person_name_to_id[surface.lower()] = person.id
        if person.display_name:
            person_name_to_id[person.display_name.lower()] = person.id
        await link_mention(db, response.id, person, alias_used=surface, confidence=0.85, role_hint=role)
        people_count += 1

    # --- Places ---
    places_count = 0
    place_name_to_id: dict[str, int] = {}
    if group_id:
        for pl in (entities.get("places") or []):
            pl_name = (pl.get("name") or "").strip()
            if not pl_name:
                continue
            try:
                place = await upsert_place_for_group(
                    db, group_id, pl_name,
                    place_type=pl.get("type") or "other",
                    city=pl.get("city"), state=pl.get("state"),
                    address=pl.get("address"), country=pl.get("country"),
                )
                await link_place_to_response(db, response.id, place.id)
                place_name_to_id[pl_name.lower()] = place.id
                places_count += 1
            except Exception:
                logger.debug("place upsert failed for %r", pl_name, exc_info=True)

    # --- Events ---
    events_count = 0
    if group_id:
        for ev in (entities.get("events") or []):
            ev_name = (ev.get("name") or "").strip()
            if not ev_name:
                continue
            try:
                event = await upsert_event_for_group(
                    db, group_id, ev_name, ev.get("year"),
                    event_type=ev.get("type") or "other",
                )
                await link_event_to_response(db, response.id, event.id)
                for pl_ref in (ev.get("places") or []):
                    pid = place_name_to_id.get((pl_ref or "").lower())
                    if pid:
                        await link_event_to_place(db, event.id, pid)
                for att_ref in (ev.get("attendees") or []):
                    ppid = person_name_to_id.get((att_ref or "").lower())
                    if ppid:
                        await link_event_to_person(db, event.id, ppid)
                events_count += 1
            except Exception:
                logger.debug("event upsert failed for %r", ev_name, exc_info=True)

    return {"people": people_count, "places": places_count, "events": events_count}


async def run_full_entity_reextraction(group_id: int | None = None) -> dict:
    """
    Background job: re-run the full people+places+events extraction pipeline
    on every response belonging to the given group (or all responses if no group).
    """
    from app.database import async_session_maker

    stats = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "group_id": group_id,
        "responses_processed": 0,
        "people_linked": 0,
        "places_linked": 0,
        "events_linked": 0,
        "errors": 0,
    }

    async with async_session_maker() as db:
        try:
            member_ids = await _member_ids_for_group(db, group_id)
            resp_stmt = select(Response.id)
            if member_ids:
                resp_stmt = resp_stmt.where(Response.user_id.in_(member_ids))
            response_ids = (await db.execute(resp_stmt)).scalars().all()
        except Exception:
            logger.exception("run_full_entity_reextraction: failed to load response ids")
            stats["error"] = "Failed to load response IDs — see server logs"
            return stats

    total = len(response_ids)
    logger.info("full_entity_reextraction: %d responses to process (group_id=%s)", total, group_id)

    for i, resp_id in enumerate(response_ids):
        try:
            from app.database import async_session_maker as _asm
            async with _asm() as inner_db:
                resp = await inner_db.get(Response, resp_id)
                if not resp:
                    continue
                counts = await _reextract_all_entities_for_response(inner_db, resp)
                await inner_db.commit()
                stats["responses_processed"] += 1
                stats["people_linked"] += counts["people"]
                stats["places_linked"] += counts["places"]
                stats["events_linked"] += counts["events"]
        except Exception:
            logger.exception("full_entity_reextraction: error on response %s", resp_id)
            stats["errors"] += 1

        if (i + 1) % 50 == 0:
            logger.info("full_entity_reextraction: %d/%d done", i + 1, total)

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    logger.info("full_entity_reextraction done: %s", stats)
    return stats
