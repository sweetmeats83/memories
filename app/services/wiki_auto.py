"""
Auto-schedule wiki regeneration after a response is enriched.

Called from enrich_after_transcription once people links are committed.
Uses a cooldown so a burst of new stories doesn't spam the LLM.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.background import spawn

logger = logging.getLogger(__name__)

# Minimum gap between regenerations for the same article.
WIKI_COOLDOWN_HOURS = 6


async def _should_refresh(article, cooldown_cutoff) -> bool:
    """Return True if article needs regenerating (missing, errored, or past cooldown)."""
    if not article:
        return True
    if article.status == "generating":
        return False
    if article.status == "error":
        return True
    return not article.generated_at or article.generated_at <= cooldown_cutoff


async def schedule_wiki_refresh_for_response(response_id: int, user_id: int) -> None:
    """
    Background-safe. Opens its own session.

    Schedules wiki refresh for:
    - All persons linked to the response (via ResponsePerson) + author's self-person
    - All places linked to the response (via ResponsePlace)
    - All events linked to the response (via ResponseEvent)
    """
    from app.database import async_session_maker
    from app.models import (
        KinMembership, ResponseEvent, ResponsePerson, ResponsePlace,
        UserProfile, WikiArticle,
    )
    from app.services.people_acl import visible_group_ids
    from app.services.wiki_generator import (
        generate_event_wiki,
        generate_person_wiki,
        generate_place_wiki,
    )

    async with async_session_maker() as db:
        gids = await visible_group_ids(db, user_id)
        if not gids:
            return
        group_id = gids[0]

        cooldown_cutoff = datetime.now(timezone.utc) - timedelta(hours=WIKI_COOLDOWN_HOURS)

        # --- Persons ---
        person_ids: set[int] = set(
            (await db.execute(
                select(ResponsePerson.person_id)
                .where(ResponsePerson.response_id == response_id)
            )).scalars().all()
        )
        profile = await db.scalar(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        if profile:
            self_pid = (profile.privacy_prefs or {}).get("self_person_id")
            if self_pid:
                person_ids.add(int(self_pid))

        for pid in person_ids:
            article = await db.scalar(
                select(WikiArticle).where(
                    WikiArticle.entity_type == "person",
                    WikiArticle.entity_id == pid,
                    WikiArticle.group_id == group_id,
                )
            )
            if not await _should_refresh(article, cooldown_cutoff):
                logger.debug("wiki_auto: person=%s skip (generating or recent)", pid)
                continue
            spawn(generate_person_wiki(pid, group_id), name=f"wiki-auto-p{pid}-g{group_id}")
            logger.info("wiki_auto: scheduled person=%s group=%s", pid, group_id)

        # --- Places ---
        place_ids: set[int] = set(
            (await db.execute(
                select(ResponsePlace.place_id)
                .where(ResponsePlace.response_id == response_id)
            )).scalars().all()
        )
        for plid in place_ids:
            article = await db.scalar(
                select(WikiArticle).where(
                    WikiArticle.entity_type == "place",
                    WikiArticle.entity_id == plid,
                    WikiArticle.group_id == group_id,
                )
            )
            if not await _should_refresh(article, cooldown_cutoff):
                logger.debug("wiki_auto: place=%s skip (generating or recent)", plid)
                continue
            spawn(generate_place_wiki(plid, group_id), name=f"wiki-auto-pl{plid}-g{group_id}")
            logger.info("wiki_auto: scheduled place=%s group=%s", plid, group_id)

        # --- Events ---
        event_ids: set[int] = set(
            (await db.execute(
                select(ResponseEvent.event_id)
                .where(ResponseEvent.response_id == response_id)
            )).scalars().all()
        )
        for eid in event_ids:
            article = await db.scalar(
                select(WikiArticle).where(
                    WikiArticle.entity_type == "event",
                    WikiArticle.entity_id == eid,
                    WikiArticle.group_id == group_id,
                )
            )
            if not await _should_refresh(article, cooldown_cutoff):
                logger.debug("wiki_auto: event=%s skip (generating or recent)", eid)
                continue
            spawn(generate_event_wiki(eid, group_id), name=f"wiki-auto-ev{eid}-g{group_id}")
            logger.info("wiki_auto: scheduled event=%s group=%s", eid, group_id)
