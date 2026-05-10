"""
Upsert helpers for Place and Event entities.

All operations are group-scoped — each KinGroup maintains its own set of
places and events so multiple families don't share records.
"""
from __future__ import annotations

import logging
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Event,
    EventPerson,
    EventPlace,
    Place,
    ResponseEvent,
    ResponsePlace,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Place
# ---------------------------------------------------------------------------

async def upsert_place_for_group(
    db: AsyncSession,
    group_id: int,
    name: str,
    *,
    place_type: str | None = None,
    city: str | None = None,
    state: str | None = None,
    address: str | None = None,
    country: str | None = None,
) -> Place:
    """Return existing Place by name (case-insensitive) or create a new one."""
    name = name.strip()
    existing = await db.scalar(
        select(Place).where(
            Place.group_id == group_id,
            Place.name.ilike(name),
        )
    )
    if existing:
        # Fill in any new detail fields we didn't have before
        changed = False
        for field, val in (
            ("place_type", place_type),
            ("city", city),
            ("state", state),
            ("address", address),
            ("country", country),
        ):
            if val and not getattr(existing, field):
                setattr(existing, field, val)
                changed = True
        if changed:
            await db.flush()
        return existing

    place = Place(
        group_id=group_id,
        name=name,
        place_type=place_type,
        city=city,
        state=state,
        address=address,
        country=country,
    )
    db.add(place)
    try:
        async with db.begin_nested():
            await db.flush()
    except IntegrityError:
        place = await db.scalar(
            select(Place).where(Place.group_id == group_id, Place.name.ilike(name))
        )
    return place


async def link_place_to_response(
    db: AsyncSession,
    response_id: int,
    place_id: int,
    role_hint: str | None = None,
) -> None:
    exists = await db.scalar(
        select(ResponsePlace).where(
            ResponsePlace.response_id == response_id,
            ResponsePlace.place_id == place_id,
        )
    )
    if not exists:
        db.add(ResponsePlace(response_id=response_id, place_id=place_id, role_hint=role_hint))
        try:
            async with db.begin_nested():
                await db.flush()
        except IntegrityError:
            pass


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

async def upsert_event_for_group(
    db: AsyncSession,
    group_id: int,
    name: str,
    year: int | None = None,
    *,
    event_type: str | None = None,
) -> Event:
    """Return existing Event by (name, year) or create a new one."""
    name = name.strip()
    stmt = select(Event).where(
        Event.group_id == group_id,
        Event.name.ilike(name),
    )
    if year is not None:
        stmt = stmt.where(Event.year == year)
    else:
        stmt = stmt.where(Event.year.is_(None))

    existing = await db.scalar(stmt)
    if existing:
        if event_type and not existing.event_type:
            existing.event_type = event_type
            await db.flush()
        return existing

    event = Event(
        group_id=group_id,
        name=name,
        year=year,
        event_type=event_type,
    )
    db.add(event)
    try:
        async with db.begin_nested():
            await db.flush()
    except IntegrityError:
        event = await db.scalar(stmt)
    return event


async def link_event_to_response(
    db: AsyncSession,
    response_id: int,
    event_id: int,
    role_hint: str | None = None,
) -> None:
    exists = await db.scalar(
        select(ResponseEvent).where(
            ResponseEvent.response_id == response_id,
            ResponseEvent.event_id == event_id,
        )
    )
    if not exists:
        db.add(ResponseEvent(response_id=response_id, event_id=event_id, role_hint=role_hint))
        try:
            async with db.begin_nested():
                await db.flush()
        except IntegrityError:
            pass


async def link_event_to_place(
    db: AsyncSession,
    event_id: int,
    place_id: int,
) -> None:
    exists = await db.scalar(
        select(EventPlace).where(
            EventPlace.event_id == event_id,
            EventPlace.place_id == place_id,
        )
    )
    if not exists:
        db.add(EventPlace(event_id=event_id, place_id=place_id))
        try:
            async with db.begin_nested():
                await db.flush()
        except IntegrityError:
            pass


async def link_event_to_person(
    db: AsyncSession,
    event_id: int,
    person_id: int,
    role_hint: str | None = None,
) -> None:
    exists = await db.scalar(
        select(EventPerson).where(
            EventPerson.event_id == event_id,
            EventPerson.person_id == person_id,
        )
    )
    if not exists:
        db.add(EventPerson(event_id=event_id, person_id=person_id, role_hint=role_hint))
        try:
            async with db.begin_nested():
                await db.flush()
        except IntegrityError:
            pass
