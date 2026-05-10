"""
Admin routes for entity cleanup (persons, places, events).

GET    /admin/people/cleanup                      — HTML cleanup dashboard
GET    /api/admin/people/audit                    — JSON person audit
POST   /api/admin/people/reprocess                — trigger background cleanup job
GET    /api/admin/people/reprocess/status         — poll job status
POST   /api/admin/people                          — create a new confirmed person
PATCH  /api/admin/people/{person_id}              — rename person / clear ghost flag
POST   /api/admin/people/{person_id}/merge        — merge ghost into real person
POST   /api/admin/people/{person_id}/reclassify   — convert ghost to Place or Event
DELETE /api/admin/people/{person_id}              — delete person record

GET    /api/admin/places/audit                    — JSON place audit
POST   /api/admin/places                          — create a new place
PATCH  /api/admin/places/{place_id}               — rename place
POST   /api/admin/places/{place_id}/merge         — merge into another place
DELETE /api/admin/places/{place_id}               — delete place

GET    /api/admin/events/audit                    — JSON event audit
POST   /api/admin/events                          — create a new event
PATCH  /api/admin/events/{event_id}               — rename / update event
POST   /api/admin/events/{event_id}/merge         — merge into another event
DELETE /api/admin/events/{event_id}               — delete event
"""
from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import (
    Event, EventPerson, EventPlace,
    Person, PersonAlias,
    Place,
    ResponseEvent, ResponsePerson, ResponsePlace,
)
from app.routes_shared import templates
from app.utils import require_admin_user
from app.background import spawn
from app.services.people_cleanup import (
    audit_people, audit_places, audit_events,
    run_people_cleanup, run_full_entity_reextraction,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory job status for both cleanup jobs.
_job_status: dict = {"running": False, "last_result": None}
_reextract_status: dict = {"running": False, "last_result": None}


async def _primary_group_id(db: AsyncSession, user_id: int) -> int | None:
    from app.services.people_acl import visible_group_ids
    gids = await visible_group_ids(db, user_id)
    return gids[0] if gids else None


@router.get("/admin/people/cleanup", response_class=HTMLResponse)
async def admin_people_cleanup_page(
    request: Request,
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    group_id = await _primary_group_id(db, user.id)
    audit = await audit_people(db, group_id=group_id)
    places = await audit_places(db, group_id=group_id)
    events = await audit_events(db, group_id=group_id)
    return templates.TemplateResponse(
        request,
        "admin_people_cleanup.html",
        {"user": user, "audit": audit, "places": places, "events": events, "job": _job_status},
    )


@router.get("/api/admin/people/audit")
async def api_audit(user=Depends(require_admin_user), db: AsyncSession = Depends(get_db)):
    group_id = await _primary_group_id(db, user.id)
    return await audit_people(db, group_id=group_id)


@router.post("/api/admin/people/reprocess")
async def api_reprocess(
    request: Request,
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    if _job_status["running"]:
        return JSONResponse({"error": "A cleanup job is already running."}, status_code=409)

    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    scope = body.get("scope", "ghosts")
    if scope not in ("ghosts", "all"):
        scope = "ghosts"

    group_id = await _primary_group_id(db, user.id)

    _job_status["running"] = True
    _job_status["last_result"] = None

    async def _run():
        try:
            result = await run_people_cleanup(scope=scope, group_id=group_id)
            _job_status["last_result"] = result
        finally:
            _job_status["running"] = False

    spawn(_run(), name="people-cleanup")
    return JSONResponse({"status": "started", "scope": scope}, status_code=202)


@router.get("/api/admin/people/reprocess/status")
async def api_reprocess_status(user=Depends(require_admin_user)):
    return _job_status


# ---------------------------------------------------------------------------
# Per-person edit operations
# ---------------------------------------------------------------------------

@router.post("/api/admin/people")
async def api_create_person(
    body: dict = Body(...),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new confirmed person in the admin's primary group."""
    group_id = await _primary_group_id(db, user.id)
    display_name = (body.get("display_name") or "").strip()
    if not display_name:
        raise HTTPException(400, "display_name is required")
    existing = (await db.execute(
        select(Person).where(Person.group_id == group_id, Person.display_name.ilike(display_name))
    )).scalars().first()
    if existing:
        raise HTTPException(409, f"A person named '{existing.display_name}' already exists")
    person = Person(display_name=display_name, group_id=group_id, meta={})
    db.add(person)
    await db.commit()
    return {"id": person.id, "display_name": person.display_name}


@router.patch("/api/admin/people/{person_id}")
async def api_rename_person(
    person_id: int,
    body: dict = Body(...),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Rename a person and optionally clear the ghost/inferred flag."""
    p = await db.get(Person, person_id)
    if not p:
        raise HTTPException(status_code=404, detail="Person not found")
    new_name = (body.get("display_name") or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="display_name is required")
    p.display_name = new_name
    meta = dict(p.meta or {})
    meta.pop("inferred", None)
    p.meta = meta
    await db.commit()
    return {"id": p.id, "display_name": p.display_name}


@router.post("/api/admin/people/{person_id}/merge")
async def api_merge_person(
    person_id: int,
    body: dict = Body(...),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Merge a ghost person into a real person.
    Migrates all ResponsePerson rows, adds ghost name as alias, then deletes ghost.
    """
    target_id = body.get("target_id")
    if not target_id:
        raise HTTPException(status_code=400, detail="target_id is required")
    try:
        target_id = int(target_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="target_id must be an integer")

    ghost = await db.get(Person, person_id)
    target = await db.get(Person, target_id)
    if not ghost:
        raise HTTPException(status_code=404, detail="Ghost person not found")
    if not target:
        raise HTTPException(status_code=404, detail="Target person not found")
    if person_id == target_id:
        raise HTTPException(status_code=400, detail="Cannot merge a person into itself")

    # Move all ResponsePerson rows from ghost → target
    await db.execute(
        sa_update(ResponsePerson)
        .where(ResponsePerson.person_id == person_id)
        .values(person_id=target_id)
    )

    # Add ghost display_name as alias on target (skip if already exists)
    existing_alias = (
        await db.execute(
            select(PersonAlias).where(
                PersonAlias.person_id == target_id,
                PersonAlias.alias == ghost.display_name,
            )
        )
    ).scalar_one_or_none()
    if not existing_alias and ghost.display_name:
        db.add(PersonAlias(person_id=target_id, alias=ghost.display_name))

    await db.delete(ghost)
    await db.commit()
    return {"merged": person_id, "into": target_id, "target_name": target.display_name}


@router.delete("/api/admin/people/{person_id}")
async def api_delete_person(
    person_id: int,
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a person record outright (cascades ResponsePerson rows)."""
    p = await db.get(Person, person_id)
    if not p:
        raise HTTPException(status_code=404, detail="Person not found")
    await db.delete(p)
    await db.commit()
    return {"deleted": person_id}


@router.post("/api/admin/people/{person_id}/reclassify")
async def api_reclassify_person(
    person_id: int,
    body: dict = Body(...),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Convert a ghost Person into a Place or Event.
    Migrates all ResponsePerson rows → ResponsePlace or ResponseEvent, then deletes the person.
    """
    entity_type = (body.get("entity_type") or "").strip().lower()
    if entity_type not in ("place", "event"):
        raise HTTPException(400, "entity_type must be 'place' or 'event'")

    ghost = await db.get(Person, person_id)
    if not ghost:
        raise HTTPException(404, "Person not found")

    name = (body.get("name") or ghost.display_name or "").strip()
    if not name:
        raise HTTPException(400, "name is required")

    # Determine group_id from ghost or fall back to first available group
    from app.models import KinGroup, KinMembership
    from app.services.places_events import (
        upsert_place_for_group, upsert_event_for_group,
        link_place_to_response, link_event_to_response,
    )

    group_id = ghost.group_id
    if not group_id:
        # Try to find any group
        gm = await db.scalar(select(KinMembership).limit(1))
        group_id = gm.group_id if gm else None
    if not group_id:
        raise HTTPException(400, "No family group found to assign entity to")

    # Get all ResponsePerson rows for this ghost
    rp_rows = (await db.execute(
        select(ResponsePerson).where(ResponsePerson.person_id == person_id)
    )).scalars().all()

    if entity_type == "place":
        place = await upsert_place_for_group(
            db, group_id, name,
            place_type=body.get("place_type"),
            city=body.get("city"),
            state=body.get("state"),
        )
        for rp in rp_rows:
            await link_place_to_response(db, rp.response_id, place.id)
        await db.delete(ghost)
        await db.commit()
        return {"reclassified": person_id, "entity_type": "place", "entity_id": place.id, "name": place.name}

    else:  # event
        year = body.get("year")
        try:
            year = int(year) if year else None
        except (TypeError, ValueError):
            year = None
        event = await upsert_event_for_group(
            db, group_id, name,
            year=year,
            event_type=body.get("event_type"),
        )
        for rp in rp_rows:
            await link_event_to_response(db, rp.response_id, event.id)
        await db.delete(ghost)
        await db.commit()
        return {"reclassified": person_id, "entity_type": "event", "entity_id": event.id, "name": event.name, "year": event.year}


# ---------------------------------------------------------------------------
# Place CRUD
# ---------------------------------------------------------------------------

@router.get("/api/admin/places/audit")
async def api_places_audit(user=Depends(require_admin_user), db: AsyncSession = Depends(get_db)):
    group_id = await _primary_group_id(db, user.id)
    return await audit_places(db, group_id=group_id)


@router.post("/api/admin/places")
async def api_create_place(
    body: dict = Body(...),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new place in the admin's primary group."""
    from app.services.places_events import upsert_place_for_group
    group_id = await _primary_group_id(db, user.id)
    if not group_id:
        raise HTTPException(400, "No group found")
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    place = await upsert_place_for_group(
        db, group_id, name,
        place_type=body.get("place_type") or "other",
    )
    await db.commit()
    return {"id": place.id, "name": place.name}


@router.patch("/api/admin/places/{place_id}")
async def api_rename_place(
    place_id: int,
    body: dict = Body(...),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    place = await db.get(Place, place_id)
    if not place:
        raise HTTPException(404, "Place not found")
    new_name = (body.get("name") or "").strip()
    if not new_name:
        raise HTTPException(400, "name is required")
    place.name = new_name
    if body.get("place_type"):
        place.place_type = body["place_type"]
    if body.get("city"):
        place.city = body["city"]
    if body.get("state"):
        place.state = body["state"]
    await db.commit()
    return {"id": place.id, "name": place.name}


@router.post("/api/admin/places/{place_id}/merge")
async def api_merge_place(
    place_id: int,
    body: dict = Body(...),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Merge place into target: migrate ResponsePlace + EventPlace rows, delete source."""
    target_id = body.get("target_id")
    if not target_id:
        raise HTTPException(400, "target_id is required")
    try:
        target_id = int(target_id)
    except (TypeError, ValueError):
        raise HTTPException(400, "target_id must be an integer")

    source = await db.get(Place, place_id)
    target = await db.get(Place, target_id)
    if not source:
        raise HTTPException(404, "Source place not found")
    if not target:
        raise HTTPException(404, "Target place not found")
    if place_id == target_id:
        raise HTTPException(400, "Cannot merge a place into itself")

    # Migrate ResponsePlace rows (ignore duplicates)
    rp_rows = (await db.execute(
        select(ResponsePlace).where(ResponsePlace.place_id == place_id)
    )).scalars().all()
    existing_resp_ids = set((await db.execute(
        select(ResponsePlace.response_id).where(ResponsePlace.place_id == target_id)
    )).scalars().all())
    for rp in rp_rows:
        if rp.response_id not in existing_resp_ids:
            db.add(ResponsePlace(response_id=rp.response_id, place_id=target_id, role_hint=rp.role_hint))

    # Migrate EventPlace rows
    ep_rows = (await db.execute(
        select(EventPlace).where(EventPlace.place_id == place_id)
    )).scalars().all()
    existing_event_ids = set((await db.execute(
        select(EventPlace.event_id).where(EventPlace.place_id == target_id)
    )).scalars().all())
    for ep in ep_rows:
        if ep.event_id not in existing_event_ids:
            db.add(EventPlace(event_id=ep.event_id, place_id=target_id))

    await db.delete(source)
    await db.commit()
    return {"merged": place_id, "into": target_id, "target_name": target.name}


@router.delete("/api/admin/places/{place_id}")
async def api_delete_place(
    place_id: int,
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    place = await db.get(Place, place_id)
    if not place:
        raise HTTPException(404, "Place not found")
    await db.delete(place)
    await db.commit()
    return {"deleted": place_id}


# ---------------------------------------------------------------------------
# Event CRUD
# ---------------------------------------------------------------------------

@router.get("/api/admin/events/audit")
async def api_events_audit(user=Depends(require_admin_user), db: AsyncSession = Depends(get_db)):
    group_id = await _primary_group_id(db, user.id)
    return await audit_events(db, group_id=group_id)


@router.post("/api/admin/events")
async def api_create_event(
    body: dict = Body(...),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new event in the admin's primary group."""
    from app.services.places_events import upsert_event_for_group
    group_id = await _primary_group_id(db, user.id)
    if not group_id:
        raise HTTPException(400, "No group found")
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    year = body.get("year")
    if year:
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None
    event = await upsert_event_for_group(
        db, group_id, name, year,
        event_type=body.get("event_type") or "other",
    )
    await db.commit()
    display_name = f"{event.name} {event.year}" if event.year else event.name
    return {"id": event.id, "name": event.name, "year": event.year, "display_name": display_name}


@router.patch("/api/admin/events/{event_id}")
async def api_rename_event(
    event_id: int,
    body: dict = Body(...),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    event = await db.get(Event, event_id)
    if not event:
        raise HTTPException(404, "Event not found")
    if body.get("name"):
        event.name = body["name"].strip()
    if "year" in body:
        try:
            event.year = int(body["year"]) if body["year"] else None
        except (TypeError, ValueError):
            pass
    if body.get("event_type"):
        event.event_type = body["event_type"]
    await db.commit()
    return {"id": event.id, "name": event.name, "year": event.year}


@router.post("/api/admin/events/{event_id}/merge")
async def api_merge_event(
    event_id: int,
    body: dict = Body(...),
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Merge event into target: migrate ResponseEvent, EventPerson, EventPlace rows, delete source."""
    target_id = body.get("target_id")
    if not target_id:
        raise HTTPException(400, "target_id is required")
    try:
        target_id = int(target_id)
    except (TypeError, ValueError):
        raise HTTPException(400, "target_id must be an integer")

    source = await db.get(Event, event_id)
    target = await db.get(Event, target_id)
    if not source:
        raise HTTPException(404, "Source event not found")
    if not target:
        raise HTTPException(404, "Target event not found")
    if event_id == target_id:
        raise HTTPException(400, "Cannot merge an event into itself")

    # Migrate ResponseEvent
    re_rows = (await db.execute(
        select(ResponseEvent).where(ResponseEvent.event_id == event_id)
    )).scalars().all()
    existing_resp_ids = set((await db.execute(
        select(ResponseEvent.response_id).where(ResponseEvent.event_id == target_id)
    )).scalars().all())
    for re in re_rows:
        if re.response_id not in existing_resp_ids:
            db.add(ResponseEvent(response_id=re.response_id, event_id=target_id, role_hint=re.role_hint))

    # Migrate EventPerson
    ep_rows = (await db.execute(
        select(EventPerson).where(EventPerson.event_id == event_id)
    )).scalars().all()
    existing_person_ids = set((await db.execute(
        select(EventPerson.person_id).where(EventPerson.event_id == target_id)
    )).scalars().all())
    for ep in ep_rows:
        if ep.person_id not in existing_person_ids:
            db.add(EventPerson(event_id=target_id, person_id=ep.person_id, role_hint=ep.role_hint))

    # Migrate EventPlace
    epl_rows = (await db.execute(
        select(EventPlace).where(EventPlace.event_id == event_id)
    )).scalars().all()
    existing_place_ids = set((await db.execute(
        select(EventPlace.place_id).where(EventPlace.event_id == target_id)
    )).scalars().all())
    for epl in epl_rows:
        if epl.place_id not in existing_place_ids:
            db.add(EventPlace(event_id=target_id, place_id=epl.place_id))

    await db.delete(source)
    await db.commit()
    display = f"{target.name} {target.year}" if target.year else target.name
    return {"merged": event_id, "into": target_id, "target_name": display}


@router.delete("/api/admin/events/{event_id}")
async def api_delete_event(
    event_id: int,
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    event = await db.get(Event, event_id)
    if not event:
        raise HTTPException(404, "Event not found")
    await db.delete(event)
    await db.commit()
    return {"deleted": event_id}


# ---------------------------------------------------------------------------
# Full entity re-extraction job (people + places + events for all responses)
# ---------------------------------------------------------------------------

@router.post("/api/admin/reextract/all")
async def api_reextract_all(
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger a background job that re-runs the full entity extraction pipeline
    (people + places + events) on every response in the admin's primary group.
    This is a long-running operation — poll /api/admin/reextract/status for progress.
    """
    if _reextract_status["running"] or _job_status["running"]:
        return JSONResponse({"error": "A job is already running."}, status_code=409)

    group_id = await _primary_group_id(db, user.id)

    _reextract_status["running"] = True
    _reextract_status["last_result"] = None

    async def _run():
        try:
            result = await run_full_entity_reextraction(group_id=group_id)
            _reextract_status["last_result"] = result
        finally:
            _reextract_status["running"] = False

    spawn(_run(), name="full-entity-reextraction")
    return JSONResponse({"status": "started", "group_id": group_id}, status_code=202)


@router.get("/api/admin/reextract/status")
async def api_reextract_status(user=Depends(require_admin_user)):
    return _reextract_status
