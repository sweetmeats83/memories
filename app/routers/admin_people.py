"""
Admin routes for people / person-detection cleanup.

GET    /admin/people/cleanup                — HTML cleanup dashboard
GET    /api/admin/people/audit              — JSON audit stats (read-only)
POST   /api/admin/people/reprocess          — trigger background cleanup job
GET    /api/admin/people/reprocess/status   — poll job status
PATCH  /api/admin/people/{person_id}        — rename person / clear ghost flag
POST   /api/admin/people/{person_id}/merge  — merge ghost into real person
DELETE /api/admin/people/{person_id}        — delete person record
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
from app.models import Person, PersonAlias, ResponsePerson
from app.routes_shared import templates
from app.utils import require_admin_user
from app.background import spawn
from app.services.people_cleanup import audit_people, run_people_cleanup

router = APIRouter()
logger = logging.getLogger(__name__)

# Simple in-memory status so the UI can poll while the job runs.
_job_status: dict = {"running": False, "last_result": None}


@router.get("/admin/people/cleanup", response_class=HTMLResponse)
async def admin_people_cleanup_page(
    request: Request,
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    audit = await audit_people(db)
    return templates.TemplateResponse(
        request,
        "admin_people_cleanup.html",
        {"user": user, "audit": audit, "job": _job_status},
    )


@router.get("/api/admin/people/audit")
async def api_audit(
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    return await audit_people(db)


@router.post("/api/admin/people/reprocess")
async def api_reprocess(
    request: Request,
    user=Depends(require_admin_user),
):
    if _job_status["running"]:
        return JSONResponse({"error": "A cleanup job is already running."}, status_code=409)

    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    scope = body.get("scope", "ghosts")
    if scope not in ("ghosts", "all"):
        scope = "ghosts"

    _job_status["running"] = True
    _job_status["last_result"] = None

    async def _run():
        try:
            result = await run_people_cleanup(scope=scope)
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
