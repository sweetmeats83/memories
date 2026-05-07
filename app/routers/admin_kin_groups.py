"""
Admin routes for KinGroup (family group) management.

GET  /admin/kin-groups                            — HTML management page
GET  /api/admin/kin-groups                        — JSON: all groups + members
POST /api/admin/kin-groups                        — create a new group
DELETE /api/admin/kin-groups/{group_id}           — delete a group
POST /api/admin/kin-groups/{group_id}/members     — add user to group
DELETE /api/admin/kin-groups/{group_id}/members/{user_id} — remove user from group
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import KinGroup, KinMembership, User
from app.routes_shared import templates
from app.utils import require_admin_user

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _groups_with_members(db: AsyncSession) -> list[dict]:
    groups = (await db.execute(select(KinGroup).order_by(KinGroup.id))).scalars().all()
    all_users = (await db.execute(select(User).order_by(User.username, User.email))).scalars().all()
    memberships = (await db.execute(select(KinMembership))).scalars().all()

    user_map = {u.id: u for u in all_users}
    mem_by_group: dict[int, list[dict]] = {}
    for m in memberships:
        mem_by_group.setdefault(m.group_id, []).append({
            "user_id": m.user_id,
            "username": user_map[m.user_id].username if m.user_id in user_map else str(m.user_id),
            "email": user_map[m.user_id].email if m.user_id in user_map else "",
            "role": m.role,
        })

    member_user_ids = {m.user_id for m in memberships}
    ungrouped = [
        {"user_id": u.id, "username": u.username, "email": u.email}
        for u in all_users
        if u.id not in member_user_ids
    ]

    return {
        "groups": [
            {
                "id": g.id,
                "name": g.name,
                "kind": g.kind,
                "join_code": g.join_code,
                "member_count": len(mem_by_group.get(g.id, [])),
                "members": mem_by_group.get(g.id, []),
            }
            for g in groups
        ],
        "all_users": [
            {"user_id": u.id, "username": u.username, "email": u.email}
            for u in all_users
        ],
        "ungrouped_users": ungrouped,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/admin/kin-groups", response_class=HTMLResponse)
async def admin_kin_groups_page(
    request: Request,
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    data = await _groups_with_members(db)
    return templates.TemplateResponse(
        request,
        "admin_kin_groups.html",
        {"user": user, **data},
    )


@router.get("/api/admin/kin-groups")
async def api_list_groups(
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    return await _groups_with_members(db)


@router.post("/api/admin/kin-groups")
async def api_create_group(
    request: Request,
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    kind = body.get("kind", "family")
    group = KinGroup(name=name, kind=kind, created_by=user.id)
    db.add(group)
    await db.commit()
    await db.refresh(group)
    return {"id": group.id, "name": group.name, "kind": group.kind}


@router.delete("/api/admin/kin-groups/{group_id}")
async def api_delete_group(
    group_id: int,
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    group = await db.get(KinGroup, group_id)
    if not group:
        raise HTTPException(404, "Group not found")
    await db.delete(group)
    await db.commit()
    return {"ok": True}


@router.post("/api/admin/kin-groups/{group_id}/members")
async def api_add_member(
    group_id: int,
    request: Request,
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    body = await request.json()
    target_user_id = body.get("user_id")
    if not target_user_id:
        raise HTTPException(400, "user_id is required")

    target = await db.get(User, target_user_id)
    if not target:
        raise HTTPException(404, "User not found")

    group = await db.get(KinGroup, group_id)
    if not group:
        raise HTTPException(404, "Group not found")

    existing = await db.scalar(
        select(KinMembership).where(
            KinMembership.group_id == group_id,
            KinMembership.user_id == target_user_id,
        )
    )
    if existing:
        return {"ok": True, "note": "already a member"}

    db.add(KinMembership(group_id=group_id, user_id=target_user_id, role=body.get("role", "member")))
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
    return {"ok": True}


@router.delete("/api/admin/kin-groups/{group_id}/members/{target_user_id}")
async def api_remove_member(
    group_id: int,
    target_user_id: int,
    user=Depends(require_admin_user),
    db: AsyncSession = Depends(get_db),
):
    await db.execute(
        delete(KinMembership).where(
            KinMembership.group_id == group_id,
            KinMembership.user_id == target_user_id,
        )
    )
    await db.commit()
    return {"ok": True}
