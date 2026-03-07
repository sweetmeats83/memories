"""Web Push subscription endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import PushSubscription
from app.utils import get_current_user, require_admin_user
from app.services.push import vapid_public_key, push_enabled, send_push_to_all

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/push/vapid-key")
async def get_vapid_key():
    """Return the VAPID public key so the browser can subscribe."""
    key = vapid_public_key()
    if not key:
        raise HTTPException(503, "Push notifications are not configured on this server")
    return {"publicKey": key}


@router.post("/api/push/subscribe")
async def push_subscribe(
    request: Request,
    payload: dict = Body(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Save or update a browser push subscription for the current user."""
    if not user:
        raise HTTPException(401, "Not authenticated")

    endpoint = (payload.get("endpoint") or "").strip()
    keys     = payload.get("keys") or {}
    p256dh   = (keys.get("p256dh") or "").strip()
    auth     = (keys.get("auth")   or "").strip()

    if not endpoint or not p256dh or not auth:
        raise HTTPException(400, "Invalid subscription payload")

    # Upsert: update keys if endpoint already exists, otherwise insert
    existing = (
        await db.execute(
            select(PushSubscription).where(PushSubscription.endpoint == endpoint)
        )
    ).scalars().first()

    ua = (request.headers.get("user-agent") or "")[:512]

    if existing:
        existing.p256dh    = p256dh
        existing.auth      = auth
        existing.user_agent = ua
        existing.user_id   = user.id   # re-associate if login changed
    else:
        db.add(PushSubscription(
            user_id=user.id,
            endpoint=endpoint,
            p256dh=p256dh,
            auth=auth,
            user_agent=ua,
        ))

    await db.commit()
    return {"ok": True}


@router.delete("/api/push/unsubscribe")
async def push_unsubscribe(
    payload: dict = Body(...),
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    """Remove the given push subscription."""
    if not user:
        raise HTTPException(401, "Not authenticated")
    endpoint = (payload.get("endpoint") or "").strip()
    if endpoint:
        await db.execute(
            delete(PushSubscription).where(
                PushSubscription.endpoint == endpoint,
                PushSubscription.user_id == user.id,
            )
        )
        await db.commit()
    return {"ok": True}


@router.get("/api/admin/push/status")
async def admin_push_status(
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    from sqlalchemy import func
    from app.models import PushSubscription as PS
    count = (await db.execute(
        select(func.count()).select_from(PS)
    )).scalar() or 0
    return {
        "enabled": push_enabled(),
        "public_key": vapid_public_key(),
        "subscriptions": count,
    }


@router.post("/api/admin/push/test")
async def admin_push_test(
    db: AsyncSession = Depends(get_db),
    admin=Depends(require_admin_user),
):
    """Send a test push to all subscribed devices."""
    if not push_enabled():
        raise HTTPException(503, "Push notifications not configured — set VAPID_PRIVATE_KEY and VAPID_PUBLIC_KEY")
    sent = await send_push_to_all(
        db,
        title="Memories — test notification",
        body="Push notifications are working.",
        url="/",
        tag="memories-test",
    )
    return {"ok": True, "sent": sent}
