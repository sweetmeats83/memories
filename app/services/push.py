"""Web Push notification helpers.

Requires VAPID keys in environment:
  VAPID_PRIVATE_KEY  — base64url-encoded private key (or path to .pem file)
  VAPID_PUBLIC_KEY   — base64url-encoded public key (returned to the browser)
  VAPID_SUBJECT      — mailto: or https: URL identifying the sender
                       (default: mailto:admin@example.com)

Generate keys:
  python -c "
  from py_vapid import Vapid
  v = Vapid()
  v.generate_keys()
  print('VAPID_PRIVATE_KEY=' + v.private_pem().decode().strip())
  print('VAPID_PUBLIC_KEY='  + v.public_key)
  "
  (py_vapid is a dependency of pywebpush)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _vapid_private_key() -> Optional[str]:
    return (os.getenv("VAPID_PRIVATE_KEY") or "").strip() or None


def vapid_public_key() -> Optional[str]:
    return (os.getenv("VAPID_PUBLIC_KEY") or "").strip() or None


def push_enabled() -> bool:
    return bool(_vapid_private_key() and vapid_public_key())


def _vapid_subject() -> str:
    return (os.getenv("VAPID_SUBJECT") or "").strip() or "mailto:admin@example.com"


def _send_one(endpoint: str, p256dh: str, auth: str, payload: dict) -> bool:
    """Blocking — call via run_sync from async code."""
    from pywebpush import webpush, WebPushException

    try:
        webpush(
            subscription_info={
                "endpoint": endpoint,
                "keys": {"p256dh": p256dh, "auth": auth},
            },
            data=json.dumps(payload),
            vapid_private_key=_vapid_private_key(),
            vapid_claims={"sub": _vapid_subject()},
            content_encoding="aes128gcm",
        )
        return True
    except WebPushException as exc:
        # 410 Gone = subscription expired/revoked; caller should delete it
        status = getattr(exc.response, "status_code", None) if exc.response else None
        if status == 410:
            raise _SubscriptionGone() from exc
        logger.warning("WebPush failed (status=%s): %s", status, exc)
        return False
    except Exception as exc:
        logger.warning("WebPush unexpected error: %s", exc)
        return False


class _SubscriptionGone(Exception):
    pass


async def send_push_to_user(
    db,
    user_id: int,
    *,
    title: str,
    body: str,
    url: str = "/",
    tag: str = "memories-prompt",
) -> int:
    """Send a push notification to all active subscriptions for a user.
    Removes expired subscriptions automatically. Returns number sent."""
    if not push_enabled():
        return 0

    from sqlalchemy import select, delete
    from app.models import PushSubscription
    from app.background import run_sync

    subs = (
        await db.execute(
            select(PushSubscription).where(PushSubscription.user_id == user_id)
        )
    ).scalars().all()

    if not subs:
        return 0

    payload = {"title": title, "body": body, "url": url, "tag": tag}
    sent = 0
    expired_ids = []

    for sub in subs:
        try:
            ok = await run_sync(_send_one, sub.endpoint, sub.p256dh, sub.auth, payload)
            if ok:
                sent += 1
        except _SubscriptionGone:
            expired_ids.append(sub.id)
        except Exception as exc:
            logger.warning("push to user %s sub %s failed: %s", user_id, sub.id, exc)

    if expired_ids:
        await db.execute(
            delete(PushSubscription).where(PushSubscription.id.in_(expired_ids))
        )
        await db.commit()

    return sent


async def send_push_to_all(
    db,
    *,
    title: str,
    body: str,
    url: str = "/",
    tag: str = "memories-broadcast",
) -> int:
    """Broadcast a push to every subscribed user. Returns total sent."""
    if not push_enabled():
        return 0

    from sqlalchemy import select
    from app.models import PushSubscription

    user_ids = list(set(
        (await db.execute(select(PushSubscription.user_id).distinct())).scalars().all()
    ))

    total = 0
    for uid in user_ids:
        total += await send_push_to_user(db, uid, title=title, body=body, url=url, tag=tag)
    return total
