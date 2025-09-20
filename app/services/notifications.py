"""Notification helpers for response events."""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.background import run_sync
from app.database import async_session_maker
from app.models import Response, User, ResponseNotificationTarget, ResponseShare
from app.services.invite import send_email

logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory="templates")


@dataclass(slots=True)
class _NotificationContext:
    recipient_name: str
    author_name: str
    response_title: str
    response_excerpt: str
    response_url: str
    prompt_title: str
    link_expires: str


def _trim(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _resolve_base_url() -> str:
    for env_key in ("BASE_URL", "PUBLIC_URL"):
        raw = os.getenv(env_key, "").strip()
        if raw:
            return raw.rstrip("/")
    return "http://localhost:8000"


def _share_ttl_days() -> int:
    raw = os.getenv("NOTIFICATION_SHARE_TTL_DAYS", "5").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 5
    return max(1, value)


def _generate_share_token() -> str:
    return secrets.token_urlsafe(22)


async def _load_watchers(response_user_id: int) -> list[User]:
    async with async_session_maker() as session:
        result = await session.execute(
            select(User)
            .join(
                ResponseNotificationTarget,
                ResponseNotificationTarget.watcher_user_id == User.id,
            )
            .where(ResponseNotificationTarget.owner_user_id == response_user_id)
            .where(User.notify_new_responses.is_(True))
            .where(User.is_active.is_(True))
        )
        users = result.scalars().all()
        unique = {u.id: u for u in users if (u.email or "").strip()}
        return list(unique.values())


async def _load_response(response_id: int) -> Response | None:
    async with async_session_maker() as session:
        result = await session.execute(
            select(Response)
            .options(selectinload(Response.user), selectinload(Response.prompt))
            .where(Response.id == response_id)
        )
        return result.scalars().first()


def _build_context(
    response: Response,
    recipient: User,
    base_url: str,
    share_token: str,
    expires_at: datetime | None,
) -> _NotificationContext:
    author = response.user
    author_label = (author.username or author.email or "A user").strip()
    resp_title = (response.title or "New response").strip() or "New response"
    excerpt_source = (response.response_text or "").strip()
    prompt_text = (getattr(response.prompt, "text", "") or "").strip()

    share_url = (
        f"{base_url}/share/r/{share_token}"
        if share_token
        else f"{base_url}/admin/users/{response.user_id}/responses/{response.id}"
    )
    expires_label = ""
    if expires_at:
        try:
            expires_local = expires_at.astimezone(timezone.utc)
        except Exception:
            expires_local = expires_at
        expires_label = expires_local.strftime("%b %d, %Y")

    return _NotificationContext(
        recipient_name=(recipient.username or recipient.email or "there").strip(),
        author_name=author_label,
        response_title=_trim(resp_title, limit=140),
        response_excerpt=_trim(excerpt_source, limit=320) if excerpt_source else "",
        response_url=share_url,
        prompt_title=_trim(prompt_text, limit=160) if prompt_text else "",
        link_expires=expires_label,
    )


async def _create_share_links(response: Response, watcher_ids: list[int]) -> dict[int, tuple[str, datetime | None]]:
    if not watcher_ids:
        return {}
    expires_at = datetime.now(timezone.utc) + timedelta(days=_share_ttl_days())
    out: dict[int, tuple[str, datetime | None]] = {}
    async with async_session_maker() as session:
        for wid in watcher_ids:
            token = _generate_share_token()
            share = ResponseShare(
                token=token,
                response_id=response.id,
                user_id=response.user_id,
                permanent=False,
                expires_at=expires_at,
            )
            session.add(share)
            out[wid] = (token, expires_at)
        await session.commit()
    return out


async def notify_new_response(response_id: int) -> None:
    """Send notifications to subscribed users about a new response."""

    response = await _load_response(response_id)
    if not response or not response.user:
        logger.debug("notify_new_response: response %s not found or missing user", response_id)
        return

    watchers = await _load_watchers(response.user_id)
    if not watchers:
        logger.debug("notify_new_response: no watchers subscribed, skipping")
        return

    base_url = _resolve_base_url()
    share_map = await _create_share_links(response, [w.id for w in watchers])
    html_template = templates.get_template("email/new_response_notification.html")
    text_template = templates.get_template("email/new_response_notification.txt")

    for watcher in watchers:
        token, expires = share_map.get(watcher.id, ("", None))
        ctx = _build_context(response, watcher, base_url, token, expires)
        context_dict = {
            "recipient_name": ctx.recipient_name,
            "author_name": ctx.author_name,
            "response_title": ctx.response_title,
            "response_excerpt": ctx.response_excerpt,
            "response_url": ctx.response_url,
            "prompt_title": ctx.prompt_title,
            "link_expires": ctx.link_expires,
        }
        try:
            html_body = html_template.render(context_dict)
        except Exception:  # noqa: BLE001
            logger.exception("notify_new_response: failed to render HTML template")
            html_body = None
        try:
            text_body = text_template.render(context_dict)
        except Exception:  # noqa: BLE001
            logger.exception("notify_new_response: failed to render text template")
            text_body = (
                f"{ctx.recipient_name},\n\n"
                f"{ctx.author_name} just added a new response titled '{ctx.response_title}'.\n"
                f"View it here: {ctx.response_url}\n"
            )

        subject = f"New response from {ctx.author_name}"
        try:
            await run_sync(
                send_email,
                watcher.email,
                subject=subject,
                text_body=text_body or "",
                html_body=html_body,
            )
        except Exception:  # noqa: BLE001
            logger.exception("notify_new_response: failed to send email to %s", watcher.email)


__all__ = ["notify_new_response"]
