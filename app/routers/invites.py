import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from passlib.hash import bcrypt
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Invite, User
from ..users import get_jwt_strategy, cookie_transport
from ..services.invite import new_token, invite_expiry, render_invite_email, send_email

router = APIRouter()
templates = Jinja2Templates(directory="templates")

INVITE_BASE_URL = os.getenv("INVITE_BASE_URL", os.getenv("BASE_URL", "")).rstrip("/")
INVITE_TTL_DAYS = int(os.getenv("INVITE_TTL_DAYS", "7"))


def _now_tz() -> datetime:
    return datetime.now(timezone.utc)


def _invite_url(request: Request, token: str) -> str:
    base = INVITE_BASE_URL
    if base:
        return f"{base}/invite/{token}"
    scheme = request.url.scheme
    host = request.headers.get("host", request.url.netloc)
    return f"{scheme}://{host}/invite/{token}"


@router.get("/invite/{token}", name="invite_accept", response_class=HTMLResponse)
async def invite_accept(
    request: Request,
    token: str,
    db: AsyncSession = Depends(get_db),
):
    invite = (await db.execute(select(Invite).where(Invite.token == token))).scalars().first()
    invalid = False
    email_for_tpl = None
    if not invite:
        invalid = True
    else:
        email_for_tpl = invite.email
        now = datetime.now(timezone.utc)
        exp = invite.expires_at
        if exp is not None and exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        used = invite.used_at
        if used is not None and used.tzinfo is None:
            used = used.replace(tzinfo=timezone.utc)
        if used or (exp and exp < now):
            invalid = True
    return templates.TemplateResponse(
        "invite_create_password.html",
        {
            "request": request,
            "user": None,
            "invalid": invalid,
            "email": email_for_tpl,
            "token": token,
        },
    )


@router.post("/invite/{token}", response_class=HTMLResponse)
async def invite_set_password(
    request: Request,
    token: str,
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    invite = (await db.execute(select(Invite).where(Invite.token == token))).scalars().first()
    now = datetime.now(timezone.utc)
    if not invite or invite.used_at or (invite.expires_at and invite.expires_at < now):
        return templates.TemplateResponse(
            "invite_create_password.html",
            {
                "request": request,
                "user": None,
                "invalid": True,
                "email": invite.email if invite else None,
            },
        )
    user = (await db.execute(select(User).where(User.email == invite.email))).scalars().first()
    if user:
        user.hashed_password = bcrypt.hash(password)
    else:
        user = User(
            email=invite.email,
            username=invite.email.split("@")[0],
            hashed_password=bcrypt.hash(password),
        )
        db.add(user)
        await db.flush()
    invite.used_at = now
    await db.commit()
    strategy = get_jwt_strategy()
    jwt_token = await strategy.write_token(user)
    response = RedirectResponse(url="/onboarding", status_code=303)
    response.set_cookie(
        key=cookie_transport.cookie_name,
        value=jwt_token,
        httponly=True,
        max_age=cookie_transport.cookie_max_age,
        secure=cookie_transport.cookie_secure,
        samesite=cookie_transport.cookie_samesite,
        path="/",
    )
    return response


@router.post("/invite")
async def send_invite(
    request: Request,
    email: str = Form(...),
    expiry_days: int = Form(7),
    db: AsyncSession = Depends(get_db),
):
    email_norm = email.strip().lower()
    invite = (
        await db.execute(select(Invite).where(func.lower(Invite.email) == email_norm))
    ).scalars().first()
    token = new_token()
    now = _now_tz()
    if invite:
        invite.token = token
        invite.expires_at = invite_expiry(expiry_days)
        invite.used_at = None
        invite.last_sent = now
        invite.sent_count = (invite.sent_count or 0) + 1
    else:
        invite = Invite(
            email=email_norm,
            token=token,
            expires_at=invite_expiry(expiry_days),
            last_sent=now,
            sent_count=1,
        )
        db.add(invite)
        await db.flush()
    await db.commit()
    link = _invite_url(request, invite.token)
    subject, text, html = render_invite_email(link)
    send_email(email_norm, subject, text, html)
    return RedirectResponse(url="/admin_dashboard?notice=Invite+sent", status_code=303)


@router.post("/resend_invite/{invite_id}")
async def resend_invite(
    request: Request,
    invite_id: int,
    db: AsyncSession = Depends(get_db),
):
    invite = (await db.execute(select(Invite).where(Invite.id == invite_id))).scalars().first()
    if not invite:
        return RedirectResponse(
            url="/admin_dashboard?notice=Invite+not+found",
            status_code=303,
        )
    invite.token = new_token()
    invite.expires_at = invite_expiry(INVITE_TTL_DAYS)
    invite.used_at = None
    invite.last_sent = _now_tz()
    invite.sent_count = (invite.sent_count or 0) + 1
    await db.commit()
    link = _invite_url(request, invite.token)
    subject, text_body, html_body = render_invite_email(link)
    send_email(invite.email, subject, text_body, html_body)
    return RedirectResponse(
        url="/admin_dashboard?notice=Invite+re-sent",
        status_code=303,
    )
