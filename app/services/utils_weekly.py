# app/services/utils_weekly.py
import base64, hmac, hashlib, os, secrets
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.models import WeeklyToken, WeeklyTokenStatus, User, Prompt, WeeklyState

HMAC_KEY = os.getenv("WEEKLY_TOKEN_HMAC", "dev-secret-change-me").encode()

def _now():
    # Return naive UTC to match DB columns (TIMESTAMP WITHOUT TIME ZONE)
    # Store and compare consistently as UTC-naive
    return datetime.utcnow()

def new_token_str(user_id: int, prompt_id: int) -> str:
    # Keep token <= 64 chars to fit DB column. Use shorter random + truncated HMAC.
    # raw is ASCII; sig is binary truncated to 16 bytes.
    rnd = secrets.token_urlsafe(12)  # ~16 chars
    raw = f"{user_id}:{prompt_id}:{rnd}".encode()
    sig_full = hmac.new(HMAC_KEY, raw, hashlib.sha256).digest()
    sig = sig_full[:16]
    token = base64.urlsafe_b64encode(raw + b"." + sig).decode().rstrip("=")
    return token

def parse_token(token: str) -> tuple[int,int] | None:
    try:
        buf = token + "=" * (-len(token) % 4)
        data = base64.urlsafe_b64decode(buf.encode())
        raw, sig = data.rsplit(b".", 1)
        exp_full = hmac.new(HMAC_KEY, raw, hashlib.sha256).digest()
        # Support both full-length (32-byte) and truncated signatures
        if not hmac.compare_digest(exp_full[: len(sig)], sig):
            return None
        s = raw.decode()
        uid, pid, _rest = s.split(":", 2)
        return int(uid), int(pid)
    except Exception:
        return None

async def get_or_refresh_active_token(db: AsyncSession, user_id: int, prompt_id: int, ttl_days: int | None = None) -> WeeklyToken:
    """
    Ensure there is exactly one row per (user_id, prompt_id) as enforced by DB.
    - If it exists, refresh in-place to an active token (optionally new token string) and clear timeline fields.
    - If not, create it.
    """
    tok = (await db.execute(select(WeeklyToken).where(
        WeeklyToken.user_id == user_id,
        WeeklyToken.prompt_id == prompt_id,
    ))).scalars().first()

    expires = (_now() + timedelta(days=ttl_days)) if ttl_days else None

    if tok:
        # Refresh in-place: new token if previously used/expired or if expired by TTL.
        need_new = (
            (tok.expires_at and tok.expires_at < _now()) or
            (tok.status in (WeeklyTokenStatus.used, WeeklyTokenStatus.expired))
        )
        if need_new:
            tok.token = new_token_str(user_id, prompt_id)
        # Activate and clear previous timeline markers for a clean send
        tok.status = WeeklyTokenStatus.active
        tok.sent_at = None
        tok.opened_at = None
        tok.clicked_at = None
        tok.used_at = None
        tok.completed_at = None
        tok.expires_at = expires
        await db.flush()
        return tok

    # No row exists yet; create it
    tok = WeeklyToken(
        token=new_token_str(user_id, prompt_id),
        user_id=user_id,
        prompt_id=prompt_id,
        status=WeeklyTokenStatus.active,
        expires_at=expires
    )
    db.add(tok)
    await db.flush()
    return tok

async def expire_active_tokens(db: AsyncSession, user_id: int, prompt_id: int):
    tok = (await db.execute(select(WeeklyToken).where(
        WeeklyToken.user_id == user_id,
        WeeklyToken.prompt_id == prompt_id,
        WeeklyToken.status.in_([WeeklyTokenStatus.active, WeeklyTokenStatus.opened, WeeklyTokenStatus.clicked])
    ))).scalars().all()
    changed = False
    for t in tok:
        t.status = WeeklyTokenStatus.expired
        changed = True
    if changed:
        await db.flush()

async def mark_opened(db: AsyncSession, token_str: str):
    tok = (await db.execute(select(WeeklyToken).where(WeeklyToken.token == token_str))).scalars().first()
    if tok and tok.status in (WeeklyTokenStatus.active, WeeklyTokenStatus.opened):
        tok.status = WeeklyTokenStatus.opened
        tok.opened_at = _now()
        await db.flush()

async def mark_clicked(db: AsyncSession, token_str: str):
    tok = (await db.execute(select(WeeklyToken).where(WeeklyToken.token == token_str))).scalars().first()
    if not tok or (tok.expires_at and tok.expires_at < _now()) or tok.status == WeeklyTokenStatus.expired:
        return None
    tok.clicked_at = _now()
    await db.flush()
    return tok

async def mark_used(db: AsyncSession, token_str: str):
    tok = (await db.execute(select(WeeklyToken).where(WeeklyToken.token == token_str))).scalars().first()
    if not tok:
        return None
    if tok.expires_at and tok.expires_at < _now():
        tok.status = WeeklyTokenStatus.expired
        await db.flush()
        return None
    if tok.status not in (WeeklyTokenStatus.active, WeeklyTokenStatus.opened, WeeklyTokenStatus.clicked):
        return None
    tok.status = WeeklyTokenStatus.used
    tok.used_at = _now()
    await db.flush()
    return tok

async def mark_completed_and_close(db: AsyncSession, token_str: str):
    tok = (await db.execute(select(WeeklyToken).where(WeeklyToken.token == token_str))).scalars().first()
    if tok:
        tok.completed_at = _now()
        tok.status = WeeklyTokenStatus.used
        await db.flush()

    # Use the token row directly (no decode) to advance weekly state
    if not tok:
        return
    user = await db.get(User, tok.user_id)
    pid = tok.prompt_id
    if user and user.weekly_current_prompt_id == pid:
        user.weekly_completed_at = _now()
        next_id = None
        if getattr(user, 'weekly_on_deck_prompt_id', None):
            next_id = user.weekly_on_deck_prompt_id
            user.weekly_on_deck_prompt_id = None
        if not next_id:
            try:
                from app.services.assignment import get_on_deck_candidates
                ids = await get_on_deck_candidates(db, user.id, k=1)
                next_id = (ids or [None])[0]
            except Exception:
                next_id = None
        if next_id:
            user.weekly_current_prompt_id = next_id
            user.weekly_state = WeeklyState.queued
            user.weekly_queued_at = _now()
        else:
            user.weekly_current_prompt_id = None
            user.weekly_state = WeeklyState.not_sent
        await db.flush()