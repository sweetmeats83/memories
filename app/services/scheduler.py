# app/services/scheduler.py
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from sqlalchemy import select
from app.models import User, WeeklyState
from app.services.utils_weekly import get_or_refresh_active_token, _now
from app.services.mailer import send_weekly_email
from app.database import async_session_maker
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # Fallback handled below

scheduler: AsyncIOScheduler | None = None

def start_scheduler():
    global scheduler
    if scheduler: return
    tz_name = os.getenv("APP_TZ") or os.getenv("TZ") or "UTC"
    tz = None
    if ZoneInfo:
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            # Fallbacks if tzdata doesn’t have the key
            try:
                tz = ZoneInfo("Etc/UTC")
            except Exception:
                tz = None  # Let APScheduler use its default
    scheduler = AsyncIOScheduler(timezone=tz)
    # Every weekday 9am local (adjust as desired)
    scheduler.add_job(job_weekly_send_scan, CronTrigger(day_of_week="mon-fri", hour=9, minute=0))
    scheduler.start()

async def job_weekly_send_scan():
    async with async_session_maker() as db:
        # Fetch users eligible to send. If a user has only on-deck, auto-promote it.
        users = (await db.execute(
            select(User).where(
                User.is_active == True,
                User.weekly_state.in_([WeeklyState.not_sent, WeeklyState.queued]),
                (User.weekly_current_prompt_id.isnot(None)) | (User.weekly_on_deck_prompt_id.isnot(None))
            )
        )).scalars().all()

        for u in users:
            # If no current but has on-deck, auto-promote on-deck to current
            if not u.weekly_current_prompt_id and u.weekly_on_deck_prompt_id:
                u.weekly_current_prompt_id = u.weekly_on_deck_prompt_id
                u.weekly_on_deck_prompt_id = None
                # mark as queued for traceability
                u.weekly_state = WeeklyState.queued
                u.weekly_queued_at = _now()

            if not u.weekly_current_prompt_id:
                # nothing to send for this user
                continue

            tok = await get_or_refresh_active_token(db, u.id, u.weekly_current_prompt_id)
            provider_id = await send_weekly_email(db, u, tok)
            u.weekly_state = WeeklyState.sent
            u.weekly_sent_at = _now()
            u.weekly_email_provider_id = provider_id

        await db.commit()

async def schedule_bulk_send(user_ids: list[int], when: datetime):
    # store as scheduled APScheduler date jobs
    for uid in user_ids:
        scheduler.add_job(job_send_one, DateTrigger(when), args=[uid])

async def job_send_one(user_id: int):
    async with async_session_maker() as db:
        u = await db.get(User, user_id)
        if not u:
            return
        # Auto-promote on-deck to current if needed
        if not u.weekly_current_prompt_id and u.weekly_on_deck_prompt_id:
            u.weekly_current_prompt_id = u.weekly_on_deck_prompt_id
            u.weekly_on_deck_prompt_id = None
            u.weekly_state = WeeklyState.queued
            u.weekly_queued_at = _now()
        if not u.weekly_current_prompt_id:
            return
        tok = await get_or_refresh_active_token(db, u.id, u.weekly_current_prompt_id)
        provider_id = await send_weekly_email(db, u, tok)
        u.weekly_state = WeeklyState.sent
        u.weekly_sent_at = _now()
        u.weekly_email_provider_id = provider_id
        await db.commit()
