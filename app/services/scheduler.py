# app/services/scheduler.py
import os
import glob
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from sqlalchemy import select
from app.models import User, WeeklyState
from app.services.utils_weekly import get_or_refresh_active_token, _now
from app.services.assignment import ensure_weekly_prompt
from app.services.mailer import send_weekly_email
from app.services.push import send_push_to_user, push_enabled
from app.models import PushSubscription
from app.database import async_session_maker
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # Fallback handled below

scheduler: AsyncIOScheduler | None = None
logger = logging.getLogger(__name__)

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

    # Schedule: configurable via WEEKLY_CRON (crontab format). Fallback: weekdays 09:00.
    cron_expr = (os.getenv("WEEKLY_CRON") or "").strip()
    try:
        if cron_expr:
            trigger = CronTrigger.from_crontab(cron_expr, timezone=tz)
            logger.info("Weekly scheduler using WEEKLY_CRON='%s' tz=%s", cron_expr, tz_name)
        else:
            trigger = CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone=tz)
            logger.info("Weekly scheduler using default Mon-Fri 09:00 tz=%s (set WEEKLY_CRON to override)", tz_name)
    except Exception:
        # On invalid crontab, fall back to default
        trigger = CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone=tz)
        logger.warning("Invalid WEEKLY_CRON; falling back to Mon-Fri 09:00")

    scheduler.add_job(job_weekly_send_scan, trigger)

    # Daily morning reminder — configurable via REMINDER_CRON, default 9 AM
    reminder_cron = (os.getenv("REMINDER_CRON") or "").strip()
    try:
        if reminder_cron:
            reminder_trigger = CronTrigger.from_crontab(reminder_cron, timezone=tz)
            logger.info("Daily reminder using REMINDER_CRON='%s' tz=%s", reminder_cron, tz_name)
        else:
            reminder_trigger = CronTrigger(hour=9, minute=0, timezone=tz)
            logger.info("Daily reminder using default 09:00 tz=%s (set REMINDER_CRON to override)", tz_name)
    except Exception:
        reminder_trigger = CronTrigger(hour=9, minute=0, timezone=tz)
        logger.warning("Invalid REMINDER_CRON; falling back to 09:00")
    scheduler.add_job(job_daily_reminder, reminder_trigger)

    # Clean up orphaned _tmp_* upload files every 6 hours
    scheduler.add_job(job_cleanup_tmp_uploads, CronTrigger(hour="*/6"))

    scheduler.start()
    logger.info("Weekly scheduler started")

async def job_weekly_send_scan():
    async with async_session_maker() as db:
        # Fetch users eligible to send. If a user has only on-deck, auto-promote it.
        now = _now()
        curr_iso = now.isocalendar()[:2]
        stale_cutoff = now - timedelta(days=5)
        users = (await db.execute(
            select(User).where(
                User.is_active == True,
                User.weekly_state.in_([
                    WeeklyState.not_sent,
                    WeeklyState.queued,
                    WeeklyState.sent,
                    WeeklyState.expired,
                ]),
                (User.weekly_current_prompt_id.isnot(None)) | (User.weekly_on_deck_prompt_id.isnot(None))
            )
        )).scalars().all()

        for u in users:
            # If we already sent this ISO week, don't resend yet.
            if u.weekly_state == WeeklyState.sent and u.weekly_sent_at:
                try:
                    sent_iso = u.weekly_sent_at.isocalendar()[:2]
                    if sent_iso == curr_iso:
                        continue
                except Exception:
                    if u.weekly_sent_at >= stale_cutoff:
                        continue
                # mark as ready to send again
                u.weekly_state = WeeklyState.queued
                u.weekly_queued_at = now

            if u.weekly_state == WeeklyState.expired:
                u.weekly_state = WeeklyState.queued
                u.weekly_queued_at = now

            # Ensure this user has a current (and on-deck) selection for this ISO week
            try:
                await ensure_weekly_prompt(db, u.id)
                await db.flush()
            except Exception:
                pass
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

            # Send push notification alongside email (best-effort)
            if push_enabled():
                try:
                    await send_push_to_user(
                        db, u.id,
                        title="Memories — new prompt waiting",
                        body="You have a new memory prompt ready to record.",
                        url=f"/respond/{tok}" if tok else "/",
                        tag="memories-weekly",
                    )
                except Exception:
                    logger.exception("Push notification failed for user %s", u.id)

        await db.commit()

_REMINDER_MESSAGES = [
    "Your memories are waiting — even a few minutes makes a difference.",
    "Someone in your family will treasure what only you can share.",
    "Take a few minutes today to record a memory.",
    "The stories that feel ordinary to you are priceless to the ones you love.",
    "You have a memory prompt ready — give it a go today.",
    "A little piece of your story is waiting to be told.",
    "Whenever you're ready, your memories are there to record.",
    "Even a short recording means the world to those who come after you.",
]

_reminder_index = 0


async def job_daily_reminder():
    """Send a gentle morning reminder push to users with an unrecorded active prompt."""
    global _reminder_index
    if not push_enabled():
        return

    async with async_session_maker() as db:
        # States where the prompt was sent but not yet recorded
        pending_states = (
            WeeklyState.sent,
            WeeklyState.queued,
            WeeklyState.opened,
            WeeklyState.clicked,
        )

        # Find users who have push subscriptions AND a pending prompt
        from sqlalchemy import exists
        users = (await db.execute(
            select(User).where(
                User.is_active == True,
                User.weekly_state.in_(pending_states),
                User.weekly_current_prompt_id.isnot(None),
                exists().where(PushSubscription.user_id == User.id),
            )
        )).scalars().all()

        if not users:
            return

        body = _REMINDER_MESSAGES[_reminder_index % len(_REMINDER_MESSAGES)]
        _reminder_index += 1

        for u in users:
            try:
                await send_push_to_user(
                    db, u.id,
                    title="Memories",
                    body=body,
                    url="/",
                    tag="memories-reminder",
                )
            except Exception:
                logger.exception("Daily reminder push failed for user %s", u.id)

        logger.info("Daily reminder sent to %d users", len(users))


async def job_cleanup_tmp_uploads():
    """Delete _tmp_* files in static/uploads older than 4 hours."""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static", "uploads")
    pattern = os.path.join(base, "_tmp_*")
    cutoff = datetime.utcnow().timestamp() - (4 * 3600)
    removed = 0
    for path in glob.glob(pattern):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except Exception:
            pass
    if removed:
        logger.info("Cleaned up %d orphaned _tmp_* upload files", removed)


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
        if push_enabled():
            try:
                await send_push_to_user(
                    db, u.id,
                    title="Memories — new prompt waiting",
                    body="You have a new memory prompt ready to record.",
                    url=f"/respond/{tok}" if tok else "/",
                    tag="memories-weekly",
                )
            except Exception:
                logger.exception("Push notification failed for user %s", u.id)
        await db.commit()

# ---- Admin-adjustable cron (runtime) ----
def _pick_tz(name: str | None):
    tz = None
    if ZoneInfo and name:
        try:
            tz = ZoneInfo(name)
        except Exception:
            try:
                tz = ZoneInfo("Etc/UTC")
            except Exception:
                tz = None
    return tz

def set_weekly_cron(days: list[int], hour: int, minute: int, tz_name: str | None = None) -> None:
    """Replace the weekly scan job using chosen days (1=Mon..7=Sun) and time.
    This affects the running process; to persist across restarts, set WEEKLY_CRON/APP_TZ.
    """
    if not scheduler:
        return
    ds = sorted({int(d) for d in (days or []) if 1 <= int(d) <= 7}) or [1,2,3,4,5]
    # compress to Cron day_of_week expression
    parts = []
    start = prev = None
    for d in ds:
        if start is None:
            start = prev = d
        elif d == prev + 1:
            prev = d
        else:
            parts.append(str(start) if start==prev else f"{start}-{prev}")
            start = prev = d
    if start is not None:
        parts.append(str(start) if start==prev else f"{start}-{prev}")
    dow = ",".join(parts)
    tz = _pick_tz(tz_name) or scheduler.timezone

    # Remove existing weekly scan jobs
    try:
        for j in list(scheduler.get_jobs()):
            func = getattr(j.func, "__name__", "")
            if func == "job_weekly_send_scan":
                scheduler.remove_job(j.id)
    except Exception:
        pass

    job = scheduler.add_job(job_weekly_send_scan, CronTrigger(day_of_week=dow, hour=hour, minute=minute, timezone=tz))
    logger.info("Updated weekly schedule: days=%s %02d:%02d tz=%s job=%s", dow, hour, minute, tz, job.id)
