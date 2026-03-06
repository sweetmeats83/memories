from fastapi import APIRouter, Depends, Request, Form, Query, HTTPException, Response as FastResponse
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, func
from typing import List, Optional
import os
from datetime import datetime, timedelta
from pydantic import BaseModel
from app.database import get_db
from app.models import User, Prompt, WeeklyToken, WeeklyState, Response
from app.utils import require_admin_user, require_authenticated_user, require_authenticated_html_user
from app.services.utils_weekly import (
    get_or_refresh_active_token, _now, mark_opened, mark_clicked,
    mark_completed_and_close, expire_active_tokens,
)
from app.services.scheduler import schedule_bulk_send, set_weekly_cron
from app.services.mailer import send_weekly_email
from app.services.assignment import get_on_deck_candidates, skip_current_prompt as _skip
from app.llm_client import make_llm_followup_prompt
from app.services.chapter_compile import compile_chapter, chapter_status
from app.schemas import ChapterCompilationDTO, ChapterStatusDTO

router = APIRouter()
templates = Jinja2Templates(directory='templates')


class WeeklyRow(BaseModel):
    user_id: int
    display_name: str
    email: str
    current_prompt: Optional[dict] = None
    on_deck_prompt: Optional[dict] = None
    on_deck_candidates: Optional[List[dict]] = None
    state: str
    queued_at: Optional[str] = None
    sent_at: Optional[str] = None
    opened_at: Optional[str] = None
    clicked_at: Optional[str] = None
    used_at: Optional[str] = None
    completed_at: Optional[str] = None
    skipped_at: Optional[str] = None
    expires_at: Optional[str] = None
    token_status: Optional[str] = None
    token_link: Optional[str] = None

class WeeklyListResp(BaseModel):
    rows: List[WeeklyRow]
    total: int

def _user_dn(u: User) -> str:
    return (u.username or u.email or '').strip() or f'User {u.id}'

@router.get('/api/admin/weekly', response_model=WeeklyListResp)
async def admin_weekly_list(page: int=Query(1, ge=1), q: Optional[str]=Query(None), status: Optional[str]=Query(None), db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    PAGE = 25
    ofs = (page - 1) * PAGE
    qstmt = select(User).where(User.is_active == True)
    if q:
        like = f'%{q}%'
        qstmt = qstmt.where(or_(User.email.ilike(like), User.username.ilike(like)))
    if status:
        try:
            st = WeeklyState(status)
            qstmt = qstmt.where(User.weekly_state == st)
        except Exception:
            if status == 'expired':
                qstmt = qstmt.where(User.weekly_state == WeeklyState.expired)
    total = (await db.execute(select(func.count()).select_from(qstmt.subquery()))).scalar_one()
    users = (await db.execute(qstmt.order_by(User.id.asc()).offset(ofs).limit(PAGE))).scalars().all()
    rows = []
    for u in users:
        cur = await db.get(Prompt, u.weekly_current_prompt_id) if u.weekly_current_prompt_id else None
        od = await db.get(Prompt, u.weekly_on_deck_prompt_id) if u.weekly_on_deck_prompt_id else None
        tok = None
        if cur:
            tok = (await db.execute(select(WeeklyToken).where(WeeklyToken.user_id == u.id, WeeklyToken.prompt_id == cur.id))).scalars().first()
        token_link = None
        token_status = None
        expires_iso = None
        if tok:
            token_status = tok.status.value
            expires_iso = tok.expires_at.isoformat() if tok.expires_at else None
            token_link = f'/weekly/t/{tok.token}'
        cand_ids: list[int] = []
        try:
            cand_ids = await get_on_deck_candidates(db, u.id, k=5)
        except Exception:
            cand_ids = []
        cands: list[dict] = []
        if cand_ids:
            prs = (await db.execute(select(Prompt).where(Prompt.id.in_(cand_ids)))).unique().scalars().all()
            pmap = {p.id: p for p in prs}
            for pid in cand_ids:
                p = pmap.get(pid)
                if p:
                    cands.append({'id': p.id, 'title': p.text})
        rows.append(WeeklyRow(user_id=u.id, display_name=_user_dn(u), email=u.email, current_prompt={'id': cur.id, 'title': cur.text, 'tags': [t.slug for t in cur.tags or []]} if cur else None, on_deck_prompt={'id': od.id, 'title': od.text} if od else None, on_deck_candidates=cands, state=u.weekly_state.value, queued_at=u.weekly_queued_at.isoformat() if u.weekly_queued_at else None, sent_at=u.weekly_sent_at.isoformat() if u.weekly_sent_at else None, opened_at=u.weekly_opened_at.isoformat() if u.weekly_opened_at else None, clicked_at=u.weekly_clicked_at.isoformat() if u.weekly_clicked_at else None, used_at=u.weekly_used_at.isoformat() if u.weekly_used_at else None, completed_at=u.weekly_completed_at.isoformat() if u.weekly_completed_at else None, skipped_at=u.weekly_skipped_at.isoformat() if u.weekly_skipped_at else None, expires_at=expires_iso, token_status=token_status, token_link=token_link))
    return WeeklyListResp(rows=rows, total=total)

class Ids(BaseModel):
    user_ids: Optional[List[int]] = None
    user_id: Optional[int] = None

@router.post('/api/admin/weekly/send')
async def admin_weekly_send(payload: Ids, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    ids = payload.user_ids or ([payload.user_id] if payload.user_id else [])
    sent = 0
    for uid in ids:
        u = await db.get(User, uid)
        if not u or not u.weekly_current_prompt_id:
            continue
        tok = await get_or_refresh_active_token(db, uid, u.weekly_current_prompt_id)
        await db.commit()
        provider_id = await send_weekly_email(db, u, tok)
        u.weekly_state = WeeklyState.sent
        u.weekly_sent_at = _now()
        u.weekly_email_provider_id = provider_id
        sent += 1
    await db.commit()
    return {'ok': True, 'sent': sent}

class ScheduleReq(BaseModel):
    user_ids: List[int]
    when: datetime

@router.post('/api/admin/weekly/schedule')
async def admin_weekly_schedule(payload: ScheduleReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    await schedule_bulk_send(payload.user_ids, payload.when)
    return {'ok': True, 'scheduled': len(payload.user_ids)}

class RecurringReq(BaseModel):
    user_ids: List[int]
    days: List[int]
    hour: int
    minute: int
    weeks: int = 12

@router.post('/api/admin/weekly/schedule_recurring')
async def admin_weekly_schedule_recurring(payload: RecurringReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    tz_name = os.getenv('APP_TZ') or os.getenv('TZ') or 'UTC'
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = None
    today = datetime.now(tz).date() if tz else datetime.utcnow().date()
    monday = today - timedelta(days=today.weekday())
    total = 0
    for w in range(max(1, min(payload.weeks, 52))):
        week_start = monday + timedelta(days=7 * w)
        for d in sorted(set((int(x) for x in payload.days if 1 <= int(x) <= 7))):
            target_date = week_start + timedelta(days=d - 1)
            dt_local = datetime.combine(target_date, datetime.min.time()).replace(hour=payload.hour, minute=payload.minute)
            dt = dt_local.replace(tzinfo=tz) if tz else dt_local
            now_cmp = datetime.now(tz) if tz else datetime.utcnow()
            if dt <= now_cmp:
                continue
            await schedule_bulk_send(payload.user_ids, dt)
            total += 1
    return {'ok': True, 'scheduled_windows': total}

class SwapReq(BaseModel):
    user_id: int
    make_on_deck_current: bool = True

@router.post('/api/admin/weekly/swap')
async def admin_weekly_swap(payload: SwapReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    u = await db.get(User, payload.user_id)
    if not u:
        raise HTTPException(404, 'User not found')
    if not u.weekly_on_deck_prompt_id:
        return {'ok': False, 'error': 'No on-deck prompt'}
    prev = u.weekly_current_prompt_id
    u.weekly_current_prompt_id, u.weekly_on_deck_prompt_id = (u.weekly_on_deck_prompt_id, u.weekly_current_prompt_id)
    u.weekly_state = WeeklyState.queued
    u.weekly_queued_at = _now()
    if prev:
        await expire_active_tokens(db, u.id, prev)
    await db.commit()
    return {'ok': True}

class ChooseReq(BaseModel):
    user_id: int
    prompt_id: int
    push_previous_to_on_deck: bool = True

@router.post('/api/admin/weekly/choose')
async def admin_weekly_choose(payload: ChooseReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    u = await db.get(User, payload.user_id)
    p = await db.get(Prompt, payload.prompt_id)
    if not u or not p:
        raise HTTPException(404, 'User or Prompt not found')
    prev = u.weekly_current_prompt_id
    u.weekly_current_prompt_id = p.id
    u.weekly_state = WeeklyState.queued
    u.weekly_queued_at = _now()
    if payload.push_previous_to_on_deck:
        u.weekly_on_deck_prompt_id = prev
    if prev:
        await expire_active_tokens(db, u.id, prev)
    await db.commit()
    return {'ok': True}

class CronReq(BaseModel):
    days: List[int]
    hour: int
    minute: int
    tz: Optional[str] = None

@router.post('/api/admin/weekly/cron')
async def admin_weekly_update_cron(payload: CronReq, admin=Depends(require_admin_user)):
    set_weekly_cron(payload.days, payload.hour, payload.minute, payload.tz)
    return {'ok': True}

class QueueReq(BaseModel):
    user_id: int
    prompt_id: int

@router.post('/api/admin/weekly/queue')
async def admin_weekly_queue(payload: QueueReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    u = await db.get(User, payload.user_id)
    p = await db.get(Prompt, payload.prompt_id)
    if not u or not p:
        raise HTTPException(404, 'User or Prompt not found')
    u.weekly_on_deck_prompt_id = p.id
    if u.weekly_state == WeeklyState.not_sent:
        u.weekly_state = WeeklyState.queued
        u.weekly_queued_at = _now()
    await db.commit()
    return {'ok': True}

class SkipReq(BaseModel):
    user_id: int

@router.post('/api/admin/weekly/skip')
async def admin_weekly_skip(payload: SkipReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    await _skip(db, payload.user_id)
    await db.commit()
    return {'ok': True}

class FollowupReq(BaseModel):
    user_id: int
    response_id: int
    style: str = 'gentle'
    max_tokens: int = 300

@router.post('/api/admin/weekly/followup')
async def admin_weekly_followup(payload: FollowupReq, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    prompt_id = await make_llm_followup_prompt(db, payload.user_id, payload.response_id, payload.style, payload.max_tokens)
    u = await db.get(User, payload.user_id)
    u.weekly_on_deck_prompt_id = prompt_id
    await db.commit()
    return {'ok': True, 'on_deck_prompt_id': prompt_id}

@router.get('/api/admin/weekly/candidates')
async def admin_weekly_candidates(user_id: int=Query(...), k: int=Query(10, ge=1, le=50), db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    ids = await get_on_deck_candidates(db, user_id, k=k)
    if not ids:
        return {'items': []}
    prs = (await db.execute(select(Prompt).where(Prompt.id.in_(ids)))).unique().scalars().all()
    pmap = {p.id: p for p in prs}
    out = []
    for pid in ids:
        p = pmap.get(pid)
        if p:
            out.append({'id': p.id, 'text': p.text})
    return {'items': out}

@router.get('/api/admin/weekly/context')
async def admin_weekly_context(user_id: int=Query(...), db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    u = await db.get(User, user_id)
    if not u:
        raise HTTPException(404, 'User not found')
    cur = await db.get(Prompt, u.weekly_current_prompt_id) if u.weekly_current_prompt_id else None
    lr = (await db.execute(select(Response).where(Response.user_id == user_id).order_by(Response.created_at.desc()).limit(1))).unique().scalars().first()
    return {'current_prompt': {'id': cur.id, 'text': cur.text} if cur else None, 'last_response': {'id': getattr(lr, 'id', None), 'text': getattr(lr, 'response_text', None) or getattr(lr, 'transcription', None)} if lr else None}

@router.post('/api/admin/weekly/copy-link')
async def admin_weekly_copy_link(payload: Ids, db: AsyncSession=Depends(get_db), admin=Depends(require_admin_user)):
    if not payload.user_id:
        raise HTTPException(400, 'user_id required')
    u = await db.get(User, payload.user_id)
    if not u or not u.weekly_current_prompt_id:
        raise HTTPException(404, 'No current prompt for user')
    tok = await get_or_refresh_active_token(db, u.id, u.weekly_current_prompt_id)
    await db.commit()
    return {'ok': True, 'link': f'/weekly/t/{tok.token}'}

def _wants_json(request: Request) -> bool:
    qp = request.query_params
    if qp.get('format') == 'json' or qp.get('json') == '1':
        return True
    accept = (request.headers.get('accept') or '').lower()
    return 'application/json' in accept and 'text/html' not in accept

@router.get('/weekly/t/{token}', include_in_schema=False)
async def weekly_token_click(token: str, request: Request, db: AsyncSession=Depends(get_db)):
    if request.method == 'HEAD':
        return FastResponse(status_code=204)
    tok = await mark_clicked(db, token)
    await db.commit()
    if not tok:
        if _wants_json(request):
            return JSONResponse({'ok': False, 'error': 'invalid_or_expired'}, status_code=400)
        return RedirectResponse(url='/login?notice=This+weekly+link+is+invalid+or+expired', status_code=303)
    return RedirectResponse(url=f'/user_record/{tok.prompt_id}?token={token}', status_code=303)

@router.get('/weekly/t/{token}.png')
async def weekly_token_pixel(token: str, db: AsyncSession=Depends(get_db)):
    await mark_opened(db, token)
    await db.commit()
    png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc``\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82'
    return FastResponse(content=png, media_type='image/png', headers={'Cache-Control': 'no-store, max-age=0', 'Pragma': 'no-cache', 'Expires': '0'})

@router.post('/weekly/token/use')
async def weekly_token_use(token: str=Form(...), db: AsyncSession=Depends(get_db)):
    tok = await mark_clicked(db, token)
    if not tok:
        await mark_opened(db, token)
        tok = (await db.execute(select(WeeklyToken).where(WeeklyToken.token == token))).scalars().first()
    await db.commit()
    if not tok:
        raise HTTPException(400, 'Token invalid or expired')
    return {'ok': True, 'user_id': tok.user_id, 'prompt_id': tok.prompt_id}

@router.get('/api/weekly/{user_id}/on_deck')
async def api_weekly_on_deck(user_id: int, k: int=Query(5, ge=1, le=25), admin=Depends(require_admin_user), db: AsyncSession=Depends(get_db)):
    ids = await get_on_deck_candidates(db, user_id, k=k)
    if not ids:
        return []
    rows = (await db.execute(select(Prompt).where(Prompt.id.in_(ids)))).unique().scalars().all()
    by_id = {p.id: p for p in rows}
    out = []
    for pid in ids:
        p = by_id.get(pid)
        if p:
            out.append({'id': p.id, 'text': p.text, 'chapter': p.chapter})
    return out

@router.get('/chapter/{chapter_id}', response_class=HTMLResponse)
async def chapter_view(chapter_id: str, request: Request, user=Depends(require_authenticated_html_user), db: AsyncSession=Depends(get_db)):
    stat = await chapter_status(db, chapter_id, user.id)
    ctx = {'request': request, 'user': user, 'chapter_key': stat.chapter, 'display_name': stat.display_name, 'ready': stat.ready, 'missing_prompts': stat.missing_prompts, 'latest_compilation': stat.latest_compilation.dict() if stat.latest_compilation else None}
    return templates.TemplateResponse('chapter_view.html', ctx)

@router.get('/api/chapter/{chapter_id}/status', response_model=ChapterStatusDTO)
async def api_chapter_status(chapter_id: str, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    return await chapter_status(db, chapter_id, user.id)

@router.get('/api/chapter/{chapter_id}/gaps', response_model=list[dict])
async def api_chapter_gaps(chapter_id: str, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    stat = await chapter_status(db, chapter_id, user.id)
    latest = stat.latest_compilation
    return [g.dict() for g in (latest.gap_questions if latest else [])]

@router.post('/api/chapter/{chapter_id}/compile', response_model=ChapterCompilationDTO)
async def api_chapter_compile(chapter_id: str, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    stat = await chapter_status(db, chapter_id, user.id)
    if not stat.ready:
        raise HTTPException(status_code=400, detail='Chapter is not ready: complete all assigned prompts.')
    dto = await compile_chapter(db, chapter_id, user.id)
    return dto

@router.post('/api/chapter/{chapter_id}/publish')
async def api_chapter_publish(chapter_id: str, user=Depends(require_authenticated_user), db: AsyncSession=Depends(get_db)):
    from sqlalchemy import update, desc
    from app.models import ChapterCompilation
    latest = (await db.execute(select(ChapterCompilation).where((ChapterCompilation.user_id == user.id) & (ChapterCompilation.chapter == chapter_id)).order_by(ChapterCompilation.version.desc(), ChapterCompilation.created_at.desc()).limit(1))).scalars().first()
    if not latest:
        raise HTTPException(status_code=404, detail='No compilation to publish.')
    latest.status = 'published'
    await db.commit()
    return {'ok': True, 'id': latest.id, 'version': latest.version}
