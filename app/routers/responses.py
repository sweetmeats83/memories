from __future__ import annotations

import os
import shutil
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy import delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.background import run_sync, spawn
from app.database import get_db, async_session_maker
from app.routes_shared import PIPELINE, templates
from app.models import (
    Prompt,
    PromptMedia,
    Response,
    ResponseSegment,
    ResponseVersion,
    SupportingMedia,
    Tag,
    User,
    UserPrompt,
)
from app.schemas import ResponseSegmentRead, ReorderSegmentsRequest
from app.services.assignment import ensure_weekly_prompt
from app.services.auto_tag import suggest_tags_rule_based
from app.services.utils_weekly import mark_completed_and_close, mark_clicked, mark_opened, mark_used
from app.transcription import enrich_after_transcription
from app.utils import get_current_user, require_authenticated_html_user, require_authenticated_user

router = APIRouter()


@router.get("/response/{response_id}", response_class=HTMLResponse, name="response_view")
async def response_view(
    response_id: int,
    request: Request,
    user=Depends(require_authenticated_html_user),
    db: AsyncSession = Depends(get_db),
):
    resp = (
        await db.execute(
            select(Response)
            .options(selectinload(Response.prompt))
            .where(Response.id == response_id, Response.user_id == user.id)
        )
    ).scalars().first()
    if not resp:
        raise HTTPException(status_code=404, detail="Response not found")

    return templates.TemplateResponse(
        "response_view.html",
        {"request": request, "user": user, "response": resp},
    )


# Additional response-related endpoints will be moved here in subsequent steps.

__all__ = ["router"]
