from fastapi import Depends, HTTPException, Path, Request, status
from fastapi_users import models
import logging
import os
from pathlib import Path
import re
import shutil
import smtplib
import ssl
import subprocess
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import quote

from .users import fastapi_users

UPLOAD_DIR = "static/uploads"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
THUMB_SUBDIR = os.path.join(BASE_DIR, "static", "uploads", "thumbs")
os.makedirs(THUMB_SUBDIR, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}
logger = logging.getLogger(__name__)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def make_video_thumbnail(src_path: str, dst_path: str, time_sec: float = 0.25, size: int = 512) -> bool:
    """Capture a centered, padded square JPEG thumbnail from a video file."""
    if not has_ffmpeg():
        return False

    vf = (
        f"scale='min({size},iw)':'min({size},ih)':force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=white"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(time_sec),
        "-i",
        src_path,
        "-vframes",
        "1",
        "-vf",
        vf,
        "-q:v",
        "3",
        dst_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


async def _attach_profile_avatar(user: models.UP | None) -> None:
    """Populate `profile_avatar_url` from the user's profile when available."""
    if not user or getattr(user, "profile_avatar_url", None):
        return
    try:
        from sqlalchemy import select

        from .database import async_session_maker
        from .models import UserProfile

        async with async_session_maker() as session:
            profile = await session.scalar(
                select(UserProfile).where(UserProfile.user_id == user.id)
            )
            if profile and isinstance(profile.privacy_prefs, dict):
                rel = (profile.privacy_prefs or {}).get("avatar_url")
                if rel:
                    rel_str = str(rel).lstrip("/")
                    if rel_str.startswith("http"):
                        setattr(user, "profile_avatar_url", rel_str)
                    else:
                        setattr(user, "profile_avatar_url", f"/static/{rel_str}")
    except Exception:
        # Avatar is best-effort only; ignore lookup issues.
        pass


# Dependency to get the currently authenticated user
async def get_current_user(
    user: models.UP = Depends(fastapi_users.current_user(optional=True)),
):
    """Return the current user (if any) enriched with avatar metadata."""
    if not user:
        return user
    await _attach_profile_avatar(user)
    return user


# Dependency to enforce authentication (non-admin user is OK)
async def require_authenticated_user(
    user: models.UP = Depends(fastapi_users.current_user(active=True)),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    await _attach_profile_avatar(user)
    return user


async def require_authenticated_html_user(
    request: Request,
    user: models.UP = Depends(fastapi_users.current_user(optional=True)),
):
    if not user:
        path = request.url.path or "/"
        query = str(request.url.query or "")
        next_rel = f"{path}?{query}" if query else path
        raise HTTPException(
            status_code=303,
            headers={"Location": f"/login?next={quote(next_rel, safe='')}"},
        )
    await _attach_profile_avatar(user)
    return user


async def require_admin_user(
    user: models.UP = Depends(fastapi_users.current_user(active=True)),
):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    if not getattr(user, "is_superuser", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    await _attach_profile_avatar(user)
    return user


async def require_super_admin(user: models.UP = Depends(require_authenticated_user)):
    if not user or not (user.is_superuser or getattr(user, "super_admin", False)):
        raise HTTPException(status_code=403, detail="Super admin access required")
    await _attach_profile_avatar(user)
    return user


def clean_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^\w\-_.]", "_", name)
    return name


def _read(path: Path, default: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9:/\-]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def slug_person(s: str) -> str:
    return f"person:{slugify(s)}"


def slug_role(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith("role:"):
        return slugify(s)
    return f"role:{slugify(s)}"


def slug_place(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith("place:"):
        return slugify(s)
    return f"place:{slugify(s)}"
