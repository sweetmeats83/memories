from fastapi import Depends, HTTPException, status, Path, Request
from fastapi_users import models
import os, re, subprocess, shutil, smtplib, ssl, logging
from .users import fastapi_users
UPLOAD_DIR = "static/uploads"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
THUMB_SUBDIR = os.path.join(BASE_DIR, "static", "uploads", "thumbs")
os.makedirs(THUMB_SUBDIR, exist_ok=True)
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import quote

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}
logger = logging.getLogger(__name__)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def make_video_thumbnail(src_path: str, dst_path: str, time_sec: float = 0.25, size: int = 512) -> bool:
    """
    Capture a centered, padded square JPEG thumbnail from a video file.
    Returns True if the thumbnail was written, False otherwise.
    """
    if not has_ffmpeg():
        return False

    # -ss before -i = fast seek; pad to square to avoid layout shift
    vf = (
        f"scale='min({size},iw)':'min({size},ih)':force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=white"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(time_sec),
        "-i", src_path,
        "-vframes", "1",
        "-vf", vf,
        "-q:v", "3",             # good quality JPEG
        dst_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

# Dependency to get the currently authenticated user
async def get_current_user(user: models.UP = Depends(fastapi_users.current_user(optional=True))):
    """
    Returns the current logged-in user, or None if not authenticated.
    Also augments the user with `profile_avatar_url` when available so templates can render the avatar site-wide.
    """
    if not user:
        return user
    # Attach avatar url from UserProfile.privacy_prefs.avatar_url if present
    try:
        from sqlalchemy import select
        from .database import async_session_maker
        from .models import UserProfile
        async with async_session_maker() as s:
            prof = await s.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
            if prof and isinstance(prof.privacy_prefs, dict):
                rel = (prof.privacy_prefs or {}).get('avatar_url')
                if rel:
                    setattr(user, 'profile_avatar_url', f"/static/{str(rel).lstrip('/')}")
    except Exception:
        pass
    return user

# Dependency to enforce authentication (non-admin user is OK)
async def require_authenticated_user(user: models.UP = Depends(fastapi_users.current_user(active=True))):
    """
    Requires a logged-in user. Raises 401 if not authenticated.
    """
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


async def require_authenticated_html_user(
    request: Request,
    user: models.UP = Depends(fastapi_users.current_user(optional=True)),
):
    if not user:
        # build a relative “next” (path + query) to avoid open redirects
        path = request.url.path or "/"
        query = str(request.url.query or "")
        next_rel = f"{path}?{query}" if query else path
        # 303 redirect to login with next
        raise HTTPException(
            status_code=303,
            headers={"Location": f"/login?next={quote(next_rel, safe='')}"}
        )
    # Attach avatar url from UserProfile.privacy_prefs.avatar_url if present
    try:
        from sqlalchemy import select
        from .database import async_session_maker
        from .models import UserProfile
        async with async_session_maker() as s:
            prof = await s.scalar(select(UserProfile).where(UserProfile.user_id == user.id))
            if prof and isinstance(prof.privacy_prefs, dict):
                rel = (prof.privacy_prefs or {}).get('avatar_url')
                if rel:
                    setattr(user, 'profile_avatar_url', f"/static/{str(rel).lstrip('/')}")
    except Exception:
        pass
    return user

# Dependency to enforce admin access
async def require_admin_user(user: models.UP = Depends(fastapi_users.current_user(active=True))):
    """
    Requires a logged-in admin user. Raises 403 if user is not admin.
    """
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    if not getattr(user, "is_superuser", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user

async def require_super_admin(user: models.UP = Depends(require_authenticated_user)):
    if not user or not (user.is_superuser or getattr(user, "super_admin", False)):
        raise HTTPException(status_code=403, detail="Super admin access required")
    return user

def clean_filename(name: str) -> str:
    name = os.path.basename(name)  # strips any path info
    name = re.sub(r'[^\w\-_.]', '_', name)  # replace unsafe characters
    return name

def _read(path: Path, default: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default


def slugify(s: str) -> str:
    """
    Lowercase, trim, replace non [a-z0-9:/-] with '-', and collapse '-'.
    Keeps ':' so we can namespace labels like 'role:father'.
    """
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9:/\-]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")

def slug_person(s: str) -> str:
    return f"person:{slugify(s)}"

def slug_role(s: str) -> str:
    # Accept either raw role (e.g., 'grandmother') or fully-labeled 'role:grandmother'
    s = (s or "").strip().lower()
    if s.startswith("role:"):
        return slugify(s)
    return f"role:{slugify(s)}"

def slug_place(s: str) -> str:
    # Freeform, normalize to 'place:<slug>'
    s = (s or "").strip().lower()
    if s.startswith("place:"):
        return slugify(s)
    return f"place:{slugify(s)}"
