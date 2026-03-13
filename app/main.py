import os, json, logging
from fastapi import FastAPI, Depends, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from .database import init_db, async_session_maker
from .routes import router
from .routers.user import router as user_router
from .routers.responses import router as responses_router
from .routers.admin_prompts import router as admin_prompts_router
from .routers.invites import router as invites_router
from .routers.people import router as people_router
from .routers.weekly import router as weekly_router
from .routers.upload import router as upload_router
from .routers.onboarding import router as onboarding_router
from .routers.admin_tags import router as admin_tags_router
from .routers.admin_responses import router as admin_responses_router
from .routers.push import router as push_router
from .users import fastapi_users, auth_backend
from .models import User, Tag
from .utils import get_current_user
from sqlalchemy import select
import bcrypt as _bcrypt_lib
from .schemas import UserRead, UserUpdate
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.exception_handlers import http_exception_handler as fastapi_http_exception_handler
from urllib.parse import quote
from app.services.scheduler import start_scheduler


logger = logging.getLogger(__name__)

def _parse_cors_origins() -> list[str]:
    """Pull an explicit allowlist from env instead of using `*`."""
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if not raw:
        # Keep sensible defaults for local development while forcing explicit origins in prod.
        return [
            "http://localhost:8000",
            "http://localhost:8003",
            "http://127.0.0.1:8000",
        ]
    # Support either comma-separated list or JSON array for convenience.
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            return [str(o).strip() for o in parsed if str(o).strip()]
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in CORS_ALLOW_ORIGINS; falling back to CSV parse")
    return [seg.strip() for seg in raw.split(",") if seg.strip()]


app = FastAPI(title="Memories App")
templates = Jinja2Templates(directory="templates")

# Static file serving
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Route Includes
# ----------------------
app.include_router(upload_router)
app.include_router(user_router)
app.include_router(responses_router)
app.include_router(people_router)
app.include_router(weekly_router)
app.include_router(onboarding_router)
app.include_router(admin_tags_router)
app.include_router(admin_responses_router)
app.include_router(push_router)
app.include_router(admin_prompts_router)
app.include_router(invites_router)
app.include_router(router)

# Authentication Routes
app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"]
)

app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"]
)
# Allow log level override via LOG_LEVEL while respecting existing handlers
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
if not logging.getLogger().handlers:
    logging.basicConfig(level=LOG_LEVEL)
logger.setLevel(LOG_LEVEL)

# -----------------------------------------------------
# Redirect unauthenticated HTML requests to /login
# Applies to 401/403 for non-API, HTML page requests.
# -----------------------------------------------------
@app.exception_handler(FastAPIHTTPException)
async def _auth_redirect_handler(request: Request, exc: FastAPIHTTPException):
    try:
        path = request.url.path or "/"
        accept = (request.headers.get("accept") or "").lower()
        is_html = "text/html" in accept or accept == "*/*"
        is_api = path.startswith("/api") or path.startswith("/auth") or path.startswith("/users")
        is_login_related = path.startswith("/login") or path.startswith("/register")
        # Only redirect browser page requests
        if exc.status_code in (401, 403) and is_html and not is_api and not is_login_related:
            query = str(request.url.query or "")
            next_rel = f"{path}?{query}" if query else path
            return RedirectResponse(url=f"/login?next={quote(next_rel, safe='')}", status_code=303)
    except Exception:
        # Fall through to default handler on any error
        pass
    return await fastapi_http_exception_handler(request, exc)
# ----------------------
# Auto-create admin user
# ----------------------
async def ensure_family_group():
    """Create the one implicit family group and add all active users to it."""
    from .models import KinGroup, KinMembership
    async with async_session_maker() as session:
        group = (await session.execute(
            select(KinGroup).where(KinGroup.kind == 'family').limit(1)
        )).scalars().first()
        if not group:
            group = KinGroup(name='Family', kind='family')
            session.add(group)
            await session.flush()
        all_user_ids = (await session.execute(
            select(User.id).where(User.is_active.is_(True))
        )).scalars().all()
        existing = set((await session.execute(
            select(KinMembership.user_id).where(KinMembership.group_id == group.id)
        )).scalars().all())
        for uid in all_user_ids:
            if uid not in existing:
                session.add(KinMembership(group_id=group.id, user_id=uid, role='member'))
        await session.commit()


async def create_admin_user():
    from .models import User
    admin_email = os.getenv("ADMIN_EMAIL")
    admin_password = os.getenv("ADMIN_PASSWORD")
    admin_username = os.getenv("ADMIN_USERNAME", "admin")

    if not admin_email or not admin_password:
        logger.warning("ADMIN_EMAIL or ADMIN_PASSWORD not set; skipping admin bootstrap")
        return

    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.email == admin_email))
        existing_admin = result.scalars().first()
        if not existing_admin:
            user = User(
                email=admin_email,
                hashed_password=_bcrypt_lib.hashpw(admin_password.encode(), _bcrypt_lib.gensalt()).decode(),
                username=admin_username,
                is_superuser=True,
                is_active=True,
            )
            session.add(user)
            await session.commit()
            logger.info("Admin user created for %s", admin_email)
        else:
            logger.info("Admin user already exists for %s", admin_email)

async def _seed_tags_from_whitelist(db: AsyncSession, path: str) -> dict:
    """
    Upsert tags from a JSON whitelist.
    Supports either:
      [{"value":"life:adult","label":"Life · Adult"}, ...]  or
      ["life:adult","topic:travel", ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    inserted = updated = skipped = 0

    # Quick guard: if Tag table isn't ready, skip seeding (avoids startup crash)
    try:
        await db.execute(select(Tag.id).limit(1))
    except Exception:
        logging.warning("Tag table not available yet; skipping whitelist seed.")
        return {"inserted": 0, "updated": 0, "skipped": 0}

    for it in items:
        if isinstance(it, str):
            slug = (it or "").strip().lower()
            name = slug.replace("-", " ").title()
        else:
            slug = (it.get("value") or "").strip().lower()
            name = (it.get("label") or slug.replace("-", " ").title()).strip()

        if not slug:
            continue

        existing = (await db.execute(select(Tag).where(Tag.slug == slug))).scalar_one_or_none()
        if existing:
            if (existing.name or "").strip() != name:
                existing.name = name
                updated += 1
            else:
                skipped += 1
            continue

        db.add(Tag(name=name, slug=slug))
        try:
            await db.flush()
            inserted += 1
        except IntegrityError:
            await db.rollback()
            skipped += 1

    await db.commit()
    return {"inserted": inserted, "updated": updated, "skipped": skipped}

async def _seed_tags_startup():
    # Default to app/data/tag_whitelist.json (override with TAG_WHITELIST_PATH env var)
    default_path = os.path.join(os.path.dirname(__file__), "data", "tag_whitelist.json")
    path = os.getenv("TAG_WHITELIST_PATH", default_path)

    if not os.path.exists(path):
        logging.info("No tag whitelist found at %s; skipping.", path)
        return

    try:
        async with async_session_maker() as db:
            res = await _seed_tags_from_whitelist(db, path)
            logging.info("Tag whitelist seed complete: %s", res)
    except Exception:
        logging.exception("Tag whitelist seed failed")

@app.on_event("startup")
async def on_startup():
    from . import models  # Required for SQLAlchemy model detection
    await init_db()
    await create_admin_user()
    await ensure_family_group()
    await _seed_tags_startup()
    start_scheduler()
# ----------------------
# Home Redirect Route
# ----------------------


@app.get("/sw.js")
async def service_worker():
    """Serve the service worker from root scope (required for full PWA push scope)."""
    from fastapi.responses import FileResponse
    return FileResponse("static/sw.js", media_type="application/javascript")


@app.get("/manifest.json")
async def pwa_manifest():
    """Serve the PWA web app manifest."""
    from fastapi.responses import FileResponse
    return FileResponse("static/manifest.json", media_type="application/manifest+json")


@app.get("/reset_password", response_class=HTMLResponse)
async def reset_password_page(request: Request):
    return templates.TemplateResponse("reset_password.html", {"request": request})

@app.get("/set_password", response_class=HTMLResponse)
async def set_password_page(request: Request):
    return templates.TemplateResponse("set_password.html", {"request": request})

@app.get("/health")
async def health_check():
    """Lightweight liveness probe used by Docker healthcheck and uptime monitors."""
    return {"status": "ok"}


@app.get("/")
async def root_redirect(user: User = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(
        url="/admin_dashboard" if user.is_superuser else "/user_dashboard",
        status_code=303,
    )
    
@app.get("/logout")
def app_logout(response: Response):
    # Clear auth cookies robustly (host-only and domain cookies)
    from .users import cookie_transport
    resp = RedirectResponse(url="/login")
    name = getattr(cookie_transport, "cookie_name", "session")
    domain = getattr(cookie_transport, "cookie_domain", None)

    # Clear session cookie
    resp.delete_cookie(name, path="/")
    if domain:
        resp.delete_cookie(name, path="/", domain=domain)
    # Belt and suspenders: explicit expire
    resp.set_cookie(
        key=name,
        value="",
        max_age=0,
        expires=0,
        httponly=True,
        secure=getattr(cookie_transport, "cookie_secure", False),
        samesite=getattr(cookie_transport, "cookie_samesite", "lax"),
        path="/",
        domain=domain,
    )
    # Clear impersonation marker if present
    for k in ("imp",):
        resp.delete_cookie(k, path="/")
        if domain:
            resp.delete_cookie(k, path="/", domain=domain)
        resp.set_cookie(k, value="", max_age=0, expires=0, httponly=True,
                        secure=getattr(cookie_transport, "cookie_secure", False),
                        samesite=getattr(cookie_transport, "cookie_samesite", "lax"),
                        path="/", domain=domain)
    return resp

