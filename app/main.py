import os, json, logging
from fastapi import FastAPI, Depends, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from .database import init_db, async_session_maker
from .routes import router
from .users import fastapi_users, auth_backend
from .models import User, Tag
from .utils import get_current_user
from sqlalchemy import select
from passlib.hash import bcrypt
from .schemas import UserRead, UserUpdate
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.exception_handlers import http_exception_handler as fastapi_http_exception_handler
from urllib.parse import quote

app = FastAPI(title="Memories App")
templates = Jinja2Templates(directory="templates")
# Static file serving
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Route Includes
# ----------------------
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
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
async def create_admin_user():
    from .models import User
    admin_email = os.getenv("ADMIN_EMAIL")
    admin_password = os.getenv("ADMIN_PASSWORD")
    admin_username = os.getenv("ADMIN_USERNAME", "admin")

    if not admin_email or not admin_password:
        print("⚠️ ADMIN_EMAIL or ADMIN_PASSWORD not set in .env, skipping admin creation.")
        return

    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.email == admin_email))
        existing_admin = result.scalars().first()
        if not existing_admin:
            user = User(
                email=admin_email,
                hashed_password=bcrypt.hash(admin_password),
                username=admin_username,
                is_superuser=True,
                is_active=True,
            )
            session.add(user)
            await session.commit()
            print(f"✅ Admin user created: {admin_email}")
        else:
            print(f"ℹ️ Admin user already exists: {admin_email}")
async def create_super_admin_user():
    super_email = os.getenv("SUPER_ADMIN_EMAIL")
    super_password = os.getenv("SUPER_ADMIN_PASSWORD")
    super_username = os.getenv("SUPER_ADMIN_USERNAME", "superadmin")

    if not super_email or not super_password:
        print("⚠️ SUPER_ADMIN_EMAIL or SUPER_ADMIN_PASSWORD not set. Skipping super admin creation.")
        return

    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.email == super_email))
        existing = result.scalars().first()

        if not existing:
            user = User(
                email=super_email,
                hashed_password=bcrypt.hash(super_password),
                username=super_username,
                is_superuser=True,
                super_admin=True,   # ✅ flag for dev/testing
                is_active=True
            )
            session.add(user)
            await session.commit()
            print(f"✅ Super admin created: {super_email}")
        else:
            print(f"ℹ️ Super admin already exists: {super_email}")

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
    await create_super_admin_user() 
    await _seed_tags_startup()
# ----------------------
# Home Redirect Route
# ----------------------


@app.get("/reset_password", response_class=HTMLResponse)
async def reset_password_page(request: Request):
    return templates.TemplateResponse("reset_password.html", {"request": request})

@app.get("/set_password", response_class=HTMLResponse)
async def set_password_page(request: Request):
    return templates.TemplateResponse("set_password.html", {"request": request})

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

#--------------------------------
#  WEEKLY PROMPT SCHEDUALER
#--------------------------------

# app/main.py (or wherever FastAPI app is created)
from app.services.scheduler import start_scheduler
@app.on_event("startup")
async def _startup():
    start_scheduler()
