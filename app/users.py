import os
import logging
from fastapi import Depends
from fastapi_users import FastAPIUsers
from fastapi_users.manager import BaseUserManager, IntegerIDMixin
from fastapi_users.authentication import CookieTransport, AuthenticationBackend, JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase
from .models import User
from .database import get_db


logger = logging.getLogger(__name__)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


SECRET = os.getenv("SECRET", "").strip()
if not SECRET or SECRET == "CHANGE_ME_SECRET":
    raise RuntimeError(
        "SECRET environment variable must be set to a strong value; the default placeholder is not allowed."
    )

# -------------------------
# Database Dependency
# -------------------------
async def get_user_db(session=Depends(get_db)):
    yield SQLAlchemyUserDatabase(session, User)

# -------------------------
# User Manager
# -------------------------
class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def on_after_register(self, user: User, request=None):
        logger.info("User %s registered", user.id)

    async def on_after_forgot_password(self, user: User, token: str, request=None):
        logger.info("Password reset requested for user %s", user.id)

    async def on_after_request_verify(self, user: User, token: str, request=None):
        logger.info("Verification email requested for user %s", user.id)

async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)

# -------------------------
# Authentication Backend
# -------------------------
cookie_transport = CookieTransport(
    cookie_name="session",
    cookie_max_age=3600 * 24,
    cookie_secure=_bool_env("COOKIE_SECURE", default=False),
    cookie_httponly=True,
)

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600 * 24)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# -------------------------
# FastAPI Users instance
# -------------------------
fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_backend],
)

# Dependency to get currently active user
current_active_user = fastapi_users.current_user(active=True)


