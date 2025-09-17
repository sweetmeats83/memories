import os
from fastapi import Depends
from fastapi_users import FastAPIUsers
from fastapi_users.manager import BaseUserManager, IntegerIDMixin
from fastapi_users.authentication import CookieTransport, AuthenticationBackend, JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase
from .models import User
from .database import get_db

SECRET = os.getenv("SECRET", "CHANGE_ME_SECRET")

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
        print(f"✅ User {user.id} has registered.")

    async def on_after_forgot_password(self, user: User, token: str, request=None):
        print(f"🔑 Password reset requested for user {user.id}. Token: {token}")

    async def on_after_request_verify(self, user: User, token: str, request=None):
        print(f"✉️ Verification requested for user {user.id}. Token: {token}")

async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)

# -------------------------
# Authentication Backend
# -------------------------
cookie_transport = CookieTransport(
    cookie_name="session",
    cookie_max_age=3600 * 24,   # 1 day
    cookie_secure=False,        # Should be True in production (HTTPS)
    cookie_httponly=True
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


