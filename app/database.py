from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import asyncio
import os

raw_url = os.getenv("DATABASE_URL", "")
if raw_url.startswith("postgresql+psycopg"):
    # if someone provided a sync URL by mistake, upgrade it to async
    DATABASE_URL = raw_url.replace("postgresql+psycopg", "postgresql+asyncpg")
else:
    DATABASE_URL = raw_url or "postgresql+asyncpg://postgres:postgres@postgres:5432/memories"


engine = create_async_engine(DATABASE_URL, echo=False, future=True)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)
Base = declarative_base()

async def get_db():
    async with async_session_maker() as session:
        yield session

async def init_db():
    # Only run create_all in dev, never in prod with Alembic
    if os.getenv("RUN_DB_CREATE_ALL") == "1":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

