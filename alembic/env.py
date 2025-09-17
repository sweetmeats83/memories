# alembic/env.py
from logging.config import fileConfig
import os
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# --- Make your app importable BEFORE importing app.* ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Import Base and ensure all models are registered on Base.metadata ---
from app.database import Base  # Base = declarative_base() defined here
from app import models         # import side-effect: registers ALL models on Base

# --- Alembic config / DB URL normalization (async -> sync for Alembic) ---
config = context.config

db_url = os.getenv("DATABASE_URL", "")
if db_url.startswith("postgresql+asyncpg"):
    db_url = db_url.replace("postgresql+asyncpg", "postgresql+psycopg2")
if not db_url:
    # fallback for local/docker compose networks
    db_url = "postgresql+psycopg2://postgres:postgres@postgres:5432/memories"

config.set_main_option("sqlalchemy.url", db_url)

# --- Logging (optional) ---
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# --- This is what Alembic inspects for autogenerate ---
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode'."""
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode'."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
	    transaction_per_migration=True, 
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
