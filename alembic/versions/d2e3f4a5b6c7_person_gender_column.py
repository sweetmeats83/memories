"""add gender column to person

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
Branch labels: None
Depends on: None

Create Date: 2026-05-08
"""
from alembic import op
import sqlalchemy as sa

revision = "d2e3f4a5b6c7"
down_revision = "c1d2e3f4a5b6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("person", sa.Column("gender", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("person", "gender")
