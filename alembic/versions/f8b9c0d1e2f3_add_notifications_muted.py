"""add notifications_muted to user

Admin-controlled flag that stops scheduled weekly emails, pushes and daily
reminders for a user.

Revision ID: f8b9c0d1e2f3
Revises: e3f4a5b6c7d8
Create Date: 2026-10-25

"""
from alembic import op
import sqlalchemy as sa

revision = "f8b9c0d1e2f3"
down_revision = "e3f4a5b6c7d8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user",
        sa.Column("notifications_muted", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("user", "notifications_muted")
