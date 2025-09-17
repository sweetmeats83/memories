"""Add UserWeeklyPrompt and UserWeeklySkip tables

Revision ID: 7520bb31a20a
Revises: f2133ddcec34
Create Date: 2025-08-13 04:30:41.942842

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "7520bb31a20a"
down_revision = "f2133ddcec34"  # keep whatever your previous good rev is
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "user_weekly_prompts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("week", sa.Integer(), nullable=False),
        sa.Column("prompt_id", sa.Integer(), sa.ForeignKey("prompt.id", ondelete="SET NULL")),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "year", "week", name="uq_user_week"),
    )
    op.create_index("ix_uwp_user_year_week", "user_weekly_prompts", ["user_id", "year", "week"])

    op.create_table(
        "user_weekly_skips",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("week", sa.Integer(), nullable=False),
        sa.Column("prompt_id", sa.Integer(), sa.ForeignKey("prompt.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "year", "week", "prompt_id", name="uq_user_week_prompt_skip"),
    )
    op.create_index(
        "ix_uws_user_year_week_prompt",
        "user_weekly_skips",
        ["user_id", "year", "week", "prompt_id"]
    )

def downgrade():
    op.drop_index("ix_uws_user_year_week_prompt", table_name="user_weekly_skips")
    op.drop_table("user_weekly_skips")
    op.drop_index("ix_uwp_user_year_week", table_name="user_weekly_prompts")
    op.drop_table("user_weekly_prompts")
