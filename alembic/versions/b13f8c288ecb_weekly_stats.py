"""Weekly stats

Revision ID: b13f8c288ecb
Revises: 5f56ecbb7de2
Create Date: 2025-08-29 03:47:41.131681
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "b13f8c288ecb"
down_revision: Union[str, Sequence[str], None] = "5f56ecbb7de2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # add nullable column first
    op.add_column("user", sa.Column("weekly_current_prompt_id", sa.Integer(), nullable=True))
    # fk to prompt, set null on delete
    op.create_foreign_key(
        "fk_user_weekly_current_prompt",
        "user",
        "prompt",
        ["weekly_current_prompt_id"],
        ["id"],
        ondelete="SET NULL",
    )
    # optional: index for faster joins
    op.create_index(
        "ix_user_weekly_current_prompt_id",
        "user",
        ["weekly_current_prompt_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_user_weekly_current_prompt_id", table_name="user")
    op.drop_constraint("fk_user_weekly_current_prompt", "user", type_="foreignkey")
    op.drop_column("user", "weekly_current_prompt_id")