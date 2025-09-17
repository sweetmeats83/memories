"""add user_prompts

Revision ID: 3ae98c47ad1c
Revises: faec801f00b3
Create Date: 2025-08-24 00:49:38.843528
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "3ae98c47ad1c"
down_revision: Union[str, Sequence[str], None] = "faec801f00b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_prompts",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
        sa.Column("prompt_id", sa.Integer(), sa.ForeignKey("prompt.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(), nullable=True, server_default="queued"),
        sa.Column("score", sa.Float(), nullable=True, server_default="0"),
        sa.Column("assigned_at", sa.DateTime(), nullable=True, server_default=sa.text("now()")),
        sa.Column("last_sent_at", sa.DateTime(), nullable=True),
        sa.Column("times_sent", sa.Integer(), nullable=True, server_default="0"),
        sa.UniqueConstraint("user_id", "prompt_id", name="uq_user_prompt"),
    )
    op.create_index("ix_user_prompts_user_id", "user_prompts", ["user_id"])
    op.create_index("ix_user_prompts_prompt_id", "user_prompts", ["prompt_id"])


def downgrade() -> None:
    op.drop_index("ix_user_prompts_prompt_id", table_name="user_prompts")
    op.drop_index("ix_user_prompts_user_id", table_name="user_prompts")
    op.drop_table("user_prompts")
