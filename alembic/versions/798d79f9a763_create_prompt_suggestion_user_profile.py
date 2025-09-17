"""create prompt_suggestion + user_profile

Revision ID: 798d79f9a763
Revises: addc09488082
Create Date: 2025-08-19 03:17:31.292294

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '798d79f9a763'
down_revision: Union[str, Sequence[str], None] = 'addc09488082'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # --- user_profile (safe to create even if you haven’t used it yet) ---
    if not op.get_bind().dialect.has_table(op.get_bind(), "user_profile"):
        op.create_table(
            "user_profile",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, unique=True, index=True),
            sa.Column("display_name", sa.String(length=128), nullable=True),
            sa.Column("birth_year", sa.Integer(), nullable=True),
            sa.Column("location", sa.String(length=128), nullable=True),
            sa.Column("relation_roles", sa.JSON(), nullable=True),       # list[str]
            sa.Column("interests", sa.JSON(), nullable=True),            # list[str]
            sa.Column("accessibility_prefs", sa.JSON(), nullable=True),  # dict
            sa.Column("consent_flags", sa.JSON(), nullable=True),        # dict
            sa.Column("bio", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        )
        op.create_index("ix_user_profile_user_id", "user_profile", ["user_id"], unique=True)

    # --- prompt_suggestion (needed by /onboarding/finish) ---
    if not op.get_bind().dialect.has_table(op.get_bind(), "prompt_suggestion"):
        op.create_table(
            "prompt_suggestion",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
            sa.Column("prompt_id", sa.Integer(), sa.ForeignKey("prompt.id", ondelete="SET NULL"), nullable=True),
            sa.Column("source", sa.String(length=16), nullable=False),   # 'tag_match' | 'llm'
            sa.Column("title", sa.String(length=200), nullable=True),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("tags", sa.JSON(), nullable=True),                 # list[str]
            sa.Column("status", sa.String(length=16), server_default="pending", nullable=False),
            sa.Column("rationale_json", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index("ix_prompt_suggestion_user_id", "prompt_suggestion", ["user_id"])
        op.create_index("ix_prompt_suggestion_status", "prompt_suggestion", ["status"])


def downgrade():
    # drop in reverse order
    if op.get_bind().dialect.has_table(op.get_bind(), "prompt_suggestion"):
        op.drop_index("ix_prompt_suggestion_status", table_name="prompt_suggestion")
        op.drop_index("ix_prompt_suggestion_user_id", table_name="prompt_suggestion")
        op.drop_table("prompt_suggestion")

    if op.get_bind().dialect.has_table(op.get_bind(), "user_profile"):
        op.drop_index("ix_user_profile_user_id", table_name="user_profile")
        op.drop_table("user_profile")