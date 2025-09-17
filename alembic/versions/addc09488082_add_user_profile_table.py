"""add user_profile table

Revision ID: addc09488082
Revises: 902e0ecaa9e0
Create Date: 2025-08-18 21:58:29.374225

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'addc09488082'
down_revision: Union[str, Sequence[str], None] = '902e0ecaa9e0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # create only if not already present (safe for dev where table might exist)
    op.create_table(
        "user_profile",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE")),
        sa.Column("display_name", sa.String(length=128)),
        sa.Column("birth_year", sa.Integer()),
        sa.Column("location", sa.String(length=128)),
        sa.Column("relation_roles", sa.JSON()),
        sa.Column("interests", sa.JSON()),
        sa.Column("accessibility_prefs", sa.JSON()),
        sa.Column("consent_flags", sa.JSON()),
        sa.Column("bio", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    # unique 1:1 profile per user
    op.create_index("ix_user_profile_user_id", "user_profile", ["user_id"], unique=True)

def downgrade() -> None:
    op.drop_index("ix_user_profile_user_id", table_name="user_profile")
    op.drop_table("user_profile")