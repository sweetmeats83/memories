"""invites: add token/expiry/used/inviter

Revision ID: 902e0ecaa9e0
Revises: d69f8c1aa2a4
Create Date: 2025-08-18 17:34:06.127980

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '902e0ecaa9e0'
down_revision: Union[str, Sequence[str], None] = 'd69f8c1aa2a4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add columns (nullable first so existing rows pass)
    op.add_column("invite", sa.Column("token", sa.String(length=64), nullable=True))
    op.add_column("invite", sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("invite", sa.Column("used_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("invite", sa.Column("invited_by_user_id", sa.Integer(), nullable=True))
    op.add_column(
        "invite",
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
    )

    # Backfill existing rows (no pgcrypto required)
    op.execute(
        """
        UPDATE invite
        SET
          token = COALESCE(
            token,
            md5(random()::text || clock_timestamp()::text) || md5(clock_timestamp()::text || random()::text)
          ),
          expires_at = COALESCE(expires_at, now() + interval '7 days')
        WHERE token IS NULL OR expires_at IS NULL
        """
    )

    # Enforce NOT NULL after backfill
    op.alter_column("invite", "token", existing_type=sa.String(length=64), nullable=False)
    op.alter_column("invite", "expires_at", existing_type=sa.DateTime(timezone=True), nullable=False)

    # Index + FK
    op.create_index(op.f("ix_invite_token"), "invite", ["token"], unique=True)
    op.create_foreign_key(
        None,
        "invite",
        "user",
        ["invited_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(None, "invite", type_="foreignkey")
    op.drop_index(op.f("ix_invite_token"), table_name="invite")
    op.drop_column("invite", "created_at")
    op.drop_column("invite", "invited_by_user_id")
    op.drop_column("invite", "used_at")
    op.drop_column("invite", "expires_at")
    op.drop_column("invite", "token")