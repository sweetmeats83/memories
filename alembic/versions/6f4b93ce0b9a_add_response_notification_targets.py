"""Add response notification targets table

Revision ID: 6f4b93ce0b9a
Revises: 2d1b1f64ef6c
Create Date: 2025-09-20 12:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "6f4b93ce0b9a"
down_revision: Union[str, Sequence[str], None] = "2d1b1f64ef6c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "response_notification_target",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("owner_user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
        sa.Column("watcher_user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("owner_user_id", "watcher_user_id", name="uq_notification_owner_watcher"),
    )
    op.create_index("ix_response_notification_target_owner", "response_notification_target", ["owner_user_id"])
    op.create_index("ix_response_notification_target_watcher", "response_notification_target", ["watcher_user_id"])

    # Seed admins as default watchers for all other users
    op.execute(
        sa.text(
            "INSERT INTO response_notification_target (owner_user_id, watcher_user_id) "
            "SELECT u.id, admin.id FROM \"user\" AS u "
            "JOIN \"user\" AS admin ON (admin.is_superuser = TRUE OR COALESCE(admin.super_admin, FALSE) = TRUE) "
            "WHERE u.id <> admin.id"
        )
    )


def downgrade() -> None:
    op.drop_index("ix_response_notification_target_watcher", table_name="response_notification_target")
    op.drop_index("ix_response_notification_target_owner", table_name="response_notification_target")
    op.drop_table("response_notification_target")

