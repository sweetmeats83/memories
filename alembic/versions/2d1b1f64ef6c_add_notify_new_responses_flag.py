"""Add notify_new_responses flag to user

Revision ID: 2d1b1f64ef6c
Revises: b77e84e87261
Create Date: 2025-09-20 15:07:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "2d1b1f64ef6c"
down_revision: Union[str, Sequence[str], None] = "b77e84e87261"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user",
        sa.Column("notify_new_responses", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    # enable notifications for existing admins by default
    op.execute(
        sa.text(
            "UPDATE \"user\" SET notify_new_responses = TRUE "
            "WHERE is_superuser = TRUE OR COALESCE(super_admin, FALSE) = TRUE"
        )
    )
    op.alter_column("user", "notify_new_responses", server_default=None)


def downgrade() -> None:
    op.drop_column("user", "notify_new_responses")
