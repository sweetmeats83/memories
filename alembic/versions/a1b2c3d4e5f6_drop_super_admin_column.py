"""Drop super_admin column from user table

Revision ID: a1b2c3d4e5f6
Revises: 7f6f6ebc6d1c
Create Date: 2026-03-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "7f6f6ebc6d1c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("user", "super_admin")


def downgrade() -> None:
    op.add_column(
        "user",
        sa.Column("super_admin", sa.Boolean(), nullable=False, server_default="false"),
    )
