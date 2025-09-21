"""Add admin flag to invites

Revision ID: 43f8c28fb600
Revises: 6f4b93ce0b9a
Create Date: 2025-09-20 13:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "43f8c28fb600"
down_revision: Union[str, Sequence[str], None] = "6f4b93ce0b9a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "invite",
        sa.Column("make_superuser", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.alter_column("invite", "make_superuser", server_default=None)


def downgrade() -> None:
    op.drop_column("invite", "make_superuser")

