"""Add processing state fields to response

Revision ID: 7f6f6ebc6d1c
Revises: 43f8c28fb600
Create Date: 2025-09-20 15:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7f6f6ebc6d1c"
down_revision: Union[str, Sequence[str], None] = "43f8c28fb600"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "response",
        sa.Column("processing_state", sa.String(length=20), nullable=False, server_default="ready"),
    )
    op.add_column("response", sa.Column("processing_error", sa.Text(), nullable=True))
    op.execute("UPDATE response SET processing_state = 'ready' WHERE processing_state IS NULL")
    op.alter_column("response", "processing_state", server_default=None)


def downgrade() -> None:
    op.drop_column("response", "processing_error")
    op.drop_column("response", "processing_state")

