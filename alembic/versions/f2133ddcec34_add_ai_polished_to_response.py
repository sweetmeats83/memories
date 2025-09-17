"""add ai_polished to response

Revision ID: f2133ddcec34
Revises: 67b636e45af1
Create Date: 2025-08-12 02:52:07.556690

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2133ddcec34'
down_revision: Union[str, Sequence[str], None] = '67b636e45af1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column("response", sa.Column("ai_polished", sa.Text(), nullable=True))
    op.add_column("response", sa.Column("ai_polished_at", sa.DateTime(timezone=True), nullable=True))

def downgrade():
    op.drop_column("response", "ai_polished_at")
    op.drop_column("response", "ai_polished")
