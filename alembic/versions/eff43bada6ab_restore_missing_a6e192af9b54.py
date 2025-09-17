"""restore missing a6e192af9b54

Revision ID: eff43bada6ab
Revises: 67b636e45af1
Create Date: 2025-08-11 22:27:36.314014

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
# revision identifiers, used by Alembic.
revision = "a6e192af9b54"   # <-- the missing one you need to restore
down_revision = None        # <-- make it the base (or the true parent if you know it)
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
