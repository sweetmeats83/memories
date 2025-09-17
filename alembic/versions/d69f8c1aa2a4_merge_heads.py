"""merge heads

Revision ID: d69f8c1aa2a4
Revises: 0001_init, c66d93087670
Create Date: 2025-08-18 17:24:44.019128

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd69f8c1aa2a4'
down_revision: Union[str, Sequence[str], None] = ('0001_init', 'c66d93087670')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
