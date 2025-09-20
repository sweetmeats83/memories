"""baseline schema

Revision ID: b77e84e87261
Revises: 
Create Date: 2025-09-20 11:21:15.878293

"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy.engine import Connection

from app.database import Base

# revision identifiers, used by Alembic.
revision: str = "b77e84e87261"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all database objects for the current metadata."""
    bind: Connection = op.get_bind()
    Base.metadata.create_all(bind)


def downgrade() -> None:
    """Drop all database objects managed by the metadata."""
    bind: Connection = op.get_bind()
    Base.metadata.drop_all(bind)
