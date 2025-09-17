"""add title to responses

Revision ID: 67b636e45af1
Revises: 
Create Date: 2025-08-11 15:28:13.122214

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '67b636e45af1'
down_revision: Union[str, Sequence[str], None] = "a6e192af9b54"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # table name is singular: "response"
    op.add_column("response", sa.Column("title", sa.String(length=200), nullable=True))
    op.create_index("ix_response_title", "response", ["title"])

def downgrade():
    op.drop_index("ix_response_title", table_name="response")
    op.drop_column("response", "title")