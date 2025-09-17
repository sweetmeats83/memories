"""add response_segments

Revision ID: c66d93087670
Revises: bf1d7949982e
Create Date: 2025-08-17 02:54:16.515025

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c66d93087670'
down_revision: Union[str, Sequence[str], None] = 'bf1d7949982e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'response_segments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('response_id', sa.Integer(), nullable=False),
        sa.Column('order_index', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('media_path', sa.String(), nullable=True),
        sa.Column('media_mime', sa.String(), nullable=True),
        sa.Column('transcript', sa.Text(), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['response_id'], ['response.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        op.f('ix_response_segments_response_id'),
        'response_segments', ['response_id'], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_response_segments_response_id'), table_name='response_segments')
    op.drop_table('response_segments')