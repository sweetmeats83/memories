"""add_chapter_compilation

Revision ID: 166e0f655b8e
Revises: ebd0895d4196
Create Date: 2025-09-02 02:09:01.994114

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql

# revision identifiers, used by Alembic.
revision: str = '166e0f655b8e'
down_revision: Union[str, Sequence[str], None] = 'ebd0895d4196'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('prompt_media') as batch:
        batch.add_column(sa.Column('assignee_user_id', sa.Integer(), nullable=True))
        batch.create_foreign_key('fk_prompt_media_user', 'user', ['assignee_user_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    with op.batch_alter_table('prompt_media') as batch:
        batch.drop_constraint('fk_prompt_media_user', type_='foreignkey')
        batch.drop_column('assignee_user_id')
