"""add assignee_user_id to prompt_media

Revision ID: 8c1b3f2add8e
Revises: ebd0895d4196
Create Date: 2025-09-06
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '8c1b3f2add8e'
down_revision = 'ebd0895d4196'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('prompt_media') as batch:
        batch.add_column(sa.Column('assignee_user_id', sa.Integer(), nullable=True))
        batch.create_foreign_key('fk_prompt_media_user', 'user', ['assignee_user_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    with op.batch_alter_table('prompt_media') as batch:
        batch.drop_constraint('fk_prompt_media_user', type_='foreignkey')
        batch.drop_column('assignee_user_id')

