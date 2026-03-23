"""add notify_daily_reminders to user

Revision ID: f7a8b9c0d1e2
Revises: c4d5e6f7a8b9
Create Date: 2026-03-15

"""
from alembic import op
import sqlalchemy as sa

revision = 'f7a8b9c0d1e2'
down_revision = 'e6f7a8b9c0d1'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('user', sa.Column('notify_daily_reminders', sa.Boolean(), nullable=False, server_default='false'))


def downgrade():
    op.drop_column('user', 'notify_daily_reminders')
