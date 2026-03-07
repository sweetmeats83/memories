"""add push_subscriptions table

Revision ID: c4d5e6f7a8b9
Revises: b3c4d5e6f7a8
Create Date: 2026-03-07
"""
from alembic import op
import sqlalchemy as sa

revision = 'c4d5e6f7a8b9'
down_revision = 'b3c4d5e6f7a8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'push_subscriptions',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('user.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('endpoint', sa.Text, nullable=False, unique=True),
        sa.Column('p256dh', sa.Text, nullable=False),
        sa.Column('auth', sa.Text, nullable=False),
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('push_subscriptions')
