"""Add wiki_article table

Stores LLM-generated biographical wiki articles for persons (and future entities).
One row per (entity_type, entity_id, user_id) — regenerated in place.

Revision ID: a9b0c1d2e3f4
Revises: f7a8b9c0d1e2
Create Date: 2026-05-06
"""
from alembic import op
import sqlalchemy as sa

revision = 'a9b0c1d2e3f4'
down_revision = 'f7a8b9c0d1e2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'wiki_article',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('entity_type', sa.String(16), nullable=False),
        sa.Column('entity_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('user.id', ondelete='CASCADE'), nullable=False),
        sa.Column('content_md', sa.Text(), nullable=True),
        sa.Column('status', sa.String(16), nullable=False, server_default='pending'),
        sa.Column('error_msg', sa.Text(), nullable=True),
        sa.Column('model_name', sa.String(64), nullable=True),
        sa.Column('source_count', sa.Integer(), nullable=True),
        sa.Column('generated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_wiki_article_entity_id', 'wiki_article', ['entity_id'])
    op.create_index('ix_wiki_article_user_id', 'wiki_article', ['user_id'])
    op.create_index('ix_wiki_entity', 'wiki_article', ['entity_type', 'entity_id'])
    op.create_unique_constraint(
        'uq_wiki_entity_user', 'wiki_article',
        ['entity_type', 'entity_id', 'user_id']
    )


def downgrade() -> None:
    op.drop_table('wiki_article')
