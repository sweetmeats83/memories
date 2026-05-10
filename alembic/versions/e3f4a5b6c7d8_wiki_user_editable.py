"""wiki user-editable notes and article override

Adds user_notes_md, user_edited_md, notes_updated_at, and edited_at columns
to wiki_article so users can annotate or override AI-generated content.

Revision ID: e3f4a5b6c7d8
Revises: d2e3f4a5b6c7
Branch labels: None
Depends on: None

Create Date: 2026-05-08
"""
from alembic import op
import sqlalchemy as sa

revision = "e3f4a5b6c7d8"
down_revision = "d2e3f4a5b6c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("wiki_article", sa.Column("user_notes_md",  sa.Text, nullable=True))
    op.add_column("wiki_article", sa.Column("user_edited_md", sa.Text, nullable=True))
    op.add_column("wiki_article", sa.Column("notes_updated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("wiki_article", sa.Column("edited_at",        sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("wiki_article", "edited_at")
    op.drop_column("wiki_article", "notes_updated_at")
    op.drop_column("wiki_article", "user_edited_md")
    op.drop_column("wiki_article", "user_notes_md")
