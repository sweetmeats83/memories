"""add chapter_meta

Revision ID: c6221c2c932a
Revises: 798d79f9a763
Create Date: 2025-08-19 15:22:53.177097

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c6221c2c932a'
down_revision: Union[str, Sequence[str], None] = '798d79f9a763'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: create chapter_meta and backfill from prompt.chapter."""
    # Create table
    op.create_table(
        "chapter_meta",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("name", sa.String(), nullable=False, unique=True, index=True),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tint", sa.String(), nullable=True),
        # LLM-useful fields (optional metadata for planning/assignment)
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("keywords", sa.Text(), nullable=True),
        sa.Column("llm_guidance", sa.Text(), nullable=True),
    )

    # Backfill meta rows for existing chapters
    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT DISTINCT chapter FROM prompt WHERE chapter IS NOT NULL")).fetchall()
    chapters = sorted([r[0] for r in rows if r[0]], key=lambda s: s.lower())

    for i, name in enumerate(chapters):
        bind.execute(
            sa.text(
                """
                INSERT INTO chapter_meta (name, display_name, "order", tint)
                VALUES (:name, :display_name, :order, :tint)
                ON CONFLICT (name) DO NOTHING
                """
            ),
            {"name": name, "display_name": name, "order": i, "tint": "#e5e7eb"},
        )


def downgrade() -> None:
    """Downgrade schema: drop chapter_meta."""
    op.drop_table("chapter_meta")