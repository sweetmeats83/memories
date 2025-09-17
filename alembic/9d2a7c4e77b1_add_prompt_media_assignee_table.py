"""add prompt_media_assignee table and backfill

Revision ID: 9d2a7c4e77b1
Revises: 8c1b3f2add8e
Create Date: 2025-09-06
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9d2a7c4e77b1'
down_revision = '8c1b3f2add8e'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'prompt_media_assignee',
        sa.Column('prompt_media_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['prompt_media_id'], ['prompt_media.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('prompt_media_id', 'user_id'),
    )
    # Backfill from single assignee_user_id into mapping table
    conn = op.get_bind()
    try:
        res = conn.execute(sa.text("SELECT id, assignee_user_id FROM prompt_media WHERE assignee_user_id IS NOT NULL"))
        rows = res.fetchall()
        if rows:
            conn.execute(sa.text("INSERT INTO prompt_media_assignee(prompt_media_id, user_id) VALUES (:pmid, :uid) ON CONFLICT DO NOTHING"),
                         [{"pmid": r[0], "uid": r[1]} for r in rows])
    except Exception:
        pass


def downgrade() -> None:
    op.drop_table('prompt_media_assignee')

