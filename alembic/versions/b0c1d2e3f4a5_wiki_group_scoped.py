"""wiki_article: group-scoped family wiki

One shared article per (person, kin_group) instead of per (person, user).
All family members contribute to and read the same article.

Revision ID: b0c1d2e3f4a5
Revises: a9b0c1d2e3f4
Create Date: 2026-05-07
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = 'b0c1d2e3f4a5'
down_revision = 'a9b0c1d2e3f4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Add group_id column (nullable while we populate it)
    op.add_column('wiki_article',
        sa.Column('group_id', sa.Integer(), nullable=True))
    op.create_index('ix_wiki_article_group_id', 'wiki_article', ['group_id'])
    op.create_foreign_key(
        'fk_wiki_article_group_id',
        'wiki_article', 'kin_group',
        ['group_id'], ['id'],
        ondelete='CASCADE',
    )

    # 2. Populate group_id from the user's primary kin membership
    conn.execute(text("""
        UPDATE wiki_article wa
        SET group_id = (
            SELECT km.group_id
            FROM kin_membership km
            WHERE km.user_id = wa.user_id
            ORDER BY km.id ASC
            LIMIT 1
        )
    """))

    # 3. Deduplicate: keep only the newest article per (entity_type, entity_id, group_id)
    conn.execute(text("""
        DELETE FROM wiki_article
        WHERE id NOT IN (
            SELECT DISTINCT ON (entity_type, entity_id, COALESCE(group_id::text, 'null'))
                id
            FROM wiki_article
            ORDER BY entity_type,
                     entity_id,
                     COALESCE(group_id::text, 'null'),
                     COALESCE(updated_at, created_at) DESC NULLS LAST
        )
    """))

    # 4. Drop old per-user unique constraint
    op.drop_constraint('uq_wiki_entity_user', 'wiki_article', type_='unique')

    # 5. Add new group-scoped unique constraint
    op.create_unique_constraint(
        'uq_wiki_entity_group', 'wiki_article',
        ['entity_type', 'entity_id', 'group_id'],
    )

    # 6. Make user_id nullable (becomes "triggered by" metadata only)
    op.alter_column('wiki_article', 'user_id', nullable=True)


def downgrade() -> None:
    op.alter_column('wiki_article', 'user_id', nullable=False)
    op.drop_constraint('uq_wiki_entity_group', 'wiki_article', type_='unique')
    op.create_unique_constraint(
        'uq_wiki_entity_user', 'wiki_article',
        ['entity_type', 'entity_id', 'user_id'],
    )
    op.drop_constraint('fk_wiki_article_group_id', 'wiki_article', type_='foreignkey')
    op.drop_index('ix_wiki_article_group_id', 'wiki_article')
    op.drop_column('wiki_article', 'group_id')
