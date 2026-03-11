"""Shared person model — group_id on Person and RelationshipEdge

Adds group_id to Person and RelationshipEdge so persons and family edges
can be scoped to a KinGroup rather than owned by a single user.
Also makes Person.owner_user_id nullable (becomes created_by attribution).

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-03-10
"""
from alembic import op
import sqlalchemy as sa

revision = 'd5e6f7a8b9c0'
down_revision = 'c4d5e6f7a8b9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- person table ---

    # Add group_id (nullable FK to kin_group)
    op.add_column('person', sa.Column('group_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_person_group_id',
        'person', 'kin_group',
        ['group_id'], ['id'],
        ondelete='SET NULL',
    )
    op.create_index('ix_person_group_id', 'person', ['group_id'])

    # Make owner_user_id nullable (was NOT NULL)
    op.alter_column('person', 'owner_user_id', existing_type=sa.Integer(), nullable=True)

    # --- relationship_edge table ---

    # Add group_id (nullable FK to kin_group)
    op.add_column('relationship_edge', sa.Column('group_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_rel_edge_group_id',
        'relationship_edge', 'kin_group',
        ['group_id'], ['id'],
        ondelete='CASCADE',
    )
    op.create_index('ix_rel_edge_group_id', 'relationship_edge', ['group_id'])

    # Make user_id nullable (family edges will have group_id instead)
    op.alter_column('relationship_edge', 'user_id', existing_type=sa.Integer(), nullable=True)

    # Partial unique index for group-scoped edges (complements existing user-scoped unique constraint)
    op.create_index(
        'uq_rel_group_once',
        'relationship_edge',
        ['group_id', 'src_id', 'dst_id', 'rel_type'],
        unique=True,
        postgresql_where=sa.text('group_id IS NOT NULL'),
    )


def downgrade() -> None:
    op.drop_index('uq_rel_group_once', table_name='relationship_edge')
    op.alter_column('relationship_edge', 'user_id', existing_type=sa.Integer(), nullable=False)
    op.drop_index('ix_rel_edge_group_id', table_name='relationship_edge')
    op.drop_constraint('fk_rel_edge_group_id', 'relationship_edge', type_='foreignkey')
    op.drop_column('relationship_edge', 'group_id')

    op.alter_column('person', 'owner_user_id', existing_type=sa.Integer(), nullable=False)
    op.drop_index('ix_person_group_id', table_name='person')
    op.drop_constraint('fk_person_group_id', 'person', type_='foreignkey')
    op.drop_column('person', 'group_id')
