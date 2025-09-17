"""add user_profile json fields

Revision ID: faec801f00b3
Revises: 53f422a64a3d
Create Date: 2025-08-20 06:13:18.609052

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'faec801f00b3'
down_revision: Union[str, Sequence[str], None] = '53f422a64a3d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(conn, table: str, col: str) -> bool:
    insp = sa.inspect(conn)
    if table not in insp.get_table_names():
        return False
    return col in [c["name"] for c in insp.get_columns(table)]

def upgrade() -> None:
    conn = op.get_bind()

    # add user_profile.tag_weights (JSON) if missing
    if not _has_column(conn, "user_profile", "tag_weights"):
        op.add_column("user_profile", sa.Column("tag_weights", sa.JSON(), nullable=True))
        # optional: seed existing rows to {}
        op.execute("UPDATE user_profile SET tag_weights = '{}'::json WHERE tag_weights IS NULL")

    # add user_profile.privacy_prefs (JSON) if missing
    if not _has_column(conn, "user_profile", "privacy_prefs"):
        op.add_column("user_profile", sa.Column("privacy_prefs", sa.JSON(), nullable=True))
        # optional: seed existing rows to {}
        op.execute("UPDATE user_profile SET privacy_prefs = '{}'::json WHERE privacy_prefs IS NULL")

def downgrade() -> None:
    conn = op.get_bind()
    # safe to drop (these are additive fields)
    insp = sa.inspect(conn)
    if "user_profile" in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns("user_profile")]
        if "privacy_prefs" in cols:
            op.drop_column("user_profile", "privacy_prefs")
        if "tag_weights" in cols:
            op.drop_column("user_profile", "tag_weights")