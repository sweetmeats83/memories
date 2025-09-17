"""people graph + kin groups + sharing (filtered)

Revision ID: 53f422a64a3d
Revises: c6221c2c932a
Create Date: 2025-08-20 05:18:46.156645
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "53f422a64a3d"
down_revision: Union[str, Sequence[str], None] = "c6221c2c932a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


"""people graph + kin groups + sharing (filtered, index-safe)

Revision ID: 53f422a64a3d
Revises: c6221c2c932a
Create Date: 2025-08-20 05:18:46.156645
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "53f422a64a3d"
down_revision: Union[str, Sequence[str], None] = "c6221c2c932a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ----- helpers -----
def has_table(conn, name: str) -> bool:
    insp = sa.inspect(conn)
    return name in insp.get_table_names()

def has_column(conn, table: str, column: str) -> bool:
    insp = sa.inspect(conn)
    if table not in insp.get_table_names():
        return False
    return column in [c["name"] for c in insp.get_columns(table)]

def index_exists(conn, table: str, index_name: str) -> bool:
    insp = sa.inspect(conn)
    try:
        idxs = insp.get_indexes(table)
    except Exception:
        return False
    return any(ix.get("name") == index_name for ix in idxs)

def create_index_safe(conn, name: str, table: str, columns: list[str]) -> None:
    if not index_exists(conn, table, name):
        op.create_index(name, table, columns)


def upgrade() -> None:
    conn = op.get_bind()

    # ---------- PERSON ----------
    if not has_table(conn, "person"):
        op.create_table(
            "person",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("owner_user_id", sa.Integer, sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
            sa.Column("display_name", sa.String(128), nullable=False),
            sa.Column("given_name", sa.String(64)),
            sa.Column("family_name", sa.String(64)),
            sa.Column("birth_year", sa.Integer),
            sa.Column("death_year", sa.Integer),
            sa.Column("notes", sa.Text),
            sa.Column("photo_url", sa.String(256)),
            sa.Column("meta", sa.JSON),
            sa.Column("visibility", sa.String(16), server_default="private"),
            sa.Column("consent_source", sa.String(32), server_default="owner"),
        )
        create_index_safe(conn, "ix_person_owner_user_id", "person", ["owner_user_id"])
    else:
        if not has_column(conn, "person", "visibility"):
            op.add_column("person", sa.Column("visibility", sa.String(16), server_default="private"))
        if not has_column(conn, "person", "consent_source"):
            op.add_column("person", sa.Column("consent_source", sa.String(32), server_default="owner"))
        if has_column(conn, "person", "shared_group_id"):
            with op.batch_alter_table("person") as b:
                b.drop_column("shared_group_id")
        # normalize any legacy value
        op.execute("UPDATE person SET visibility='groups' WHERE visibility='group'")

    # ---------- PERSON ALIAS ----------
    if not has_table(conn, "person_alias"):
        op.create_table(
            "person_alias",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("person_id", sa.Integer, sa.ForeignKey("person.id", ondelete="CASCADE"), nullable=False),
            sa.Column("alias", sa.String(128), nullable=False),
        )
        create_index_safe(conn, "ix_person_alias_person_id", "person_alias", ["person_id"])
        create_index_safe(conn, "ix_person_alias_alias", "person_alias", ["alias"])

    # ---------- RELATIONSHIP EDGE ----------
    if not has_table(conn, "relationship_edge"):
        op.create_table(
            "relationship_edge",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("user_id", sa.Integer, sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
            sa.Column("src_id", sa.Integer, sa.ForeignKey("person.id", ondelete="CASCADE"), nullable=False),
            sa.Column("dst_id", sa.Integer, sa.ForeignKey("person.id", ondelete="CASCADE"), nullable=False),
            sa.Column("rel_type", sa.String(48), nullable=False),
            sa.Column("confidence", sa.Float, server_default="0.8"),
            sa.Column("notes", sa.Text),
            sa.Column("meta", sa.JSON),
        )
        create_index_safe(conn, "ix_relationship_edge_user_id", "relationship_edge", ["user_id"])
        create_index_safe(conn, "ix_relationship_edge_src_id", "relationship_edge", ["src_id"])
        create_index_safe(conn, "ix_relationship_edge_dst_id", "relationship_edge", ["dst_id"])
        create_index_safe(conn, "ix_relationship_edge_rel_type", "relationship_edge", ["rel_type"])
        try:
            op.create_unique_constraint("uq_rel_once", "relationship_edge", ["user_id", "src_id", "dst_id", "rel_type"])
        except Exception:
            pass

    # ---------- RESPONSE PERSON ----------
    if not has_table(conn, "response_person"):
        op.create_table(
            "response_person",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("response_id", sa.Integer, sa.ForeignKey("response.id", ondelete="CASCADE"), nullable=False),
            sa.Column("person_id", sa.Integer, sa.ForeignKey("person.id", ondelete="CASCADE"), nullable=False),
            sa.Column("alias_used", sa.String(128)),
            sa.Column("start_char", sa.Integer),
            sa.Column("end_char", sa.Integer),
            sa.Column("confidence", sa.Float, server_default="0.7"),
            sa.Column("role_hint", sa.String(48)),
        )
        create_index_safe(conn, "ix_response_person_response_id", "response_person", ["response_id"])
        create_index_safe(conn, "ix_response_person_person_id", "response_person", ["person_id"])

    # ---------- KIN GROUP ----------
    if not has_table(conn, "kin_group"):
        op.create_table(
            "kin_group",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("name", sa.String(128), nullable=False),
            sa.Column("kind", sa.String(32), server_default="family"),
            sa.Column("join_code", sa.String(16), unique=True),
            sa.Column("created_by", sa.Integer, sa.ForeignKey("user.id", ondelete="SET NULL")),
            sa.Column("created_at", sa.DateTime, server_default=sa.text("CURRENT_TIMESTAMP")),
        )

    # ---------- KIN MEMBERSHIP ----------
    if not has_table(conn, "kin_membership"):
        op.create_table(
            "kin_membership",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("group_id", sa.Integer, sa.ForeignKey("kin_group.id", ondelete="CASCADE"), nullable=False),
            sa.Column("user_id", sa.Integer, sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
            sa.Column("role", sa.String(32), server_default="member"),
        )
        create_index_safe(conn, "ix_kin_membership_group_id", "kin_membership", ["group_id"])
        create_index_safe(conn, "ix_kin_membership_user_id", "kin_membership", ["user_id"])
        try:
            op.create_unique_constraint("uq_kin_group_user", "kin_membership", ["group_id", "user_id"])
        except Exception:
            pass

    # ---------- PERSON SHARE ----------
    if not has_table(conn, "person_share"):
        op.create_table(
            "person_share",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("person_id", sa.Integer, sa.ForeignKey("person.id", ondelete="CASCADE"), nullable=False),
            sa.Column("group_id", sa.Integer, sa.ForeignKey("kin_group.id", ondelete="CASCADE"), nullable=False),
            sa.Column("shared_by_user_id", sa.Integer, sa.ForeignKey("user.id", ondelete="SET NULL")),
            sa.Column("created_at", sa.DateTime, server_default=sa.text("CURRENT_TIMESTAMP")),
        )
        create_index_safe(conn, "ix_person_share_person_id", "person_share", ["person_id"])
        create_index_safe(conn, "ix_person_share_group_id", "person_share", ["group_id"])
        try:
            op.create_unique_constraint("uq_person_group_once", "person_share", ["person_id", "group_id"])
        except Exception:
            pass

    # ---------- CLEANUP LEGACY (optional) ----------
    if has_table(conn, "household_membership"):
        op.drop_table("household_membership")
    if has_table(conn, "household_group"):
        op.drop_table("household_group")


def downgrade() -> None:
    conn = op.get_bind()

    # Drop in reverse dependency order; indexes drop with their tables.
    if has_table(conn, "person_share"):
        op.drop_table("person_share")
    if has_table(conn, "kin_membership"):
        op.drop_table("kin_membership")
    if has_table(conn, "kin_group"):
        op.drop_table("kin_group")
    if has_table(conn, "response_person"):
        op.drop_table("response_person")
    if has_table(conn, "relationship_edge"):
        op.drop_table("relationship_edge")
    # Intentionally leaving person/person_alias intact by default.
    # Uncomment if you truly want full rollback:
    # if has_table(conn, "person_alias"):
    #     op.drop_table("person_alias")
    # if has_table(conn, "person"):
    #     op.drop_table("person")
