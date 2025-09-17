"""goodnight

Revision ID: ebd0895d4196
Revises: e8746b121024
Create Date: 2025-08-29 05:28:56.388777

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ebd0895d4196'
down_revision: Union[str, Sequence[str], None] = 'e8746b121024'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# --- helpers ---
def insp():
    return sa.inspect(op.get_bind())

def has_table(name: str) -> bool:
    return name in insp().get_table_names()

def has_column(table: str, col: str) -> bool:
    return col in [c["name"] for c in insp().get_columns(table)]

def fk_names(table: str) -> set[str]:
    return {fk.get("name") for fk in insp().get_foreign_keys(table) if fk.get("name")}

def has_index(table: str, name: str) -> bool:
    return any(ix["name"] == name for ix in insp().get_indexes(table))

def enum_type_exists(enum_name: str) -> bool:
    conn = op.get_bind()
    return conn.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = :n)"
        ),
        {"n": enum_name},
    ).scalar()

def create_enum_if_missing(name: str, values: list[str]) -> None:
    if not enum_type_exists(name):
        op.execute(
            sa.text(
                "CREATE TYPE " + sa.text(name).text + " AS ENUM (" +
                ", ".join(sa.literal(v).text for v in values) + ")"
            )
        )

def upgrade() -> None:
    # 1) Enums (names must match models.py)
    create_enum_if_missing("weeklystate",
        ["not_sent","queued","sent","opened","clicked","used","recorded","responded","skipped","expired"]
    )
    create_enum_if_missing("weeklytokenstatus",
        ["active","opened","used","expired"]
    )

    # 2) Columns on user
    #    (Your model has both weekly_current_prompt_id and weekly_on_deck_prompt_id)
    if has_table("user"):
        if not has_column("user", "weekly_current_prompt_id"):
            op.add_column("user", sa.Column("weekly_current_prompt_id", sa.Integer(), nullable=True))
        if not has_column("user", "weekly_on_deck_prompt_id"):
            op.add_column("user", sa.Column("weekly_on_deck_prompt_id", sa.Integer(), nullable=True))

        # weekly_state (enum)
        if not has_column("user", "weekly_state"):
            op.add_column(
                "user",
                sa.Column("weekly_state", sa.Enum(name="weeklystate"), nullable=False, server_default=sa.text("'not_sent'"))
            )
            # drop default to avoid future autogen noise
            op.alter_column("user", "weekly_state", server_default=None)

        # timeline / misc fields
        for col, typ in [
            ("weekly_queued_at", sa.DateTime()),
            ("weekly_sent_at", sa.DateTime()),
            ("weekly_opened_at", sa.DateTime()),
            ("weekly_clicked_at", sa.DateTime()),
            ("weekly_used_at", sa.DateTime()),
            ("weekly_completed_at", sa.DateTime()),
            ("weekly_skipped_at", sa.DateTime()),
            ("weekly_expires_at", sa.DateTime()),
            ("weekly_email_provider_id", sa.String()),
        ]:
            if not has_column("user", col):
                op.add_column("user", sa.Column(col, typ, nullable=True))

        # FKs (safe if missing)
        fks = fk_names("user")
        if "fk_user_weekly_current_prompt" not in fks and has_column("user", "weekly_current_prompt_id"):
            op.create_foreign_key(
                "fk_user_weekly_current_prompt",
                "user", "prompt",
                ["weekly_current_prompt_id"], ["id"],
                ondelete="SET NULL",
            )
        if "fk_user_weekly_on_deck_prompt" not in fks and has_column("user", "weekly_on_deck_prompt_id"):
            op.create_foreign_key(
                "fk_user_weekly_on_deck_prompt",
                "user", "prompt",
                ["weekly_on_deck_prompt_id"], ["id"],
                ondelete="SET NULL",
            )

        # Indexes (skip if present)
        if not has_index("user", "ix_user_weekly_current_prompt_id") and has_column("user", "weekly_current_prompt_id"):
            op.create_index("ix_user_weekly_current_prompt_id", "user", ["weekly_current_prompt_id"], unique=False)
        if not has_index("user", "ix_user_weekly_on_deck_prompt_id") and has_column("user", "weekly_on_deck_prompt_id"):
            op.create_index("ix_user_weekly_on_deck_prompt_id", "user", ["weekly_on_deck_prompt_id"], unique=False)

    # 3) weekly_token table (if missing)
    if not has_table("weekly_token"):
        op.create_table(
            "weekly_token",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("token", sa.String(64), nullable=False, unique=True, index=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("prompt_id", sa.Integer(), sa.ForeignKey("prompt.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("status", sa.Enum(name="weeklytokenstatus"), nullable=False, server_default=sa.text("'active'")),
            sa.Column("sent_at", sa.DateTime(), nullable=True),
            sa.Column("opened_at", sa.DateTime(), nullable=True),
            sa.Column("clicked_at", sa.DateTime(), nullable=True),
            sa.Column("used_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("expires_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("user_id", "prompt_id", name="uq_weeklytoken_user_prompt"),
        )
        # Drop default to keep schema quiet for autogenerate diffs
        op.alter_column("weekly_token", "status", server_default=None)

def downgrade() -> None:
    # Be conservative: only drop what we created, and only if it exists.
    if has_table("weekly_token"):
        try:
            op.drop_table("weekly_token")
        except Exception:
            pass

    if has_table("user"):
        # Indexes & FKs
        for name in [
            "ix_user_weekly_on_deck_prompt_id",
            "ix_user_weekly_current_prompt_id",
        ]:
            try:
                op.drop_index(name, table_name="user")
            except Exception:
                pass
        for name in [
            "fk_user_weekly_on_deck_prompt",
            "fk_user_weekly_current_prompt",
        ]:
            try:
                op.drop_constraint(name, "user", type_="foreignkey")
            except Exception:
                pass

        # Columns (drop if present)
        for col in [
            "weekly_email_provider_id", "weekly_expires_at", "weekly_skipped_at",
            "weekly_completed_at", "weekly_used_at", "weekly_clicked_at",
            "weekly_opened_at", "weekly_sent_at", "weekly_queued_at",
            "weekly_state", "weekly_on_deck_prompt_id", "weekly_current_prompt_id",
        ]:
            if has_column("user", col):
                try:
                    op.drop_column("user", col)
                except Exception:
                    pass

    # Enum types (optional: keep to avoid dependency issues if other revs reference them)
    # Uncomment to drop:
    # if enum_type_exists("weeklytokenstatus"):
    #     op.execute(sa.text("DROP TYPE weeklytokenstatus"))
    # if enum_type_exists("weeklystate"):
    #     op.execute(sa.text("DROP TYPE weeklystate"))