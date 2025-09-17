"""fluffer

Revision ID: e8746b121024
Revises: ceeae8478dd1
Create Date: 2025-08-29 04:41:11.704919

"""
from typing import Sequence, Union, Iterable

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e8746b121024'
down_revision: Union[str, Sequence[str], None] = 'ceeae8478dd1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _insp():
    return sa.inspect(op.get_bind())

def _has_column(table: str, col: str) -> bool:
    return col in [c["name"] for c in _insp().get_columns(table)]

def _fk_names(table: str) -> set[str]:
    return {fk.get("name") for fk in _insp().get_foreign_keys(table) if fk.get("name")}

def _has_index(table: str, name: str) -> bool:
    return any(ix["name"] == name for ix in _insp().get_indexes(table))

def upgrade() -> None:
    # Ensure user.weekly_current_prompt_id (+ FK + index)
    if not _has_column("user", "weekly_current_prompt_id"):
        op.add_column("user", sa.Column("weekly_current_prompt_id", sa.Integer(), nullable=True))
    fks = _fk_names("user")
    if "fk_user_weekly_current_prompt" not in fks and _has_column("user", "weekly_current_prompt_id"):
        op.create_foreign_key(
            "fk_user_weekly_current_prompt", "user", "prompt",
            ["weekly_current_prompt_id"], ["id"], ondelete="SET NULL"
        )
    if not _has_index("user", "ix_user_weekly_current_prompt_id") and _has_column("user","weekly_current_prompt_id"):
        op.create_index("ix_user_weekly_current_prompt_id", "user", ["weekly_current_prompt_id"], unique=False)

    # Ensure on-deck FK exists for weekly_on_deck_prompt_id (column may already exist)
    if _has_column("user", "weekly_on_deck_prompt_id"):
        fks = _fk_names("user")
        if "fk_user_weekly_on_deck_prompt" not in fks and "fk_user_weekly_ondeck_prompt" not in fks:
            op.create_foreign_key(
                "fk_user_weekly_on_deck_prompt", "user", "prompt",
                ["weekly_on_deck_prompt_id"], ["id"], ondelete="SET NULL"
            )

def downgrade() -> None:
    # Only undo what we did here
    try:
        op.drop_index("ix_user_weekly_current_prompt_id", table_name="user")
    except Exception:
        pass
    try:
        op.drop_constraint("fk_user_weekly_current_prompt", "user", type_="foreignkey")
    except Exception:
        pass
    try:
        op.drop_column("user", "weekly_current_prompt_id")
    except Exception:
        pass