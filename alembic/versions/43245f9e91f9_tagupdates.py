"""tagupdates

Revision ID: 43245f9e91f9
Revises: 3ae98c47ad1c
Create Date: 2025-08-24 02:42:40.407389

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "43245f9e91f9"
down_revision: str | None = "3ae98c47ad1c"
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa

def upgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)

    # create table only if it doesn't exist
    if "user_prompts" not in insp.get_table_names():
        op.create_table(
            "user_prompts",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("user_id", sa.Integer, sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
            sa.Column("prompt_id", sa.Integer, sa.ForeignKey("prompt.id", ondelete="CASCADE"), nullable=False),
            sa.Column("status", sa.String),
            sa.Column("score", sa.Float),
            sa.Column("assigned_at", sa.DateTime),
            sa.Column("last_sent_at", sa.DateTime),
            sa.Column("times_sent", sa.Integer),
        )

    # ensure the unique constraint exists
    uqs = {u["name"] for u in insp.get_unique_constraints("user_prompts")}
    if "uq_user_prompt" not in uqs:
        op.create_unique_constraint("uq_user_prompt", "user_prompts", ["user_id", "prompt_id"])

def downgrade():
    bind = op.get_bind()
    insp = sa.inspect(bind)
    # drop only if exists
    if "user_prompts" in insp.get_table_names():
        op.drop_table("user_prompts")