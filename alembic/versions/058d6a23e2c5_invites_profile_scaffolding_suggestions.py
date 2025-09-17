"""init clean state

Revision ID: 0001_init
Revises: 
Create Date: 2025-08-18

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- users table ---
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean, server_default=sa.sql.expression.true()),
        sa.Column("is_admin", sa.Boolean, server_default=sa.sql.expression.false()),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # --- invites table ---
    op.create_table(
        "invites",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("token", sa.String(255), nullable=False, unique=True),
        sa.Column("sent_count", sa.Integer, default=0),
        sa.Column("last_sent", sa.DateTime),
    )

    # --- prompts ---
    op.create_table(
        "prompts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("chapter", sa.String(255)),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # --- tags ---
    op.create_table(
        "tags",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), unique=True, nullable=False),
        sa.Column("slug", sa.String(120), unique=True, nullable=False),
    )

    # --- prompt_tags (m2m) ---
    op.create_table(
        "prompt_tags",
        sa.Column("prompt_id", sa.Integer, sa.ForeignKey("prompts.id", ondelete="CASCADE")),
        sa.Column("tag_id", sa.Integer, sa.ForeignKey("tags.id", ondelete="CASCADE")),
        sa.PrimaryKeyConstraint("prompt_id", "tag_id"),
    )

    # --- media ---
    op.create_table(
        "media",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("prompt_id", sa.Integer, sa.ForeignKey("prompts.id", ondelete="CASCADE")),
        sa.Column("response_id", sa.Integer, sa.ForeignKey("responses.id", ondelete="CASCADE"), nullable=True),
        sa.Column("file_path", sa.String(255), nullable=False),
        sa.Column("media_type", sa.String(50)),
        sa.Column("thumbnail_url", sa.String(255)),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # --- responses ---
    op.create_table(
        "responses",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("prompt_id", sa.Integer, sa.ForeignKey("prompts.id", ondelete="SET NULL")),
        sa.Column("title", sa.String(255)),
        sa.Column("response_text", sa.Text),
        sa.Column("transcription", sa.Text),
        sa.Column("ai_polished", sa.Text),
        sa.Column("primary_media_url", sa.String(255)),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # --- response_tags (m2m) ---
    op.create_table(
        "response_tags",
        sa.Column("response_id", sa.Integer, sa.ForeignKey("responses.id", ondelete="CASCADE")),
        sa.Column("tag_id", sa.Integer, sa.ForeignKey("tags.id", ondelete="CASCADE")),
        sa.PrimaryKeyConstraint("response_id", "tag_id"),
    )

    # --- weekly prompt assignment ---
    op.create_table(
        "user_weekly_prompts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("prompt_id", sa.Integer, sa.ForeignKey("prompts.id", ondelete="CASCADE")),
        sa.Column("year", sa.Integer, nullable=False),
        sa.Column("week", sa.Integer, nullable=False),
        sa.Column("status", sa.String(50)),
    )

    op.create_table(
        "user_weekly_skips",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("prompt_id", sa.Integer, sa.ForeignKey("prompts.id", ondelete="CASCADE")),
        sa.Column("year", sa.Integer, nullable=False),
        sa.Column("week", sa.Integer, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("user_weekly_skips")
    op.drop_table("user_weekly_prompts")
    op.drop_table("response_tags")
    op.drop_table("responses")
    op.drop_table("media")
    op.drop_table("prompt_tags")
    op.drop_table("tags")
    op.drop_table("prompts")
    op.drop_table("invites")
    op.drop_table("users")
