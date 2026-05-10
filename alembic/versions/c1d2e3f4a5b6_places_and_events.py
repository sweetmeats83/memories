"""places and events entities with kin-group scope

Adds Place, PlaceAlias, ResponsePlace, Event, EventAlias, ResponseEvent,
EventPlace (event↔place), and EventPerson (event↔person) tables.

Revision ID: c1d2e3f4a5b6
Revises: b0c1d2e3f4a5
Create Date: 2026-05-07
"""
from alembic import op
import sqlalchemy as sa

revision = "c1d2e3f4a5b6"
down_revision = "b0c1d2e3f4a5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------ place
    op.create_table(
        "place",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("group_id", sa.Integer, sa.ForeignKey("kin_group.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("place_type", sa.String(32), nullable=True),
        sa.Column("address", sa.String(256), nullable=True),
        sa.Column("city", sa.String(128), nullable=True),
        sa.Column("state", sa.String(64), nullable=True),
        sa.Column("country", sa.String(64), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.UniqueConstraint("group_id", "name", name="uq_place_group_name"),
    )

    op.create_table(
        "place_alias",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("place_id", sa.Integer, sa.ForeignKey("place.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("alias", sa.String(200), nullable=False),
    )

    op.create_table(
        "response_place",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("response_id", sa.Integer, sa.ForeignKey("response.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("place_id", sa.Integer, sa.ForeignKey("place.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("role_hint", sa.String(64), nullable=True),
        sa.UniqueConstraint("response_id", "place_id", name="uq_response_place"),
    )

    # ------------------------------------------------------------------ event
    op.create_table(
        "event",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("group_id", sa.Integer, sa.ForeignKey("kin_group.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("year", sa.Integer, nullable=True),
        sa.Column("event_type", sa.String(32), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.UniqueConstraint("group_id", "name", "year", name="uq_event_group_name_year"),
    )

    op.create_table(
        "event_alias",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("event_id", sa.Integer, sa.ForeignKey("event.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("alias", sa.String(200), nullable=False),
    )

    op.create_table(
        "response_event",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("response_id", sa.Integer, sa.ForeignKey("response.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("event_id", sa.Integer, sa.ForeignKey("event.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("role_hint", sa.String(64), nullable=True),
        sa.UniqueConstraint("response_id", "event_id", name="uq_response_event"),
    )

    # --------------------------------------------------------- event relations
    op.create_table(
        "event_place",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("event_id", sa.Integer, sa.ForeignKey("event.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("place_id", sa.Integer, sa.ForeignKey("place.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.UniqueConstraint("event_id", "place_id", name="uq_event_place"),
    )

    op.create_table(
        "event_person",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("event_id", sa.Integer, sa.ForeignKey("event.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("person_id", sa.Integer, sa.ForeignKey("person.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("role_hint", sa.String(64), nullable=True),
        sa.UniqueConstraint("event_id", "person_id", name="uq_event_person"),
    )


def downgrade() -> None:
    op.drop_table("event_person")
    op.drop_table("event_place")
    op.drop_table("response_event")
    op.drop_table("event_alias")
    op.drop_table("event")
    op.drop_table("response_place")
    op.drop_table("place_alias")
    op.drop_table("place")
