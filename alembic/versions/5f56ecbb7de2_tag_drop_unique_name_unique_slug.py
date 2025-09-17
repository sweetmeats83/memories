"""tag: drop unique(name), unique(slug)

Revision ID: 5f56ecbb7de2
Revises: 43245f9e91f9
Create Date: 2025-08-24 03:08:51.994764

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5f56ecbb7de2'
down_revision: Union[str, Sequence[str], None] = '43245f9e91f9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # 1) drop any legacy unique on name
    # the default name from SA is usually "tag_name_key"; if you used a custom name, drop that instead
    with op.batch_alter_table("tag") as b:
        try:
            b.drop_constraint("tag_name_key", type_="unique")
        except Exception:
            pass  # already gone

    # 2) ensure a unique on slug exists (idempotent-ish)
    # if you previously created an unnamed unique constraint, use this named one going forward
    try:
        op.create_unique_constraint("uq_tag_slug", "tag", ["slug"])
    except Exception:
        pass  # already exists

def downgrade():
    with op.batch_alter_table("tag") as b:
        try:
            b.drop_constraint("uq_tag_slug", type_="unique")
        except Exception:
            pass
        b.create_unique_constraint("tag_name_key", ["name"])