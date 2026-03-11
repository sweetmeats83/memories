"""Migrate existing persons and family edges to group scope

For every user who belongs to exactly one KinGroup:
  - Set group_id on all their Person rows
  - Set group_id (and null user_id) on all their family RelationshipEdge rows
Users with no group or multiple groups are left unchanged (private persons).

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-03-10
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = 'e6f7a8b9c0d1'
down_revision = 'd5e6f7a8b9c0'
branch_labels = None
depends_on = None

# Edge types treated as shared family facts
FAMILY_EDGE_TYPES = (
    'parent-of', 'mother-of', 'father-of', 'child-of', 'son-of', 'daughter-of',
    'adoptive-parent-of', 'step-parent-of',
    'sibling-of', 'half-sibling-of', 'step-sibling-of',
    'aunt-of', 'uncle-of', 'niece-of', 'nephew-of', 'cousin-of',
    'spouse-of', 'partner-of', 'ex-partner-of', 'ex-spouse-of',
)


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Find users who belong to exactly one kin group
    rows = conn.execute(text("""
        SELECT user_id, group_id
        FROM kin_membership
        WHERE user_id IN (
            SELECT user_id FROM kin_membership
            GROUP BY user_id
            HAVING COUNT(*) = 1
        )
    """)).fetchall()

    for user_id, group_id in rows:
        # 2. Move that user's Person rows to the group
        conn.execute(text("""
            UPDATE person
            SET group_id = :gid
            WHERE owner_user_id = :uid
              AND group_id IS NULL
        """), {'gid': group_id, 'uid': user_id})

        # 3. Move that user's family RelationshipEdges to the group
        family_placeholders = ', '.join(f"'{t}'" for t in FAMILY_EDGE_TYPES)
        conn.execute(text(f"""
            UPDATE relationship_edge
            SET group_id = :gid,
                user_id  = NULL
            WHERE user_id = :uid
              AND group_id IS NULL
              AND rel_type IN ({family_placeholders})
        """), {'gid': group_id, 'uid': user_id})

    # 4. Deduplicate persons within each group: for each group, find persons
    #    with identical normalised display_name and merge the newer one into
    #    the older one (keep lowest id, transfer ResponsePerson + edges).
    groups = conn.execute(text("SELECT id FROM kin_group")).fetchall()
    for (gid,) in groups:
        # Fetch all persons in this group ordered by id (oldest first)
        persons = conn.execute(text("""
            SELECT id, lower(regexp_replace(display_name, '[^a-z0-9]', '', 'gi')) AS norm_name
            FROM person
            WHERE group_id = :gid
            ORDER BY id
        """), {'gid': gid}).fetchall()

        # Group by norm_name
        seen: dict[str, int] = {}  # norm_name -> keep_id
        for pid, norm in persons:
            if not norm:
                continue
            if norm not in seen:
                seen[norm] = pid
            else:
                keep_id = seen[norm]
                gone_id = pid
                # Re-point ResponsePerson
                conn.execute(text("""
                    UPDATE response_person SET person_id = :keep WHERE person_id = :gone
                """), {'keep': keep_id, 'gone': gone_id})
                # Re-point edges (src)
                conn.execute(text("""
                    UPDATE relationship_edge SET src_id = :keep
                    WHERE src_id = :gone AND group_id = :gid
                      AND NOT EXISTS (
                          SELECT 1 FROM relationship_edge r2
                          WHERE r2.src_id = :keep AND r2.dst_id = relationship_edge.dst_id
                            AND r2.rel_type = relationship_edge.rel_type
                            AND r2.group_id = :gid
                      )
                """), {'keep': keep_id, 'gone': gone_id, 'gid': gid})
                # Re-point edges (dst)
                conn.execute(text("""
                    UPDATE relationship_edge SET dst_id = :keep
                    WHERE dst_id = :gone AND group_id = :gid
                      AND NOT EXISTS (
                          SELECT 1 FROM relationship_edge r2
                          WHERE r2.dst_id = :keep AND r2.src_id = relationship_edge.src_id
                            AND r2.rel_type = relationship_edge.rel_type
                            AND r2.group_id = :gid
                      )
                """), {'keep': keep_id, 'gone': gone_id, 'gid': gid})
                # Delete self-loops left over
                conn.execute(text("""
                    DELETE FROM relationship_edge
                    WHERE group_id = :gid AND (src_id = dst_id)
                """), {'gid': gid})
                # Delete duplicate edges
                conn.execute(text("""
                    DELETE FROM relationship_edge
                    WHERE src_id = :gone OR dst_id = :gone
                """), {'gone': gone_id})
                # Add alias for the deleted name
                conn.execute(text("""
                    INSERT INTO person_alias (person_id, alias)
                    SELECT :keep, display_name FROM person WHERE id = :gone
                    ON CONFLICT DO NOTHING
                """), {'keep': keep_id, 'gone': gone_id})
                # Delete the duplicate person
                conn.execute(text("DELETE FROM person WHERE id = :gone"), {'gone': gone_id})


def downgrade() -> None:
    # Reversing the merge is not safely possible; only undo the group_id assignment.
    conn = op.get_bind()
    conn.execute(text("UPDATE relationship_edge SET user_id = NULL, group_id = NULL WHERE group_id IS NOT NULL"))
    conn.execute(text("UPDATE person SET group_id = NULL WHERE group_id IS NOT NULL"))
