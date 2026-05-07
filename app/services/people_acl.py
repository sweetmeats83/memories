# services/people_acl.py
from sqlalchemy import select, or_
from app.models import Person, PersonShare, KinMembership
from app.settings.config import settings


async def visible_group_ids(db, user_id: int) -> list[int]:
    """Return all KinGroup IDs the user belongs to."""
    rows = (await db.execute(
        select(KinMembership.group_id).where(KinMembership.user_id == user_id)
    )).scalars().all()
    return list(rows)


def person_visibility_filter(user_id: int, group_ids: list[int]):
    """SQLAlchemy WHERE expression covering all Person rows visible to this user:
    their own private persons plus any group-scoped persons from their groups."""
    private = (Person.owner_user_id == user_id) & Person.group_id.is_(None)
    if group_ids:
        return or_(private, Person.group_id.in_(group_ids))
    return private


async def can_view_person(db, viewer_user_id: int, person: Person) -> bool:
    # Owner can always view
    if person.owner_user_id == viewer_user_id:
        return True

    # Feature switch
    if getattr(settings, "SHARE_PEOPLE_SCOPE", "off") != "group":
        return False

    # Only shared persons are visible to groups
    if getattr(person, "visibility", "private") != "groups":
        return False

    # Viewer must be a member of ANY group that person is shared to
    q = (
        select(PersonShare.id)
        .join(KinMembership, KinMembership.group_id == PersonShare.group_id)
        .where(PersonShare.person_id == person.id)
        .where(KinMembership.user_id == viewer_user_id)
        .limit(1)
    )
    return bool((await db.execute(q)).scalar_one_or_none())

