# services/people_acl.py
from sqlalchemy import select
from app.models import Person, PersonShare, KinMembership
from app.settings.config import settings

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

