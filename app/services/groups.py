# services/groups.py
import secrets, string
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import KinGroup, KinMembership

def _join_code(n: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))

async def create_group(db: AsyncSession, name: str, kind: str, created_by_user_id: int) -> KinGroup:
    g = KinGroup(name=(name or "Family Group").strip(), kind=(kind or "family"), created_by=created_by_user_id)
    g.join_code = _join_code(8)
    db.add(g)
    await db.flush()
    db.add(KinMembership(group_id=g.id, user_id=created_by_user_id, role="admin"))
    return g

async def add_member_by_code(db: AsyncSession, join_code: str, user_id: int) -> KinMembership | None:
    code = (join_code or "").strip().upper()
    g = (await db.execute(select(KinGroup).where(KinGroup.join_code == code))).scalars().first()
    if not g:
        return None
    existing = (await db.execute(
        select(KinMembership).where(KinMembership.group_id == g.id, KinMembership.user_id == user_id)
    )).scalars().first()
    if existing:
        return existing
    m = KinMembership(group_id=g.id, user_id=user_id, role="member")
    db.add(m)
    await db.flush()
    return m

async def user_group_ids(db: AsyncSession, user_id: int) -> list[int]:
    rows = (await db.execute(select(KinMembership.group_id).where(KinMembership.user_id == user_id))).scalars().all()
    return list(rows or [])
