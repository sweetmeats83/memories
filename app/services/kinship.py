from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Tuple, Set

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RelationshipEdge, Person


# Canonical edge kinds used to derive kinship. Keep using RelationshipEdge as-is.
PARENT_EDGE_TYPES_FWD: Set[str] = {
    "parent-of",            # biological/unspecified
    "mother-of",
    "father-of",
    "adoptive-parent-of",
    "step-parent-of",
}
PARENT_EDGE_TYPES_REV: Set[str] = {
    "child-of",
    "son-of",
    "daughter-of",
}

PARTNER_EDGE_TYPES: Set[str] = {
    "spouse-of",
    "partner-of",
    "ex-partner-of",
    "ex-spouse-of",
}


@dataclass
class KinshipResult:
    ego_id: int
    alter_id: int
    label_neutral: str
    label_gendered: Optional[str]
    cousin_degree: Optional[int]
    removed: Optional[int]
    ancestor_steps: Optional[int]
    descendant_steps: Optional[int]
    is_half: bool
    is_step: bool
    is_adoptive: bool
    mrca_id: Optional[int]
    is_affinal: bool = False  # True for in-law relationships


async def _parents_with_types(db: AsyncSession, user_id: int, person_id: int) -> list[tuple[int, str]]:
    """Return list of (parent_id, rel_type) for the given person, accepting both directions."""
    # Case 1: stored as parent -> child
    rows1 = await db.execute(
        select(RelationshipEdge.src_id, RelationshipEdge.rel_type)
        .where(
            RelationshipEdge.user_id == user_id,
            RelationshipEdge.dst_id == person_id,
            RelationshipEdge.rel_type.in_(PARENT_EDGE_TYPES_FWD),
        )
    )
    out = [(int(r[0]), str(r[1])) for r in rows1.all()]
    # Case 2: stored as child -> parent
    rows2 = await db.execute(
        select(RelationshipEdge.dst_id, RelationshipEdge.rel_type)
        .where(
            RelationshipEdge.user_id == user_id,
            RelationshipEdge.src_id == person_id,
            RelationshipEdge.rel_type.in_(PARENT_EDGE_TYPES_REV),
        )
    )
    out.extend((int(r[0]), str(r[1])) for r in rows2.all())
    return out


async def _parents_set(db: AsyncSession, user_id: int, person_id: int) -> set[int]:
    # Accept both storage directions
    rows1 = await db.execute(
        select(RelationshipEdge.src_id)
        .where(
            RelationshipEdge.user_id == user_id,
            RelationshipEdge.dst_id == person_id,
            RelationshipEdge.rel_type.in_(PARENT_EDGE_TYPES_FWD),
        )
    )
    rows2 = await db.execute(
        select(RelationshipEdge.dst_id)
        .where(
            RelationshipEdge.user_id == user_id,
            RelationshipEdge.src_id == person_id,
            RelationshipEdge.rel_type.in_(PARENT_EDGE_TYPES_REV),
        )
    )
    return {int(r[0]) for r in rows1.all()} | {int(r[0]) for r in rows2.all()}


async def _ancestor_map(
    db: AsyncSession, user_id: int, seed_id: int, max_depth: int = 10
) -> Dict[int, tuple[int, bool, bool]]:
    """
    BFS upward via parent edges.
    Returns {person_id: (steps_up, saw_step, saw_adoptive)} including parents, grandparents, ...
    Does not include the seed itself.
    """
    out: Dict[int, tuple[int, bool, bool]] = {}
    frontier: list[tuple[int, int, bool, bool]] = [(seed_id, 0, False, False)]
    seen: Set[int] = {seed_id}

    while frontier:
        node, d, saw_step, saw_adopt = frontier.pop(0)
        if d >= max_depth:
            continue
        parents = await _parents_with_types(db, user_id, node)
        for pid, rtype in parents:
            step_flag = saw_step or (rtype == "step-parent-of")
            adopt_flag = saw_adopt or (rtype == "adoptive-parent-of")
            if pid not in seen:
                seen.add(pid)
                out[pid] = (d + 1, step_flag, adopt_flag)
                frontier.append((pid, d + 1, step_flag, adopt_flag))
    return out


def _gender_from_meta(person: Optional[Person]) -> Optional[str]:
    meta = getattr(person, "meta", None) or {}
    g = (meta.get("gender") or meta.get("sex") or "").strip().lower()
    if g in {"male", "m", "man", "boy"}:
        return "male"
    if g in {"female", "f", "woman", "girl"}:
        return "female"
    return None


def _format_ancestor_term(steps: int) -> str:
    if steps == 1:
        return "parent"
    if steps == 2:
        return "grandparent"
    # steps >= 3
    greats = steps - 2
    return ("great-" * greats) + "grandparent"


def _format_descendant_term(steps: int) -> str:
    if steps == 1:
        return "child"
    if steps == 2:
        return "grandchild"
    greats = steps - 2
    return ("great-" * greats) + "grandchild"


def _format_aunt_uncle_term(younger_steps: int) -> str:
    """
    Label for the older side relative to the younger side.
    When the older side is 1 step from MRCA and the younger side is 2 steps, it's 'aunt/uncle'.
    Each additional step on the younger side adds a 'great-'.
    Mapping: 2 -> aunt/uncle; 3 -> grandaunt/uncle; 4 -> great-grandaunt/uncle; ...
    """
    if younger_steps <= 2:
        return "aunt/uncle"
    if younger_steps == 3:
        return "grandaunt/uncle"
    greats = younger_steps - 3
    return ("great-" * (greats - 1)) + "grandaunt/uncle"


def _format_niece_nephew_term(younger_steps: int) -> str:
    """
    Label for the younger side relative to the older side, parameterized by the younger side's distance.
    Mapping: 2 -> niece/nephew; 3 -> grandniece/nephew; 4 -> great-grandniece/nephew; ...
    """
    if younger_steps <= 2:
        return "niece/nephew"
    if younger_steps == 3:
        return "grandniece/nephew"
    greats = younger_steps - 3
    return ("great-" * (greats - 1)) + "grandniece/nephew"


def _gendered_variant(neutral: str, alter_gender: Optional[str]) -> Optional[str]:
    if not alter_gender:
        return None
    g = alter_gender
    m = {
        "parent": "father",
        "grandparent": "grandfather",
        "child": "son",
        "grandchild": "grandson",
        "sibling": "brother",
        "aunt/uncle": "uncle" if g == "male" else "aunt",
        "grandaunt/uncle": "granduncle" if g == "male" else "grandaunt",
        "grandniece/nephew": "grandnephew" if g == "male" else "grandniece",
    }
    # handle great- chains by replacing base token
    if neutral.startswith("great-") and neutral.endswith("grandparent"):
        return neutral.replace("grandparent", "grandfather" if g == "male" else "grandmother")
    if neutral.startswith("great-") and neutral.endswith("grandchild"):
        return neutral.replace("grandchild", "grandson" if g == "male" else "granddaughter")
    if neutral.endswith("grandaunt/uncle"):
        return neutral.replace(
            "grandaunt/uncle", "granduncle" if g == "male" else "grandaunt"
        )
    if neutral.endswith("grandniece/nephew"):
        return neutral.replace(
            "grandniece/nephew", "grandnephew" if g == "male" else "grandniece"
        )
    return m.get(neutral)


async def _partners(db: AsyncSession, user_id: int, person_id: int) -> Set[int]:
    rows1 = await db.execute(
        select(RelationshipEdge.dst_id)
        .where(
            RelationshipEdge.user_id == user_id,
            RelationshipEdge.src_id == person_id,
            RelationshipEdge.rel_type.in_(PARTNER_EDGE_TYPES),
        )
    )
    rows2 = await db.execute(
        select(RelationshipEdge.src_id)
        .where(
            RelationshipEdge.user_id == user_id,
            RelationshipEdge.dst_id == person_id,
            RelationshipEdge.rel_type.in_(PARTNER_EDGE_TYPES),
        )
    )
    return {int(r[0]) for r in rows1.all()} | {int(r[0]) for r in rows2.all()}


async def _classify_blood(
    db: AsyncSession, *, user_id: int, ego_id: int, alter_id: int
) -> Optional[KinshipResult]:
    """Pure consanguine classification using only parent edges."""
    if ego_id == alter_id:
        return KinshipResult(
            ego_id=ego_id, alter_id=alter_id, label_neutral="self", label_gendered=None,
            cousin_degree=None, removed=None, ancestor_steps=0, descendant_steps=0,
            is_half=False, is_step=False, is_adoptive=False, mrca_id=ego_id, is_affinal=False,
        )

    alter = await db.get(Person, alter_id)
    alter_gender = _gender_from_meta(alter)

    anc_ego = await _ancestor_map(db, user_id, ego_id)
    anc_alt = await _ancestor_map(db, user_id, alter_id)

    if alter_id in anc_ego:
        steps, saw_step, saw_adopt = anc_ego[alter_id]
        neutral = _format_ancestor_term(steps)
        return KinshipResult(
            ego_id=ego_id, alter_id=alter_id, label_neutral=neutral,
            label_gendered=_gendered_variant(neutral, alter_gender), cousin_degree=None, removed=None,
            ancestor_steps=steps, descendant_steps=None, is_half=False, is_step=saw_step, is_adoptive=saw_adopt,
            mrca_id=alter_id, is_affinal=False,
        )

    if ego_id in anc_alt:
        steps, saw_step, saw_adopt = anc_alt[ego_id]
        neutral = _format_descendant_term(steps)
        return KinshipResult(
            ego_id=ego_id, alter_id=alter_id, label_neutral=neutral,
            label_gendered=_gendered_variant(neutral, alter_gender), cousin_degree=None, removed=None,
            ancestor_steps=None, descendant_steps=steps, is_half=False, is_step=saw_step, is_adoptive=saw_adopt,
            mrca_id=ego_id, is_affinal=False,
        )

    common = set(anc_ego.keys()) & set(anc_alt.keys())
    best: Optional[Tuple[int, int, int, bool, bool]] = None
    for aid in common:
        a, s_step_a, s_adopt_a = anc_ego[aid]
        b, s_step_b, s_adopt_b = anc_alt[aid]
        total = a + b
        step_path = s_step_a or s_step_b
        adopt_path = s_adopt_a or s_adopt_b
        if best is None or total < (best[1] + best[2]):
            best = (aid, a, b, step_path, adopt_path)

    if best is None:
        return None

    mrca_id, a, b, step_path, adopt_path = best

    if a == 1 and b == 1:
        p_ego = await _parents_set(db, user_id, ego_id)
        p_alt = await _parents_set(db, user_id, alter_id)
        shared = len(p_ego & p_alt)
        neutral = "sibling"
        return KinshipResult(
            ego_id=ego_id, alter_id=alter_id, label_neutral=neutral,
            label_gendered=_gendered_variant(neutral, alter_gender), cousin_degree=None, removed=None,
            ancestor_steps=None, descendant_steps=None, is_half=(shared == 1), is_step=step_path, is_adoptive=adopt_path,
            mrca_id=mrca_id, is_affinal=False,
        )

    if min(a, b) == 1 and max(a, b) >= 2:
        # Label describes ALTER relative to EGO.
        # a = ego->MRCA steps; b = alter->MRCA steps
        # If ego is closer (a==1) and alter is further (b>=2), alter is the younger side -> niece/nephew
        # If alter is closer (b==1) and ego is further (a>=2), alter is the older side -> aunt/uncle
        if a == 1 and b >= 2:
            neutral = _format_niece_nephew_term(b)
        else:
            neutral = _format_aunt_uncle_term(a)
        return KinshipResult(
            ego_id=ego_id, alter_id=alter_id, label_neutral=neutral,
            label_gendered=_gendered_variant(neutral, alter_gender), cousin_degree=None, removed=None,
            ancestor_steps=None, descendant_steps=None, is_half=False, is_step=step_path, is_adoptive=adopt_path,
            mrca_id=mrca_id, is_affinal=False,
        )

    degree = min(a, b) - 1
    removed = abs(a - b)
    if degree <= 0:
        neutral = "related"
    else:
        ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
        base = f"{ordinals.get(degree, str(degree)+'th')} cousin"
        if removed == 0:
            neutral = base
        elif removed == 1:
            neutral = base + " once removed"
        elif removed == 2:
            neutral = base + " twice removed"
        else:
            neutral = base + f" {removed} times removed"

    return KinshipResult(
        ego_id=ego_id, alter_id=alter_id, label_neutral=neutral,
        label_gendered=None, cousin_degree=degree, removed=removed,
        ancestor_steps=None, descendant_steps=None, is_half=False, is_step=step_path, is_adoptive=adopt_path,
        mrca_id=mrca_id, is_affinal=False,
    )


async def classify_kinship(
    db: AsyncSession, *, user_id: int, ego_id: int, alter_id: int
) -> KinshipResult:
    # Quick direct-edge check for partner/social relations to avoid generic 'related'
    try:
        rows = await db.execute(
            select(RelationshipEdge.rel_type).where(
                RelationshipEdge.user_id == user_id,
                (
                    (RelationshipEdge.src_id == ego_id) & (RelationshipEdge.dst_id == alter_id)
                ) | (
                    (RelationshipEdge.src_id == alter_id) & (RelationshipEdge.dst_id == ego_id)
                )
            )
        )
        direct = {str(r[0]).strip().lower() for r in rows.all()}
        if direct:
            if ("ex-partner-of" in direct) or ("ex-spouse-of" in direct):
                return KinshipResult(
                    ego_id=ego_id, alter_id=alter_id,
                    label_neutral="ex-spouse", label_gendered=None,
                    cousin_degree=None, removed=None,
                    ancestor_steps=None, descendant_steps=None,
                    is_half=False, is_step=False, is_adoptive=False,
                    mrca_id=None, is_affinal=False,
                )
            if any(rt in direct for rt in ("spouse-of", "partner-of")):
                return KinshipResult(
                    ego_id=ego_id, alter_id=alter_id,
                    label_neutral="spouse", label_gendered=None,
                    cousin_degree=None, removed=None,
                    ancestor_steps=None, descendant_steps=None,
                    is_half=False, is_step=False, is_adoptive=False,
                    mrca_id=None, is_affinal=False,
                )
            # Common social relations
            social_map = {
                "friend-of": "friend",
                "neighbor-of": "neighbor",
                "coworker-of": "coworker",
                "mentor-of": "mentor",
                "teacher-of": "teacher",
                "student-of": "student",
            }
            for k, v in social_map.items():
                if k in direct:
                    return KinshipResult(
                        ego_id=ego_id, alter_id=alter_id,
                        label_neutral=v, label_gendered=None,
                        cousin_degree=None, removed=None,
                        ancestor_steps=None, descendant_steps=None,
                        is_half=False, is_step=False, is_adoptive=False,
                        mrca_id=None, is_affinal=False,
                    )
    except Exception:
        pass
    """
    Compute a precise kinship label between two persons owned by `user_id`.
    Uses only RelationshipEdge with rel_type in PARENT_EDGE_TYPES and PARTNER_EDGE_TYPES.
    The result provides a neutral label and an optional gendered variant derived from alter's gender.
    """
    # First, consanguine classification
    base = await _classify_blood(db, user_id=user_id, ego_id=ego_id, alter_id=alter_id)
    if base is not None:
        return base

    # Affinal (in-law) detection via partner links
    alter = await db.get(Person, alter_id)
    alter_gender = _gender_from_meta(alter)

    ego_partners = await _partners(db, user_id, ego_id)
    for p in ego_partners:
        b = await _classify_blood(db, user_id=user_id, ego_id=p, alter_id=alter_id)
        if not b:
            continue
        neutral = None
        # Generalize: append "-in-law" to any consanguine label beyond self
        L = (b.label_neutral or '').lower()
        if L == 'parent': neutral = 'parent-in-law'
        elif L == 'grandparent' or L.endswith('grandparent') or L.endswith('great-grandparent'):
            neutral = f"{b.label_neutral}-in-law"
        elif L == 'child': neutral = 'child-in-law'
        elif L == 'grandchild' or L.endswith('grandchild') or L.endswith('great-grandchild'):
            neutral = f"{b.label_neutral}-in-law"
        elif L == 'sibling': neutral = 'sibling-in-law'
        elif 'aunt/uncle' in L: neutral = b.label_neutral  # treat affinal aunt/uncle as aunt/uncle
        elif 'niece/nephew' in L: neutral = b.label_neutral  # treat affinal niece/nephew as niece/nephew
        elif 'cousin' in L: neutral = f"{b.label_neutral}-in-law"
        if neutral:
            return KinshipResult(
                ego_id=ego_id, alter_id=alter_id, label_neutral=neutral,
                label_gendered=None if neutral.endswith("-in-law") else _gendered_variant(neutral, alter_gender),
                cousin_degree=None, removed=None, ancestor_steps=None, descendant_steps=None,
                is_half=False, is_step=False, is_adoptive=False, mrca_id=None, is_affinal=True,
            )

    # Symmetric: alter's partners
    alter_partners = await _partners(db, user_id, alter_id)
    for q in alter_partners:
        b = await _classify_blood(db, user_id=user_id, ego_id=ego_id, alter_id=q)
        if not b:
            continue
        neutral = None
        L = (b.label_neutral or '').lower()
        if L == 'parent': neutral = 'parent-in-law'
        elif L == 'grandparent' or L.endswith('grandparent') or L.endswith('great-grandparent'):
            neutral = f"grandparent-in-law"
        elif L == 'child': neutral = 'child-in-law'
        elif L == 'grandchild' or L.endswith('grandchild') or L.endswith('great-grandchild'):
            neutral = f"grandchild-in-law"
        elif L == 'sibling': neutral = 'sibling-in-law'
        elif 'aunt/uncle' in L: neutral = b.label_neutral
        elif 'niece/nephew' in L: neutral = b.label_neutral
        elif 'cousin' in L: neutral = f"{b.label_neutral}-in-law"
        if neutral:
            return KinshipResult(
                ego_id=ego_id, alter_id=alter_id, label_neutral=neutral,
                label_gendered=None, cousin_degree=None, removed=None,
                ancestor_steps=None, descendant_steps=None,
                is_half=False, is_step=False, is_adoptive=False, mrca_id=None, is_affinal=True,
            )

    # Fallback when no blood MRCA and no direct in-law mapping
    return KinshipResult(
        ego_id=ego_id, alter_id=alter_id, label_neutral="related",
        label_gendered=None, cousin_degree=None, removed=None,
        ancestor_steps=None, descendant_steps=None,
        is_half=False, is_step=False, is_adoptive=False, mrca_id=None, is_affinal=False,
    )
