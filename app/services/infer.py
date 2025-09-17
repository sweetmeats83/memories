from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Tuple, Set, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RelationshipEdge, Person, ResponsePerson, Response


# Canonical sets
PARENT_EDGE_TYPES_FWD: Set[str] = {
    "parent-of",
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
}

SIBLING_EDGE_TYPES: Set[str] = {
    "sibling-of",
    "half-sibling-of",
    "step-sibling-of",
    "brother-of",
    "sister-of",
}


def _canon_rel(rt: str) -> str:
    r = (rt or "").strip().lower()
    if r in {"wife-of", "husband-of"}:  # normalize synonyms
        return "spouse-of"
    # parent/child
    if r in {"mother-of", "father-of", "parent-of"}: return "parent-of"
    if r in {"son-of", "daughter-of", "child-of"}: return "child-of"
    # sibling
    if r in {"brother-of", "sister-of", "sibling-of"}: return "sibling-of"
    # passthrough common types; ensure -of suffix for role bases
    if r in {"spouse", "partner", "friend", "neighbor", "coworker", "mentor", "teacher", "student"}:
        return f"{r}-of"
    return r


def _inverse_rel(rt: str) -> str:
    rt = (rt or "").strip().lower()
    mapping = {
        # Immediate family
        "mother-of": "child-of",
        "father-of": "child-of",
        "parent-of": "child-of",
        "child-of": "parent-of",
        "son-of": "parent-of",
        "daughter-of": "parent-of",
        # Extended family
        "grandparent-of": "grandchild-of",
        "grandchild-of": "grandparent-of",
        "sibling-of": "sibling-of",
        "half-sibling-of": "half-sibling-of",
        "step-sibling-of": "step-sibling-of",
        "aunt-of": "niece-of",
        "uncle-of": "nephew-of",
        "niece-of": "aunt-of",
        "nephew-of": "uncle-of",
        "cousin-of": "cousin-of",
        # In-laws (keep mapped)
        "parent-in-law-of": "parent-in-law-of",
        "child-in-law-of": "child-in-law-of",
        "sister-in-law-of": "sister-in-law-of",
        "brother-in-law-of": "brother-in-law-of",
        # Partners / social
        "spouse-of": "spouse-of",
        "partner-of": "partner-of",
        "friend-of": "friend-of",
        "mentor-of": "student-of",
        "student-of": "teacher-of",
        "teacher-of": "student-of",
        "coworker-of": "coworker-of",
        "neighbor-of": "neighbor-of",
        "coach-of": "student-of",
    }
    return mapping.get(rt, rt)


@dataclass
class CandidateEdge:
    src_id: int
    dst_id: int
    rel_type: str
    confidence: float
    source: str  # closure|profile|text
    explain: str
    meta: dict


async def _load_people_for_user(db: AsyncSession, user_id: int) -> Dict[int, Person]:
    rows = (await db.execute(select(Person).where(Person.owner_user_id == user_id))).scalars().all()
    return {int(p.id): p for p in rows}


async def _load_edges_for_user(db: AsyncSession, user_id: int) -> List[RelationshipEdge]:
    return (await db.execute(select(RelationshipEdge).where(RelationshipEdge.user_id == user_id))).scalars().all()


async def _parents_index(db: AsyncSession, user_id: int) -> Dict[int, list[tuple[int, str]]]:
    """Map child_id -> list of (parent_id, rel_type). Accept both storage directions."""
    out: Dict[int, list[tuple[int, str]]] = {}
    # parent -> child
    rows1 = await db.execute(
        select(RelationshipEdge.src_id, RelationshipEdge.dst_id, RelationshipEdge.rel_type).where(
            RelationshipEdge.user_id == user_id,
            RelationshipEdge.rel_type.in_(PARENT_EDGE_TYPES_FWD),
        )
    )
    for src, dst, rt in rows1.all():
        out.setdefault(int(dst), []).append((int(src), _canon_rel(rt)))
    # child -> parent
    rows2 = await db.execute(
        select(RelationshipEdge.src_id, RelationshipEdge.dst_id, RelationshipEdge.rel_type).where(
            RelationshipEdge.user_id == user_id,
            RelationshipEdge.rel_type.in_(PARENT_EDGE_TYPES_REV),
        )
    )
    for src, dst, rt in rows2.all():
        out.setdefault(int(src), []).append((int(dst), _canon_rel(rt)))
    return out


async def _partners_index(db: AsyncSession, user_id: int) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    rows = await db.execute(
        select(RelationshipEdge.src_id, RelationshipEdge.dst_id, RelationshipEdge.rel_type)
        .where(RelationshipEdge.user_id == user_id)
        .where(RelationshipEdge.rel_type.in_(PARTNER_EDGE_TYPES))
    )
    for s, d, _ in rows.all():
        s = int(s); d = int(d)
        out.setdefault(s, set()).add(d)
        out.setdefault(d, set()).add(s)
    return out


async def _siblings_index(db: AsyncSession, user_id: int) -> Dict[int, Set[int]]:
    """Undirected sibling adjacency from any sibling edge types."""
    out: Dict[int, Set[int]] = {}
    rows = await db.execute(
        select(RelationshipEdge.src_id, RelationshipEdge.dst_id, RelationshipEdge.rel_type)
        .where(RelationshipEdge.user_id == user_id)
        .where(RelationshipEdge.rel_type.in_(SIBLING_EDGE_TYPES))
    )
    for s, d, rt in rows.all():
        s = int(s); d = int(d)
        out.setdefault(s, set()).add(d)
        out.setdefault(d, set()).add(s)
    return out


def _has_explicit_conflict(existing: Dict[tuple[int, int], Set[str]], a: int, b: int, rel_type: str) -> bool:
    """
    Very light conflict detection: avoid inferring parent/child if a spouse edge exists between the pair.
    Extendable with more rules if needed.
    """
    rset = existing.get((a, b), set()) | existing.get((b, a), set())
    rt = (rel_type or "").strip().lower()
    if rt in {"parent-of", "child-of"} and ("spouse-of" in rset or "partner-of" in rset):
        return True
    return False


async def _find_me_person_id(db: AsyncSession, user_id: int) -> Optional[int]:
    rows = (await db.execute(select(Person.id, Person.meta).where(Person.owner_user_id == user_id))).all()
    for pid, meta in rows:
        if isinstance(meta, dict) and meta.get("connect_to_owner"):
            rh = str(meta.get("role_hint", "")).strip().lower()
            if rh in {"you", "self", "me"}:
                return int(pid)
    return None


async def infer_edges_for_person(db: AsyncSession, user_id: int, person_id: int) -> list[CandidateEdge]:
    """
    Compute candidate inferred edges around `person_id` using local closure rules and hints.
    Does not write to DB. Returns CandidateEdge items.
    """
    # Load indices needed
    people = await _load_people_for_user(db, user_id)
    parents_of = await _parents_index(db, user_id)
    partners = await _partners_index(db, user_id)
    siblings_idx = await _siblings_index(db, user_id)
    edges = await _load_edges_for_user(db, user_id)

    # Build existing map for quick checks (explicit/inferred both)
    existing: Dict[tuple[int, int], Set[str]] = {}
    for e in edges:
        a = int(e.src_id); b = int(e.dst_id); r = _canon_rel(e.rel_type)
        existing.setdefault((a, b), set()).add(r)

    cand: list[CandidateEdge] = []
    seen_keys: Set[tuple[int, int, str]] = set()

    def _push(src: int, dst: int, rel: str, conf: float, source: str, explain: str, meta: dict):
        r = _canon_rel(rel)
        key = (int(src), int(dst), r)
        if key in seen_keys:
            return
        seen_keys.add(key)
        cand.append(CandidateEdge(src_id=int(src), dst_id=int(dst), rel_type=r, confidence=conf, source=source, explain=explain, meta=meta or {"inferred": True, "source": source}))
    pid = int(person_id)

    # 1) Siblings from shared parents (half/step/adoptive flags)
    # Map parent -> children
    parent_to_children: Dict[int, list[tuple[int, str]]] = {}
    for child, plist in parents_of.items():
        for p, rtype in plist:
            parent_to_children.setdefault(p, []).append((child, rtype))

    # Work set limited to neighbors of person_id radius 2 for simplicity
    focus_children = set([pid])
    for p, r in parents_of.get(pid, []):
        for c, rt2 in parent_to_children.get(p, []):
            focus_children.add(c)

    for child in focus_children:
        plist = parents_of.get(child, [])
        # For each sibling candidate via any shared parent
        sibs: Set[int] = set()
        par_ids = [p for (p, _t) in plist]
        for p in par_ids:
            for c2, rt2 in parent_to_children.get(p, []):
                if c2 == child:
                    continue
                sibs.add(c2)
        for sib in sibs:
            if _has_explicit_conflict(existing, child, sib, "sibling-of"):
                continue
            if "sibling-of" in existing.get((child, sib), set()) or "sibling-of" in existing.get((sib, child), set()):
                continue
            # Determine half/step/adoptive
            shared = set([p for (p, _t) in parents_of.get(child, [])]) & set([p for (p, _t) in parents_of.get(sib, [])])
            shared_types_child = {t for (p, t) in parents_of.get(child, []) if p in shared}
            shared_types_sib = {t for (p, t) in parents_of.get(sib, []) if p in shared}
            is_half = (len(shared) == 1)
            is_step = ("step-parent-of" in shared_types_child) or ("step-parent-of" in shared_types_sib)
            is_adopt = ("adoptive-parent-of" in shared_types_child) or ("adoptive-parent-of" in shared_types_sib)
            rel = "half-sibling-of" if is_half else "sibling-of"
            meta = {"inferred": True, "source": "closure", "kind": "sibling", "half": is_half, "step": is_step, "adoptive": is_adopt}
            expl = f"Sibling via shared parent(s) {sorted(shared)}"
            # add both directions once (avoid double-adding when iterating the other child)
            if int(child) < int(sib):
                _push(child, sib, rel, 0.7, "closure", expl, meta)
                _push(sib, child, rel, 0.7, "closure", expl, meta)

    # 2) Grandparent/Grandchild: parent-of(parent) (upward path)
    for (mid_parent, _rt) in parents_of.get(pid, []):
        for (gp, gp_rt) in parents_of.get(mid_parent, []):
            if "grandparent-of" in existing.get((gp, pid), set()):
                continue
            is_step = gp_rt == "step-parent-of"
            is_adopt = gp_rt == "adoptive-parent-of"
            meta = {"inferred": True, "source": "closure", "kind": "grandparent", "step": is_step, "adoptive": is_adopt}
            expl = f"Grandparent via {gp}->{mid_parent}->{pid}"
            _push(gp, pid, "grandparent-of", 0.65, "closure", expl, meta)
            _push(pid, gp, "grandchild-of", 0.65, "closure", expl, meta)

    # 2b) Grandchildren (downward path): child-of(child)
    # For each child of pid, and for each child of that child, infer grandparent-of
    for (child, rt1) in parent_to_children.get(pid, []):
        gkids = parent_to_children.get(child, [])
        for (gk, rt2) in gkids:
            if gk == pid:
                continue
            if "grandparent-of" in existing.get((pid, gk), set()):
                continue
            is_step = (rt1 == "step-parent-of") or (rt2 == "step-parent-of")
            is_adopt = (rt1 == "adoptive-parent-of") or (rt2 == "adoptive-parent-of")
            meta = {"inferred": True, "source": "closure", "kind": "grandparent", "step": is_step, "adoptive": is_adopt}
            expl = f"Grandparent via {pid}->{child}->{gk}"
            _push(pid, gk, "grandparent-of", 0.65, "closure", expl, meta)
            _push(gk, pid, "grandchild-of", 0.65, "closure", expl, meta)

    # 2c) Great-grandparent and deeper (upward path >= 3 steps)
    # BFS ancestors up to depth 4 and add grandparent-of edges for depth >= 3 (great-, great-great-, ...)
    from collections import deque
    dq_up: deque[tuple[int, int]] = deque()
    dq_up.append((pid, 0))
    seen_up_g: Set[int] = {pid}
    MAX_UP = 4
    while dq_up:
        node, d = dq_up.popleft()
        if d >= MAX_UP:
            continue
        for (par, prt) in parents_of.get(node, []):
            if par in seen_up_g:
                continue
            seen_up_g.add(par)
            depth = d + 1
            dq_up.append((par, depth))
            if depth >= 3:  # great-grandparent or deeper
                is_step = prt == "step-parent-of"
                is_adopt = prt == "adoptive-parent-of"
                meta = {"inferred": True, "source": "closure", "kind": "grandparent", "generation": int(depth), "step": is_step, "adoptive": is_adopt}
                expl = f"Ancestor depth {depth}: {par} is grandparent-of pid={pid}"
                _push(par, pid, "grandparent-of", 0.6 if depth == 3 else 0.55, "closure", expl, meta)
                _push(pid, par, "grandchild-of", 0.6 if depth == 3 else 0.55, "closure", expl, meta)

    # 2d) Great-grandchildren (downward path >= 3 steps)
    dq_down: deque[tuple[int, int]] = deque()
    dq_down.append((pid, 0))
    seen_down_g: Set[int] = {pid}
    MAX_DOWN = 4
    while dq_down:
        node, d = dq_down.popleft()
        if d >= MAX_DOWN:
            continue
        for (ch, rtch) in parent_to_children.get(node, []):
            if ch in seen_down_g:
                continue
            seen_down_g.add(ch)
            depth = d + 1
            dq_down.append((ch, depth))
            if depth >= 3:  # great-grandchild or deeper
                meta = {"inferred": True, "source": "closure", "kind": "grandparent", "generation": int(depth)}
                expl = f"Descendant depth {depth}: pid={pid} grandparent-of {ch}"
                _push(pid, ch, "grandparent-of", 0.58 if depth == 3 else 0.54, "closure", expl, meta)
                _push(ch, pid, "grandchild-of", 0.58 if depth == 3 else 0.54, "closure", expl, meta)

    # 3) Aunt/Uncle and Niece/Nephew via parent's siblings
    # For each parent of pid, compute their siblings (via their parents)
    for (par, _pr) in parents_of.get(pid, []):
        for (gpar, _gr) in parents_of.get(par, []):
            # siblings of parent: other children of gpar
            for (sibcand, _rt) in parent_to_children.get(gpar, []):
                if sibcand == par:
                    continue
                # sibcand is aunt/uncle of pid
                if "aunt-of" in existing.get((sibcand, pid), set()) or "uncle-of" in existing.get((sibcand, pid), set()):
                    continue
                meta = {"inferred": True, "source": "closure", "kind": "aunt/uncle"}
                expl = f"Parent's sibling via shared grandparent {gpar}"
                # Use neutral aunt-of/uncle-of as generic 'aunt-of' label (gender unknown); keep one rel 'aunt-of'
                _push(sibcand, pid, "aunt-of", 0.6, "closure", expl, meta)
                _push(pid, sibcand, "niece-of", 0.6, "closure", expl, meta)

    # 3b) Aunt/Uncle and Niece/Nephew via explicit sibling-of edges for the parent
    for (par, _pr) in parents_of.get(pid, []):
        for sibcand in siblings_idx.get(par, set()):
            if sibcand == par:
                continue
            if "aunt-of" in existing.get((sibcand, pid), set()):
                continue
            meta = {"inferred": True, "source": "closure", "kind": "aunt/uncle"}
            expl = f"Parent's sibling via sibling-of edge {par}<->{sibcand}"
            _push(sibcand, pid, "aunt-of", 0.6, "closure", expl, meta)
            _push(pid, sibcand, "niece-of", 0.6, "closure", expl, meta)

    # 3c) Grandaunt/Granduncle and deeper: ancestor's siblings (depth >= 2)
    # Build ancestors map (person -> steps up), up to depth 4
    anc_map: Dict[int, int] = {}
    from collections import deque
    dq: deque[tuple[int, int]] = deque()
    dq.append((pid, 0))
    seen_up: Set[int] = {pid}
    MAX_UP = 4
    while dq:
        node, d = dq.popleft()
        if d >= MAX_UP:
            continue
        for (par, _t) in parents_of.get(node, []):
            if par in seen_up:
                continue
            seen_up.add(par)
            anc_map[par] = d + 1
            dq.append((par, d + 1))
    # For each ancestor at depth >= 2, add their siblings as aunt/uncle (generation via depth)
    for anc, depth_up in anc_map.items():
        if depth_up < 2:
            continue  # parents handled above
        # siblings of this ancestor
        sibs_set: Set[int] = set(siblings_idx.get(anc, set()))
        for (gpar, _rt) in parents_of.get(anc, []):
            for (c2, _t) in parent_to_children.get(gpar, []):
                if c2 != anc:
                    sibs_set.add(c2)
        for s in sibs_set:
            if s == anc:
                continue
            meta = {"inferred": True, "source": "closure", "kind": "aunt/uncle", "generation": int(depth_up)}
            conf = 0.55 if depth_up == 2 else (0.5 if depth_up == 3 else 0.45)
            expl = f"Ancestor sibling at depth {depth_up}: anc={anc} sibling={s}"
            _push(s, pid, "aunt-of", conf, "closure", expl, meta)
            _push(pid, s, "niece-of", conf, "closure", expl, meta)
            # spouses (including ex) of the ancestor's sibling are also aunts/uncles by marriage
            for sp in partners.get(s, set()):
                _push(sp, pid, "aunt-of", conf - 0.02, "closure", expl+" (partner)", meta)
                _push(pid, sp, "niece-of", conf - 0.02, "closure", expl+" (partner)", meta)

    # 4) First cousins: children of siblings
    # For each parent's sibling, their children are first cousins of pid
    cousins: Set[tuple[int, int]] = set()
    ego_cousins: Set[int] = set()
    for (par, _pr) in parents_of.get(pid, []):
        # parent siblings via their parents
        for (gpar, _gr) in parents_of.get(par, []):
            sibs = [c for (c, _t) in parent_to_children.get(gpar, []) if c != par]
            for s in sibs:
                # children of s
                kids = [c for (c, _t) in parent_to_children.get(s, [])]
                for k in kids:
                    if k == pid:
                        continue
                    key = (min(pid, k), max(pid, k))
                    if key in cousins:
                        continue
                    cousins.add(key)
                    ego_cousins.add(int(k))
                    if "cousin-of" in existing.get((pid, k), set()) or "cousin-of" in existing.get((k, pid), set()):
                        continue
                    meta = {"inferred": True, "source": "closure", "kind": "cousin", "degree": 1}
                    expl = f"First cousin via parent's sibling {s}"
                    _push(pid, k, "cousin-of", 0.55, "closure", expl, meta)
                    _push(k, pid, "cousin-of", 0.55, "closure", expl, meta)

    # 4b) Niece/Nephew via your siblings' children
    for sib in siblings_idx.get(pid, set()):
        for (kid, _t) in parent_to_children.get(sib, []):
            if "niece-of" in existing.get((kid, pid), set()):
                continue
            meta = {"inferred": True, "source": "closure", "kind": "niece/nephew"}
            expl = f"Sibling's child via {sib}"
            _push(pid, kid, "aunt-of", 0.6, "closure", expl, meta)
            _push(kid, pid, "niece-of", 0.6, "closure", expl, meta)

    # 4c) Grandniece/Grandnephew and deeper via siblings' descendants
    for sib in siblings_idx.get(pid, set()):
        # BFS downwards from sibling up to depth 3
        from collections import deque
        dq2: deque[tuple[int, int]] = deque()
        dq2.append((sib, 0))
        seen_down: Set[int] = {sib}
        MAX_DOWN = 3
        while dq2:
            node, dd = dq2.popleft()
            if dd >= MAX_DOWN:
                continue
            for (child, _t) in parent_to_children.get(node, []):
                if child in seen_down:
                    continue
                seen_down.add(child)
                next_d = dd + 1
                dq2.append((child, next_d))
                if next_d >= 2:  # depth 2 => grandniece/nephew and deeper
                    meta = {"inferred": True, "source": "closure", "kind": "niece/nephew", "generation": int(next_d)}
                    conf = 0.55 if next_d == 2 else 0.5
                    expl = f"Sibling descendant at depth {next_d} via {sib}"
                    _push(pid, child, "aunt-of", conf, "closure", expl, meta)
                    _push(child, pid, "niece-of", conf, "closure", expl, meta)

    # 4d) Cousins (general): degree and removed via MRCA within bounded depth
    # We derive cousin relationships by picking an ancestor of ego (depth a) and a descendant of that ancestor (depth b)
    # with a>=2 and b>=2 and computing degree=min(a,b)-1, removed=|a-b|. We cap degree<=2 and removed<=2.
    from collections import deque as _dq
    # Build ancestors (depth up) for ego if not already
    anc_up: Dict[int, int] = {}
    q = _dq([(pid, 0)])
    seen_u: Set[int] = {pid}
    MAX_A = 4
    while q:
        node, d = q.popleft()
        if d >= MAX_A:
            continue
        for (par, _t) in parents_of.get(node, []):
            if par in seen_u:
                continue
            seen_u.add(par)
            anc_up[par] = d + 1
            q.append((par, d + 1))

    # For each ancestor, walk down to descendants as candidate cousins
    for anc, a_depth in anc_up.items():
        if a_depth < 2:
            continue  # cousins require >= 2 on both sides
        # BFS down from this ancestor up to 4 levels
        qd = _dq([(anc, 0)])
        seen_d: Set[int] = {anc}
        MAX_B = 4
        while qd:
            node, d = qd.popleft()
            if d >= MAX_B:
                continue
            for (child, _t) in parent_to_children.get(node, []):
                if child in seen_d:
                    continue
                seen_d.add(child)
                nextd = d + 1
                qd.append((child, nextd))
                if child == pid:
                    continue
                b_depth = nextd
                if b_depth < 2:
                    continue
                degree = min(a_depth, b_depth) - 1
                removed = abs(a_depth - b_depth)
                if degree < 1:
                    continue
                # Target the requested set: second cousins, and once/twice removed for first cousins
                if not (
                    (degree == 1 and removed in (1, 2)) or
                    (degree == 2 and removed in (0, 1, 2))
                ):
                    continue
                if "cousin-of" in existing.get((pid, child), set()) or "cousin-of" in existing.get((child, pid), set()):
                    continue
                meta = {"inferred": True, "source": "closure", "kind": "cousin", "degree": int(degree), "removed": int(removed), "mrca": int(anc)}
                expl = f"Cousin via MRCA {anc} (a={a_depth}, b={b_depth})"
                _push(pid, child, "cousin-of", 0.52, "closure", expl, meta)
                _push(child, pid, "cousin-of", 0.52, "closure", expl, meta)

    # 4c) Aunt/Uncle-in-law via parent's sibling spouse: cousin's parents' spouse
    # If X is a cousin of pid, any partner of X's parent is an aunt/uncle-in-law of pid
    for k in list(ego_cousins):
        for (par_k, _rtk) in parents_of.get(k, []):
            for q in partners.get(par_k, set()):
                meta = {"inferred": True, "source": "closure", "kind": "aunt/uncle-in-law"}
                expl = f"Aunt/uncle-in-law via cousin {k} parent {par_k} partner {q}"
                _push(q, pid, "aunt-in-law-of", 0.55, "closure", expl, meta)
                _push(pid, q, "niece-in-law-of", 0.55, "closure", expl, meta)

    # 5) In-laws: from partners of pid
    for sp in partners.get(pid, set()):
        # spouse's parents -> parent-in-law
        for (sp_par, _rt) in parents_of.get(sp, []):
            if _has_explicit_conflict(existing, sp_par, pid, "parent-in-law-of"):
                continue
            if "parent-in-law-of" in existing.get((sp_par, pid), set()):
                continue
            meta = {"inferred": True, "source": "closure", "kind": "in-law"}
            expl = f"Parent-in-law via spouse {sp}"
            _push(sp_par, pid, "parent-in-law-of", 0.6, "closure", expl, meta)
            _push(pid, sp_par, "child-in-law-of", 0.6, "closure", expl, meta)
        # spouse's siblings -> sibling-in-law
        sp_sibs = set()
        for (sp_par, _rt) in parents_of.get(sp, []):
            for (sib, _rt2) in parent_to_children.get(sp_par, []):
                if sib != sp:
                    sp_sibs.add(sib)
        for sib in sp_sibs:
            if "sibling-in-law-of" in existing.get((pid, sib), set()) or "sibling-in-law-of" in existing.get((sib, pid), set()):
                continue
            meta = {"inferred": True, "source": "closure", "kind": "in-law"}
            expl = f"Sibling-in-law via spouse {sp}"
            _push(pid, sib, "sibling-in-law-of", 0.55, "closure", expl, meta)
            _push(sib, pid, "sibling-in-law-of", 0.55, "closure", expl, meta)
        # spouse's siblings' children -> niece/nephew (treat as aunt/uncle without in-law)
        for sib in sp_sibs:
            for (kid, _t) in parent_to_children.get(sib, []):
                meta = {"inferred": True, "source": "closure", "kind": "niece/nephew"}
                expl = f"Niece/nephew via spouse {sp} and sibling {sib}"
                _push(pid, kid, "aunt-of", 0.5, "closure", expl, meta)
                _push(kid, pid, "niece-of", 0.5, "closure", expl, meta)

    # 6) Profile hints anchored to owner (You)
    me_pid = await _find_me_person_id(db, user_id)
    if me_pid is not None:
        # scan people with role_hint relative to You
        for p in people.values():
            m = getattr(p, "meta", None)
            if not isinstance(m, dict):
                continue
            base = str(m.get("role_hint", "")).strip().lower()
            if not base:
                continue
            rid = int(p.id)
            if rid == me_pid:
                continue
            if base in {"mother", "father", "parent"}:
                if "parent-of" not in existing.get((rid, me_pid), set()):
                    meta = {"inferred": True, "source": "profile", "kind": "parent"}
                    _push(rid, me_pid, "parent-of", 0.9, "profile", "role_hint parent->you", meta)
                    _push(me_pid, rid, "child-of", 0.9, "profile", "role_hint parent->you", meta)
            elif base in {"son", "daughter", "child"}:
                if "parent-of" not in existing.get((me_pid, rid), set()):
                    meta = {"inferred": True, "source": "profile", "kind": "child"}
                    _push(me_pid, rid, "parent-of", 0.9, "profile", "role_hint you->child", meta)
                    _push(rid, me_pid, "child-of", 0.9, "profile", "role_hint you->child", meta)
            elif base in {"spouse", "partner", "husband", "wife"}:
                if "spouse-of" not in existing.get((me_pid, rid), set()) and "partner-of" not in existing.get((me_pid, rid), set()):
                    meta = {"inferred": True, "source": "profile", "kind": "spouse"}
                    _push(me_pid, rid, "spouse-of", 0.85, "profile", "role_hint spouse", meta)
                    _push(rid, me_pid, "spouse-of", 0.85, "profile", "role_hint spouse", meta)
            elif base in {"brother", "sister", "sibling"}:
                if "sibling-of" not in existing.get((me_pid, rid), set()) and "sibling-of" not in existing.get((rid, me_pid), set()):
                    meta = {"inferred": True, "source": "profile", "kind": "sibling"}
                    _push(me_pid, rid, "sibling-of", 0.7, "profile", "role_hint sibling", meta)
                    _push(rid, me_pid, "sibling-of", 0.7, "profile", "role_hint sibling", meta)

    # 7) Light text hints (mentions with role_hint near) — optional, conservative
    # If multiple mentions agree on parent/child/spouse, add with lower confidence
    # We scope to the focus person to keep this light.
    try:
        # Count role_hint mentions for this person
        rows = await db.execute(
            select(ResponsePerson.role_hint, ResponsePerson.person_id)
            .join(Response, Response.id == ResponsePerson.response_id)
            .where(Response.user_id == user_id, ResponsePerson.person_id == pid)
        )
        counts: Dict[str, int] = {}
        for rh, _pid in rows.all():
            if not rh:
                continue
            k = str(rh).strip().lower()
            if not k:
                continue
            counts[k] = counts.get(k, 0) + 1
        # Only act on strong hints
        rh_sorted = sorted(((k, c) for k, c in counts.items()), key=lambda x: -x[1])
        if rh_sorted:
            me = await _find_me_person_id(db, user_id)
            if me is not None:
                top, c = rh_sorted[0]
                if top in {"mother", "father", "parent"} and c >= 2:
                    if "parent-of" not in existing.get((pid, me), set()):
                        meta = {"inferred": True, "source": "text", "kind": "parent"}
                        _push(pid, me, "parent-of", 0.6, "text", "mentions: 'my mother/father'", meta)
                        _push(me, pid, "child-of", 0.6, "text", "mentions: 'my mother/father'", meta)
                if top in {"son", "daughter", "child"} and c >= 2:
                    if "parent-of" not in existing.get((me, pid), set()):
                        meta = {"inferred": True, "source": "text", "kind": "child"}
                        _push(me, pid, "parent-of", 0.6, "text", "mentions: 'my child'", meta)
                        _push(pid, me, "child-of", 0.6, "text", "mentions: 'my child'", meta)
                if top in {"spouse", "partner", "husband", "wife"} and c >= 2:
                    if "spouse-of" not in existing.get((me, pid), set()):
                        meta = {"inferred": True, "source": "text", "kind": "spouse"}
                        _push(me, pid, "spouse-of", 0.55, "text", "mentions: spouse/partner", meta)
                        _push(pid, me, "spouse-of", 0.55, "text", "mentions: spouse/partner", meta)
    except Exception:
        pass

    # Final pass: drop any candidate that exactly exists already
    result: list[CandidateEdge] = []
    for ce in cand:
        r = _canon_rel(ce.rel_type)
        if r in existing.get((ce.src_id, ce.dst_id), set()):
            continue
        # Basic sanity to avoid absurd self edges
        if int(ce.src_id) == int(ce.dst_id):
            continue
        result.append(ce)
    return result


async def commit_inferred_edges(db: AsyncSession, user_id: int, candidates: list[CandidateEdge]) -> int:
    """Persist candidates using upsert (ignore conflicts). Returns number attempted."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    n = 0
    for ce in candidates:
        try:
            meta = dict(ce.meta or {})
            meta.setdefault("inferred", True)
            meta.setdefault("source", ce.source)
            meta.setdefault("explain", ce.explain)
            stmt = pg_insert(RelationshipEdge.__table__).values(
                user_id=user_id,
                src_id=ce.src_id,
                dst_id=ce.dst_id,
                rel_type=_canon_rel(ce.rel_type),
                confidence=ce.confidence,
                meta=meta,
            ).on_conflict_do_nothing(index_elements=["user_id", "src_id", "dst_id", "rel_type"])
            await db.execute(stmt)
            n += 1
        except Exception:
            # keep going on individual failures
            continue
    return n


async def infer_all_for_user(db: AsyncSession, user_id: int, max_people: Optional[int] = None) -> Dict[int, list[CandidateEdge]]:
    out: Dict[int, list[CandidateEdge]] = {}
    rows = (await db.execute(select(Person.id).where(Person.owner_user_id == user_id))).scalars().all()
    ids = [int(x) for x in rows]
    if max_people:
        ids = ids[: max_people]
    for pid in ids:
        out[pid] = await infer_edges_for_person(db, user_id=user_id, person_id=pid)
    return out
