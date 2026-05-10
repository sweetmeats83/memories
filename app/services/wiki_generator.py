"""Wiki article generation service.

Generates third-person biographical wiki articles for Person records
by synthesizing all story excerpts that mention them.

Each article is stored in WikiArticle (one row per person+user). Generation
runs as a background task; status field tracks progress.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from html.parser import HTMLParser

import httpx
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.llm_client import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.settings.config import settings
from app.models import (
    Event,
    EventPerson,
    EventPlace,
    KinMembership,
    Person,
    PersonAlias,
    Place,
    PlaceAlias,
    Prompt,
    RelationshipEdge,
    Response,
    ResponseEvent,
    ResponsePerson,
    ResponsePlace,
    User,
    UserProfile,
    WikiArticle,
)
from app.services.people_acl import visible_group_ids

logger = logging.getLogger(__name__)

# Separate model env var so admins can point heavy wiki jobs at the 14B model
# while keeping fast ops on the smaller one.
WIKI_MODEL = os.getenv("WIKI_OLLAMA_MODEL", OLLAMA_MODEL)

_WIKILINK_RE = re.compile(r'\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]')


def resolve_wikilinks(content: str, link_map: dict[str, str]) -> str:
    """Replace [[Name]] or [[Name|Display]] with [Display](url).

    Names not found in link_map are rendered as plain text (brackets stripped).
    Call this at render time so links always reflect the current wiki state.
    """
    def _replace(m: re.Match) -> str:
        name = m.group(1).strip()
        display = (m.group(2) or name).strip()
        url = link_map.get(name)
        return f"[{display}]({url})" if url else display
    return _WIKILINK_RE.sub(_replace, content)


async def build_link_map(db: AsyncSession, group_id: int) -> dict[str, str]:
    """Return {display_name: wiki_url} for every entity with a ready article in the group."""
    link_map: dict[str, str] = {}

    for pid, name in (await db.execute(
        select(Person.id, Person.display_name)
        .join(WikiArticle, (WikiArticle.entity_type == "person") & (WikiArticle.entity_id == Person.id))
        .where(WikiArticle.group_id == group_id, WikiArticle.status == "ready")
    )).all():
        if name:
            link_map[name] = f"/wiki/people/{pid}"

    for pid, name in (await db.execute(
        select(Place.id, Place.name)
        .join(WikiArticle, (WikiArticle.entity_type == "place") & (WikiArticle.entity_id == Place.id))
        .where(WikiArticle.group_id == group_id, WikiArticle.status == "ready")
    )).all():
        if name:
            link_map[name] = f"/wiki/places/{pid}"

    for eid, name, year in (await db.execute(
        select(Event.id, Event.name, Event.year)
        .join(WikiArticle, (WikiArticle.entity_type == "event") & (WikiArticle.entity_id == Event.id))
        .where(WikiArticle.group_id == group_id, WikiArticle.status == "ready")
    )).all():
        if name:
            link_map[f"{name} {year}" if year else name] = f"/wiki/events/{eid}"

    return link_map


async def _narrator_ids_for_group(db: AsyncSession, group_id: int) -> list[int]:
    """Return user_ids of all KinGroup members, optionally excluding superusers."""
    stmt = select(KinMembership.user_id).where(KinMembership.group_id == group_id)
    if settings.WIKI_EXCLUDE_SUPERUSERS:
        stmt = (
            stmt.join(User, User.id == KinMembership.user_id)
            .where(User.is_superuser.is_(False))
        )
    return list(set((await db.execute(stmt)).scalars().all()))

_INVERT_REL: dict[str, str] = {
    "mother-of": "child-of",
    "father-of": "child-of",
    "parent-of": "child-of",
    "step-parent-of": "child-of",
    "adoptive-parent-of": "child-of",
    "child-of": "parent-of",
    "son-of": "parent-of",
    "daughter-of": "parent-of",
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
    "spouse-of": "spouse-of",
    "partner-of": "partner-of",
    "ex-spouse-of": "ex-spouse-of",
    "ex-partner-of": "ex-partner-of",
    "friend-of": "friend-of",
    "neighbor-of": "neighbor-of",
    "coworker-of": "coworker-of",
    "mentor-of": "student-of",
    "student-of": "mentor-of",
}


def _invert_rel(rt: str) -> str:
    return _INVERT_REL.get((rt or "").strip().lower(), rt)


# Higher = preferred when multiple edges exist for the same pair
_REL_SPECIFICITY: dict[str, int] = {
    "father-of": 10, "mother-of": 10,
    "son-of": 10, "daughter-of": 10,
    "grandfather-of": 10, "grandmother-of": 10,
    "grandson-of": 10, "granddaughter-of": 10,
    "brother-of": 10, "sister-of": 10,
    "half-brother-of": 10, "half-sister-of": 10,
    "step-father-of": 10, "step-mother-of": 10,
    "step-son-of": 10, "step-daughter-of": 10,
    "step-brother-of": 10, "step-sister-of": 10,
    "adoptive-father-of": 10, "adoptive-mother-of": 10,
    "spouse-of": 9, "partner-of": 9,
    "ex-spouse-of": 8, "ex-partner-of": 8,
    "aunt-of": 8, "uncle-of": 8,
    "niece-of": 8, "nephew-of": 8,
    "cousin-of": 7,
    "grandparent-of": 5, "grandchild-of": 5,
    "parent-of": 5, "child-of": 5,
    "sibling-of": 5, "half-sibling-of": 5,
    "step-parent-of": 5, "step-sibling-of": 5,
    "adoptive-parent-of": 5,
    "mentor-of": 4, "student-of": 4,
    "friend-of": 3, "neighbor-of": 3, "coworker-of": 3,
}

_GENDER_TO_MALE: dict[str, str] = {
    "parent-of": "father-of",
    "child-of": "son-of",
    "sibling-of": "brother-of",
    "half-sibling-of": "half-brother-of",
    "grandparent-of": "grandfather-of",
    "grandchild-of": "grandson-of",
    "step-parent-of": "step-father-of",
    "step-sibling-of": "step-brother-of",
    "adoptive-parent-of": "adoptive-father-of",
}

_GENDER_TO_FEMALE: dict[str, str] = {
    "parent-of": "mother-of",
    "child-of": "daughter-of",
    "sibling-of": "sister-of",
    "half-sibling-of": "half-sister-of",
    "grandparent-of": "grandmother-of",
    "grandchild-of": "granddaughter-of",
    "step-parent-of": "step-mother-of",
    "step-sibling-of": "step-sister-of",
    "adoptive-parent-of": "adoptive-mother-of",
}


def _gender_resolve_rel(rel_type: str, gender: str | None) -> str:
    """Map generic rel types (parent-of, child-of, sibling-of) to gendered versions."""
    g = (gender or "").lower()
    if g == "male":
        return _GENDER_TO_MALE.get(rel_type, rel_type)
    if g == "female":
        return _GENDER_TO_FEMALE.get(rel_type, rel_type)
    return rel_type


def dedupe_edges(
    src_rows: list[tuple],  # (rel_type, other_id, other_name)  — edge FROM subject
    dst_rows: list[tuple],  # (rel_type, other_id, other_name)  — edge TO subject (needs inversion)
    subject_gender: str | None,
) -> list[dict]:
    """
    Build a deduplicated, gender-resolved edge list for a person.

    Groups by other_person_id and keeps the highest-specificity relationship.
    Returns list of dicts: {person_id, name, rel_type, label}.
    """
    by_id: dict[int, list[tuple[str, str]]] = {}  # {other_id: [(resolved_rel_type, other_name)]}

    for rel_type, other_id, other_name in src_rows:
        rt = _gender_resolve_rel(rel_type, subject_gender)
        by_id.setdefault(other_id, []).append((rt, other_name))

    for rel_type, other_id, other_name in dst_rows:
        inv = _invert_rel(rel_type)
        rt = _gender_resolve_rel(inv, subject_gender)
        by_id.setdefault(other_id, []).append((rt, other_name))

    result: list[dict] = []
    for other_id, entries in by_id.items():
        best_rel, other_name = max(entries, key=lambda e: _REL_SPECIFICITY.get(e[0], 1))
        result.append({
            "person_id": other_id,
            "name": other_name,
            "rel_type": best_rel,
            "label": best_rel.replace("-", " "),
        })
    return result


async def _narrator_relationship_to_subject(
    db: AsyncSession, narrator_user_id: int, subject_person_id: int
) -> str | None:
    """
    Return a human-readable relationship label describing how the narrator relates
    to the wiki subject (e.g. "son of", "daughter of", "grandchild of").

    Looks up the narrator's self-person via UserProfile, checks for a direct
    edge in either direction, and applies gender resolution using the narrator's gender.
    """
    profile = await db.scalar(
        select(UserProfile).where(UserProfile.user_id == narrator_user_id)
    )
    self_pid = (profile.privacy_prefs or {}).get("self_person_id") if profile else None
    if not self_pid:
        return None

    narrator_person = await db.get(Person, self_pid)
    narrator_gender = getattr(narrator_person, "gender", None) if narrator_person else None

    # Direct edge: narrator → subject
    edge = await db.scalar(
        select(RelationshipEdge.rel_type)
        .where(RelationshipEdge.src_id == self_pid)
        .where(RelationshipEdge.dst_id == subject_person_id)
        .limit(1)
    )
    if edge:
        return _gender_resolve_rel(edge, narrator_gender).replace("-", " ")

    # Inverse edge: subject → narrator (invert, then gender-resolve from narrator's perspective)
    edge = await db.scalar(
        select(RelationshipEdge.rel_type)
        .where(RelationshipEdge.src_id == subject_person_id)
        .where(RelationshipEdge.dst_id == self_pid)
        .limit(1)
    )
    if edge:
        return _gender_resolve_rel(_invert_rel(edge), narrator_gender).replace("-", " ")

    return None


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|h[1-6]|blockquote)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&#39;", "'")
        .replace('&quot;', '"')
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def _narrator_names(db: AsyncSession, user_ids: list[int]) -> dict[int, str]:
    """
    Returns {user_id: first_name} for citation labels.
    Priority: first word of UserProfile.display_name → User.username → email prefix.
    """
    if not user_ids:
        return {}
    profiles = (await db.execute(
        select(UserProfile.user_id, UserProfile.display_name)
        .where(UserProfile.user_id.in_(user_ids))
    )).all()
    profile_map = {uid: dn for uid, dn in profiles if dn}

    users = (await db.execute(
        select(User.id, User.username, User.email)
        .where(User.id.in_(user_ids))
    )).all()
    user_map = {uid: (username, email) for uid, username, email in users}

    out: dict[int, str] = {}
    for uid in user_ids:
        if uid in profile_map:
            out[uid] = profile_map[uid].split()[0]
        else:
            username, email = user_map.get(uid, (None, ""))
            out[uid] = username or (email or "").split("@")[0]
    return out


async def _llm_json(system: str, user: str, timeout: int = 120) -> dict:
    prompt = f"{system}\n\nReturn ONLY valid JSON, no commentary.\n\nUSER:\n{user}"
    payload = {
        "model": WIKI_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": 16384},
        "think": False,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            r.raise_for_status()
            raw = _strip_thinking((r.json() or {}).get("response", ""))
        return json.loads(raw)
    except Exception as e:
        logger.warning("wiki _llm_json failed: %s", e)
        return {}


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks that Qwen3 may emit even with think=False."""
    return _THINK_TAG_RE.sub("", text).strip()


async def _llm_prose(system: str, user: str, timeout: int = 240) -> str:
    prompt = f"{system}\n\nUSER:\n{user}"
    payload = {
        "model": WIKI_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.25, "num_ctx": 16384},
        "think": False,
    }
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        r.raise_for_status()
        raw = (r.json() or {}).get("response", "")
        return _strip_thinking(raw)


def _resolve_citations(md: str, citation_map: dict, subject: str) -> str:
    """
    Replace [Name N] citation keys with numbered superscript anchors [1][2]…
    that link to a Sources section appended at the bottom.
    Each source entry links to the story response page.
    """
    key_to_num: dict[str, int] = {}

    def _replace(m: re.Match) -> str:
        key = m.group(1)
        if key not in citation_map:
            return m.group(0)
        if key not in key_to_num:
            key_to_num[key] = len(key_to_num) + 1
        n = key_to_num[key]
        return f'<sup class="wiki-cite"><a href="#ref-{n}">[{n}]</a></sup>'

    # Match [Name N] patterns — narrator name (one or more words) + space + integer
    result = re.sub(r"\[([A-Za-z][A-Za-z0-9 ]*?\s\d+)\]", _replace, md)

    if not key_to_num:
        return result

    # Build sources list ordered by first citation number
    ordered = sorted(key_to_num.items(), key=lambda x: x[1])
    rows: list[str] = []
    for key, num in ordered:
        meta = citation_map[key]
        narrator = meta.get("narrator", "")
        rel = meta.get("narrator_rel") or ""
        rel_label = f" · {rel} of {subject}" if rel and rel != "self" else ""
        title = meta.get("title") or "Untitled story"
        rows.append(
            f'<div class="wiki-ref" id="ref-{num}">'
            f'<span class="wiki-ref-n">[{num}]</span>'
            f'<span class="wiki-ref-body">'
            f'<a href="/response/{meta["response_id"]}" class="wiki-ref-link">{title}</a>'
            f'<span class="wiki-ref-meta"> — {narrator}{rel_label}</span>'
            f'</span>'
            f'</div>'
        )

    sources_block = (
        '\n\n<div class="wiki-sources">'
        '<div class="wiki-sources-heading">Sources</div>'
        + "".join(rows)
        + '</div>'
    )
    return result + sources_block


def _lede(content_md: str | None, max_chars: int = 350) -> str:
    """Extract the first meaningful paragraph from a wiki article (strip headings)."""
    if not content_md:
        return ""
    lines = []
    for line in content_md.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if lines:
                break  # stop at first heading after content
            continue
        lines.append(stripped)
        if sum(len(l) for l in lines) >= max_chars:
            break
    text = " ".join(lines)
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


async def _load_wiki_world(
    db: AsyncSession,
    group_id: int,
    focal_person_id: int | None = None,
    focal_place_id: int | None = None,
    focal_event_id: int | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """
    Returns (link_map, related_summaries) for use in generation prompts.

    link_map       — {display_name: wiki_url} for every entity with a ready article.
                     Lets the LLM emit markdown cross-links.
    related_summaries — [{name, relationship, url, lede}] for entities directly
                     connected to the focal entity (top ~6). Used as consistency context.
    """
    # ── 1. Build link map for all ready articles ──────────────────────────────
    link_map = await build_link_map(db, group_id)

    # ── 2. Load related entity summaries ─────────────────────────────────────
    related: list[dict] = []

    if focal_person_id:
        # Direct family / relationship edges
        edge_scope = or_(
            RelationshipEdge.group_id == group_id,
            RelationshipEdge.user_id.in_(
                select(KinMembership.user_id).where(KinMembership.group_id == group_id)
            ),
        )
        DstP = aliased(Person)
        SrcP = aliased(Person)

        for rel_type, other_name, other_id in (await db.execute(
            select(RelationshipEdge.rel_type, DstP.display_name, DstP.id)
            .join(DstP, DstP.id == RelationshipEdge.dst_id)
            .where(RelationshipEdge.src_id == focal_person_id)
            .where(edge_scope)
            .limit(8)
        )).all():
            art = await db.scalar(
                select(WikiArticle).where(
                    WikiArticle.entity_type == "person",
                    WikiArticle.entity_id == other_id,
                    WikiArticle.group_id == group_id,
                    WikiArticle.status == "ready",
                )
            )
            if art and art.content_md:
                related.append({
                    "name": other_name,
                    "url": f"/wiki/people/{other_id}",
                    "relationship": rel_type.replace("-", " "),
                    "lede": _lede(art.content_md),
                })

        for rel_type, other_name, other_id in (await db.execute(
            select(RelationshipEdge.rel_type, SrcP.display_name, SrcP.id)
            .join(SrcP, SrcP.id == RelationshipEdge.src_id)
            .where(RelationshipEdge.dst_id == focal_person_id)
            .where(edge_scope)
            .limit(8)
        )).all():
            if any(r["url"] == f"/wiki/people/{other_id}" for r in related):
                continue
            art = await db.scalar(
                select(WikiArticle).where(
                    WikiArticle.entity_type == "person",
                    WikiArticle.entity_id == other_id,
                    WikiArticle.group_id == group_id,
                    WikiArticle.status == "ready",
                )
            )
            if art and art.content_md:
                inv = _invert_rel(rel_type)
                related.append({
                    "name": other_name,
                    "url": f"/wiki/people/{other_id}",
                    "relationship": inv.replace("-", " "),
                    "lede": _lede(art.content_md),
                })

        # Events this person attended
        for ev_id, ev_name, ev_year in (await db.execute(
            select(Event.id, Event.name, Event.year)
            .join(EventPerson, EventPerson.event_id == Event.id)
            .where(EventPerson.person_id == focal_person_id)
            .limit(5)
        )).all():
            art = await db.scalar(
                select(WikiArticle).where(
                    WikiArticle.entity_type == "event",
                    WikiArticle.entity_id == ev_id,
                    WikiArticle.group_id == group_id,
                    WikiArticle.status == "ready",
                )
            )
            if art and art.content_md:
                label = f"{ev_name} {ev_year}" if ev_year else ev_name
                related.append({"name": label, "url": f"/wiki/events/{ev_id}", "relationship": "attended event", "lede": _lede(art.content_md)})

    elif focal_place_id:
        # Events at this place
        for ev_id, ev_name, ev_year in (await db.execute(
            select(Event.id, Event.name, Event.year)
            .join(EventPlace, EventPlace.event_id == Event.id)
            .where(EventPlace.place_id == focal_place_id)
            .limit(5)
        )).all():
            art = await db.scalar(
                select(WikiArticle).where(WikiArticle.entity_type == "event", WikiArticle.entity_id == ev_id, WikiArticle.group_id == group_id, WikiArticle.status == "ready")
            )
            if art and art.content_md:
                label = f"{ev_name} {ev_year}" if ev_year else ev_name
                related.append({"name": label, "url": f"/wiki/events/{ev_id}", "relationship": "event held here", "lede": _lede(art.content_md)})

    elif focal_event_id:
        # Places for this event
        for pl_id, pl_name in (await db.execute(
            select(Place.id, Place.name)
            .join(EventPlace, EventPlace.place_id == Place.id)
            .where(EventPlace.event_id == focal_event_id)
            .limit(3)
        )).all():
            art = await db.scalar(
                select(WikiArticle).where(WikiArticle.entity_type == "place", WikiArticle.entity_id == pl_id, WikiArticle.group_id == group_id, WikiArticle.status == "ready")
            )
            if art and art.content_md:
                related.append({"name": pl_name, "url": f"/wiki/places/{pl_id}", "relationship": "event location", "lede": _lede(art.content_md)})

        # Attendees of this event
        AttP = aliased(Person)
        for p_id, p_name in (await db.execute(
            select(AttP.id, AttP.display_name)
            .join(EventPerson, EventPerson.person_id == AttP.id)
            .where(EventPerson.event_id == focal_event_id)
            .limit(6)
        )).all():
            art = await db.scalar(
                select(WikiArticle).where(WikiArticle.entity_type == "person", WikiArticle.entity_id == p_id, WikiArticle.group_id == group_id, WikiArticle.status == "ready")
            )
            if art and art.content_md:
                related.append({"name": p_name, "url": f"/wiki/people/{p_id}", "relationship": "attendee", "lede": _lede(art.content_md)})

    return link_map, related[:8]  # cap to keep context manageable


def _build_world_prompt_block(link_map: dict[str, str], related: list[dict], exclude_name: str = "") -> str:
    """
    Builds the WIKI WORLD CONTEXT block injected into generation prompts.
    Instructs the LLM to use [[Name]] wikilink syntax; resolved to URLs at render time.
    """
    lines: list[str] = []

    if link_map:
        people_names = [n for n, u in link_map.items() if u.startswith("/wiki/people/") and n != exclude_name]
        place_names  = [n for n, u in link_map.items() if u.startswith("/wiki/places/")]
        event_names  = [n for n, u in link_map.items() if u.startswith("/wiki/events/")]

        if people_names or place_names or event_names:
            lines.append("CROSS-LINKING:")
            lines.append(
                "When you mention any of the names listed below, wrap the FIRST occurrence "
                "in double square brackets: [[Name]]. Do not wrap the article's own subject."
            )
            if people_names:
                lines.append("  People:  " + ", ".join(people_names[:20]))
            if place_names:
                lines.append("  Places:  " + ", ".join(place_names[:15]))
            if event_names:
                lines.append("  Events:  " + ", ".join(event_names[:15]))

    if related:
        lines.append("")
        lines.append(
            "RELATED WIKI ARTICLES (read for consistency — do not copy, do not contradict):"
        )
        for r in related:
            lines.append(f"  {r['name']} ({r['relationship']}): {r['lede']}")

    return "\n".join(lines)


def _stub_article(person: Person, aliases: list[str], relationships: list[str]) -> str:
    lines = [f"## {person.display_name}"]
    if aliases:
        lines.append(f"*Also known as: {', '.join(aliases)}*\n")
    if person.birth_year or person.death_year:
        yr = f"{person.birth_year or '?'}–{person.death_year or 'present'}"
        lines.append(f"**{yr}**\n")
    if relationships:
        lines.append("**Relationships:** " + "; ".join(relationships) + "\n")
    if person.notes:
        lines.append(person.notes + "\n")
    lines.append(
        "*No stories mention this person yet. "
        "Record your memories to generate a richer article.*"
    )
    return "\n".join(lines)


async def _do_generate(db: AsyncSession, article: WikiArticle, person_id: int, group_id: int) -> None:
    """Inner generation logic — db session is already open and article row exists."""

    # --- Load person ---
    person = await db.get(Person, person_id)
    if not person:
        raise ValueError(f"Person {person_id} not found")

    # --- Load aliases ---
    alias_rows = (
        await db.execute(select(PersonAlias.alias).where(PersonAlias.person_id == person_id))
    ).scalars().all()
    aliases = [a for a in alias_rows if a]

    # --- All group narrators ---
    narrator_ids = await _narrator_ids_for_group(db, group_id)
    narrator_name_map = await _narrator_names(db, narrator_ids)

    # --- Load relationships (group-scoped edges + private social edges from any narrator) ---
    edge_scope = or_(
        RelationshipEdge.group_id == group_id,
        RelationshipEdge.user_id.in_(narrator_ids),
    )

    DstPerson = aliased(Person)
    SrcPerson = aliased(Person)

    src_rows = (
        await db.execute(
            select(RelationshipEdge.rel_type, DstPerson.id, DstPerson.display_name)
            .join(DstPerson, DstPerson.id == RelationshipEdge.dst_id)
            .where(RelationshipEdge.src_id == person_id)
            .where(edge_scope)
        )
    ).all()

    dst_rows = (
        await db.execute(
            select(RelationshipEdge.rel_type, SrcPerson.id, SrcPerson.display_name)
            .join(SrcPerson, SrcPerson.id == RelationshipEdge.src_id)
            .where(RelationshipEdge.dst_id == person_id)
            .where(edge_scope)
        )
    ).all()

    clean_edges = dedupe_edges(
        [(r[0], r[1], r[2]) for r in src_rows],
        [(r[0], r[1], r[2]) for r in dst_rows],
        getattr(person, "gender", None),
    )
    relationships = [e["label"] + " " + e["name"] for e in clean_edges]

    # Pre-compute each narrator's relationship to the subject
    narrator_rel_cache: dict[int, str | None] = {}
    for nid in narrator_ids:
        narrator_rel_cache[nid] = await _narrator_relationship_to_subject(db, nid, person_id)

    # --- Load story mentions from all family narrators ---
    NarratorUser = aliased(User)
    mention_rows = (
        await db.execute(
            select(ResponsePerson, Response, Prompt, NarratorUser)
            .join(Response, Response.id == ResponsePerson.response_id)
            .join(Prompt, Prompt.id == Response.prompt_id, isouter=True)
            .join(NarratorUser, NarratorUser.id == Response.user_id)
            .where(ResponsePerson.person_id == person_id)
            .where(Response.user_id.in_(narrator_ids))
            .order_by(Response.created_at.asc())
        )
    ).unique().all()

    # Mention stories: other family members wrote about this person
    mention_stories: list[dict] = []
    seen_response_ids: set[int] = set()
    for rp, resp, prompt, narrator in mention_rows:
        raw = resp.ai_polished or resp.response_text or resp.transcription or ""
        text = _strip_html(raw)
        narrator_label = narrator_name_map.get(narrator.id) or narrator.username or narrator.email.split("@")[0]
        narrator_rel = narrator_rel_cache.get(narrator.id)
        if text:
            mention_stories.append(
                {
                    "prompt": prompt.text if prompt else "",
                    "title": resp.title or "",
                    "text": text,
                    "role_hint": rp.role_hint or "",
                    "response_id": resp.id,
                    "narrator": narrator_label,
                    "narrator_rel": narrator_rel,
                }
            )
            seen_response_ids.add(resp.id)

    # Subject's own stories: if this person is a registered user, pull their authored responses.
    # These are the most valuable source — the subject writing about their own life in the first person.
    own_stories: list[dict] = []
    all_profiles = (await db.execute(select(UserProfile))).scalars().all()
    subject_user_id: int | None = next(
        (p.user_id for p in all_profiles
         if (p.privacy_prefs or {}).get("self_person_id") == person_id),
        None,
    )
    if subject_user_id:
        SubjectUser = aliased(User)
        own_stmt = (
            select(Response, Prompt, SubjectUser)
            .join(Prompt, Prompt.id == Response.prompt_id, isouter=True)
            .join(SubjectUser, SubjectUser.id == Response.user_id)
            .where(Response.user_id == subject_user_id)
            .order_by(Response.created_at.asc())
        )
        if seen_response_ids:
            own_stmt = own_stmt.where(Response.id.notin_(list(seen_response_ids)))
        own_rows = (await db.execute(own_stmt)).unique().all()
        for resp, prompt, narrator in own_rows:
            raw = resp.ai_polished or resp.response_text or resp.transcription or ""
            text = _strip_html(raw)
            narrator_label = narrator_name_map.get(narrator.id) or narrator.username or narrator.email.split("@")[0]
            if text:
                own_stories.append(
                    {
                        "prompt": prompt.text if prompt else "",
                        "title": resp.title or "",
                        "text": text,
                        "role_hint": "subject",
                        "response_id": resp.id,
                        "narrator": narrator_label,
                        "narrator_rel": "self",
                    }
                )
                seen_response_ids.add(resp.id)

    # Own stories go first (primary source), then mentions by others
    stories = own_stories + mention_stories
    source_response_ids = list(seen_response_ids)

    subject = person.display_name

    # Build the subject's relationship graph so the LLM can resolve first-person
    # role references ("my dad", "my mom", "my wife") to actual names.
    subject_narrator_graph: list[dict] = []
    if subject_user_id:
        try:
            from app.services.people import build_narrator_graph as _build_narrator_graph
            subject_narrator_graph = await _build_narrator_graph(db, subject_user_id)
        except Exception:
            pass

    _grouped_rels: dict[str, list[str]] = {}
    for entry in subject_narrator_graph:
        _grouped_rels.setdefault(entry["role"], []).append(entry["display_name"])
    subject_rel_context = (
        f"\n{subject}'s actual relatives (use these when resolving first-person role references):\n"
        + "\n".join(f"  {role}: {', '.join(names)}" for role, names in _grouped_rels.items())
    ) if _grouped_rels else ""

    # Load cross-wiki world context before committing the session
    link_map, related_wiki = await _load_wiki_world(db, group_id, focal_person_id=person_id)
    world_block = _build_world_prompt_block(link_map, related_wiki, exclude_name=subject)

    # No stories → stub
    if not stories:
        article.content_md = _stub_article(person, aliases, relationships)
        article.status = "ready"
        article.source_count = 0
        article.model_name = WIKI_MODEL
        article.generated_at = datetime.now(timezone.utc)
        await db.commit()
        return

    # Release connection before long LLM calls
    await db.commit()

    # --- Pass 1: extract structured facts ---
    alias_str = f" (also known as: {', '.join(aliases)})" if aliases else ""

    p1_sys = f"""\
You are extracting verifiable biographical facts about ONE person: {subject}{alias_str}.
The sources are personal memoir stories written by family members. Each story is emotionally rich \
but may mention many people — focus EXCLUSIVELY on concrete, factual details about {subject}.
Ignore feelings, narrative, and details about other people.
Extract ONLY facts that are directly and explicitly stated. Do not infer or guess.
Return JSON with this exact shape (use null for any field not found):
{{
  "birth_year": null,
  "birth_place": null,
  "death_year": null,
  "death_place": null,
  "occupation": null,
  "education": null,
  "places_lived": [],
  "key_traits": [],
  "notable_facts": []
}}"""

    def _narrator_label(s: dict) -> str:
        rel = s.get("narrator_rel")
        if rel == "self":
            return f"{s['narrator']} (the subject, writing in the first person)"
        return f"{s['narrator']}{f' ({rel} of {subject})' if rel else ''}"

    p1_user = (
        f"Subject: {subject}{alias_str}\n"
        f"Known relationships: {', '.join(relationships) if relationships else 'unknown'}"
        f"{subject_rel_context}\n\n"
        f"Source stories (each labelled with the narrator and their relationship to {subject}):\n\n"
        + "\n\n---\n\n".join(
            f"NARRATOR: {_narrator_label(s)}\n"
            + (f"PROMPT: {s['prompt']}\n" if s["prompt"] else "")
            + (f"TITLE: {s['title']}\n" if s["title"] else "")
            + (f"SUBJECT'S ROLE IN THIS STORY: {s['role_hint']}\n" if s["role_hint"] and s['role_hint'] != 'subject' else "")
            + f"STORY:\n{s['text'][:2500]}"
            for s in stories[:15]
        )
    )

    facts = await _llm_json(p1_sys, p1_user, timeout=120)

    # --- Pass 2: generate factual wiki article ---
    narrators = sorted({s["narrator"] for s in stories})
    def _narrator_ctx_label(name: str) -> str:
        rel = next((s["narrator_rel"] for s in stories if s["narrator"] == name and s.get("narrator_rel")), None)
        if rel == "self":
            return f"{name} (the subject)"
        return f"{name} ({rel})" if rel else f"{name} (family member)"
    narrator_ctx = (
        f"Sources: {len(stories)} family stories from {len(narrators)} contributor(s) — "
        + ", ".join(_narrator_ctx_label(n) for n in narrators)
        + "."
    )

    p2_sys = f"""\
You are writing a biographical wiki article for a private family archive.
The article is about: {subject}{alias_str}.
{narrator_ctx}

IMPORTANT DISTINCTION: The source material is personal memoir stories — vivid, emotional, \
first-person accounts. Your job is to distill the FACTS out of those stories into a \
concise, encyclopedic third-person article. Do NOT retell the stories. Do NOT quote \
emotional passages. Write the way a careful family historian or Wikipedia editor would.

RULES:
1. State only facts that are directly supported by the source stories. Do not invent, infer, or fill gaps.
2. Write in third person ("She was born…", "He worked as…").
3. Cite inline using the provided citation keys in square brackets immediately after the fact they support. \
   Example: "She was born in 1932 in Cork [Sarah 1] and later moved to Dublin [Nathan 2]."
4. PRONOUN RESOLUTION — resolve relationship words using the narrator's perspective:
   - If a narrator is marked "the subject, writing in the first person": \
     "I", "me", "my" all refer to {subject} themselves. \
     Use the relative list below to map role words to actual names — \
     e.g. "my dad" = {subject}'s father, "my mom" = {subject}'s mother.{subject_rel_context}
   - If a narrator has a stated relationship (e.g. "son of {subject}"): \
     "my dad" or "my father" refers to {subject}; \
     "my mom" refers to {subject}'s spouse/partner.
   - Never confuse the narrator's relatives with the subject's relatives.
5. Organise into 2–4 ## sections based on the actual facts available (e.g. Early Life, Career, Family).
6. If information is thin, write a shorter article rather than padding with vague statements.
7. Do NOT include a references or sources section — that will be added automatically.
8. Output clean markdown only — no JSON, no preamble, no closing commentary.
9. Target 250–500 words.
{(chr(10) + world_block) if world_block else ""}"""

    list_keys = {"key_traits", "notable_facts", "places_lived"}
    facts_block = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}"
        for k, v in facts.items()
        if v and k not in list_keys
    )
    for list_key in ("places_lived", "key_traits"):
        vals = facts.get(list_key)
        if vals:
            facts_block += f"\n- {list_key.replace('_', ' ').title()}: {', '.join(vals)}"
    if facts.get("notable_facts"):
        facts_block += "\n- Notable facts:\n" + "\n".join(f"  • {f}" for f in facts["notable_facts"])

    eff_birth = facts.get("birth_year") or person.birth_year
    eff_death = facts.get("death_year") or person.death_year

    header = f"**{person.display_name}**"
    if eff_birth:
        header += f" (born {eff_birth}" + (f", died {eff_death}" if eff_death else "") + ")"
    if aliases:
        header += f"\nAlso known as: {', '.join(aliases)}"
    # Relationships intentionally omitted — sidebar shows them; LLM should mention
    # only the closest ones organically when supported by the source stories.

    # Build per-narrator citation keys: [Nathan 1], [Nathan 2], [Sarah 1], …
    narrator_counters: dict[str, int] = {}
    citation_map: dict[str, dict] = {}  # key → {response_id, narrator, narrator_rel, title, prompt}
    cited_stories = stories[:20]
    for s in cited_stories:
        n = s["narrator"]
        narrator_counters[n] = narrator_counters.get(n, 0) + 1
        key = f"{n} {narrator_counters[n]}"
        citation_map[key] = {
            "response_id": s["response_id"],
            "narrator": n,
            "narrator_rel": s["narrator_rel"],
            "title": s["title"] or s["prompt"] or "Untitled",
        }
        s["cite_key"] = key  # attach to story dict for use in prompt

    cite_index = "\n".join(
        f"[{key}] — {meta['narrator']}"
        + (f" ({meta['narrator_rel']} of {subject})" if meta["narrator_rel"] else "")
        + (f': "{meta["title"]}"' if meta["title"] else "")
        for key, meta in citation_map.items()
    )

    p2_user = (
        f"{header}\n\n"
        + (f"Extracted facts:\n{facts_block}\n\n" if facts_block else "")
        + f"Citation index (use these keys inline when you state a fact):\n{cite_index}\n\n"
        + f"Source stories:\n\n"
        + "\n\n---\n\n".join(
            f"Citation key: [{s['cite_key']}]\nNarrator: {_narrator_label(s)}\n"
            + (f'Story prompt: "{s["prompt"]}"\n' if s["prompt"] else "")
            + (f"Subject's role: {s['role_hint']}\n" if s["role_hint"] and s['role_hint'] != 'subject' else "")
            + s["text"][:2500]
            for s in cited_stories
        )
    )

    content_md = await _llm_prose(p2_sys, p2_user, timeout=270)

    if not content_md.strip():
        content_md = _stub_article(person, aliases, relationships)
    else:
        content_md = _resolve_citations(content_md, citation_map, subject)

    # --- Save ---
    # Save a revision snapshot of the outgoing AI article before overwriting.
    # user_edited_md is NEVER cleared by AI regeneration — the user's override
    # persists until they explicitly revert to AI via the UI.
    await db.refresh(article)
    if article.content_md:
        from app.models import WikiRevision
        db.add(WikiRevision(
            wiki_article_id=article.id,
            content_md=article.content_md,
            source="ai",
        ))
    article.content_md = content_md
    article.status = "ready"
    article.source_count = len(stories)
    article.model_name = WIKI_MODEL
    article.generated_at = datetime.now(timezone.utc)
    await db.commit()


async def generate_person_wiki(person_id: int, group_id: int) -> None:
    """Background-safe entry point: creates its own DB session.

    Upserts one shared WikiArticle per (person, kin_group).
    All family members in the group see the same article.
    """
    from app.database import async_session_maker

    async with async_session_maker() as db:
        article = await db.scalar(
            select(WikiArticle).where(
                WikiArticle.entity_type == "person",
                WikiArticle.entity_id == person_id,
                WikiArticle.group_id == group_id,
            )
        )
        if not article:
            article = WikiArticle(
                entity_type="person",
                entity_id=person_id,
                group_id=group_id,
                status="generating",
            )
            db.add(article)
        else:
            article.status = "generating"
            article.error_msg = None
        await db.commit()

        try:
            await _do_generate(db, article, person_id, group_id)
        except Exception as exc:
            logger.exception(
                "wiki generation failed for person=%s group=%s", person_id, group_id
            )
            try:
                await db.refresh(article)
                article.status = "error"
                article.error_msg = str(exc)[:500]
                await db.commit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Place wiki
# ---------------------------------------------------------------------------

async def _do_generate_place(db: AsyncSession, article: WikiArticle, place_id: int, group_id: int) -> None:
    place = await db.get(Place, place_id)
    if not place:
        raise ValueError(f"Place {place_id} not found")

    alias_rows = (await db.execute(
        select(PlaceAlias.alias).where(PlaceAlias.place_id == place_id)
    )).scalars().all()
    aliases = [a for a in alias_rows if a]

    narrator_ids = await _narrator_ids_for_group(db, group_id)
    narrator_name_map = await _narrator_names(db, narrator_ids)

    NarrUser = aliased(User)
    story_rows = (await db.execute(
        select(Response, Prompt, NarrUser)
        .join(ResponsePlace, ResponsePlace.response_id == Response.id)
        .join(Prompt, Prompt.id == Response.prompt_id, isouter=True)
        .join(NarrUser, NarrUser.id == Response.user_id)
        .where(ResponsePlace.place_id == place_id)
        .where(Response.user_id.in_(narrator_ids))
        .order_by(Response.created_at.asc())
    )).unique().all()

    stories = []
    for resp, prompt, narrator in story_rows:
        raw = resp.ai_polished or resp.response_text or resp.transcription or ""
        text = _strip_html(raw)
        if text:
            stories.append({
                "response_id": resp.id,
                "title": resp.title or "",
                "prompt": prompt.text if prompt else "",
                "text": text,
                "narrator": narrator_name_map.get(narrator.id) or narrator.username or narrator.email.split("@")[0],
            })

    name = place.display_name if hasattr(place, "display_name") else place.name
    alias_str = f" (also known as: {', '.join(aliases)})" if aliases else ""
    location_parts = [p for p in [place.address, place.city, place.state, place.country] if p]
    location_str = ", ".join(location_parts) if location_parts else ""

    link_map, related_wiki = await _load_wiki_world(db, group_id, focal_place_id=place_id)
    world_block = _build_world_prompt_block(link_map, related_wiki, exclude_name=name)

    if not stories:
        md = f"## {name}{alias_str}\n"
        if location_str:
            md += f"*{location_str}*\n\n"
        if place.notes:
            md += place.notes + "\n\n"
        md += "*No stories mention this place yet.*"
        article.content_md = md
        article.status = "ready"
        article.source_count = 0
        article.model_name = WIKI_MODEL
        article.generated_at = datetime.now(timezone.utc)
        await db.commit()
        return

    await db.commit()

    sys_prompt = f"""\
You are writing a wiki entry about a place for a private family archive.
The place is: {name}{alias_str}{f' — located in {location_str}' if location_str else ''}.
The sources are personal family stories describing events and memories at this place.

Write a warm, narrative entry describing what this place was like, what happened there, and
what role it played in family life. Use third person. Cite stories inline with the narrator's
name in square brackets, e.g. [Nathan 1].

RULES:
1. Only state facts directly from the source stories.
2. Organise into 2–3 ## sections (e.g. About the Place, Memories, Events Held Here).
3. Do not invent details. If stories are sparse, write shorter.
4. Output clean markdown only — no preamble, no closing commentary.
5. Target 150–400 words.
{(chr(10) + world_block) if world_block else ""}"""

    narrator_counters: dict[str, int] = {}
    citation_map: dict[str, dict] = {}
    for s in stories[:10]:
        n = s["narrator"]
        narrator_counters[n] = narrator_counters.get(n, 0) + 1
        key = f"{n} {narrator_counters[n]}"
        citation_map[key] = {"response_id": s["response_id"], "narrator": n, "narrator_rel": None, "title": s["title"] or s["prompt"] or "Untitled"}
        s["cite_key"] = key

    cite_index = "\n".join(
        f"[{k}] — {v['narrator']}: \"{v['title']}\""
        for k, v in citation_map.items()
    )
    user_prompt = (
        f"Place: {name}{alias_str}\n"
        + (f"Location: {location_str}\n" if location_str else "")
        + (f"Notes: {place.notes}\n" if place.notes else "")
        + f"\nCitation index:\n{cite_index}\n\nSource stories:\n\n"
        + "\n\n---\n\n".join(
            f"Citation key: [{s['cite_key']}]\nNarrator: {s['narrator']}\n"
            + (f'Prompt: "{s["prompt"]}"\n' if s["prompt"] else "")
            + s["text"][:2500]
            for s in stories[:20]
        )
    )

    content_md = await _llm_prose(sys_prompt, user_prompt, timeout=180)
    if not content_md.strip():
        content_md = f"## {name}\n*Article generation produced no content.*"
    else:
        content_md = _resolve_citations(content_md, citation_map, name)

    await db.refresh(article)
    article.content_md = content_md
    article.user_edited_md = None
    article.status = "ready"
    article.source_count = len(stories)
    article.model_name = WIKI_MODEL
    article.generated_at = datetime.now(timezone.utc)
    await db.commit()


async def generate_place_wiki(place_id: int, group_id: int) -> None:
    from app.database import async_session_maker
    async with async_session_maker() as db:
        article = await db.scalar(
            select(WikiArticle).where(
                WikiArticle.entity_type == "place",
                WikiArticle.entity_id == place_id,
                WikiArticle.group_id == group_id,
            )
        )
        if not article:
            article = WikiArticle(entity_type="place", entity_id=place_id, group_id=group_id, status="generating")
            db.add(article)
        else:
            article.status = "generating"
            article.error_msg = None
        await db.commit()
        try:
            await _do_generate_place(db, article, place_id, group_id)
        except Exception as exc:
            logger.exception("wiki generation failed for place=%s group=%s", place_id, group_id)
            try:
                await db.refresh(article)
                article.status = "error"
                article.error_msg = str(exc)[:500]
                await db.commit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Event wiki
# ---------------------------------------------------------------------------

async def _do_generate_event(db: AsyncSession, article: WikiArticle, event_id: int, group_id: int) -> None:
    event = await db.get(Event, event_id)
    if not event:
        raise ValueError(f"Event {event_id} not found")

    alias_rows = (await db.execute(
        select(EventAlias.alias).where(EventAlias.event_id == event_id)
    )).scalars().all()
    aliases = [a for a in alias_rows if a]

    # Linked places
    PlaceJoin = aliased(Place)
    place_rows = (await db.execute(
        select(PlaceJoin.name)
        .join(EventPlace, EventPlace.place_id == PlaceJoin.id)
        .where(EventPlace.event_id == event_id)
    )).scalars().all()
    linked_places = [p for p in place_rows if p]

    # Linked attendees
    AttPerson = aliased(Person)
    attendee_rows = (await db.execute(
        select(AttPerson.display_name, EventPerson.role_hint)
        .join(AttPerson, AttPerson.id == EventPerson.person_id)
        .where(EventPerson.event_id == event_id)
    )).all()
    attendees = [
        f"{name}{f' ({role})' if role else ''}"
        for name, role in attendee_rows
        if name
    ]

    narrator_ids = await _narrator_ids_for_group(db, group_id)
    narrator_name_map = await _narrator_names(db, narrator_ids)

    NarrUser = aliased(User)
    story_rows = (await db.execute(
        select(Response, Prompt, NarrUser)
        .join(ResponseEvent, ResponseEvent.response_id == Response.id)
        .join(Prompt, Prompt.id == Response.prompt_id, isouter=True)
        .join(NarrUser, NarrUser.id == Response.user_id)
        .where(ResponseEvent.event_id == event_id)
        .where(Response.user_id.in_(narrator_ids))
        .order_by(Response.created_at.asc())
    )).unique().all()

    stories = []
    for resp, prompt, narrator in story_rows:
        raw = resp.ai_polished or resp.response_text or resp.transcription or ""
        text = _strip_html(raw)
        if text:
            stories.append({
                "response_id": resp.id,
                "title": resp.title or "",
                "prompt": prompt.text if prompt else "",
                "text": text,
                "narrator": narrator_name_map.get(narrator.id) or narrator.username or narrator.email.split("@")[0],
            })

    label = event.name
    if event.year:
        label = f"{event.name} {event.year}"
    alias_str = f" (also known as: {', '.join(aliases)})" if aliases else ""

    link_map, related_wiki = await _load_wiki_world(db, group_id, focal_event_id=event_id)
    world_block = _build_world_prompt_block(link_map, related_wiki, exclude_name=label)

    if not stories:
        md = f"## {label}{alias_str}\n"
        if linked_places:
            md += f"*Location(s): {', '.join(linked_places)}*\n\n"
        if attendees:
            md += f"*Attendees: {', '.join(attendees)}*\n\n"
        if event.notes:
            md += event.notes + "\n\n"
        md += "*No stories describe this event yet.*"
        article.content_md = md
        article.status = "ready"
        article.source_count = 0
        article.model_name = WIKI_MODEL
        article.generated_at = datetime.now(timezone.utc)
        await db.commit()
        return

    await db.commit()

    sys_prompt = f"""\
You are writing a wiki entry about a family event for a private family archive.
The event is: {label}{alias_str}.
{f'Known locations: {", ".join(linked_places)}.' if linked_places else ''}
{f'Known attendees: {", ".join(attendees)}.' if attendees else ''}
The sources are personal family stories describing this event.

Write a warm, narrative entry covering what happened at this event — the setting, who was there,
what took place, and why it was memorable. Use third person. Cite stories inline, e.g. [Nathan 1].

RULES:
1. Only state facts directly from the source stories.
2. Organise into 2–3 ## sections (e.g. Overview, What Happened, People & Places).
3. Do not invent details or fill gaps with assumptions.
4. Output clean markdown only — no preamble, no closing commentary.
5. Target 150–400 words.
{(chr(10) + world_block) if world_block else ""}"""

    narrator_counters: dict[str, int] = {}
    citation_map: dict[str, dict] = {}
    for s in stories[:10]:
        n = s["narrator"]
        narrator_counters[n] = narrator_counters.get(n, 0) + 1
        key = f"{n} {narrator_counters[n]}"
        citation_map[key] = {"response_id": s["response_id"], "narrator": n, "narrator_rel": None, "title": s["title"] or s["prompt"] or "Untitled"}
        s["cite_key"] = key

    cite_index = "\n".join(
        f"[{k}] — {v['narrator']}: \"{v['title']}\""
        for k, v in citation_map.items()
    )
    user_prompt = (
        f"Event: {label}{alias_str}\n"
        + (f"Locations: {', '.join(linked_places)}\n" if linked_places else "")
        + (f"Attendees: {', '.join(attendees)}\n" if attendees else "")
        + (f"Notes: {event.notes}\n" if event.notes else "")
        + f"\nCitation index:\n{cite_index}\n\nSource stories:\n\n"
        + "\n\n---\n\n".join(
            f"Citation key: [{s['cite_key']}]\nNarrator: {s['narrator']}\n"
            + (f'Prompt: "{s["prompt"]}"\n' if s["prompt"] else "")
            + s["text"][:2500]
            for s in stories[:20]
        )
    )

    content_md = await _llm_prose(sys_prompt, user_prompt, timeout=180)
    if not content_md.strip():
        content_md = f"## {label}\n*Article generation produced no content.*"
    else:
        content_md = _resolve_citations(content_md, citation_map, label)

    await db.refresh(article)
    article.content_md = content_md
    article.user_edited_md = None
    article.status = "ready"
    article.source_count = len(stories)
    article.model_name = WIKI_MODEL
    article.generated_at = datetime.now(timezone.utc)
    await db.commit()


async def generate_event_wiki(event_id: int, group_id: int) -> None:
    from app.database import async_session_maker
    async with async_session_maker() as db:
        article = await db.scalar(
            select(WikiArticle).where(
                WikiArticle.entity_type == "event",
                WikiArticle.entity_id == event_id,
                WikiArticle.group_id == group_id,
            )
        )
        if not article:
            article = WikiArticle(entity_type="event", entity_id=event_id, group_id=group_id, status="generating")
            db.add(article)
        else:
            article.status = "generating"
            article.error_msg = None
        await db.commit()
        try:
            await _do_generate_event(db, article, event_id, group_id)
        except Exception as exc:
            logger.exception("wiki generation failed for event=%s group=%s", event_id, group_id)
            try:
                await db.refresh(article)
                article.status = "error"
                article.error_msg = str(exc)[:500]
                await db.commit()
            except Exception:
                pass
