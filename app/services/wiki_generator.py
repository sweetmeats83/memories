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
from app.models import (
    KinMembership,
    Person,
    PersonAlias,
    Prompt,
    RelationshipEdge,
    Response,
    ResponsePerson,
    User,
    UserProfile,
    WikiArticle,
)
from app.services.people_acl import visible_group_ids

logger = logging.getLogger(__name__)

# Separate model env var so admins can point heavy wiki jobs at the 14B model
# while keeping fast ops on the smaller one.
WIKI_MODEL = os.getenv("WIKI_OLLAMA_MODEL", OLLAMA_MODEL)

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


async def _narrator_relationship_to_subject(
    db: AsyncSession, narrator_user_id: int, subject_person_id: int
) -> str | None:
    """
    Return a human-readable relationship label describing how the narrator relates
    to the wiki subject (e.g. "son of", "daughter of", "grandchild of").

    Looks up the narrator's self-person via UserProfile, then checks for a direct
    edge in either direction. Returns None if no direct edge exists.
    """
    profile = await db.scalar(
        select(UserProfile).where(UserProfile.user_id == narrator_user_id)
    )
    self_pid = (profile.privacy_prefs or {}).get("self_person_id") if profile else None
    if not self_pid:
        return None

    # Direct edge: narrator → subject
    edge = await db.scalar(
        select(RelationshipEdge.rel_type)
        .where(RelationshipEdge.src_id == self_pid)
        .where(RelationshipEdge.dst_id == subject_person_id)
        .limit(1)
    )
    if edge:
        return edge.replace("-", " ")

    # Inverse edge: subject → narrator
    edge = await db.scalar(
        select(RelationshipEdge.rel_type)
        .where(RelationshipEdge.src_id == subject_person_id)
        .where(RelationshipEdge.dst_id == self_pid)
        .limit(1)
    )
    if edge:
        return _invert_rel(edge).replace("-", " ")

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


async def _llm_json(system: str, user: str, timeout: int = 120) -> dict:
    prompt = f"{system}\n\nReturn ONLY valid JSON, no commentary.\n\nUSER:\n{user}"
    payload = {
        "model": WIKI_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": 8192},
        "think": False,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            r.raise_for_status()
            raw = (r.json() or {}).get("response", "").strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning("wiki _llm_json failed: %s", e)
        return {}


async def _llm_prose(system: str, user: str, timeout: int = 240) -> str:
    prompt = f"{system}\n\nUSER:\n{user}"
    payload = {
        "model": WIKI_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.25, "num_ctx": 8192},
        "think": False,
    }
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        r.raise_for_status()
        return (r.json() or {}).get("response", "").strip()


def _resolve_citations(md: str, citation_map: dict, subject: str) -> str:
    """
    Replace [Nathan 1] style citation keys with markdown links to the source story,
    then append a ## Sources section listing all cited references.
    """
    cited_keys: list[str] = []

    def _replace(m: re.Match) -> str:
        key = m.group(1)
        if key not in citation_map:
            return m.group(0)  # leave unknown brackets untouched
        meta = citation_map[key]
        if key not in cited_keys:
            cited_keys.append(key)
        return f"[[{key}]](/response/{meta['response_id']})"

    # Match [Name N] patterns — narrator name (one or more words) + space + integer
    result = re.sub(r"\[([A-Za-z][A-Za-z0-9 ]*?\s\d+)\]", _replace, md)

    if not cited_keys:
        return result

    # Append sources section
    lines = ["\n\n## Sources"]
    for key in cited_keys:
        meta = citation_map[key]
        rel = f" ({meta['narrator_rel']} of {subject})" if meta["narrator_rel"] else ""
        title = meta["title"] or "Untitled story"
        lines.append(
            f"- [[{key}]](/response/{meta['response_id']}) "
            f"— {meta['narrator']}{rel}: *{title}*"
        )
    return result + "\n".join(lines)


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


async def _do_generate(db: AsyncSession, article: WikiArticle, person_id: int, user_id: int) -> None:
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

    # --- Load visible relationships ---
    group_ids = await visible_group_ids(db, user_id)
    if group_ids:
        edge_scope = or_(
            RelationshipEdge.user_id == user_id,
            RelationshipEdge.group_id.in_(group_ids),
        )
    else:
        edge_scope = RelationshipEdge.user_id == user_id

    DstPerson = aliased(Person)
    SrcPerson = aliased(Person)

    src_rows = (
        await db.execute(
            select(RelationshipEdge.rel_type, DstPerson.display_name)
            .join(DstPerson, DstPerson.id == RelationshipEdge.dst_id)
            .where(RelationshipEdge.src_id == person_id)
            .where(edge_scope)
        )
    ).all()

    dst_rows = (
        await db.execute(
            select(RelationshipEdge.rel_type, SrcPerson.display_name)
            .join(SrcPerson, SrcPerson.id == RelationshipEdge.src_id)
            .where(RelationshipEdge.dst_id == person_id)
            .where(edge_scope)
        )
    ).all()

    relationships: list[str] = []
    for rel_type, other_name in src_rows:
        relationships.append(f"{rel_type.replace('-', ' ')} {other_name}")
    for rel_type, other_name in dst_rows:
        inv = _invert_rel(rel_type)
        relationships.append(f"{inv.replace('-', ' ')} {other_name}")

    # --- Collect narrator IDs: requesting user + all family group members ---
    narrator_ids = [user_id]
    if group_ids:
        member_ids = (
            await db.execute(
                select(KinMembership.user_id).where(KinMembership.group_id.in_(group_ids))
            )
        ).scalars().all()
        narrator_ids = list({user_id} | set(member_ids))

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

    stories: list[dict] = []
    source_response_ids: list[int] = []
    for rp, resp, prompt, narrator in mention_rows:
        raw = resp.ai_polished or resp.response_text or resp.transcription or ""
        text = _strip_html(raw)
        narrator_label = narrator.username or narrator.email.split("@")[0]
        narrator_rel = narrator_rel_cache.get(narrator.id)
        if text:
            stories.append(
                {
                    "prompt": prompt.text if prompt else "",
                    "title": resp.title or "",
                    "text": text,
                    "role_hint": rp.role_hint or "",
                    "response_id": resp.id,
                    "narrator": narrator_label,
                    "narrator_rel": narrator_rel,  # e.g. "son of", "daughter of", None
                }
            )
            source_response_ids.append(resp.id)

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
    subject = person.display_name
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

    p1_user = (
        f"Subject: {subject}{alias_str}\n"
        f"Known relationships: {', '.join(relationships) if relationships else 'unknown'}\n\n"
        f"Source stories (each labelled with the narrator and their relationship to {subject}):\n\n"
        + "\n\n---\n\n".join(
            "NARRATOR: {name}{rel}\n".format(
                name=s["narrator"],
                rel=f" ({s['narrator_rel']} of {subject})" if s["narrator_rel"] else "",
            )
            + (f"PROMPT: {s['prompt']}\n" if s["prompt"] else "")
            + (f"TITLE: {s['title']}\n" if s["title"] else "")
            + (f"SUBJECT'S ROLE IN THIS STORY: {s['role_hint']}\n" if s["role_hint"] else "")
            + f"STORY:\n{s['text'][:1500]}"
            for s in stories[:8]
        )
    )

    facts = await _llm_json(p1_sys, p1_user, timeout=120)

    # --- Pass 2: generate factual wiki article ---
    narrators = sorted({s["narrator"] for s in stories})
    narrator_ctx = (
        f"Sources: {len(stories)} family stories from {len(narrators)} contributor(s) — "
        + ", ".join(
            "{name}{rel}".format(
                name=n,
                rel=f" ({next((s['narrator_rel'] for s in stories if s['narrator'] == n and s['narrator_rel']), None) or 'family member'})"
            )
            for n in narrators
        )
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
4. When a narrator's relationship to the subject is known, use it to resolve pronouns: \
   e.g. if Nathan (grandson of {subject}) says "my dad," that refers to {subject}'s son, not {subject}.
5. Organise into 2–4 ## sections based on the actual facts available (e.g. Early Life, Career, Family).
6. If information is thin, write a shorter article rather than padding with vague statements.
7. Do NOT include a references or sources section — that will be added automatically.
8. Output clean markdown only — no JSON, no preamble, no closing commentary.
9. Target 250–500 words."""

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
    if relationships:
        header += f"\nRelationships: {', '.join(relationships)}"

    # Build per-narrator citation keys: [Nathan 1], [Nathan 2], [Sarah 1], …
    narrator_counters: dict[str, int] = {}
    citation_map: dict[str, dict] = {}  # key → {response_id, narrator, narrator_rel, title, prompt}
    cited_stories = stories[:10]
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
            "Citation key: [{key}]\nNarrator: {name}{rel}\n".format(
                key=s["cite_key"],
                name=s["narrator"],
                rel=f" ({s['narrator_rel']} of {subject})" if s["narrator_rel"] else "",
            )
            + (f'Story prompt: "{s["prompt"]}"\n' if s["prompt"] else "")
            + (f"Subject's role: {s['role_hint']}\n" if s["role_hint"] else "")
            + s["text"][:1400]
            for s in cited_stories
        )
    )

    content_md = await _llm_prose(p2_sys, p2_user, timeout=270)

    if not content_md.strip():
        content_md = _stub_article(person, aliases, relationships)
    else:
        content_md = _resolve_citations(content_md, citation_map, subject)

    # --- Save ---
    await db.refresh(article)
    article.content_md = content_md
    article.status = "ready"
    article.source_count = len(stories)
    article.model_name = WIKI_MODEL
    article.generated_at = datetime.now(timezone.utc)
    await db.commit()


async def generate_person_wiki(person_id: int, user_id: int) -> None:
    """Background-safe entry point: creates its own DB session."""
    from app.database import async_session_maker

    async with async_session_maker() as db:
        # Upsert article row and mark as generating
        article = await db.scalar(
            select(WikiArticle).where(
                WikiArticle.entity_type == "person",
                WikiArticle.entity_id == person_id,
                WikiArticle.user_id == user_id,
            )
        )
        if not article:
            article = WikiArticle(
                entity_type="person",
                entity_id=person_id,
                user_id=user_id,
                status="generating",
            )
            db.add(article)
        else:
            article.status = "generating"
            article.error_msg = None
        await db.commit()

        try:
            await _do_generate(db, article, person_id, user_id)
        except Exception as exc:
            logger.exception(
                "wiki generation failed for person=%s user=%s", person_id, user_id
            )
            try:
                await db.refresh(article)
                article.status = "error"
                article.error_msg = str(exc)[:500]
                await db.commit()
            except Exception:
                pass
