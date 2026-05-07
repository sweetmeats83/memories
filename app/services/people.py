import re, json, logging
from typing import Optional, Iterable, Tuple
from sqlalchemy import select, func
from app.models import ResponsePerson, PersonShare, KinMembership, Person, PersonAlias
from app.services.people_acl import visible_group_ids, person_visibility_filter
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils import slugify

logger = logging.getLogger(__name__)
ROLE_WORDS = {
    # GRANDPARENTS
    "grandmother": ["grandma","grandmother","granny","gran","nana","nanna","nonna","abuela","oma","lola","yiayia","yaya","bubbe"],
    "grandfather": ["grandpa","grandfather","granddad","granddaddy","grampa","gramps","pop-pop","pawpaw","papaw","pappy","abuelo","opa","lolo","zayde"],
    "grandparent": ["grandparent","grandparents","granparent","grand folks"],

    "great-grandmother": ["great-grandmother","great grandma","great-grandma","great gran","bisabuela"],
    "great-grandfather": ["great-grandfather","great grandpa","great-grandpa","bisabuelo"],
    "great-grandparent": ["great-grandparent","great grandparents"],

    "step-grandmother": ["step-grandmother","step grandma","step-grandma","stepgran"],
    "step-grandfather": ["step-grandfather","step grandpa","step-grandpa"],
    "step-grandparent": ["step-grandparent","step grandparents"],

    # PARENTS
    "mother": ["mother","mom","mum","mam","mama","ma","amá","mami"],
    "father": ["father","dad","daddy","papa","pa","papá","papi"],
    "parent": ["parent","parents","guardian","legal guardian","caregiver","co-parent","coparent"],

    "stepmother": ["stepmother","step-mom","stepmom","madrastra"],
    "stepfather": ["stepfather","step-dad","stepdad","padrastro"],
    "adoptive mother": ["adoptive mother","adopted mother","amom"],
    "adoptive father": ["adoptive father","adopted father","adad"],
    "adoptive parent": ["adoptive parent","adopted parent"],
    "foster mother": ["foster mother","foster mom"],
    "foster father": ["foster father","foster dad"],
    "foster parent": ["foster parent"],

    "mother-in-law": ["mother-in-law","mom-in-law","mil","suegra"],
    "father-in-law": ["father-in-law","dad-in-law","fil","suegro"],
    "parent-in-law": ["parent-in-law","in-law parent"],

    # CHILDREN
    "child": ["child","kid","offspring"],
    "son": ["son"],
    "daughter": ["daughter"],
    "stepson": ["stepson","step-son"],
    "stepdaughter": ["stepdaughter","step-daughter"],
    "stepchild": ["stepchild","step child","step-children","stepchildren"],
    "adopted son": ["adopted son","adoptive son"],
    "adopted daughter": ["adopted daughter","adoptive daughter"],
    "adopted child": ["adopted child","adoptive child"],
    "foster son": ["foster son"],
    "foster daughter": ["foster daughter"],
    "foster child": ["foster child"],

    "grandchild": ["grandchild","grandchildren"],
    "grandson": ["grandson","grand-son","nieto"],
    "granddaughter": ["granddaughter","grand-daughter","nieta"],
    "great-grandchild": ["great-grandchild","great grandchildren","great-grandchildren"],
    "great-grandson": ["great-grandson","great grandson"],
    "great-granddaughter": ["great-granddaughter","great granddaughter"],
    "step-grandchild": ["step-grandchild","step grandchild","stepgrandchild"],
    "step-grandson": ["step-grandson","step grandson"],
    "step-granddaughter": ["step-granddaughter","step granddaughter"],

    # SIBLINGS
    "sibling": ["sibling","sib","brother or sister","twin"],
    "brother": ["brother","bro","hermano","twin brother"],
    "sister": ["sister","sis","hermana","twin sister"],
    "half-brother": ["half-brother","half brother","halfbrother","medio hermano"],
    "half-sister": ["half-sister","half sister","halfsister","media hermana"],
    "stepbrother": ["stepbrother","step-brother","hermanastro"],
    "stepsister": ["stepsister","step-sister","hermanastra"],
    "stepsibling": ["stepsibling","step-sibling","step sib"],

    "brother-in-law": ["brother-in-law","bro-in-law","bil","cuñado"],
    "sister-in-law": ["sister-in-law","sis-in-law","cuñada"],
    "sibling-in-law": ["sibling-in-law","in-law sibling"],

    # AUNTS/UNCLES & NIECES/NEPHEWS
    "uncle": ["uncle","unc","tio","tío"],
    "aunt": ["aunt","auntie","aunty","tia","tía"],
    "great-uncle": ["great-uncle","great uncle","granduncle"],
    "great-aunt": ["great-aunt","great aunt","grandaunt"],
    "uncle-in-law": ["uncle-in-law","uncle in law"],
    "aunt-in-law": ["aunt-in-law","aunt in law"],

    "nephew": ["nephew","sobrino"],
    "niece": ["niece","sobrina"],
    "nibling": ["nibling"],  # gender-neutral niece/nephew
    "great-nephew": ["great-nephew","great nephew","grandnephew","grand nephew"],
    "great-niece": ["great-niece","great niece","grandniece","grand niece"],

    # COUSINS
    "cousin": ["cousin","cuz","primo","prima"],
    "second cousin": ["second cousin","2nd cousin"],
    "first cousin once removed": ["first cousin once removed","1st cousin once removed","1c1r"],
    "cousin-in-law": ["cousin-in-law","cousin in law"],
    "step-cousin": ["step-cousin","step cousin"],

    # PARTNERS / SPOUSES
    "spouse": ["spouse","partner","wife","husband"],
    "wife": ["wife","wifey"],
    "husband": ["husband","hubby"],
    "partner": ["partner","domestic partner","significant other","life partner","common-law partner","common law partner"],
    "boyfriend": ["boyfriend","bf"],
    "girlfriend": ["girlfriend","gf"],
    "fiancé": ["fiancé","fiance"],
    "fiancée": ["fiancée","fiancee"],
    "son-in-law": ["son-in-law","son in law","yerno"],
    "daughter-in-law": ["daughter-in-law","daughter in law","dil","nuera"],
    "child-in-law": ["child-in-law","child in law"],
    "ex-spouse": ["ex-spouse","ex husband","ex-husband","ex wife","ex-wife","former spouse"],

    # GODFAMILY
    "godparent": ["godparent","god parent"],
    "godmother": ["godmother","madrina"],
    "godfather": ["godfather","padrino"],
    "godchild": ["godchild","god child","godson","goddaughter","ahijado","ahijada"],

    # NEUTRAL / MISC
    "pibling": ["pibling"],  # gender-neutral aunt/uncle
    "relative": ["relative","relation","family member","kin"],
    "ancestor": ["ancestor","forebear","forefather","foremother"],
    "descendant": ["descendant"],
    "household member": ["household member","housemate","roommate","flatmate"],

    # keep your non-family roles too
    "friend": ["friend","buddy","pal","bestie"],
    "mentor": ["coach","mentor","teacher"],
    "neighbor": ["neighbor","neighbour"],
}
# Flat list of role-word prefixes, longest-first so "great-grandmother" matches before "grandmother"
_ROLE_PREFIXES: list[str] = sorted(
    {w.lower() for words in ROLE_WORDS.values() for w in words},
    key=len, reverse=True,
)

# Maps narrator-graph role labels (from _SRC_EDGE_TO_ROLE / _DST_EDGE_TO_ROLE) to
# all synonyms a narrator might use for that role in text.
# "parent" covers "mom", "dad", "mama", "papa" etc. so the fast-path doesn't miss them.
def _build_narrator_role_synonyms() -> dict[str, frozenset[str]]:
    def _words(*keys: str) -> frozenset[str]:
        out: set[str] = set()
        for k in keys:
            out.update(w.lower() for w in ROLE_WORDS.get(k, []))
        return frozenset(out)

    return {
        "parent":       _words("mother", "father", "parent"),
        "step-parent":  _words("stepmother", "stepfather", "step-parent"),
        "child":        _words("son", "daughter") | frozenset(["child", "kid"]),
        "spouse":       _words("wife", "husband", "spouse"),
        "partner":      _words("partner"),
        "ex-spouse":    _words("ex-spouse", "ex-partner"),
        "sibling":      _words("brother", "sister", "sibling", "half-sibling"),
        "half-sibling": _words("half-sibling"),
        "step-sibling": _words("step-sibling"),
        "grandparent":  _words("grandmother", "grandfather", "grandparent"),
        "grandchild":   frozenset(["grandson", "granddaughter", "grandchild", "grandkid"]),
        "aunt or uncle":    _words("aunt", "uncle"),
        "niece or nephew":  _words("niece", "nephew"),
        "cousin":       _words("cousin"),
        "mentor":       _words("mentor"),
        "friend":       _words("friend"),
        "neighbor":     _words("neighbor"),
    }

_NARRATOR_ROLE_SYNONYMS: dict[str, frozenset[str]] = _build_narrator_role_synonyms()


def _split_display_name(display_name: str) -> tuple[str | None, str | None]:
    """
    Very small heuristic: split the last token as family_name, the rest as given_name.
    If there is only one token, store it as given_name and leave family_name None.
    """
    if not display_name:
        return None, None
    parts = display_name.strip().split()
    if len(parts) == 1:
        return parts[0], None
    return " ".join(parts[:-1]), parts[-1]

def guess_role_hint(text_around_name: str) -> Optional[str]:
    s = text_around_name.lower()
    for role, words in ROLE_WORDS.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", s):
                return role
    return None

def _role_matches(stored: str | None, hint: str | None) -> bool:
    """True when two role strings are the same or one contains the other (handles 'cousin' vs 'cousin')."""
    if not stored or not hint:
        return False
    s, h = stored.lower().strip(), hint.lower().strip()
    return s == h or s in h or h in s


# Maps a relationship edge label (from the narrator's perspective) to a
# plain role word the LLM and users will recognise.
_SRC_EDGE_TO_ROLE: dict[str, str] = {
    "child-of": "parent", "son-of": "parent", "daughter-of": "parent",
    "spouse-of": "spouse", "partner-of": "partner",
    "ex-spouse-of": "ex-spouse", "ex-partner-of": "ex-partner",
    "sibling-of": "sibling", "half-sibling-of": "half-sibling",
    "step-sibling-of": "step-sibling",
    "grandchild-of": "grandparent",
    "nephew-of": "aunt or uncle", "niece-of": "aunt or uncle",
    "cousin-of": "cousin",
    "student-of": "mentor", "friend-of": "friend",
}
_DST_EDGE_TO_ROLE: dict[str, str] = {
    "parent-of": "parent", "mother-of": "parent", "father-of": "parent",
    "adoptive-parent-of": "parent", "step-parent-of": "step-parent",
    "child-of": "child", "son-of": "child", "daughter-of": "child",
    "spouse-of": "spouse", "partner-of": "partner",
    "sibling-of": "sibling",
    "grandparent-of": "grandchild", "grandchild-of": "grandparent",
    "aunt-of": "niece or nephew", "uncle-of": "niece or nephew",
}


async def build_narrator_graph(db: AsyncSession, user_id: int) -> list[dict]:
    """
    Return a list of {role, display_name, person_id} dicts describing the
    narrator's direct family/social relationships, derived from their self-person
    and the relationship graph.

    Used to ground LLM extraction: when Nathan writes "my dad", the LLM can
    substitute "John Smith" because the narrator graph says Nathan's parent is John Smith.
    """
    from app.models import UserProfile, RelationshipEdge

    profile = await db.scalar(
        select(UserProfile).where(UserProfile.user_id == user_id)
    )
    self_pid = (profile.privacy_prefs or {}).get("self_person_id") if profile else None
    if not self_pid:
        return []

    src_rows = (await db.execute(
        select(RelationshipEdge.rel_type, RelationshipEdge.dst_id)
        .where(RelationshipEdge.src_id == self_pid)
    )).all()

    dst_rows = (await db.execute(
        select(RelationshipEdge.rel_type, RelationshipEdge.src_id)
        .where(RelationshipEdge.dst_id == self_pid)
    )).all()

    entries: list[dict] = []
    seen_pids: set[int] = set()

    for rel_type, other_id in src_rows:
        role = _SRC_EDGE_TO_ROLE.get((rel_type or "").lower())
        if role and other_id not in seen_pids:
            p = await db.get(Person, other_id)
            if p and p.display_name:
                entries.append({"role": role, "display_name": p.display_name, "person_id": other_id})
                seen_pids.add(other_id)

    for rel_type, other_id in dst_rows:
        role = _DST_EDGE_TO_ROLE.get((rel_type or "").lower())
        if role and other_id not in seen_pids:
            p = await db.get(Person, other_id)
            if p and p.display_name:
                entries.append({"role": role, "display_name": p.display_name, "person_id": other_id})
                seen_pids.add(other_id)

    return entries


async def _llm_pick(candidates: list, mention: str, context_text: str | None) -> "Person | None":
    """Call the LLM disambiguator and return the winning Person, or None."""
    try:
        from app.llm_client import disambiguate_person_mention
        cand_dicts = [
            {"id": p.id, "name": p.display_name, "role": (p.meta or {}).get("role_hint")}
            for p in candidates
        ]
        winner_id = await disambiguate_person_mention(mention, cand_dicts, context_text=context_text)
        if winner_id is not None:
            return next((p for p in candidates if p.id == winner_id), None)
    except Exception:
        pass
    return None


async def resolve_person(
    db,
    user_id: int,
    display_or_alias: str,
    role_hint: str | None = None,
    context_text: str | None = None,
    narrator_graph: list[dict] | None = None,
) -> "Person | None":
    name = (display_or_alias or "").strip()
    if not name:
        p = Person(owner_user_id=user_id, display_name="(Unknown)")
        db.add(p)
        await db.flush()
        return p

    name_lower = name.lower()
    _group_ids = await visible_group_ids(db, user_id)
    _vis = person_visibility_filter(user_id, _group_ids)

    # 0) Narrator-graph fast-path: if the extracted name is a role word that maps
    #    unambiguously to ONE person in the narrator's graph, return that person.
    #    Uses _NARRATOR_ROLE_SYNONYMS so "dad"/"mom" match the generic "parent" role,
    #    and collects all candidates before deciding — two parents both matching
    #    "parent" is ambiguous, so we skip rather than return the wrong one.
    if narrator_graph:
        graph_candidates: list[int] = []
        for entry in narrator_graph:
            synonyms = _NARRATOR_ROLE_SYNONYMS.get(entry["role"], frozenset())
            if name_lower in synonyms or name_lower == entry["role"].lower():
                graph_candidates.append(entry["person_id"])
        if len(graph_candidates) == 1:
            p = await db.get(Person, graph_candidates[0])
            if p:
                await add_alias_if_new(db, p.id, name)
                return p
        # Multiple matches → ambiguous role (e.g. two people with role "parent");
        # fall through to name-based matching where the LLM already substituted
        # a full name, or to fuzzy matching as last resort.

    # 1) Alias exact (case-insensitive)
    alias_row = await db.scalar(
        select(PersonAlias)
        .join(Person, Person.id == PersonAlias.person_id)
        .where(func.lower(PersonAlias.alias) == name_lower)
        .where(_vis)
        .limit(1)
    )
    if alias_row:
        return await db.get(Person, alias_row.person_id)

    # 2) Person display_name exact (case-insensitive)
    person = await db.scalar(
        select(Person)
        .where(func.lower(Person.display_name) == name_lower)
        .where(_vis)
        .limit(1)
    )
    if person:
        return person

    # Load all persons and aliases once; reused across all fuzzy passes below.
    all_persons = (await db.execute(
        select(Person).where(_vis)
    )).scalars().all()
    alias_rows = (await db.execute(
        select(PersonAlias)
        .join(Person, Person.id == PersonAlias.person_id)
        .where(_vis)
    )).scalars().all()

    def _collect_fuzzy(search_lc: str) -> list:
        """Return every Person whose display_name or alias contains search_lc as a whole word."""
        seen: set[int] = set()
        hits: list = []
        for p in all_persons:
            dn = (p.display_name or "").lower()
            if re.search(rf"\b{re.escape(search_lc)}\b", dn) or re.search(rf"\b{re.escape(dn)}\b", search_lc):
                if p.id not in seen:
                    hits.append(p); seen.add(p.id)
        for ar in alias_rows:
            al = (ar.alias or "").lower()
            if re.search(rf"\b{re.escape(search_lc)}\b", al) or re.search(rf"\b{re.escape(al)}\b", search_lc):
                if ar.person_id not in seen:
                    seen.add(ar.person_id)
                    # fetch person object (already loaded in all_persons if owner matches)
                    found = next((p for p in all_persons if p.id == ar.person_id), None)
                    if found:
                        hits.append(found)
        return hits

    async def _disambiguate(candidates: list, mention: str, effective_role: str | None) -> "Person | None":
        """Role check → LLM → give up."""
        if len(candidates) == 1:
            return candidates[0]
        # 1) Role stored on person
        if effective_role:
            role_matched = [p for p in candidates if _role_matches((p.meta or {}).get("role_hint"), effective_role)]
            if len(role_matched) == 1:
                return role_matched[0]
        # 2) LLM reads the surrounding context
        winner = await _llm_pick(candidates, mention, context_text)
        return winner  # None if LLM is also uncertain

    # 3) Fuzzy on the full name as given ("Rosa", "Josh", "Grandma Rosa")
    candidates = _collect_fuzzy(name_lower)
    if candidates:
        winner = await _disambiguate(candidates, name, role_hint)
        if winner:
            await add_alias_if_new(db, winner.id, name)
            return winner
        # Multiple matches and couldn't disambiguate — fall through to role-prefix stripping
        # before creating a new person; if that also fails, return None.
        if not any(name_lower.startswith(p + " ") for p in _ROLE_PREFIXES):
            return None  # no prefix to strip; truly ambiguous

    # 4b) Role-prefix stripping: "cousin Josh" → bare "Josh"
    #     Lets the role word itself serve as a disambiguation signal even if the caller
    #     didn't pass role_hint.
    for prefix in _ROLE_PREFIXES:
        if name_lower.startswith(prefix + " "):
            bare_lc = name_lower[len(prefix) + 1:]
            if not bare_lc:
                continue
            bare_name = name[len(prefix) + 1:]
            candidates = _collect_fuzzy(bare_lc)
            if candidates:
                effective_role = role_hint or prefix
                winner = await _disambiguate(candidates, bare_name, effective_role)
                if winner:
                    await add_alias_if_new(db, winner.id, name)
                    return winner
                return None  # multiple matches, couldn't resolve
            break  # recognised the prefix but no person found — create new below

    # 5) Create new person + seed alias (mark as inferred from mention)
    p = Person(owner_user_id=user_id, display_name=name, meta={"inferred": True})
    db.add(p)
    await db.flush()
    db.add(PersonAlias(person_id=p.id, alias=name))
    return p
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

# Simple capitalized-sequence fallback: require at least two tokens to reduce false positives
# e.g., "Maria Gomez", "Mary Anne Smith" (allow hyphen/apostrophe)
_NAME_REGEX = re.compile(r"\b([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){1,2})\b")

def extract_name_spans(text: str, aliases: list[str] | None = None) -> list[tuple[str,int,int]]:
    """
    Returns a list of (surface_name, start, end) spans.
    - Uses spaCy PERSON if available
    - Boosts with user aliases (exact, whole-word, case-insensitive)
    - Falls back to conservative regex (multi-token) when spaCy unavailable
    """
    if not text:
        return []
    spans: list[tuple[int,int,str]] = []
    if _NLP:
        doc = _NLP(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and ent.text.strip():
                spans.append((ent.start_char, ent.end_char, ent.text))
    else:
        for m in _NAME_REGEX.finditer(text):
            spans.append((m.start(), m.end(), m.group(0)))

    # Alias overlay (whole-word, case-insensitive)
    try:
        for a in (aliases or []):
            a = (a or "").strip()
            if not a:
                continue
            pat = re.compile(rf"\b{re.escape(a)}\b", re.IGNORECASE)
            for m in pat.finditer(text):
                spans.append((m.start(), m.end(), text[m.start():m.end()]))
    except Exception:
        pass

    if not spans:
        return []

    spans.sort(key=lambda t: (t[0], -(t[1]-t[0])))
    merged: list[tuple[int,int,str]] = []
    for s,e,txt in spans:
        if not merged:
            merged.append((s,e,txt)); continue
        ls, le, _ = merged[-1]
        if s >= ls and e <= le:
            continue
        if s <= le and e > le:
            if (e - s) > (le - ls):
                merged[-1] = (s,e,txt)
            continue
        merged.append((s,e,txt))

    return [(txt, s, e) for (s,e,txt) in merged]

def role_hint_near(text: str, start: int, end: int, window: int = 80) -> str | None:
    """
    Looks ±window chars around a mention for role words (grandma, uncle, etc.).
    """
    around = text[max(0, start-window): min(len(text), end+window)]
    return guess_role_hint(around)


_LLM_EXTRACT_SYSTEM = """You extract people mentioned in a personal story or memoir response.
Given the story text, a list of people already in the narrator's family tree, and the narrator's
known relationships, identify each distinct person mentioned.

NARRATOR CONTEXT: If the narrator says "my dad" and the narrator's father is listed as "John Smith",
output "John Smith" as the name — not "my dad". Always prefer the full name from the narrator's
relationships when a role-based reference matches.

Return STRICT JSON only:
{
  "people": [
    {"name": "full name from narrator relationships if known, otherwise the name used in text",
     "role": "their relationship to the narrator, or null"}
  ]
}

Rules:
- Resolve role-based references ("my mom", "dad", "Uncle Dave") to full names using narrator_relationships
- If no match in narrator_relationships, use the most specific name/nickname from the text
- If the same person is mentioned multiple ways, pick the most specific / full name
- Do NOT include the narrator themselves
- Limit to 20 people maximum
- If no people are mentioned, return {"people": []}"""


async def llm_extract_people(
    text: str,
    known_names: list[str] | None = None,
    narrator_graph: list[dict] | None = None,
) -> list[dict]:
    """
    Use the LLM to extract (name, role) pairs from a response/transcript.
    Returns a list of dicts: [{"name": str, "role": str | None}, ...]
    Falls back to empty list on any error so callers can degrade gracefully.
    """
    if not text or not text.strip():
        return []
    try:
        from app.llm_client import your_chat_completion
        # narrator_relationships: [{role, display_name}] — tells LLM to resolve
        # "my dad" → "John Smith" rather than outputting the role word as a name.
        narrator_relationships = [
            {"role": e["role"], "name": e["display_name"]}
            for e in (narrator_graph or [])
        ]
        user_payload = {
            "story": text.strip(),
            "known_family_tree": known_names or [],
            "narrator_relationships": narrator_relationships,
        }
        raw = await your_chat_completion(
            system=_LLM_EXTRACT_SYSTEM,
            user=json.dumps(user_payload, ensure_ascii=False),
            response_format="json",
            temperature=0.1,
        )
        data = json.loads(raw)
        people = data.get("people") or []
        result = []
        for p in people:
            n = (p.get("name") or "").strip()
            r = (p.get("role") or None)
            if r:
                r = r.strip() or None
            if n:
                result.append({"name": n, "role": r})
        return result
    except Exception:
        logger.debug("llm_extract_people failed; caller will fall back", exc_info=True)
        return []

async def ensure_self_person(db: AsyncSession, user_id: int, display_name: str) -> int:
    """
    Guarantee that a user has a self-person node in the graph.
    Returns the person_id.

    Priority:
    1. self_person_id already pinned in UserProfile.privacy_prefs → use it
    2. An owned person with connect_to_owner=True + role_hint in {you, self, me} → pin it
    3. Create a new private self-person and pin it

    Safe to call on every login/dashboard load — is a no-op if already set.
    """
    from app.models import UserProfile

    profile = await db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    if profile is None:
        profile = UserProfile(user_id=user_id, privacy_prefs={}, tag_weights={})
        db.add(profile)
        await db.flush()

    prefs: dict = profile.privacy_prefs or {}

    # 1) Already pinned
    if prefs.get("self_person_id"):
        existing = await db.get(Person, prefs["self_person_id"])
        if existing:
            return existing.id

    # 2) Scan owned persons for a flagged self-person
    owned = (await db.execute(
        select(Person).where(Person.owner_user_id == user_id)
    )).scalars().all()
    for p in owned:
        m = p.meta or {}
        if isinstance(m, dict) and m.get("connect_to_owner") and str(m.get("role_hint", "")).lower() in {"you", "self", "me"}:
            prefs["self_person_id"] = p.id
            profile.privacy_prefs = prefs
            await db.flush()
            return p.id

    # 3) Create a new self-person
    self_person = Person(
        owner_user_id=user_id,
        display_name=display_name,
        meta={"connect_to_owner": True, "role_hint": "you"},
        visibility="private",
    )
    db.add(self_person)
    await db.flush()
    db.add(PersonAlias(person_id=self_person.id, alias=display_name))
    prefs["self_person_id"] = self_person.id
    profile.privacy_prefs = prefs
    await db.commit()
    return self_person.id


async def add_alias_if_new(db: AsyncSession, person_id: int, alias: str) -> None:
    a = (alias or "").strip()
    if not a:
        return
    exists = await db.scalar(
        select(PersonAlias.id)
        .where(PersonAlias.person_id == person_id)
        .where(func.lower(PersonAlias.alias) == func.lower(a))
        .limit(1)
    )
    if not exists:
        db.add(PersonAlias(person_id=person_id, alias=a))
        await db.flush()

async def link_mention(
    db: AsyncSession, response_id: int, person: Person, alias_used: str,
    start_char: int | None = None, end_char: int | None = None,
    confidence: float = 0.75, role_hint: str | None = None
) -> ResponsePerson:
    """
    Records a mention of a person in a response and ensures the alias is saved.
    """
    await add_alias_if_new(db, person.id, alias_used)
    rp = ResponsePerson(
        response_id=response_id,
        person_id=person.id,
        alias_used=alias_used,
        start_char=start_char,
        end_char=end_char,
        confidence=confidence,
        role_hint=role_hint,
    )
    db.add(rp)
    await db.flush()
    return rp

async def apply_person_visibility(
    db: AsyncSession,
    person: Person,
    owner_user_id: int,
    target_group_ids: list[int] | None,
    share_default: bool
) -> None:
    """
    If share_default and no specific target groups are passed, share to ALL of owner's groups.
    Otherwise, share to exactly target_group_ids. If neither, keep private.
    """
    # Determine groups
    if not share_default and not target_group_ids:
        person.visibility = "private"
        return

    if not target_group_ids:
        rows = (await db.execute(
            select(KinMembership.group_id).where(KinMembership.user_id == owner_user_id)
        )).scalars().all()
        target_group_ids = list(rows or [])

    if not target_group_ids:
        person.visibility = "private"
        return

    person.visibility = "groups"
    for gid in target_group_ids:
        exists = (await db.execute(
            select(PersonShare.id).where(PersonShare.person_id == person.id, PersonShare.group_id == gid)
        )).scalar_one_or_none()
        if not exists:
            db.add(PersonShare(person_id=person.id, group_id=gid, shared_by_user_id=owner_user_id))
    # caller commits


async def upsert_person_for_user(
    db: AsyncSession,
    owner_user_id: int,
    display_name: str,
    role_hint: str | None = None,
) -> Person:
    """
    Idempotently ensure a Person exists for this owner and surface name, and add an alias.
    - If the user belongs to a KinGroup, search the shared group pool first.
    - Prefer matching by exact display_name (case-insensitive).
    - If not found, match by alias.
    - Otherwise create a new Person scoped to the group (or privately to the user).
    - Always ensure a PersonAlias row exists for the submitted surface string.
    - Store/merge role_hint into Person.meta without clobbering other meta keys.
    """
    name = (display_name or "").strip()
    if not name:
        raise ValueError("display_name is required")

    # 0) Look up the user's KinGroup membership (exactly one group → shared scope)
    memberships = (await db.execute(
        select(KinMembership.group_id).where(KinMembership.user_id == owner_user_id)
    )).scalars().all()
    group_id: int | None = memberships[0] if len(memberships) == 1 else None

    person: Person | None = None

    if group_id is not None:
        # 1a) Exact display_name match in the shared group pool
        person = (await db.execute(
            select(Person)
            .where(
                Person.group_id == group_id,
                func.lower(Person.display_name) == func.lower(name),
            )
            .limit(1)
        )).scalars().first()

        # 1b) Alias match in the shared group pool
        if not person:
            person = (await db.execute(
                select(Person)
                .join(PersonAlias, PersonAlias.person_id == Person.id)
                .where(
                    Person.group_id == group_id,
                    func.lower(PersonAlias.alias) == func.lower(name),
                )
                .limit(1)
            )).scalars().first()
    else:
        # 1) Try exact display_name match for this owner (case-insensitive)
        person = (await db.execute(
            select(Person)
            .where(
                Person.owner_user_id == owner_user_id,
                func.lower(Person.display_name) == func.lower(name),
            )
            .limit(1)
        )).scalars().first()

        # 2) Otherwise, try alias match for this owner
        if not person:
            person = (await db.execute(
                select(Person)
                .join(PersonAlias, PersonAlias.person_id == Person.id)
                .where(
                    Person.owner_user_id == owner_user_id,
                    func.lower(PersonAlias.alias) == func.lower(name),
                )
                .limit(1)
            )).scalars().first()

    # 3) Create if still missing
    if not person:
        given, family = _split_display_name(name)
        if group_id is not None:
            person = Person(
                group_id=group_id,
                owner_user_id=owner_user_id,  # attribution only
                display_name=name,
                given_name=given,
                family_name=family,
                meta={},
            )
        else:
            person = Person(
                owner_user_id=owner_user_id,
                display_name=name,
                given_name=given,
                family_name=family,
                meta={},
            )
        db.add(person)
        await db.flush()  # ensure person.id

    # 4) Ensure an alias row for the exact submitted surface form
    has_alias = (
        await db.execute(
            select(PersonAlias.id).where(
                PersonAlias.person_id == person.id,
                func.lower(PersonAlias.alias) == func.lower(name),
            )
        )
    ).first()
    if not has_alias:
        db.add(PersonAlias(person_id=person.id, alias=name))

    # 5) Merge role_hint into meta (non-destructive)
    try:
        m = person.meta or {}
        if role_hint:
            # keep the most recent role hint (or append a history list if you prefer)
            m["role_hint"] = role_hint
        person.meta = m
    except Exception:
        # ensure meta is a dict on odd DB contents
        person.meta = {"role_hint": role_hint} if role_hint else (person.meta or {})

    await db.flush()
    return person


async def add_alias(db: AsyncSession, person_id: int, alias_text: str) -> None:
    alias_text = (alias_text or "").strip()
    if not alias_text:
        return
    exists = await db.scalar(
        select(PersonAlias).where(PersonAlias.person_id == person_id, PersonAlias.alias == alias_text)
    )
    if not exists:
        db.add(PersonAlias(person_id=person_id, alias=alias_text))
        await db.flush()
