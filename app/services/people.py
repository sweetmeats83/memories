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
        # gender-specific roles (from _DST_EDGE_TO_ROLE "father-of"/"mother-of")
        "father":       _words("father"),
        "mother":       _words("mother"),
        # generic fallback (when edge type is just "parent-of")
        "parent":       _words("mother", "father", "parent"),
        "step-parent":  _words("stepmother", "stepfather"),
        "child":        _words("son", "daughter") | frozenset(["child", "kid"]),
        "spouse":       _words("wife", "husband", "spouse"),
        "partner":      _words("partner"),
        "ex-spouse":    _words("ex-spouse", "ex-partner"),
        "sibling":      _words("brother", "sister", "sibling", "half-sibling"),
        "half-sibling": _words("half-sibling"),
        "step-sibling": _words("step-sibling"),
        "grandmother":  _words("grandmother"),
        "grandfather":  _words("grandfather"),
        "grandparent":  _words("grandmother", "grandfather", "grandparent"),
        "grandchild":   frozenset(["grandson", "granddaughter", "grandchild", "grandkid"]),
        "aunt or uncle":    _words("aunt", "uncle"),
        "aunt":         _words("aunt"),
        "uncle":        _words("uncle"),
        "niece or nephew":  _words("niece", "nephew"),
        "niece":        _words("niece"),
        "nephew":       _words("nephew"),
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
    "parent-of": "parent",
    "mother-of": "mother",          # gender-specific → resolves "mom/mother/my mom"
    "father-of": "father",          # gender-specific → resolves "dad/father/my dad"
    "adoptive-parent-of": "parent",
    "step-parent-of": "step-parent",
    "child-of": "child", "son-of": "child", "daughter-of": "child",
    "spouse-of": "spouse", "partner-of": "partner",
    "sibling-of": "sibling",
    "grandparent-of": "grandchild", "grandchild-of": "grandparent",
    "aunt-of": "niece or nephew", "uncle-of": "niece or nephew",
}

# Synonyms the narrator might use in text for each role — used for rule-based fallback
_ROLE_TEXT_SYNONYMS: dict[str, list[str]] = {
    "father":      ["dad", "father", "my dad", "my father", "pop", "papa", "pa", "old man"],
    "mother":      ["mom", "mother", "my mom", "my mother", "mama", "ma", "momma"],
    "parent":      ["parents", "my parents", "mom and dad", "dad and mom", "my folks", "folks"],
    "spouse":      ["wife", "my wife", "husband", "my husband"],
    "sibling":     ["brother", "my brother", "sister", "my sister", "bro", "sis"],
    "grandmother": ["grandma", "grandmother", "nana", "gran", "grannie", "granny", "my grandma"],
    "grandfather": ["grandpa", "grandfather", "gramps", "pop-pop", "pawpaw", "my grandpa"],
    "grandparent": ["grandparents", "my grandparents", "grandma and grandpa"],
}

# Roles where a bare role word is too ambiguous — a family can have many people in this
# category. Only a concrete title+name ("Aunt Michelle", "Uncle Bob") triggers a match.
# Roles NOT in this set (father, mother, spouse, grandma/grandpa) are unique enough
# that a bare role word is sufficient evidence.
_TITLE_REQUIRED_ROLES: frozenset[str] = frozenset({
    # Extended family — many possible
    "aunt", "uncle", "cousin", "niece", "nephew",
    "great-aunt", "great-uncle", "aunt-in-law", "uncle-in-law",
    # Siblings — a person can have several brothers/sisters
    "sibling", "brother", "sister", "half-sibling", "step-sibling",
    "brother-in-law", "sister-in-law",
    # Children — a person can have several
    "child", "son", "daughter", "stepchild", "step-son", "step-daughter",
    # Grandparents — paternal + maternal = up to 4
    "grandparent", "grandmother", "grandfather",
    "great-grandparent", "great-grandmother", "great-grandfather",
    # Social roles — inherently many-to-many
    "friend", "colleague", "coworker", "roommate", "neighbor",
    "mentor", "mentee", "acquaintance",
})

# Title words that appear directly before a first name ("Aunt Michelle", "Uncle Bob").
# Used for the concrete title+name extraction pass for _TITLE_REQUIRED_ROLES.
_TITLE_TO_ROLE: dict[str, str] = {
    "aunt": "aunt", "auntie": "aunt", "aunty": "aunt",
    "uncle": "uncle",
    "cousin": "cousin",
    "niece": "niece",
    "nephew": "nephew",
    "brother": "sibling",
    "sister": "sibling",
    "son": "child",
    "daughter": "child",
    "grandma": "grandmother", "grandmom": "grandmother", "nana": "grandmother",
    "grandpa": "grandfather", "gramps": "grandfather",
}

# Social-role words where the name appears alongside the role in text.
# These are used for the named-social-role extraction pass in _apply_role_fallback.
_SOCIAL_ROLE_WORDS: dict[str, str] = {
    "friend": "friend", "buddy": "friend", "pal": "friend", "bestie": "friend",
    "neighbor": "neighbor", "neighbour": "neighbor",
    "colleague": "colleague", "coworker": "colleague", "co-worker": "colleague",
    "classmate": "classmate", "roommate": "roommate", "teammate": "teammate",
    "mentor": "mentor",
}

_DAD_WORDS  = r"(?:dad|father|pop|papa|pa)"
_MOM_WORDS  = r"(?:mom|mother|mama|ma)"
_GRANDMA_W  = r"(?:grandma|grandmother|nana|gran|grannie|granny)"
_GRANDPA_W  = r"(?:grandpa|grandfather|gramps|pop-pop|pawpaw)"
_APOS       = r"['’]?"  # straight or curly apostrophe

# Two-hop possessive patterns → (regex, narrator_graph_role_to_match)
# build_narrator_graph labels grandparents as "paternal/maternal grandmother/grandfather"
# so once the narrator_graph is enriched these patterns resolve cleanly.
_TWO_HOP_PATTERNS: list[tuple[str, str]] = [
    # "my dad's mom" / "my father's mother"
    (rf"my\s+{_DAD_WORDS}{_APOS}s\s+{_MOM_WORDS}",         "paternal grandmother"),
    (rf"my\s+{_DAD_WORDS}{_APOS}s\s+{_DAD_WORDS}",         "paternal grandfather"),
    (rf"my\s+{_MOM_WORDS}{_APOS}s\s+{_MOM_WORDS}",         "maternal grandmother"),
    (rf"my\s+{_MOM_WORDS}{_APOS}s\s+{_DAD_WORDS}",         "maternal grandfather"),
    # "grandma on my dad's side"
    (rf"{_GRANDMA_W}\s+on\s+my\s+{_DAD_WORDS}{_APOS}s\s+side", "paternal grandmother"),
    (rf"{_GRANDPA_W}\s+on\s+my\s+{_DAD_WORDS}{_APOS}s\s+side", "paternal grandfather"),
    (rf"{_GRANDMA_W}\s+on\s+my\s+{_MOM_WORDS}{_APOS}s\s+side", "maternal grandmother"),
    (rf"{_GRANDPA_W}\s+on\s+my\s+{_MOM_WORDS}{_APOS}s\s+side", "maternal grandfather"),
    # "my grandma that is my dad's mom" / "my grandma who is my dad's mom"
    (rf"{_GRANDMA_W}\s+(?:that\s+is|who\s+is)\s+my\s+{_DAD_WORDS}{_APOS}s\s+{_MOM_WORDS}", "paternal grandmother"),
    (rf"{_GRANDPA_W}\s+(?:that\s+is|who\s+is)\s+my\s+{_DAD_WORDS}{_APOS}s\s+{_DAD_WORDS}", "paternal grandfather"),
    (rf"{_GRANDMA_W}\s+(?:that\s+is|who\s+is)\s+my\s+{_MOM_WORDS}{_APOS}s\s+{_MOM_WORDS}", "maternal grandmother"),
    (rf"{_GRANDPA_W}\s+(?:that\s+is|who\s+is)\s+my\s+{_MOM_WORDS}{_APOS}s\s+{_DAD_WORDS}", "maternal grandfather"),
]


def _apply_role_fallback(
    text: str,
    llm_people: list[dict],
    narrator_graph: list[dict],
) -> list[dict]:
    """
    Post-LLM pass: scan story text for role synonyms (dad, mom, grandma…) that
    the LLM missed. For each matched synonym, if narrator_graph has exactly one
    person with that role and they're not already in the result, add them.

    For roles in _TITLE_REQUIRED_ROLES (aunt, uncle, cousin…) a bare role word is
    too ambiguous. Only a concrete "Aunt [Name]" / "Uncle [Name]" pattern is used.
    """
    if not narrator_graph or not text:
        return llm_people

    already_named = {item["name"].lower() for item in llm_people}
    text_lc = text.lower()
    to_add: list[dict] = []

    # --- Generic role-word scan (dad, mom, grandma, etc.) ---
    for role, synonyms in _ROLE_TEXT_SYNONYMS.items():
        if role in _TITLE_REQUIRED_ROLES:
            continue  # handled separately below

        found_in_text = any(
            re.search(rf"(?<![a-z]){re.escape(s)}(?![a-z])", text_lc)
            for s in synonyms
        )
        if not found_in_text:
            continue

        synonyms_set = _NARRATOR_ROLE_SYNONYMS.get(role, frozenset())
        candidates = [
            e for e in narrator_graph
            if e["role"] == role
            or e["role"] in synonyms_set
            or role in _NARRATOR_ROLE_SYNONYMS.get(e["role"], frozenset())
        ]

        if not candidates and role in ("father", "mother"):
            parent_candidates = [e for e in narrator_graph if e["role"] == "parent"]
            if len(parent_candidates) > 1:
                target_gender = "male" if role == "father" else "female"
                parent_candidates = [
                    e for e in parent_candidates
                    if (e.get("gender") or "").lower() in (target_gender, target_gender[0])
                ]
            candidates = parent_candidates

        if len(candidates) == 1:
            name = candidates[0]["display_name"]
            if name and name.lower() not in already_named:
                to_add.append({"name": name, "role": role})
                already_named.add(name.lower())

    # --- Title+name scan for ambiguous roles (Aunt Michelle, Uncle Bob, etc.) ---
    # Build a lookup: first-name (lowered) → narrator_graph entry for title-required roles
    title_graph: dict[str, dict] = {}
    for entry in narrator_graph:
        if entry.get("role") not in _TITLE_REQUIRED_ROLES:
            continue
        dn = (entry.get("display_name") or "").strip()
        if not dn:
            continue
        first = dn.split()[0].lower()
        title_graph[first] = entry

    if title_graph:
        title_pattern = "|".join(re.escape(t) for t in _TITLE_TO_ROLE)
        for m in re.finditer(
            rf"\b({title_pattern})\s+([A-Z][a-z]{{1,20}})\b", text
        ):
            role = _TITLE_TO_ROLE[m.group(1).lower()]
            first_name = m.group(2).lower()
            entry = title_graph.get(first_name)
            if entry and entry["display_name"].lower() not in already_named:
                to_add.append({"name": entry["display_name"], "role": role})
                already_named.add(entry["display_name"].lower())

    # --- Named social-role scan ("my friend Neal", "a friend named Neal", etc.) ---
    # Extracts the proper name directly from text so it can go through resolve_person.
    # Unlike title+name scan, this doesn't require a narrator_graph hit — it just
    # produces {"name": ..., "role": ...} entries from surface patterns the LLM may miss.
    _sw_pat = "|".join(re.escape(w) for w in _SOCIAL_ROLE_WORDS)
    _nm = r"([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?)"  # first or full name (capitalized)
    _social_patterns: list[tuple[str, str, str]] = [
        # (regex, role_group, name_group) — group numbers within each sub-pattern
        (rf"\bmy\s+({_sw_pat})\s+{_nm}",                              "1", "2"),
        (rf"\ba\s+(?:good\s+)?({_sw_pat})\s+named\s+{_nm}",          "1", "2"),
        (rf"\bI\s+(?:had|have|knew)\s+a\s+(?:good\s+)?({_sw_pat})\s+named\s+{_nm}", "1", "2"),
        (rf"\b{_nm},?\s+(?:my|a)\s+({_sw_pat})\b",                   "2", "1"),  # "Neal, my friend"
    ]
    for pat, rg, ng in _social_patterns:
        ri, ni = int(rg) - 1, int(ng) - 1
        for m in re.finditer(pat, text):
            g = m.groups()
            role_raw = g[ri] if ri < len(g) else None
            extracted_name = g[ni] if ni < len(g) else None
            if role_raw and extracted_name:
                canonical_role = _SOCIAL_ROLE_WORDS.get(role_raw.lower(), role_raw.lower())
                if extracted_name.lower() not in already_named:
                    to_add.append({"name": extracted_name, "role": canonical_role})
                    already_named.add(extracted_name.lower())

    # --- Two-hop possessive scan ("my dad's mom", "grandma on my dad's side", etc.) ---
    # Relies on build_narrator_graph having labeled grandparents as
    # "paternal/maternal grandmother/grandfather".
    two_hop_graph: dict[str, dict] = {
        e["role"]: e for e in narrator_graph
        if e.get("role", "").endswith(("grandmother", "grandfather", "grandparent"))
        and e.get("role", "").startswith(("paternal", "maternal"))
    }
    if two_hop_graph:
        for pattern, target_role in _TWO_HOP_PATTERNS:
            if re.search(pattern, text_lc):
                entry = two_hop_graph.get(target_role)
                if entry and entry["display_name"].lower() not in already_named:
                    to_add.append({"name": entry["display_name"], "role": target_role})
                    already_named.add(entry["display_name"].lower())

    return llm_people + to_add


def _gender_refine_parent(role: str, gender: str | None) -> str:
    """Upgrade generic 'parent' to 'father'/'mother' when Person.gender is known."""
    if role != "parent" or not gender:
        return role
    g = gender.lower()
    if g in ("male", "m", "man"):
        return "father"
    if g in ("female", "f", "woman"):
        return "mother"
    return role


async def build_narrator_graph(db: AsyncSession, user_id: int) -> list[dict]:
    """
    Return a list of {role, display_name, person_id, gender} dicts describing the
    narrator's direct and 2-hop family relationships.

    Direct (1-hop): father, mother, spouse, siblings, children, aunts/uncles, etc.
    Two-hop: grandparents are resolved with side labels — "paternal grandmother",
    "maternal grandfather" — so the LLM can resolve "my dad's mom" or
    "my grandma on my mom's side" to the right person.
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
                refined = _gender_refine_parent(role, getattr(p, "gender", None))
                entries.append({"role": refined, "display_name": p.display_name, "person_id": other_id, "gender": getattr(p, "gender", None)})
                seen_pids.add(other_id)

    for rel_type, other_id in dst_rows:
        role = _DST_EDGE_TO_ROLE.get((rel_type or "").lower())
        if role and other_id not in seen_pids:
            p = await db.get(Person, other_id)
            if p and p.display_name:
                refined = _gender_refine_parent(role, getattr(p, "gender", None))
                entries.append({"role": refined, "display_name": p.display_name, "person_id": other_id, "gender": getattr(p, "gender", None)})
                seen_pids.add(other_id)

    # --- 2-hop: grandparents with paternal/maternal side labels ---
    _PARENT_ROLES = {"father", "mother", "parent"}
    for parent_entry in [e for e in entries if e["role"] in _PARENT_ROLES]:
        parent_pid = parent_entry["person_id"]
        parent_role = parent_entry["role"]
        # Determine side from the parent's role/gender
        parent_gender = (parent_entry.get("gender") or "").lower()
        if parent_role == "father" or parent_gender in ("male", "m"):
            side = "paternal"
        elif parent_role == "mother" or parent_gender in ("female", "f"):
            side = "maternal"
        else:
            continue  # unknown-gender generic parent — can't label side reliably

        gp_src = (await db.execute(
            select(RelationshipEdge.rel_type, RelationshipEdge.dst_id)
            .where(RelationshipEdge.src_id == parent_pid)
        )).all()
        gp_dst = (await db.execute(
            select(RelationshipEdge.rel_type, RelationshipEdge.src_id)
            .where(RelationshipEdge.dst_id == parent_pid)
        )).all()

        for rel_type, gp_id in gp_src:
            gp_role = _SRC_EDGE_TO_ROLE.get((rel_type or "").lower())
            if gp_role in _PARENT_ROLES and gp_id not in seen_pids:
                gp = await db.get(Person, gp_id)
                if gp and gp.display_name:
                    refined = _gender_refine_parent(gp_role, getattr(gp, "gender", None))
                    if refined == "father":
                        labeled = f"{side} grandfather"
                    elif refined == "mother":
                        labeled = f"{side} grandmother"
                    else:
                        labeled = f"{side} grandparent"
                    entries.append({"role": labeled, "display_name": gp.display_name, "person_id": gp_id, "gender": getattr(gp, "gender", None)})
                    seen_pids.add(gp_id)

        for rel_type, gp_id in gp_dst:
            gp_role = _DST_EDGE_TO_ROLE.get((rel_type or "").lower())
            if gp_role in _PARENT_ROLES and gp_id not in seen_pids:
                gp = await db.get(Person, gp_id)
                if gp and gp.display_name:
                    refined = _gender_refine_parent(gp_role, getattr(gp, "gender", None))
                    if refined == "father":
                        labeled = f"{side} grandfather"
                    elif refined == "mother":
                        labeled = f"{side} grandmother"
                    else:
                        labeled = f"{side} grandparent"
                    entries.append({"role": labeled, "display_name": gp.display_name, "person_id": gp_id, "gender": getattr(gp, "gender", None)})
                    seen_pids.add(gp_id)

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
            text_syns = frozenset(s.lower() for s in _ROLE_TEXT_SYNONYMS.get(entry["role"], []))
            if name_lower in synonyms or name_lower in text_syns or name_lower == entry["role"].lower():
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
    _ROLE_TO_GENDER: dict[str, str] = {
        "father": "male", "grandfather": "male", "great-grandfather": "male",
        "uncle": "male", "brother": "male", "son": "male", "husband": "male",
        "nephew": "male", "godfather": "male", "stepfather": "male",
        "mother": "female", "grandmother": "female", "great-grandmother": "female",
        "aunt": "female", "sister": "female", "daughter": "female", "wife": "female",
        "niece": "female", "godmother": "female", "stepmother": "female",
    }
    inferred_gender = _ROLE_TO_GENDER.get((role_hint or "").lower().strip())
    p = Person(
        owner_user_id=user_id,
        display_name=name,
        gender=inferred_gender,
        meta={"inferred": True},
    )
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


_LLM_EXTRACT_SYSTEM = """You extract people, places, and events mentioned in a personal story or memoir response.

NARRATOR CONTEXT — CRITICAL:
The narrator_relationships list tells you the narrator's actual family members.
You MUST resolve informal role references to the real names listed there.

Role reference → how to resolve using narrator_relationships:
  "dad", "father", "my dad", "my father", "pop", "papa", "pa"
      → find entry with role "father" in narrator_relationships
      → if no "father" entry, find "parent" with gender "male" (or "m")
  "mom", "mother", "my mom", "my mother", "mama", "ma"
      → find entry with role "mother" in narrator_relationships
      → if no "mother" entry, find "parent" with gender "female" (or "f")
  "parents", "mom and dad", "my folks"
      → include BOTH the father AND mother entries
  "wife", "my wife", "husband", "my husband"
      → find entry with role "spouse" or "partner"
  "brother", "my brother", "sister", "my sister"
      → find entry with role "sibling"
  "grandma", "grandmother", "nana", "grandpa", "grandfather", "gramps"
      → find entry with role "grandmother", "grandfather", or "grandparent"
  "my dad's mom", "my grandma on my dad's side", "my grandma that is my dad's mom"
      → find entry with role "paternal grandmother"
  "my dad's dad", "my grandpa on my dad's side"
      → find entry with role "paternal grandfather"
  "my mom's mom", "my grandma on my mom's side"
      → find entry with role "maternal grandmother"
  "my mom's dad", "my grandpa on my mom's side"
      → find entry with role "maternal grandfather"

When narrator_relationships has a "gender" field, use it to distinguish father (male) from mother (female) when both appear with role "parent".
Always output the full name from narrator_relationships, never the role word itself.

Return STRICT JSON only — exactly this shape:
{
  "people": [
    {"name": "full name or best known name", "role": "relationship to narrator or null"}
  ],
  "places": [
    {"name": "place name", "type": "home|farm|city|landmark|region|other",
     "address": null, "city": null, "state": null, "country": null}
  ],
  "events": [
    {"name": "event name without year e.g. Christmas", "year": 1985,
     "type": "holiday|birthday|vacation|reunion|wedding|other",
     "places": ["place name as it appears in places array"],
     "attendees": ["person name as it appears in people array"]}
  ]
}

Rules — PEOPLE:
- Resolve role-based references ("my mom", "dad", "Uncle Dave") to full names using narrator_relationships
- If no match in narrator_relationships, use the most specific name/nickname from the text
- If the same person is mentioned multiple ways, pick the most specific / full name
- Do NOT include the narrator themselves
- Limit to 20 people maximum

Rules — PLACES:
- Include specific locations: houses, farms, towns, cities, landmarks, schools, churches
- Use the family's informal name if that's what's used ("Grandma's farm", "the lake house")
- Do NOT include vague directions like "home", "outside", "somewhere"
- city/state/country: fill only if explicitly stated in the text
- Limit to 15 places maximum

Rules — EVENTS:
- Include named occasions tied to a specific year if mentioned: "Christmas 1985", "our vacation to Florida"
- Name should be the event type only ("Christmas", "Florida Vacation") — year goes in the year field
- year: integer if stated or clearly implied, null if unknown
- type: pick the closest match from the allowed values
- places/attendees: reference names exactly as they appear in the people/places arrays above
- Do NOT create events for generic mentions like "last summer" with no specific occasion
- Limit to 10 events maximum

If a category has nothing to report, return an empty array for it."""


_PLACES_EVENTS_SYSTEM = """Extract specific named places and named occasions from a family memoir story.

Return STRICT JSON only — no commentary, no markdown:
{
  "places": [
    {"name": "place name as used in the story", "type": "home|farm|city|landmark|school|church|other",
     "city": null, "state": null, "country": null}
  ],
  "events": [
    {"name": "event name without year", "year": null,
     "type": "holiday|birthday|vacation|reunion|wedding|funeral|other"}
  ]
}

PLACES — include named or described specific locations:
- Family properties: "Grandma's farm", "the lake house", "the old homestead", "Uncle Bill's place"
- Towns, cities, states, countries when named as destinations or origins
- Buildings by name: schools, churches, hospitals, businesses
- Geographic features when named: "the creek", "Mohawk Lake", "the back forty"

PLACES — exclude:
- Vague words: "home", "outside", "somewhere", "there", "the store" (unnamed)
- Pure directions: "down the road", "next door"

EVENTS — include named family occasions (year optional):
- Recurring holidays: Christmas, Thanksgiving, Easter, Fourth of July
- Life events: weddings, funerals, births, graduations, baptisms
- Named trips or gatherings: "the Florida vacation", "the family reunion", "Sunday dinners"
- Be liberal — if the story is clearly about a specific occasion, include it

EVENTS — exclude:
- Pure time references with no occasion: "last summer", "back then", "a few years ago"
- Generic daily activities: "we had dinner", "went fishing" (unless it's a named tradition)

Limit: 10 places, 8 events. Return empty arrays if nothing qualifies."""


async def llm_extract_places_events(text: str) -> dict:
    """
    Dedicated LLM call for places and events — simpler prompt, no narrator context.
    Returns {"places": [...], "events": [...]}.
    """
    if not text or not text.strip():
        return {"places": [], "events": []}
    try:
        from app.llm_client import your_chat_completion
        raw = await your_chat_completion(
            system=_PLACES_EVENTS_SYSTEM,
            user=text.strip()[:6000],  # cap to keep context manageable
            response_format="json",
            temperature=0.1,
        )
        data = json.loads(raw)

        places = []
        for pl in (data.get("places") or []):
            n = (pl.get("name") or "").strip()
            if n and len(n) > 1:
                places.append({
                    "name": n,
                    "type": (pl.get("type") or "other").strip(),
                    "address": None,
                    "city": pl.get("city") or None,
                    "state": pl.get("state") or None,
                    "country": pl.get("country") or None,
                })

        events = []
        for ev in (data.get("events") or []):
            n = (ev.get("name") or "").strip()
            if n and len(n) > 1:
                yr = ev.get("year")
                events.append({
                    "name": n,
                    "year": int(yr) if yr else None,
                    "type": (ev.get("type") or "other").strip(),
                    "places": [],
                    "attendees": [],
                })

        return {"places": places, "events": events}

    except Exception:
        logger.warning("llm_extract_places_events failed", exc_info=True)
        return {"places": [], "events": []}


async def llm_extract_entities(
    text: str,
    known_names: list[str] | None = None,
    narrator_graph: list[dict] | None = None,
) -> dict:
    """
    LLM call extracting people from a story (with narrator-graph resolution).
    Places and events are extracted separately via llm_extract_places_events().

    Returns {"people": [...], "places": [], "events": []} — callers should
    merge with llm_extract_places_events() results for the full picture.
    """
    if not text or not text.strip():
        return {"people": [], "places": [], "events": []}
    people: list[dict] = []
    try:
        from app.llm_client import your_chat_completion
        narrator_relationships = [
            {"role": e["role"], "name": e["display_name"], **({"gender": e["gender"]} if e.get("gender") else {})}
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

        for p in (data.get("people") or []):
            n = (p.get("name") or "").strip()
            r = (p.get("role") or None)
            if r:
                r = r.strip() or None
            if n:
                people.append({"name": n, "role": r})

    except Exception:
        logger.warning("llm_extract_entities failed; falling back to role rules only", exc_info=True)

    # Always run role-word fallback — ensures "dad"/"mom"/etc. map to narrator graph
    # even when the LLM call fails or the model ignores narrator_relationships
    people = _apply_role_fallback(text, people, narrator_graph or [])

    return {"people": people, "places": [], "events": []}


# Keep old name as a shim so existing callers don't break
async def llm_extract_people(
    text: str,
    known_names: list[str] | None = None,
    narrator_graph: list[dict] | None = None,
) -> list[dict]:
    result = await llm_extract_entities(text, known_names=known_names, narrator_graph=narrator_graph)
    return result["people"]

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
