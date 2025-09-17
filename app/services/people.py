import re
from typing import Optional, Iterable, Tuple
from sqlalchemy import select, func
from app.models import ResponsePerson, PersonShare, KinMembership, Person, PersonAlias
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils import slugify
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

async def resolve_person(db, user_id: int, display_or_alias: str) -> Person:
    name = (display_or_alias or "").strip()
    if not name:
        # defensive fallback: create a placeholder
        p = Person(owner_user_id=user_id, display_name="(Unknown)")
        db.add(p)
        await db.flush()
        return p

    # 1) Alias exact (case-insensitive) WITH join to Person for owner filter
    alias_row = await db.scalar(
        select(PersonAlias)
        .join(Person, Person.id == PersonAlias.person_id)
        .where(func.lower(PersonAlias.alias) == func.lower(name))
        .where(Person.owner_user_id == user_id)
        .limit(1)
    )
    if alias_row:
        return await db.get(Person, alias_row.person_id)

    # 2) Person display_name exact (case-insensitive) for this owner
    person = await db.scalar(
        select(Person)
        .where(func.lower(Person.display_name) == func.lower(name))
        .where(Person.owner_user_id == user_id)
        .limit(1)
    )
    if person:
        return person

    # 3) Create new person + seed alias (mark as inferred from mention)
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

def role_hint_near(text: str, start: int, end: int, window: int = 24) -> str | None:
    """
    Looks ±window chars around a mention for role words (grandma, uncle, etc.).
    """
    around = text[max(0, start-window): min(len(text), end+window)]
    return guess_role_hint(around)

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
    - Prefer matching by exact display_name (case-insensitive) for this owner.
    - If not found, match by alias for this owner.
    - Otherwise create a new Person.
    - Always ensure a PersonAlias row exists for the submitted surface string.
    - Store/merge role_hint into Person.meta without clobbering other meta keys.
    """
    name = (display_name or "").strip()
    if not name:
        raise ValueError("display_name is required")

    # 1) Try exact display_name match for this owner (case-insensitive)
    q1 = (
        select(Person)
        .where(
            Person.owner_user_id == owner_user_id,
            func.lower(Person.display_name) == func.lower(name),
        )
        .limit(1)
    )
    person = (await db.execute(q1)).scalars().first()

    # 2) Otherwise, try alias match for this owner
    if not person:
        q2 = (
            select(Person)
            .join(PersonAlias, PersonAlias.person_id == Person.id)
            .where(
                Person.owner_user_id == owner_user_id,
                func.lower(PersonAlias.alias) == func.lower(name),
            )
            .limit(1)
        )
        person = (await db.execute(q2)).scalars().first()

    # 3) Create if still missing
    if not person:
        given, family = _split_display_name(name)
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
