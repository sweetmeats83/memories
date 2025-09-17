"""
Deterministic auto-tagger for Prompts/Responses.

- Loads a shared Tagify whitelist from: app/data/tag_whitelist.json
- Suggests tags using keywords, decade/year heuristics, length estimation
- Returns a list of (tag, score) sorted by score desc

Usage:
    from app.services.auto_tag import suggest_tags_rule_based
    tags = suggest_tags_rule_based(text, word_count=len(text.split()), language_code="en")
"""

from __future__ import annotations
import json
import re, os, json
import pathlib
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Iterable
from app.llm_client import guess_for_gates
import asyncio, logging, json

# ------------------------------------------------------------
# Load whitelist (works with:
#   1) ["life:adult", "theme:school", ...] OR
#   2) [{"value":"life:adult","label":"Life · Adult"}, ...]
# ------------------------------------------------------------
def _load_whitelist() -> set[str]:
    wl_path = pathlib.Path(__file__).resolve().parents[1] / "data" / "tag_whitelist.json"
    data = json.loads(wl_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and "value" in data[0]:
            return {d["value"] for d in data if isinstance(d, dict) and "value" in d}
        elif all(isinstance(x, str) for x in data):
            return set(data)
    raise ValueError("tag_whitelist.json must be a list of strings OR a list of objects with a 'value' field.")

WHITELIST: set[str] = _load_whitelist()

# Hot-reload support for roles/relationship whitelist
def reload_whitelist(path: str | None = None) -> int:
    """
    Reload the tag whitelist from disk and replace the global WHITELIST.
    If `path` is None, uses the same default as _load_whitelist().
    Returns the number of entries loaded.
    """
    global WHITELIST
    try:
        if path is None:
            wl_path = pathlib.Path(__file__).resolve().parents[1] / "data" / "tag_whitelist.json"
            path = str(wl_path)
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            if data and isinstance(data[0], dict) and "value" in data[0]:
                WHITELIST = {d["value"] for d in data if isinstance(d, dict) and "value" in d}
            elif all(isinstance(x, str) for x in data):
                WHITELIST = set(data)
            else:
                raise ValueError
        else:
            raise ValueError
        return len(WHITELIST)
    except Exception:
        # Do not clear existing on failure
        return len(WHITELIST)

# ------------------------------------------------------------
# Keyword maps (extend safely over time)
# ------------------------------------------------------------
THEME_KEYWORDS: Dict[str, List[str]] = {
    "theme:school":        [r"\bschool\b", r"\bclass(es)?\b", r"\bteacher(s)?\b", r"\bhomework\b"],
    "theme:parenting":     [r"\b(parenting|newborn|diaper|nursery)\b"],
    "theme:first-job":     [r"\b(first job|hired|interview|resume)\b"],
    "theme:career":        [r"\b(career|promotion|coworker|office|workplace)\b"],
    "theme:migration":     [r"\b(moved? (to|from)|relocat(e|ed|ion))\b"],
    "theme:immigration":   [r"\b(immigra(nt|tion)|visa|green card|citizenship)\b"],
    "theme:travel":        [r"\b(road trip|flight|airport|train|backpack(ing)?)\b"],
    "theme:food":          [r"\b(food|meal|restaurant|dinner|lunch|breakfast)\b"],
    "theme:recipes":       [r"\b(recipe|cook|kitchen|bake|ingredients)\b"],
    "theme:music":         [r"\b(band|album|concert|playlist|cassette|vinyl|CDs?)\b"],
    "theme:arts":          [r"\b(paint(ing)?|draw(ing)?|sculpt(ure|ing)|gallery)\b"],
    "theme:crafts":        [r"\b(knit|crochet|quilt|woodwork|handmade)\b"],
    "theme:sports":        [r"\b(game|team|coach|tournament|practice|league)\b"],
    "theme:faith":         [r"\b(church|mosque|synagogue|temple|faith|prayer)\b"],
    "theme:traditions":    [r"\b(tradition(s)?|custom(s)?|ritual(s)?)\b"],
    "theme:holidays":      [r"\b(holiday|christmas|hanukkah|eid|diwali|thanksgiving)\b"],
    "theme:technology":    [r"\b(computer|internet|smartphone|software|programming)\b"],
    "theme:health":        [r"\b(doctor|hospital|surgery|diagnosis|recovery)\b"],
    "theme:resilience":    [r"\b(recovery|overcame|resilien(t|ce)|bounce\s?back)\b"],
    "theme:love":          [r"\b(love|romance|partner|relationship)\b"],
    "theme:dating":        [r"\b(date|dating|boyfriend|girlfriend)\b"],
    "theme:marriage":      [r"\b(married|wedding|spouse)\b"],
    "theme:home":          [r"\b(home|house|apartment|neighborhood)\b"],
    "theme:moving-house":  [r"\b(move|moved|packing|unpack(ing)?)\b"],
    "theme:community":     [r"\b(community|volunteer|neighborhood|local)\b"],
    "theme:activism":      [r"\b(protest|activis(m|t)|campaign|organize)\b"],
    "theme:military":      [r"\b(deployed|boot camp|basic training|unit|platoon|veteran)\b"],
    "theme:nature":        [r"\b(hike|forest|mountain|river|camp(ing)?)\b"],
    "theme:pets":          [r"\b(cat|dog|pet|puppy|kitten)\b"],
    "theme:vehicles":      [r"\b(car|truck|motorcycle|bicycle|bus)\b"],
    "theme:finance":       [r"\b(budget|savings|debt|loan|finance)\b"],
    "theme:pandemic":      [r"\b(pandemic|covid|lockdown|quarantine)\b"],
    "theme:disasters":     [r"\b(earthquake|flood|hurricane|wildfire|tornado)\b"],
    "theme:grief":         [r"\b(grief|passed away|funeral|bereavement)\b"],
    "theme:language":      [r"\b(bilingual|learn(ing)? (english|spanish|french|german|italian|portuguese|russian|chinese|japanese|korean))\b"],
    "theme:customs":       [r"\b(custom(s)?|etiquette|tradition(s)?)\b"],
    "theme:humor":         [r"\b(funny|laugh|joke|humor)\b"],
    "theme:firsts":        [r"\b(first time|first\b.*(day|car|job|home))\b"],
    "theme:failures":      [r"\b(fail(ed|ure)|mistake|error|setback)\b"],
    "theme:successes":     [r"\b(success|achieve(d|ment)|win|award)\b"],
    "theme:lessons":       [r"\b(lesson(s)? learned|what I learned|taught me)\b"],
    "theme:heroes":        [r"\b(hero|role model|inspiration)\b"],
}

RELATIONSHIP_KEYWORDS: Dict[str, List[str]] = {
    "relationship:grandparent":    [r"\b(grandma|grandpa|grandparent(s)?)\b"],
    "relationship:parent":         [r"\b(mom|mother|dad|father|parent(s)?)\b"],
    "relationship:child":          [r"\b(son|daughter|child|children)\b"],
    "relationship:sibling":        [r"\b(sister|brother|sibling(s)?)\b"],
    "relationship:spouse-partner": [r"\b(wife|husband|spouse|partner)\b"],
    "relationship:friend":         [r"\b(friend|buddy|pal|bestie)\b"],
    "relationship:mentor":         [r"\b(mentor|coach|teacher)\b"],
    "relationship:colleague":      [r"\b(colleague|coworker)\b"],
    "relationship:neighbor":       [r"\b(neighbor|neighbour)\b"],
    "relationship:caregiver-role": [r"\b(caregiver|caretaker)\b"],
}

REGION_KEYWORDS: Dict[str, List[str]] = {
    "region:us-midwest":   [r"\b(Chicago|Detroit|Cleveland|Milwaukee|Minneapolis|St\.?\s?Paul)\b"],
    "region:us-northeast": [r"\b(New York|Boston|Philadelphia|Pittsburgh|Providence)\b"],
    "region:us-south":     [r"\b(Atlanta|Nashville|Dallas|Houston|Miami|Charlotte)\b"],
    "region:us-west":      [r"\b(Los Angeles|San Francisco|Seattle|Portland|Phoenix|Denver)\b"],
    "region:east-asia":    [r"\b(Seoul|Tokyo|Osaka|Beijing|Shanghai|Taipei)\b"],
    "region:europe-west":  [r"\b(London|Paris|Madrid|Lisbon|Dublin|Amsterdam|Brussels)\b"],
    "region:latin-america":[r"\b(Mexico City|Bogotá|Lima|Santiago|Buenos Aires|São Paulo)\b"],
}

DECADE_PAT = re.compile(r"\b(19|20)\d0s\b", re.IGNORECASE)  # “1990s”
YEAR_PAT   = re.compile(r"\b(19|20)\d{2}\b")                # “1998”

# ------------------------------------------------------------
# Heuristics
# ------------------------------------------------------------
def _score_hits(text: str, mapping: Dict[str, List[str]], base: float = 0.65) -> Dict[str, float]:
    text_lc = text.lower()
    out: Dict[str, float] = {}
    for tag, pats in mapping.items():
        for p in pats:
            if re.search(p, text_lc, flags=re.IGNORECASE):
                out[tag] = max(out.get(tag, 0.0), base)
    return out

def _infer_era(text: str) -> Dict[str, float]:
    text_lc = text.lower()
    hits: Dict[str, float] = {}
    for m in DECADE_PAT.finditer(text_lc):
        decade = m.group(0).lower()  # e.g., "1990s"
        tag = f"era:{decade}"
        if tag in WHITELIST:
            hits[tag] = max(hits.get(tag, 0.0), 0.7)
    for m in YEAR_PAT.finditer(text_lc):
        y = int(m.group(0))
        decade = (y // 10) * 10
        tag = f"era:{decade}s"
        if tag in WHITELIST:
            hits[tag] = max(hits.get(tag, 0.0), 0.5)
    return hits

def _infer_length(word_count: Optional[int]) -> Dict[str, float]:
    if word_count is None:
        return {}
    if word_count < 120:
        return {"length:short": 0.7}
    if word_count < 400:
        return {"length:medium": 0.7}
    return {"length:long": 0.7}

def _clamp_whitelist(scored: Dict[str, float], *, max_tags: int = 12, min_score: float = 0.5) -> List[Tuple[str, float]]:
    items = [(t, s) for t, s in scored.items() if t in WHITELIST and s >= min_score]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:max_tags]

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def suggest_tags_rule_based(
    text: str,
    *,
    word_count: Optional[int] = None,
    locale_hint: Optional[str] = None,
    language_code: Optional[str] = None
) -> List[Tuple[str, float]]:
    """
    Returns: list of (tag, score) sorted desc, max 12 tags.
    Only emits tags present in WHITELIST.
    """
    scores: Dict[str, float] = defaultdict(float)

    # Themes / relationships / regions
    for t, s in _score_hits(text, THEME_KEYWORDS, 0.65).items():
        scores[t] = max(scores[t], s)
    for t, s in _score_hits(text, RELATIONSHIP_KEYWORDS, 0.65).items():
        scores[t] = max(scores[t], s)
    for t, s in _score_hits(text, REGION_KEYWORDS, 0.70).items():
        scores[t] = max(scores[t], s)

    # Era / length
    for t, s in _infer_era(text).items():
        scores[t] = max(scores[t], s)
    for t, s in _infer_length(word_count).items():
        scores[t] = max(scores[t], s)

    # Language (if provided by transcription or UI)
    if language_code:
        tag = f"language:{language_code}"
        if tag in WHITELIST:
            scores[tag] = max(scores[tag], 0.9)

    # Locale hints (optional)
    if locale_hint:
        l = locale_hint.lower()
        if "rural" in l:
            scores["region:rural"] = max(scores["region:rural"], 0.6)
        if "urban" in l or "city" in l:
            scores["region:urban"] = max(scores["region:urban"], 0.6)

    return _clamp_whitelist(scores, max_tags=12, min_score=0.5)

# ------------------------------------------------------------
# Prompt-aware tagger (adds for:* guesses when text is prompt-like)
# ------------------------------------------------------------
_PROMPT_LIKE_PAT = re.compile(
    r"\b(tell|write|share|describe|explain|what|how|when|did you|have you|remember)\b",
    re.I,
)

def _is_prompt_like(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(_PROMPT_LIKE_PAT.search(t) and re.search(r"\byou\b", t, re.I))

async def suggest_tags_for_prompt(
    text: str,
    *,
    word_count: Optional[int] = None,
    locale_hint: Optional[str] = None,
    language_code: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Rule-based tags + LLM 'for:*' guess (only from whitelist). If LLM unavailable,
    falls back to 'for:all' when allowed. Returns up to 12 items (slug, score).
    """
    base = suggest_tags_rule_based(
        text,
        word_count=word_count,
        locale_hint=locale_hint,
        language_code=language_code,
    )
    # Only try LLM if text looks like a prompt AND we have for:* in whitelist
    FOR_ALLOWED: list[str] = sorted([v for v in WHITELIST if v.startswith("for:")])
    if not FOR_ALLOWED or not _is_prompt_like(text):
        return base

    # Feature flag (default ON). Set MEMORIES_AUTOTAG_GUESS_FOR=0 to disable.
    if os.getenv("MEMORIES_AUTOTAG_GUESS_FOR", "1") in ("0", "false", "False"):
        return base

    try:
        for_tags = await guess_for_gates(text, FOR_ALLOWED)
    except Exception:
        for_tags = ["for:all"] if "for:all" in FOR_ALLOWED else []

    if not for_tags:
        return base

    # Merge with high confidence for specificity; keep set & clamp
    merged: Dict[str, float] = {k: v for k, v in base}
    for t in for_tags:
        merged[t] = max(merged.get(t, 0.0), 0.92 if t != "for:all" else 0.80)
    # return in (slug, score) format, bounded to whitelist/max count
    items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    # Respect global clamp behavior
    return items[:12]
