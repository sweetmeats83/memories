import os, re, json
import httpx
from typing import Optional, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models import Response, Prompt, Tag

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # adjust to your model tag

class OllamaError(RuntimeError):
    pass

#----------text polish---------------

def _sanitize_llm_text(out: str) -> str:
    """Remove assistant-y prefaces and unwrap code fences/quotes."""
    if not out:
        return ""
    s = out.strip()
    # Prefer content inside triple backticks if present
    m = re.search(r"```(?:\w+)?\s*([\s\S]*?)```", s)
    if m and m.group(1).strip():
        s = m.group(1).strip()
    # Drop common preface lines like "Here you go:", "Cleaned transcript:", etc.
    lines = [ln.rstrip() for ln in s.splitlines()]
    while lines:
        head = lines[0].strip()
        if not head:
            lines.pop(0)
            continue
        low = head.lower().rstrip(":")
        boiler = (
            "here you go" in low or
            "here is" in low or
            "here's" in low or
            "cleaned transcript" in low or
            "polished transcript" in low or
            "edited version" in low or
            "revised version" in low or
            "final version" in low or
            low.endswith("cleaned transcript") or
            low.endswith("transcript")
        )
        if boiler and len(head) <= 120:
            lines.pop(0)
            continue
        break
    s = "\n".join(lines).strip()
    # Unwrap matching surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("“") and s.endswith("”")) or (s.startswith("'") and s.endswith("'")):
        inner = s[1:-1].strip()
        if inner:
            s = inner
    return s


async def polish_text(text: str, style: str = "clean") -> str:
    """
    Send text to Ollama /api/generate for a light-touch cleanup suitable for transcripts.
    """
    if not text.strip():
        return text

    # Prompt engineered for transcript cleanup (punctuation, casing, light disfluency removal)
    system = (
        "You are a transcription editor. "
        "Lightly fix punctuation, casing, and obvious transcription errors. "
        "Keep the speaker's voice and meaning. Do not add facts. "
        "Do not invent content. Keep it as a single paragraph unless clear sentence breaks."
    )
    if style == "paragraphs":
        system += " Break into short readable paragraphs when appropriate."

    prompt = f"{system}\n\n---\nOriginal transcript:\n{text}\n---\nCleaned transcript:\n"

    url = f"{OLLAMA_BASE_URL}/api/generate"
    # Non-streaming for simplicity
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Tweak temperature etc. for stability
        "options": {"temperature": 0.2}
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        raise OllamaError(f"Ollama request failed: {e}") from e

    out = (data or {}).get("response", "")
    if not out:
        raise OllamaError("Empty response from Ollama.")
    return _sanitize_llm_text(out)
# llm_client.py (add)

#------Prompt curation------------

async def curate_prompts_for_user(profile_summary: str,
                                  top_tags: Sequence[str],
                                  exemplars: Sequence[dict],
                                  max_new: int = 5) -> list[dict]:
    """
    Returns a list of {title, text, tags, rationale} from Ollama.
    """
    system = (
        "You are a family-history prompt curator. Given user profile and tags, "
        "select relevant existing prompts by tag overlap and propose up to {max_new} new prompts (if gaps exist). "
        "Keep prompts kind, clear, specific; avoid sensitive topics unless explicitly opted-in. "
        "Return ONLY strict JSON list of objects with fields: title, text, tags (<=3), rationale."
    )
    sys_prompt = system.format(max_new=max_new)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": (
            f"{sys_prompt}\n\n"
            f"User summary:\n{profile_summary}\n\n"
            f"Top tags: {', '.join(top_tags)}\n\n"
            f"Exemplar prompts (JSON):\n{exemplars}\n\n"
            "Output JSON:"
        ),
        "stream": False,
        "options": {"temperature": 0.2}
    }
    try:
        async with httpx.AsyncClient(timeout=90) as c:
            r = await c.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        raise OllamaError(f"Ollama curator failed: {e}") from e

    import json
    raw = (data or {}).get("response", "").strip()
    try:
        out = json.loads(raw)
    except Exception as e:
        raise OllamaError(f"Curator returned non-JSON: {raw[:200]}") from e
    # Normalize
    cleaned = []
    for item in out or []:
        cleaned.append({
            "title": (item.get("title") or "").strip() or None,
            "text": (item.get("text") or "").strip(),
            "tags": [t.strip() for t in (item.get("tags") or [])][:3],
            "rationale": (item.get("rationale") or "").strip(),
        })
    return [x for x in cleaned if x["text"]]
    
async def your_chat_completion(system: str, user: str, response_format: str = "json", temperature: float = 0.2) -> str:
    import httpx, json
    prompt = (
        f"{system}\n\n"
        f"USER:\n{user}\n\n"
        "Return ONLY the response in strict JSON."
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
    return (data or {}).get("response", "").strip()
#-----------tag refineing-------------



async def refine_tags_with_llm(text: str, candidate_tags: list[str], whitelist: list[str]) -> list[tuple[str, float]]:
    """
    Returns [(tag, confidence_0to1)].
    Must only include tags from whitelist.
    """
    system = (
        "You are a tagger. Output JSON with {\"tags\":[{\"value\":\"family:value\",\"confidence\":0..1}]}.\n"
        "Only use tags from the whitelist. Do not invent new values. Max 12 tags."
    )
    user = json.dumps({
        "whitelist": whitelist,
        "draft_tags": candidate_tags,
        "text": text
    }, ensure_ascii=False)

    # Replace this with your actual chat call:
    raw = await your_chat_completion(system=system, user=user, response_format="json", temperature=0.2)

    try:
        data = json.loads(raw)
        out = []
        for item in data.get("tags", []):
            v = item.get("value")
            c = float(item.get("confidence", 0))
            if v in whitelist:
                out.append((v, max(0.0, min(1.0, c))))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:12]
    except Exception:
        return []

# ---------- for:* gate guesser ----------
async def guess_for_gates(text: str, allowed: list[str]) -> list[str]:
    """
    Return up to 2 items from `allowed` (e.g. ["for:grandmother"]) that indicate
    WHO should answer the prompt. Prefer specific roles; use "for:all" only if
    universal or no clear signal.
    """
    if not text or not allowed:
        return []

    # Prefer neutral families if present
    family_neutral_order = ["for:parent", "for:grandparent", "for:sibling", "for:spouse", "for:child"]

    # Few-shot + strict JSON
    examples = [
        {
            "prompt": "As a mother, what surprised you most about raising children?",
            "tags": ["for:mother"]
        },
        {
            "prompt": "What do your grandchildren call you?",
            "tags": ["for:grandparent"]
        },
        {
            "prompt": "Tell your son about the day he was born.",
            "tags": ["for:parent"]
        },
        {
            "prompt": "Who was your best friend in school?",
            "tags": ["for:all"]
        },
    ]
    # Filter example tags to only include allowed ones (keep shape)
    filtered_examples = []
    for ex in examples:
        keep = [t for t in ex["tags"] if t in allowed]
        if not keep:
            # if example's canonical not allowed, try neutral fallbacks
            for n in family_neutral_order:
                if n in allowed:
                    keep = [n]
                    break
            if not keep and "for:all" in allowed:
                keep = ["for:all"]
        filtered_examples.append({"prompt": ex["prompt"], "tags": keep})

    sys = (
        "You are tagging WRITING PROMPTS by WHO should answer (the writer's role). "
        "Pick AT MOST TWO tags ONLY from the allowed list. Prefer specific roles when clearly indicated "
        "(e.g., 'As a mother…' => for:mother). Use 'for:parent' or other neutral roles when gender is unknown. "
        "Use 'for:all' only when universal or unclear.\n\n"
        f"ALLOWED TAGS: {json.dumps(allowed, ensure_ascii=False)}\n"
        "Return STRICT JSON only: {\"tags\":[\"for:role\", ...]} with no commentary."
    )

    fewshot = "\n\n".join(
        [f"EXAMPLE\nPROMPT: {ex['prompt']}\nJSON: {json.dumps({'tags': ex['tags']}, ensure_ascii=False)}"
         for ex in filtered_examples]
    )

    full_prompt = f"{sys}\n\n{fewshot}\n\nPROMPT: {text}\nJSON:"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9},
    }

    raw = ""
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            r.raise_for_status()
            raw = (r.json() or {}).get("response", "") or ""
    except Exception:
        raw = ""

    # parse JSON object anywhere in the output (be strict but forgiving)
    tags: list[str] = []
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            for t in (data.get("tags") or []):
                if t in allowed and t not in tags:
                    tags.append(t)
        except Exception:
            pass

    # keep at most 2; if none, prefer 'for:all' if allowed
    if not tags and "for:all" in allowed:
        return ["for:all"]

    # prefer specificity; if both specific & 'for:all', drop 'for:all'
    if "for:all" in tags and len(tags) > 1:
        tags = [t for t in tags if t != "for:all"]

    return tags[:2]

# -------------------------------
# Follow-up question generation
# -------------------------------

async def ask_llm(prompt: str, *, max_tokens: int = 120, temperature: float = 0.2) -> str:
    """
    Minimal wrapper around Ollama /api/generate (non-streaming), consistent with polish_text().
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            # Some Ollama builds accept num_predict to limit tokens. If unsupported, it's ignored.
            "num_predict": max_tokens
        },
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        raise OllamaError(f"Ollama request failed: {e}") from e

    out = (data or {}).get("response", "") or ""
    return out.strip()


FOLLOWUP_TEMPLATE = (
    "You are writing a gentle follow-up question for an elder based on their story below.\n"
    "Be concise, kind, and focused. Avoid repetition. Ask ONE question only, ending with a question mark.\n\n"
    "Story:\n"
    "---\n{story}\n---\n\n"
    "Write exactly ONE clear follow-up question:\n"
)

async def generate_followup_question(story: str, *, style: str = "gentle", max_tokens: int = 120) -> str:
    """
    Returns a single-question follow-up string for the provided story text.
    """
    story = (story or "").strip()
    if not story:
        raise ValueError("No story text to seed")

    # We keep style for future variants; for now, it just nudges the temperature slightly.
    temperature = 0.2 if style == "gentle" else 0.3
    prompt = FOLLOWUP_TEMPLATE.format(story=story)
    question = await ask_llm(prompt, max_tokens=max_tokens, temperature=temperature)

    # Post-trim to one line / single question
    q = question.strip()
    # Heuristic: keep first line that ends with a question mark
    for line in q.splitlines():
        line = line.strip()
        if line.endswith("?"):
            return line
    # Fallback: if model didn't end with '?', force it
    return (q.splitlines()[0].strip().rstrip(".") + "?") if q else "Could you share a little more about that?"


# Keep the exact signature you proposed, but implemented here to use the helpers above.
async def make_llm_followup_prompt(db, user_id: int, response_id: int, style: str, max_tokens: int) -> int:
    """
    Create a new Prompt from an LLM-generated follow-up question seeded with the user's response text.
    Returns the new Prompt.id.
    """
    # Local imports to avoid circular imports at module import time.
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.models import Response, Prompt, Tag

    if not isinstance(db, AsyncSession):
        # Safety—won't block, just clearer error
        raise TypeError("db must be an AsyncSession")

    r = await db.get(Response, response_id)
    if not r or r.user_id != user_id:
        raise ValueError("Response not found")

    story = (r.response_text or r.transcription or "").strip()
    if not story:
        raise ValueError("No story text to seed")

    prompt_text = await generate_followup_question(story, style=style, max_tokens=max_tokens)

    # Create a Prompt; store the question text; chapter defaults to "Follow-ups" if missing.
    chapter_name = (r.prompt.chapter if getattr(r, "prompt", None) and r.prompt and r.prompt.chapter else "Follow-ups")
    p = Prompt(text=prompt_text, chapter=chapter_name)
    db.add(p)
    await db.flush()  # get p.id

    # Optional: tag as follow-up if tag exists
    try:
        tag = (await db.execute(select(Tag).where(Tag.slug == "followup"))).scalars().first()
        if tag:
            p.tags = (p.tags or []) + [tag]
    except Exception:
        # non-fatal
        pass

    await db.flush()
    return p.id
