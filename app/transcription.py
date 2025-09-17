# app/transcription.py
import os, re, shutil, subprocess, logging, difflib
from statistics import mean
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Iterable, Set

# Whisper backend
from faster_whisper import WhisperModel

# DB / models / services
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.database import async_session_maker
from app.models import (
    Response as ResponseModel,
    ResponseSegment,
    Tag,
    Person,
    PersonAlias,
    UserProfile,
    Prompt,
)
from app.services.auto_tag import suggest_tags_rule_based
from app.services.assignment_core import morph_profile
from app.settings.config import settings

# Optional people helpers (we degrade gracefully if missing)
try:
    from app.services.people import extract_name_spans, role_hint_near, resolve_person, link_mention
    _PEOPLE_ENABLED = True
except Exception:
    _PEOPLE_ENABLED = False

# ---------------------------------------------------------------------------
# Paths / logging
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]   # /app/app
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("transcription.log")]
)

# ---------------------------------------------------------------------------
# Settings for Whisper (env-compatible)
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("WHISPER_MODEL", getattr(settings, "WHISPER_MODEL", "small"))
PREFERRED_DEVICE = os.getenv("WHISPER_DEVICE", getattr(settings, "WHISPER_DEVICE", "auto")).lower()
PREFERRED_COMPUTE = os.getenv("WHISPER_COMPUTE", getattr(settings, "WHISPER_COMPUTE", "auto")).lower()


# ---------------------------------------------------------------------------
# Small slug helpers (local to avoid import loops)
# ---------------------------------------------------------------------------
def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\- ]+", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s

def slug_person(name: str) -> str:
    return f"person:{slugify(name)}"

def slug_place(name: str) -> str:
    return f"place:{slugify(name)}"

def slug_role(role: str) -> str:
    return f"role:{slugify(role)}"


# ---------------------------------------------------------------------------
# Name lexicon (bias) + tiny proper noun harvesting
# ---------------------------------------------------------------------------
async def _collect_user_vocabulary(db: Optional[AsyncSession], user_id: Optional[int]) -> List[str]:
    """
    Build a per-user 'proper nouns' list to bias transcription:
      - PersonAlias for People owned by user
      - person's display_name
      - person:* tags inferred from profile.tag_weights
      - extras in profile.privacy_prefs['user_vocab_extra'] (if any)
    """
    if not db or not user_id:
        return []

    vocab: Set[str] = set()

    # Aliases for user's people
    try:
        people = (await db.execute(
            select(Person).where(Person.owner_user_id == user_id)
        )).scalars().all()
        pids = [p.id for p in (people or [])] or [-1]
        aliases = (await db.execute(
            select(PersonAlias.alias).where(PersonAlias.person_id.in_(pids))
        )).scalars().all() or []
        for a in aliases:
            a = (a or "").strip()
            if a:
                vocab.add(a)
        for p in (people or []):
            if p.display_name:
                vocab.add(p.display_name.strip())
    except Exception:
        pass

    # From profile
    try:
        profile: Optional[UserProfile] = (await db.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )).scalars().first()
    except Exception:
        profile = None

    if profile:
        tw = (profile.tag_weights or {}).get("tagWeights", {}) or {}
        for slug in tw.keys():
            if isinstance(slug, str) and slug.startswith("person:"):
                label = slug.split(":", 1)[1].replace("-", " ").strip()
                if label:
                    vocab.add(label.title())
        extras = ((profile.privacy_prefs or {}).get("user_vocab_extra") or [])
        for e in extras:
            e = (e or "").strip()
            if e:
                vocab.add(e)

    words = sorted(vocab, key=lambda s: (len(s), s.lower()))
    return words[:80] if len(words) > 80 else words


def _build_initial_prompt_from_vocab(words: List[str]) -> Optional[str]:
    if not words:
        return None
    joined = ", ".join(words)
    return f"Names and places in this recording may include: {joined}. Use these spellings."


def _harvest_proper_nouns(text: str) -> List[str]:
    """
    Stricter: keep two-token names; keep single tokens only if frequent.
    """
    if not text:
        return []
    candidates = set()

    # First Last (allow hyphen/apostrophe)
    for m in re.finditer(r"\b([A-Z][a-z'’-]+)\s+([A-Z][a-z'’-]+)\b", text):
        candidates.add(f"{m.group(1)} {m.group(2)}")

    # Single capitalized tokens (>=3 letters)
    for t in re.findall(r"\b[A-Z][a-z'’-]{2,}\b", text):
        candidates.add(t)

    STOP = {
        "I","We","My","The","A","An","And","But","Or","If","So","OK","Okay","Yes","No","Thanks","Hello","Hi","Hey",
        "January","February","March","April","May","June","July","August","September","October","November","December",
        "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday",
        "Christmas","Thanksgiving","Easter","Halloween","New","Year",
        # add/remove role nouns depending on preference:
        "Mom","Dad","Grandma","Grandpa","Mother","Father",
    }

    # basic screen
    screened = []
    for c in candidates:
        if c in STOP:
            continue
        if " " in c:
            screened.append(c)
        else:
            if c.isupper():         # drop ALLCAPS
                continue
            screened.append(c)

    # frequency gate for singletons
    freq = {w: len(re.findall(rf"\b{re.escape(w)}\b", text)) for w in screened}
    return [w for w in screened if (" " in w) or freq.get(w, 0) > 1]


def _append_graylist(profile: UserProfile, new_slugs: Iterable[str]) -> None:
    pp = profile.privacy_prefs or {}
    gl = pp.get("graylist_tags") or []
    seen = set(gl)
    changed = False
    for s in new_slugs:
        if s and s not in seen:
            gl.append(s)
            seen.add(s)
            changed = True
    if changed:
        pp["graylist_tags"] = gl
        profile.privacy_prefs = pp


async def _ensure_tag(db: AsyncSession, slug: str) -> Optional[Tag]:
    t = (await db.execute(select(Tag).where(Tag.slug == slug))).scalars().first()
    if t:
        return t
    t = Tag(name=slug.split(":", 1)[-1].replace("-", " ").title(), slug=slug)
    db.add(t)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        t = (await db.execute(select(Tag).where(Tag.slug == slug))).scalars().first()
    return t


# ---------------------------------------------------------------------------
# FFmpeg + transcription core
# ---------------------------------------------------------------------------
def convert_to_wav(input_path: Path, target_sr: int = 16000, mono: bool = True) -> Path:
    """
    Convert to mono 16kHz 16-bit WAV (audio-only).
    """
    assert input_path.exists() and input_path.stat().st_size > 0, f"Missing file: {input_path}"
    out = input_path.with_suffix(".wav")
    ac = "1" if mono else "2"
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-i", str(input_path),
        "-vn", "-map", "a:0?",
        "-ar", str(target_sr), "-ac", ac, "-c:a", "pcm_s16le", "-f", "wav", str(out)
    ]
    logging.info(f"🎯 ffmpeg: {input_path.name} → {out.name}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode('utf-8','ignore')}") from e
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced no output")
    return out


def _punctuation_ratio(text: str) -> float:
    if not text:
        return 1.0
    return len(re.findall(r"\W", text)) / max(1, len(text))


def _compute_params(device_hint: str) -> tuple[str, str]:
    if device_hint == "cuda":
        return "cuda", ("float16" if PREFERRED_COMPUTE in ("auto", "float16") else PREFERRED_COMPUTE)
    return "cpu", ("int8" if PREFERRED_COMPUTE in ("auto", "int8") else PREFERRED_COMPUTE)


def _load_model() -> WhisperModel:
    first = "cuda" if PREFERRED_DEVICE in ("auto", "cuda") else PREFERRED_DEVICE
    try:
        dev, ctype = _compute_params(first)
        m = WhisperModel(MODEL_NAME, device=dev, compute_type=ctype)
        logging.info(f"✅ Whisper '{MODEL_NAME}' on {dev} ({ctype})")
        return m
    except Exception as e:
        logging.warning(f"⚠️ Failed on {first}: {e}; falling back to CPU.")
        dev, ctype = _compute_params("cpu")
        m = WhisperModel(MODEL_NAME, device=dev, compute_type=ctype)
        logging.info(f"✅ Whisper '{MODEL_NAME}' on cpu ({ctype})")
        return m


_model_singleton: Optional[WhisperModel] = None
def _model() -> WhisperModel:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = _load_model()
    return _model_singleton


def _run_transcription(path: Path, force_lang: Optional[str], initial_prompt: Optional[str]) -> str:
    segments, info = _model().transcribe(
        str(path),
        language=force_lang,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        temperature=[0.0],
        beam_size=1,
        compression_ratio_threshold=2.6,
        log_prob_threshold=-1.5,
        no_speech_threshold=0.7,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
    )

    cleaned = []
    kept_lp = []
    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        pr = _punctuation_ratio(text)
        avg_lp = getattr(seg, "avg_logprob", None)

        # Drop very low-confidence punctuation noise; keep the rest
        if (avg_lp is not None and avg_lp < -5.0) and pr > 0.80:
            continue
        if len(text) < 2 and pr > 0.5:
            continue

        cleaned.append(text)
        kept_lp.append(avg_lp)

    out = " ".join(cleaned).strip()
    if out:
        return out

    # fallback: raw join
    raw = " ".join([(getattr(s, "text", "") or "").strip() for s in segments]).strip()
    return raw


async def transcribe_file(filename_or_path: str | Path, db: Optional[AsyncSession] = None, user_id: Optional[int] = None) -> str:
    """
    Transcribe an uploaded media file (relative to uploads or absolute path).
    Bias with user's vocabulary when available.
    """
    # Resolve absolute path
    p = Path(filename_or_path)
    if not p.is_absolute():
        rel = str(p).lstrip("/").replace("\\", "/")
        # accept both "uploads/..." and "static/uploads/..."
        rel = rel.split("uploads/", 1)[-1] if "uploads/" in rel else rel
        p = UPLOAD_DIR / rel

    if not p.exists():
        logging.error(f"❌ File not found: {p}")
        return "[Transcription failed: file not found]"

    # initial prompt from vocab
    initial_prompt = None
    try:
        if db and user_id:
            words = await _collect_user_vocabulary(db, user_id)
            initial_prompt = _build_initial_prompt_from_vocab(words)
    except Exception as e:
        logging.warning(f"⚠️ initial_prompt build failed: {e}")

    try:
        wav = convert_to_wav(p)
        text = _run_transcription(wav, force_lang="en", initial_prompt=initial_prompt)
        if not text or text.startswith("[No clear speech"):
            text = _run_transcription(wav, force_lang=None, initial_prompt=None)
        try:
            wav.unlink(missing_ok=True)
        except Exception:
            pass
        return text or "[No clear speech detected in the recording.]"
    except Exception as e:
        logging.exception("❌ Transcription error")
        return f"[Transcription failed: {e}]"


# ---------------------------------------------------------------------------
# Enrichment (auto-tag + morph + people extraction)
# ---------------------------------------------------------------------------
def _text_for_tagging(resp: ResponseModel) -> str:
    parts = []
    try:
        if resp.prompt and getattr(resp.prompt, "text", None):
            parts.append(resp.prompt.text)
    except Exception:
        pass
    if getattr(resp, "title", None):
        parts.append(resp.title)
    if getattr(resp, "response_text", None):
        parts.append(resp.response_text)
    if getattr(resp, "transcription", None):
        parts.append(resp.transcription)
    return " \n".join(p for p in parts if p and str(p).strip())


async def enrich_after_transcription(db: AsyncSession, response: ResponseModel) -> None:
    """
    Keeps your previous enrichment behavior:
      1) auto-tag using rule-based suggester
      2) morph user profile
      3) people extraction + link mentions (if app.services.people is available)
    """
    try:
        # ensure prompt loaded
        try:
            await db.refresh(response, attribute_names=["prompt"])
        except Exception:
            pass

        # reload with tags to avoid lazy loads
        loaded = (await db.execute(
            select(ResponseModel)
            .options(selectinload(ResponseModel.tags))
            .where(ResponseModel.id == response.id)
        )).scalars().first() or response
        response = loaded

        text = (_text_for_tagging(response) or "").strip()
        if not text:
            logging.info("ℹ️ enrich_after_transcription: empty text; skip.")
            return

        # 1) auto-tag
        wc = len(text.split())
        suggestions = suggest_tags_rule_based(text, word_count=wc)
        added = 0
        for slug, _score in suggestions:
            tag = (await db.execute(select(Tag).where(Tag.slug == slug))).scalars().first()
            if not tag:
                tag = Tag(name=slug.split(":",1)[-1].replace("-", " ").title(), slug=slug)
                db.add(tag)
                try:
                    await db.flush()
                except IntegrityError:
                    await db.rollback()
                    tag = (await db.execute(select(Tag).where(Tag.slug == slug))).scalars().first()
            if tag and tag not in response.tags:
                response.tags.append(tag)
                added += 1
        logging.info(f"🏷️ Auto-tag added {added} tag(s) on response {response.id}")

        # 2) morph profile
        profile = (await db.execute(
            select(UserProfile).where(UserProfile.user_id == response.user_id)
        )).scalars().first()
        used = {slug for slug, _ in suggestions}
        if profile is not None:
            current = getattr(profile, "tag_weights", None) or {
                "tagWeights": {},
                "recentHistory": [],
                "targets": {},
                "knobs": {"epsilon": 0.15, "perDayLimit": 2, "minDifficulty": "light", "maxDifficulty": "deep"},
            }
            profile.tag_weights = morph_profile(current, used)

        # 3) people extraction (optional; no dependency on response.user object)
        if _PEOPLE_ENABLED:
            people_text = ((getattr(response, "response_text", "") or "").strip()
                        or (getattr(response, "transcription", "") or "").strip())
            if people_text:
                # Gather user's known aliases/display names to boost NER
                aliases: list[str] = []
                try:
                    people = (await db.execute(
                        select(Person).where(Person.owner_user_id == response.user_id)
                    )).scalars().all()
                    alias_rows: list[str] = []
                    if people:
                        pids = [p.id for p in people]
                        alias_rows = (await db.execute(
                            select(PersonAlias.alias).where(PersonAlias.person_id.in_(pids))
                        )).scalars().all()
                    for p in (people or []):
                        if p.display_name:
                            aliases.append(p.display_name)
                    aliases.extend([a for a in alias_rows or []])
                except Exception:
                    aliases = []

                name_spans = extract_name_spans(people_text, aliases=aliases)
                seen_person_ids = set()
                for surface, s, e in name_spans:
                    person = await resolve_person(db, response.user_id, surface)
                    role = role_hint_near(people_text, s, e)
                    await link_mention(
                        db, response.id, person,
                        alias_used=surface, start_char=s, end_char=e,
                        confidence=0.75, role_hint=role
                    )
                    if person.id not in seen_person_ids:
                        seen_person_ids.add(person.id)
        await db.commit()
    except Exception:
        logging.exception("❌ enrich_after_transcription failed")


# ---------------------------------------------------------------------------
# Orchestrators (update Response + ResponseSegment)
# ---------------------------------------------------------------------------
async def transcribe_and_update(response_id: int, media_relpath: str, user_id: Optional[int] = None) -> None:
    """
    Background-safe:
      - open a session
      - transcribe primary media
      - update Response.transcription
      - ensure/refresh ResponseSegment[0]
      - graylist unknown person names
      - run enrichment (auto-tag, morph, people)
    """
    async with async_session_maker() as db:
        resp: ResponseModel | None = (await db.execute(
            select(ResponseModel).where(ResponseModel.id == response_id)
        )).scalars().first()
        if not resp:
            return

        # resolve input path under uploads
        rel = media_relpath.lstrip("/").replace("\\", "/")
        if rel.startswith("uploads/"):
            rel = rel[len("uploads/"):]
        absolute_media = UPLOAD_DIR / rel

        text = await transcribe_file(absolute_media, db=db, user_id=user_id or resp.user_id)

        if text:
            resp.transcription = text

        # seed/update segment 0
        seg0 = (await db.execute(
            select(ResponseSegment).where(
                ResponseSegment.response_id == response_id,
                ResponseSegment.order_index == 0
            )
        )).scalars().first()
        if not seg0:
            seg0 = ResponseSegment(
                response_id=response_id,
                order_index=0,
                media_path=rel,
                media_mime=resp.primary_mime_type or None,
                transcript=text or "",
            )
            db.add(seg0)
        else:
            seg0.transcript = text or (seg0.transcript or "")

        # graylist unknown names (simple harvest)
        full_text = " ".join(s for s in [
            resp.response_text or "",
            resp.transcription or "",
            getattr(resp, "title", "") or "",
        ] if s).strip()

        if full_text and resp.user_id:
            # candidate mentions (prefer NER-based when available)
            try:
                if _PEOPLE_ENABLED:
                    mentions = [s for (s, _s, _e) in extract_name_spans(full_text)]  # type: ignore[name-defined]
                else:
                    mentions = _harvest_proper_nouns(full_text)
            except Exception:
                mentions = _harvest_proper_nouns(full_text)
            # Add conservative single-token names in context (e.g., "my friend Sarah")
            try:
                ctx_re = re.compile(r"\b(?:(?:my|our|mr\.|mrs\.|ms\.|dr\.|aunt|uncle|grandma|grandpa|grandmother|grandfather|cousin|friend)\s+)([A-Z][a-z'\-]{2,})\b", re.IGNORECASE)
                singles = [m.group(1) for m in ctx_re.finditer(full_text)]
                seenm = set(m.lower() for m in mentions)
                for s in singles:
                    if s.lower() not in seenm:
                        mentions.append(s)
                        seenm.add(s.lower())
            except Exception:
                pass
            # collect aliases for user
            aliases = []
            try:
                people = (await db.execute(
                    select(Person).where(Person.owner_user_id == resp.user_id)
                )).scalars().all()
                alias_rows = []
                if people:
                    pids = [p.id for p in people]
                    alias_rows = (await db.execute(
                        select(PersonAlias.alias).where(PersonAlias.person_id.in_(pids))
                    )).scalars().all()
                for p in (people or []):
                    if p.display_name:
                        aliases.append(p.display_name)
                aliases.extend([a for a in alias_rows or []])
            except Exception:
                pass

            alias_set = set((a or "").lower() for a in aliases if a)
            # Force-include any known aliases that appear in text (whole-word, case-insensitive)
            try:
                seen = set(m.lower() for m in mentions)
                for a in aliases:
                    aa = (a or "").strip()
                    if not aa:
                        continue
                    if re.search(rf"\b{re.escape(aa)}\b", full_text, flags=re.IGNORECASE):
                        if aa.lower() not in seen:
                            mentions.append(aa)
                            seen.add(aa.lower())
            except Exception:
                pass
            matched, unknown = [], []
            for m in mentions:
                ml = m.lower().strip()
                if ml in alias_set:
                    matched.append(m); continue
                if " " in ml and ml.split(" ",1)[0] in alias_set:
                    matched.append(m); continue
                unknown.append(m)

            # ensure person tags for matched; graylist unknown
            prof = (await db.execute(
                select(UserProfile).where(UserProfile.user_id == resp.user_id)
            )).scalars().first()
            if prof:
                for m in matched:
                    try:
                        await _ensure_tag(db, slug_person(m))
                    except Exception:
                        pass
                _append_graylist(prof, [slug_person(u) for u in unknown])

        await db.commit()

        # enrichment (safe-guarded)
        try:
            if (resp.transcription or "").strip() or (resp.response_text or "").strip():
                await enrich_after_transcription(db, resp)
                await db.commit()
        except Exception:
            pass


async def transcribe_segment_and_attach(
    response_id: int,
    media_relpath: str,
    order_index: int,
    user_id: Optional[int] = None,
) -> None:
    """
    For *additional recorded segments*:
      - transcribe file at media_relpath
      - create/update ResponseSegment at given order_index
      - DO NOT overwrite Response.transcription (leave full text to primary)
      - still run enrichment pass using the segment text appended to response_text for tagging
    """
    async with async_session_maker() as db:
        resp: ResponseModel | None = (await db.execute(
            select(ResponseModel).where(ResponseModel.id == response_id)
        )).scalars().first()
        if not resp:
            return

        rel = media_relpath.lstrip("/").replace("\\", "/")
        if rel.startswith("uploads/"):
            rel = rel[len("uploads/"):]
        absolute_media = UPLOAD_DIR / rel

        text = await transcribe_file(absolute_media, db=db, user_id=user_id or resp.user_id)

        seg = (await db.execute(
            select(ResponseSegment).where(
                ResponseSegment.response_id == response_id,
                ResponseSegment.order_index == order_index
            )
        )).scalars().first()
        if not seg:
            seg = ResponseSegment(
                response_id=response_id,
                order_index=order_index,
                media_path=rel,
                media_mime=None,
                transcript=text or "",
            )
            db.add(seg)
        else:
            seg.transcript = text or (seg.transcript or "")

        await db.commit()

        # Light enrichment based on segment text (don’t clobber resp.transcription)
        try:
            # temp assemble a view for tagging
            pseudo = ResponseModel(
                id=resp.id,
                user_id=resp.user_id,
                prompt=resp.prompt,
                title=resp.title,
                response_text=(resp.response_text or "") + "\n" + (text or ""),
                transcription=(resp.transcription or ""),
            )
            await enrich_after_transcription(db, pseudo)  # uses only getters
            await db.commit()
        except Exception:
            pass
