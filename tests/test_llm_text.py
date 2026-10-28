"""Tests for transcript cleanup: output sanitising and the Ollama call (mocked)."""
import asyncio

import pytest

from app import llm_client
from app.llm_client import OllamaError, _sanitize_llm_text, _strip_thinking, polish_text


def test_strip_thinking_removes_reasoning_block():
    raw = "<think>let me consider\nthe options</think>\nWe moved to Ohio in 1962."
    assert _strip_thinking(raw) == "We moved to Ohio in 1962."


def test_sanitize_drops_preface_line():
    out = "Here is the cleaned transcript:\n\nWe moved to Ohio in 1962."
    assert _sanitize_llm_text(out) == "We moved to Ohio in 1962."


def test_sanitize_unwraps_code_fence_and_quotes():
    assert _sanitize_llm_text("```\nHello there.\n```") == "Hello there."
    assert _sanitize_llm_text('"We had a dog named Rex."') == "We had a dog named Rex."


def test_sanitize_keeps_real_first_line():
    # A long first line that merely contains "here is" must not be treated as boilerplate.
    text = (
        "Here is what I remember about the summer we spent at the lake with my cousins "
        "and their dog, Rex, every single July until I was twelve."
    )
    assert _sanitize_llm_text(text) == text


def test_sanitize_empty():
    assert _sanitize_llm_text("") == ""


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _fake_client(body=None, error=None, sent=None):
    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            if sent is not None:
                sent.update(url=url, payload=json)
            if error:
                raise error
            return _FakeResponse(body)

    return _Client


def test_polish_text_prompt_preserves_voice_and_cleans_output(monkeypatch):
    sent = {}
    body = {"response": "<think>hmm</think>Cleaned transcript:\nWe moved to Ohio."}
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _fake_client(body, sent=sent))

    result = asyncio.run(polish_text("um we moved to ohio"))

    assert result == "We moved to Ohio."
    prompt = sent["payload"]["prompt"]
    assert "Keep the speaker's voice and meaning" in prompt
    assert "Do not add facts" in prompt
    assert "um we moved to ohio" in prompt
    assert sent["payload"]["options"]["temperature"] <= 0.3
    assert sent["payload"]["stream"] is False


def test_polish_text_paragraph_style_extends_prompt(monkeypatch):
    sent = {}
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _fake_client({"response": "ok"}, sent=sent))
    asyncio.run(polish_text("text", style="paragraphs"))
    assert "short readable paragraphs" in sent["payload"]["prompt"]


def test_polish_text_blank_input_skips_llm(monkeypatch):
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _fake_client(error=AssertionError("should not call")))
    assert asyncio.run(polish_text("   ")) == "   "


def test_polish_text_wraps_transport_errors(monkeypatch):
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _fake_client(error=RuntimeError("down")))
    with pytest.raises(OllamaError):
        asyncio.run(polish_text("hello"))


def test_polish_text_empty_response_is_an_error(monkeypatch):
    monkeypatch.setattr(llm_client.httpx, "AsyncClient", _fake_client({"response": "<think>x</think>"}))
    with pytest.raises(OllamaError):
        asyncio.run(polish_text("hello"))


# --- fidelity guard ---------------------------------------------------------

ORIGINAL = (
    "so um we moved to ohio in nineteen sixty two and my dad got a job at the steel mill "
    "and mom stayed home with me and my brother"
)


def test_fidelity_accepts_light_cleanup():
    cleaned = (
        "We moved to Ohio in 1962, and my dad got a job at the steel mill. "
        "Mom stayed home with me and my brother."
    )
    assert llm_client.check_fidelity(ORIGINAL, cleaned) == []


def test_fidelity_flags_summary():
    problems = llm_client.check_fidelity(ORIGINAL, "The family relocated to Ohio.")
    assert problems


def test_fidelity_flags_embellishment():
    cleaned = ORIGINAL + " and we visited Aunt Margaret in Cleveland every Christmas with Uncle Walter"
    problems = llm_client.check_fidelity(ORIGINAL, cleaned)
    assert any("longer" in p or "names" in p for p in problems)


def test_fidelity_flags_invented_names_and_numbers():
    original = "we lived on a farm and my dad grew corn and we had a dog and some chickens out back"
    cleaned = "We lived on a farm in 1950, and my dad, Walter, grew corn. We had a dog, Rex, and some chickens out back."
    problems = llm_client.check_fidelity(original, cleaned)
    assert any("numbers" in p for p in problems)
    assert any("names" in p for p in problems)


def test_fidelity_skips_very_short_text():
    assert llm_client.check_fidelity("um hello there", "Completely different.") == []


def test_polish_text_rejects_unfaithful_output(monkeypatch):
    monkeypatch.setattr(
        llm_client.httpx, "AsyncClient", _fake_client({"response": "The family relocated to Ohio."})
    )
    with pytest.raises(llm_client.UnfaithfulOutputError):
        asyncio.run(polish_text(ORIGINAL))
