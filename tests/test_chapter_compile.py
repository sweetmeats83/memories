"""Tests for the text helpers used by the chapter compilation pipeline."""
from types import SimpleNamespace

from app.services.chapter_compile import _format_responses, _strip_html


def test_strip_html_converts_blocks_to_newlines_and_decodes_entities():
    html = "<p>Mom &amp; Dad</p><p>We lived on Elm&nbsp;St.<br>It was small.</p>"
    assert _strip_html(html) == "Mom & Dad\nWe lived on Elm St.\nIt was small."


def test_strip_html_collapses_blank_lines_and_handles_empty():
    assert _strip_html("<p>a</p><p></p><p></p><p></p><p>b</p>") == "a\n\nb"
    assert _strip_html("") == ""
    assert _strip_html(None) == ""


def _pair(prompt, *, text=None, transcription=None, title=None):
    return (
        SimpleNamespace(text=prompt),
        SimpleNamespace(response_text=text, transcription=transcription, title=title),
    )


def test_format_responses_numbers_pairs_and_prefers_typed_text():
    pairs = [
        _pair("Where were you born?", text="In Dayton.", transcription="<p>ignored</p>", title="Origins"),
        _pair("Favorite food?", transcription="<p>Pot roast.</p>"),
    ]
    out = _format_responses(pairs)
    assert "[1] PROMPT: Where were you born?" in out
    assert "TITLE: Origins" in out
    assert "In Dayton." in out and "ignored" not in out
    assert "[2] PROMPT: Favorite food?" in out
    assert "Pot roast." in out


def test_format_responses_skips_empty_but_keeps_original_numbering():
    pairs = [_pair("Skipped?", text="", transcription=None), _pair("Kept?", text="Yes.")]
    out = _format_responses(pairs)
    assert "Skipped?" not in out
    # Numbers must match positions in the full list, since the pipeline maps them back to response ids.
    assert "[2] PROMPT: Kept?" in out
