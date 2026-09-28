"""CPU-only tests for structured JSON parse/repair helpers."""

from __future__ import annotations

from docintel.structured import parse_structured_output, strip_markdown_fences


def test_strip_markdown_fences_json_fence():
    raw = '```json\n{"answer": "ok"}\n```'
    assert strip_markdown_fences(raw) == '{"answer": "ok"}'


def test_strip_markdown_fences_plain_fence():
    raw = '```\n[1, 2]\n```'
    assert strip_markdown_fences(raw) == "[1, 2]"


def test_strip_markdown_fences_unfenced():
    assert strip_markdown_fences('{"a": 1}') == '{"a": 1}'


def test_parse_structured_output_valid_object():
    raw = '{"answer": "42", "key_values": [], "tables": [], "citations": ["page 1"]}'
    ok, display, parsed = parse_structured_output(raw)
    assert ok is True
    assert parsed["answer"] == "42"
    assert parsed["citations"] == ["page 1"]
    assert '"answer": "42"' in display


def test_parse_structured_output_strips_fences_without_inventing_fields():
    raw = '```json\n{"answer": "yes"}\n```'
    ok, display, parsed = parse_structured_output(raw)
    assert ok is True
    assert parsed == {"answer": "yes"}
    assert "key_values" not in parsed
    assert '"answer": "yes"' in display


def test_parse_structured_output_invalid_shows_error_and_raw():
    raw = "This is not JSON at all."
    ok, display, parsed = parse_structured_output(raw)
    assert ok is False
    assert parsed is None
    assert "Structured JSON parse failed" in display
    assert "This is not JSON at all." in display


def test_parse_structured_output_empty():
    ok, display, parsed = parse_structured_output("   ")
    assert ok is False
    assert parsed is None
    assert "empty output" in display
