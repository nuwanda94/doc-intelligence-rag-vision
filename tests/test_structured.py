"""CPU-only tests for structured JSON parse/repair helpers."""

from __future__ import annotations

import json
from pathlib import Path

from docintel.structured import (
    build_structured_exports,
    key_values_to_json,
    parse_structured_output,
    strip_markdown_fences,
    tables_to_csv,
)


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


def test_tables_to_csv_writes_headers_and_rows():
    parsed = {
        "tables": [
            {
                "title": "Revenue",
                "page": "page 2",
                "headers": ["Year", "Amount"],
                "rows": [["2023", "10"], ["2024", "12"]],
            }
        ]
    }
    csv_text = tables_to_csv(parsed)
    assert csv_text is not None
    assert "Revenue" in csv_text
    assert "page 2" in csv_text
    assert "Year" in csv_text
    assert "2024" in csv_text


def test_tables_to_csv_missing_or_empty_is_none():
    assert tables_to_csv({"answer": "x"}) is None
    assert tables_to_csv({"tables": []}) is None
    assert tables_to_csv([1, 2]) is None


def test_key_values_to_json_preserves_list():
    parsed = {"key_values": [{"key": "total", "value": "12", "page": "p1"}]}
    text = key_values_to_json(parsed)
    assert text is not None
    assert json.loads(text) == parsed["key_values"]


def test_key_values_to_json_missing_is_none():
    assert key_values_to_json({"answer": "x"}) is None
    assert key_values_to_json({"key_values": {"k": "v"}}) is None


def test_build_structured_exports_noop_when_off_or_failed():
    parsed = {
        "key_values": [{"key": "a", "value": "1"}],
        "tables": [{"headers": ["h"], "rows": [["1"]]}],
    }
    assert build_structured_exports(False, True, parsed) == (None, None)
    assert build_structured_exports(True, False, parsed) == (None, None)
    assert build_structured_exports(True, True, None) == (None, None)


def test_build_structured_exports_writes_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parsed = {
        "key_values": [{"key": "total", "value": "9"}],
        "tables": [{"title": "T", "headers": ["A"], "rows": [["1"]]}],
    }
    csv_path, json_path = build_structured_exports(True, True, parsed)
    assert csv_path is not None and json_path is not None
    assert Path(csv_path).read_text(encoding="utf-8").startswith("T")
    assert json.loads(Path(json_path).read_text(encoding="utf-8")) == parsed["key_values"]
