from __future__ import annotations

import csv
import io
import json
import tempfile
from typing import Any, Optional, Tuple


def strip_markdown_fences(text: str) -> str:
    """Remove a wrapping ``` / ```json fence if present. Do not rewrite JSON."""
    raw = (text or "").strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    first = lines[0].strip().lower()
    if first in ("```", "```json"):
        lines = lines[1:]
    elif first.startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_structured_output(text: str) -> Tuple[bool, str, Optional[Any]]:
    """Parse model output as JSON after light fence stripping.

    Returns (ok, display_text, parsed_value).
    On failure, display_text is a clear error plus the raw model text.
    Does not invent or coerce missing schema fields.
    """
    raw = text if text is not None else ""
    candidate = strip_markdown_fences(raw)
    if not candidate:
        display = (
            "Structured JSON parse failed: model returned empty output.\n\n"
            "Raw model output:\n(empty)"
        )
        return False, display, None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        display = (
            f"Structured JSON parse failed: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno}).\n\n"
            f"Raw model output:\n{raw.strip() or '(empty)'}"
        )
        return False, display, None
    pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
    return True, pretty, parsed


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def tables_to_csv(parsed: Any) -> Optional[str]:
    """Serialize `tables` from parsed structured JSON to CSV text.

    Returns None when there is no tables list (or it is empty).
    Does not invent headers or cells.
    """
    if not isinstance(parsed, dict):
        return None
    tables = parsed.get("tables")
    if not isinstance(tables, list) or not tables:
        return None
    buf = io.StringIO()
    writer = csv.writer(buf)
    wrote_any = False
    for idx, table in enumerate(tables):
        if not isinstance(table, dict):
            continue
        title = _stringify(table.get("title", ""))
        page = _stringify(table.get("page", ""))
        headers = table.get("headers")
        rows = table.get("rows")
        if not isinstance(headers, list):
            headers = []
        if not isinstance(rows, list):
            rows = []
        if idx > 0:
            writer.writerow([])
        meta = []
        if title:
            meta.append(title)
        if page:
            meta.append(page)
        if meta:
            writer.writerow(meta)
        if headers:
            writer.writerow([_stringify(h) for h in headers])
            wrote_any = True
        for row in rows:
            if isinstance(row, list):
                writer.writerow([_stringify(c) for c in row])
                wrote_any = True
            elif row is not None:
                writer.writerow([_stringify(row)])
                wrote_any = True
    text = buf.getvalue()
    return text if wrote_any or text.strip() else None


def key_values_to_json(parsed: Any) -> Optional[str]:
    """Serialize `key_values` from parsed structured JSON.

    Returns None when the field is missing or not a list.
    Does not invent keys.
    """
    if not isinstance(parsed, dict):
        return None
    if "key_values" not in parsed:
        return None
    key_values = parsed.get("key_values")
    if not isinstance(key_values, list):
        return None
    return json.dumps(key_values, indent=2, ensure_ascii=False)


def _write_temp(suffix: str, content: str) -> str:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=suffix,
        prefix="docintel-",
        delete=False,
    )
    with handle:
        handle.write(content)
    return handle.name


def build_structured_exports(
    structured_output: bool,
    parse_ok: bool,
    parsed: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Return (csv_path, json_path) for Gradio download components.

    No-op (None, None) when structured mode is off or parse failed.
    """
    if not structured_output or not parse_ok or parsed is None:
        return None, None
    csv_text = tables_to_csv(parsed)
    kv_text = key_values_to_json(parsed)
    csv_path = _write_temp(".csv", csv_text) if csv_text is not None else None
    json_path = _write_temp(".json", kv_text) if kv_text is not None else None
    return csv_path, json_path
