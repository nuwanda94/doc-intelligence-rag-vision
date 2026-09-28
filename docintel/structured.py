from __future__ import annotations

import json
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
