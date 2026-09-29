from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image

from docintel.constants import (
    COMPARE_SYSTEM_PROMPT,
    MAX_HISTORY_TURNS,
    STRUCTURED_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)


def message_text(content: Any) -> str:
    """Normalize a Gradio chat message content field to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def build_vlm_messages(
    history: List[Dict[str, Any]],
    page_images: List[Image.Image],
    source_info: List[str],
    question: str,
    structured: bool = False,
    compare: bool = False,
) -> List[Dict[str, Any]]:
    """System + prior text turns + current user turn with ranked page images."""
    if structured:
        system = STRUCTURED_SYSTEM_PROMPT
    elif compare:
        system = COMPARE_SYSTEM_PROMPT
    else:
        system = SYSTEM_PROMPT
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
    ]

    prior = history or []
    if MAX_HISTORY_TURNS > 0 and len(prior) > MAX_HISTORY_TURNS * 2:
        prior = prior[-(MAX_HISTORY_TURNS * 2):]

    for turn in prior:
        role = turn.get("role")
        text = message_text(turn.get("content")).strip()
        if role not in ("user", "assistant") or not text:
            continue
        messages.append({"role": role, "content": text})

    content: List[Dict[str, Any]] = []
    for img, src in zip(page_images, source_info):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[Source: {src}]"})
    user_text = question
    if compare:
        user_text = (
            question
            + "\n\nCompare the uploaded documents. Contrast them on this question "
            "and cite every claim with the full source label (filename and page)."
        )
    if structured:
        user_text = (
            user_text
            + "\n\nReturn only valid JSON matching the required schema "
            "(answer, key_values, tables, citations)."
        )
    content.append({"type": "text", "text": user_text})
    messages.append({"role": "user", "content": content})
    return messages
