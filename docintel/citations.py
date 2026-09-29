from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

from docintel.constants import PAGE_REF_RE

CITED_BADGE = "Cited"
CITED_BORDER_PX = 8
CITED_BORDER_COLOR = (37, 99, 235)  # blue, matches Gradio Soft primary


def extract_cited_page_numbers(text: str) -> Set[int]:
    """Page numbers mentioned as 'page N' in the answer or citations."""
    return {int(n) for n in PAGE_REF_RE.findall(text or "")}


def page_number_from_label(label: str) -> Optional[int]:
    nums = [int(n) for n in PAGE_REF_RE.findall(label or "")]
    return nums[-1] if nums else None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("\u2014", "—").strip())


def source_is_cited(label: str, answer: str) -> bool:
    """True when the answer cites this gallery source label.

    Matches:
    - the full source label as a substring (case-insensitive)
    - an explicit 'page N' mention that matches the label's page number
    """
    if not label or not answer:
        return False
    hay = _normalize(answer)
    needle = _normalize(label)
    if needle and needle in hay:
        return True
    page_n = page_number_from_label(label)
    if page_n is not None and page_n in extract_cited_page_numbers(answer):
        return True
    return False


def cited_source_indices(source_info: Sequence[str], answer: str) -> List[int]:
    return [i for i, src in enumerate(source_info) if source_is_cited(src, answer)]


def _add_cited_badge(caption: str) -> str:
    cap = caption or ""
    if cap.startswith(f"{CITED_BADGE} "):
        return cap
    return f"{CITED_BADGE} · {cap}" if cap else CITED_BADGE


def emphasize_cited_image(image: Any) -> Any:
    """Draw a simple colored border. No-op if the object is not a PIL image."""
    try:
        from PIL import Image, ImageOps
    except Exception:
        return image
    if image is None or not isinstance(image, Image.Image):
        return image
    try:
        rgb = image.convert("RGB")
        return ImageOps.expand(rgb, border=CITED_BORDER_PX, fill=CITED_BORDER_COLOR)
    except Exception:
        return image


def apply_citation_highlights(
    gallery: Optional[List[Tuple[Any, str]]],
    source_info: Sequence[str],
    answer: str,
    sources_text: str = "",
) -> Tuple[Optional[List[Tuple[Any, str]]], str]:
    """Badge + border gallery items whose labels are cited in the answer."""
    if not gallery:
        return gallery, sources_text
    cited = set(cited_source_indices(source_info, answer))
    if not cited:
        return gallery, sources_text

    highlighted: List[Tuple[Any, str]] = []
    for i, item in enumerate(gallery):
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            img, cap = item[0], item[1]
        else:
            img, cap = item, ""
        if i in cited:
            highlighted.append((emphasize_cited_image(img), _add_cited_badge(str(cap))))
        else:
            highlighted.append((img, str(cap)))

    cited_labels = [source_info[i] for i in sorted(cited) if i < len(source_info)]
    extra = "Cited in answer: " + "; ".join(cited_labels)
    text = sources_text or ""
    if extra not in text:
        text = (text + ("\n\n" if text else "") + extra).strip()
    return highlighted, text
