from __future__ import annotations

import re
from typing import List, Optional, Tuple

from PIL import Image

from docintel.constants import PAGE_REF_RE, STOPWORDS, TOKEN_RE, TOP_K_PAGES


def tokenize_query(text: str) -> set:
    """Lowercase alphanumeric tokens with stopwords removed."""
    tokens = set(TOKEN_RE.findall((text or "").lower()))
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def page_relevance_score(source_label: str, question: str, page_index: int) -> float:
    """Cheap relevance: keyword overlap with the source label plus explicit page refs.\n\n    Images have no OCR yet, so the score uses the question text and the\n    human-readable source label (filename + page number).\n    """
    q = question or ""
    q_tokens = tokenize_query(q)
    label_tokens = tokenize_query(source_label.replace("\u2014", " ").replace("-", " ").replace(".", " "))

    overlap = len(q_tokens & label_tokens)
    score = float(overlap)

    mentioned_pages = {int(n) for n in PAGE_REF_RE.findall(q)}
    page_nums = {int(n) for n in re.findall(r"\bpage\s*(\d+)\b", source_label, flags=re.IGNORECASE)}
    if mentioned_pages and page_nums & mentioned_pages:
        score += 5.0

    score += max(0.0, 0.05 * (1.0 / (page_index + 1)))
    return score


def rank_pages(
    page_images: List[Image.Image],
    source_info: List[str],
    question: str,
    top_k: int = TOP_K_PAGES,
) -> Tuple[List[Image.Image], List[str], List[float], List[str]]:
    """Split pages into top-k for the VLM and the remainder not sent.\n\n    Returns (selected_images, selected_labels, selected_scores, omitted_labels).\n    Selected lists are ordered by descending relevance score.\n    """
    if not page_images:
        return [], [], [], []

    scored = []
    for i, (img, src) in enumerate(zip(page_images, source_info)):
        scored.append((page_relevance_score(src, question, i), i, img, src))

    scored.sort(key=lambda row: (-row[0], row[1]))
    k = max(1, min(int(top_k), len(scored)))
    selected = scored[:k]
    omitted = scored[k:]
    images = [row[2] for row in selected]
    labels = [row[3] for row in selected]
    scores = [row[0] for row in selected]
    omitted_labels = [row[3] for row in omitted]
    return images, labels, scores, omitted_labels


def label_sent_page(src: str, index: int, total: int, score: Optional[float] = None) -> str:
    """Caption for a page that was actually sent to the VLM."""
    base = f"Sent to model \u00b7 {index}/{total} \u00b7 {src}"
    if score is None:
        return base
    return f"{base} \u00b7 score {score:.2f}"


def build_gallery_and_sources(
    page_images: List[Image.Image],
    source_info: List[str],
    scores: Optional[List[float]] = None,
    omitted_labels: Optional[List[str]] = None,
    total_extracted: Optional[int] = None,
    truncation_note: Optional[str] = None,
) -> Tuple[List[Tuple[Image.Image, str]], str]:
    """Gallery of pages forwarded to the model, plus a ranking-aware source list."""
    sent = len(source_info)
    extracted = total_extracted if total_extracted is not None else sent
    gallery = []
    for i, (img, src) in enumerate(zip(page_images, source_info)):
        score = scores[i] if scores and i < len(scores) else None
        gallery.append((img, label_sent_page(src, i + 1, sent, score)))

    header = f"Pages sent to the model ({sent} of {extracted} extracted):"
    lines = [header]
    for i, src in enumerate(source_info):
        if scores and i < len(scores):
            lines.append(f"- {src} (relevance {scores[i]:.2f})")
        else:
            lines.append(f"- {src}")
    if omitted_labels:
        lines.append("")
        lines.append(f"Not sent to the model ({len(omitted_labels)}):")
        lines.extend([f"- {src}" for src in omitted_labels])
    if truncation_note:
        lines.append("")
        lines.append(truncation_note)
    return gallery, "\n".join(lines)
