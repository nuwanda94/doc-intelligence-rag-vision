from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from docintel.constants import (
    DEFAULT_RANKING_MODE,
    OCR_CANDIDATE_LIMIT,
    OCR_MAX_SIDE,
    OCR_TEXT_WEIGHT,
    PAGE_REF_RE,
    RANKING_MODE_AUTO,
    RANKING_MODE_LABEL,
    RANKING_MODE_OCR,
    RANKING_MODE_UI,
    STOPWORDS,
    TOKEN_RE,
    TOP_K_PAGES,
)
from docintel.ingest import extract_native_pdf_page_text


def tokenize_query(text: str) -> set:
    """Lowercase alphanumeric tokens with stopwords removed."""
    tokens = set(TOKEN_RE.findall((text or "").lower()))
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def normalize_ranking_mode(value: Optional[str]) -> str:
    """Map a UI label or short code to label | ocr | auto."""
    raw = (value or DEFAULT_RANKING_MODE).strip().lower()
    if raw in {RANKING_MODE_LABEL, RANKING_MODE_OCR, RANKING_MODE_AUTO}:
        return raw
    if raw.startswith("label"):
        return RANKING_MODE_LABEL
    if raw.startswith("ocr"):
        return RANKING_MODE_OCR
    if raw.startswith("auto"):
        return RANKING_MODE_AUTO
    return DEFAULT_RANKING_MODE


def page_relevance_score(source_label: str, question: str, page_index: int) -> float:
    """Cheap relevance: keyword overlap with the source label plus explicit page refs.\n\n    Images have no OCR yet, so the score uses the question text and the
    human-readable source label (filename + page number).
    """
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


def page_text_relevance_score(
    source_label: str,
    page_text: str,
    question: str,
    page_index: int,
) -> float:
    """Label score plus keyword overlap against cheap OCR/extracted page text."""
    score = page_relevance_score(source_label, question, page_index)
    q_tokens = tokenize_query(question)
    text_tokens = tokenize_query(page_text or "")
    if q_tokens and text_tokens:
        score += OCR_TEXT_WEIGHT * float(len(q_tokens & text_tokens))
    return score


def labels_are_informative(source_info: List[str], question: str) -> bool:
    """True when filename/page labels already overlap the question usefully."""
    q_tokens = tokenize_query(question)
    if not q_tokens or not source_info:
        return False
    for src in source_info:
        label_tokens = tokenize_query(src.replace("\u2014", " ").replace("-", " ").replace(".", " "))
        if q_tokens & label_tokens:
            return True
    return False


def _downscale_for_ocr(image: Image.Image) -> Image.Image:
    img = image
    try:
        img = image.convert("RGB")
    except Exception:
        pass
    try:
        w, h = img.size
    except Exception:
        return img
    longest = max(w, h)
    if longest <= OCR_MAX_SIDE:
        return img
    scale = OCR_MAX_SIDE / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
    try:
        return img.resize(new_size, resample)
    except Exception:
        return img


def ocr_page_text(image: Image.Image) -> str:
    """Best-effort OCR on a downscaled page. Returns empty string if unavailable."""
    try:
        import pytesseract
    except Exception:
        return ""
    try:
        small = _downscale_for_ocr(image)
        text = pytesseract.image_to_string(small) or ""
        return text.strip()
    except Exception:
        return ""


def _resolve_use_ocr(ranking_mode: str, source_info: List[str], question: str) -> bool:
    mode = normalize_ranking_mode(ranking_mode)
    if mode == RANKING_MODE_LABEL:
        return False
    if mode == RANKING_MODE_OCR:
        return True
    # auto: OCR only when labels carry no useful question overlap
    return not labels_are_informative(source_info, question)


def _origin_for(page_origins: Optional[List[Dict[str, Any]]], idx: int) -> Optional[Dict[str, Any]]:
    if not page_origins or idx < 0 or idx >= len(page_origins):
        return None
    origin = page_origins[idx]
    return origin if isinstance(origin, dict) else None


def page_text_for_ranking(
    idx: int,
    image: Image.Image,
    page_origins: Optional[List[Dict[str, Any]]] = None,
    ocr_fn: Optional[Callable[[Image.Image], str]] = None,
    native_text_fn: Optional[Callable[[str, int], str]] = None,
    text_cache: Optional[Dict[int, str]] = None,
) -> str:
    """Native PDF text layer first; Tesseract only if that layer is empty."""
    cache = text_cache if text_cache is not None else {}
    if idx in cache:
        return cache.get(idx) or ""

    text = ""
    origin = _origin_for(page_origins, idx)
    if origin and origin.get("kind") == "pdf" and origin.get("path"):
        extract = native_text_fn or extract_native_pdf_page_text
        try:
            page_no = int(origin.get("page") or 0)
        except (TypeError, ValueError):
            page_no = 0
        if page_no >= 1:
            try:
                text = extract(str(origin["path"]), page_no) or ""
            except Exception:
                text = ""

    if not (text or "").strip():
        extract_ocr = ocr_fn or ocr_page_text
        try:
            text = extract_ocr(image) or ""
        except Exception:
            text = ""

    text = (text or "").strip()
    cache[idx] = text
    return text


def rank_pages(
    page_images: List[Image.Image],
    source_info: List[str],
    question: str,
    top_k: int = TOP_K_PAGES,
    ranking_mode: str = DEFAULT_RANKING_MODE,
    ocr_fn: Optional[Callable[[Image.Image], str]] = None,
    ocr_cache: Optional[Dict[int, str]] = None,
    page_origins: Optional[List[Dict[str, Any]]] = None,
    native_text_fn: Optional[Callable[[str, int], str]] = None,
) -> Tuple[List[Image.Image], List[str], List[float], List[str], str]:
    """Split pages into top-k for the VLM and the remainder not sent.\n\n    Returns (selected_images, selected_labels, selected_scores, omitted_labels, mode_note).
    Selected lists are ordered by descending relevance score.
    Text ranking, when used, runs only on a bounded candidate set after a cheap
    label pass. Digital PDFs use the native text layer first; OCR is fallback.
    """
    if not page_images:
        return [], [], [], [], RANKING_MODE_LABEL

    label_scored = []
    for i, (img, src) in enumerate(zip(page_images, source_info)):
        label_scored.append((page_relevance_score(src, question, i), i, img, src))
    label_scored.sort(key=lambda row: (-row[0], row[1]))

    use_ocr = _resolve_use_ocr(ranking_mode, source_info, question)
    mode_used = RANKING_MODE_LABEL
    scored = label_scored

    if use_ocr:
        cache = ocr_cache if ocr_cache is not None else {}
        candidate_n = max(1, min(int(OCR_CANDIDATE_LIMIT), len(label_scored)))
        candidates = label_scored[:candidate_n]
        rest = label_scored[candidate_n:]
        rescored = []
        any_text = False
        for _ls, idx, img, src in candidates:
            text = page_text_for_ranking(
                idx,
                img,
                page_origins=page_origins,
                ocr_fn=ocr_fn,
                native_text_fn=native_text_fn,
                text_cache=cache,
            )
            if text.strip():
                any_text = True
            rescored.append((page_text_relevance_score(src, text, question, idx), idx, img, src))
        if any_text:
            mode_used = RANKING_MODE_OCR
            scored = rescored + rest
            scored.sort(key=lambda row: (-row[0], row[1]))
        else:
            mode_used = RANKING_MODE_LABEL
            scored = label_scored

    k = max(1, min(int(top_k), len(scored)))
    selected = scored[:k]
    omitted = scored[k:]
    images = [row[2] for row in selected]
    labels = [row[3] for row in selected]
    scores = [row[0] for row in selected]
    omitted_labels = [row[3] for row in omitted]
    return images, labels, scores, omitted_labels, mode_used


def label_sent_page(src: str, index: int, total: int, score: Optional[float] = None) -> str:
    """Caption for a page that was actually sent to the VLM."""
    base = f"Sent to model \u00b7 {index}/{total} \u00b7 {src}"
    if score is None:
        return base
    return f"{base} \u00b7 score {score:.2f}"


def _mode_used_line(ranking_mode: Optional[str]) -> str:
    if ranking_mode == RANKING_MODE_OCR:
        return "Ranking used: page text overlap (PDF text layer, then OCR; label score as fallback)."
    if ranking_mode:
        return "Ranking used: keyword overlap on file/page labels."
    return ""


def build_gallery_and_sources(
    page_images: List[Image.Image],
    source_info: List[str],
    scores: Optional[List[float]] = None,
    omitted_labels: Optional[List[str]] = None,
    total_extracted: Optional[int] = None,
    truncation_note: Optional[str] = None,
    ranking_mode: Optional[str] = None,
    requested_mode: Optional[str] = None,
    top_k: Optional[int] = None,
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
    requested = normalize_ranking_mode(requested_mode) if requested_mode else None
    if requested:
        requested_label = RANKING_MODE_UI.get(requested, requested)
        lines.append(f"Ranking requested: {requested_label}.")
    used_line = _mode_used_line(ranking_mode)
    if used_line:
        lines.append(used_line)
    if top_k is not None:
        lines.append(f"Top-k pages sent: {sent} (budget {int(top_k)}).")
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
