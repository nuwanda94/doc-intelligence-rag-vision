from __future__ import annotations

import hashlib
import os
from typing import List, Optional, Tuple

import gradio as gr
from pdf2image import convert_from_path
from PIL import Image

from docintel.constants import (
    ALLOWED_EXTENSIONS,
    DPI_HIGH,
    DPI_LOW,
    DPI_PAGE_THRESHOLD,
    MAX_FILE_SIZE_BYTES,
    MAX_IMAGE_SIDE,
    MAX_PAGES_DEFAULT,
)

_HASH_CHUNK_BYTES = 1024 * 1024


def prepare_image(img: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Convert to RGB and downscale so the longest side is at most max_side."""
    img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def adaptive_dpi(max_pages: int) -> int:
    """Choose PDF rasterization DPI from how many pages will be converted.\n\n    Higher page counts use a lower DPI to limit memory and conversion time
    on the free tier, while short documents keep higher detail.
    """
    if max_pages > DPI_PAGE_THRESHOLD:
        return DPI_LOW
    return DPI_HIGH


def file_path(file_obj) -> str:
    path = file_obj.name if hasattr(file_obj, "name") else file_obj
    if not path or not isinstance(path, str):
        raise gr.Error("Could not read an uploaded file. Please try uploading again.")
    return path


def file_content_key(path: str) -> str:
    """Stable cache key from file size and SHA-256 of bytes (path-independent)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return f"missing:{path}"
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        return f"unreadable:{size}:{path}"
    return f"{size}:{hasher.hexdigest()}"


def file_signature(files: Optional[List]) -> Tuple[str, ...]:
    """Content-based signature so identical bytes reuse the page cache across paths."""
    if not files:
        return ()
    return tuple(file_content_key(file_path(f)) for f in files)


def validate_uploads(files: Optional[List], question: str) -> List[str]:
    """Raise clear Gradio errors for empty, unsupported, empty-byte, or oversized uploads."""
    if not files:
        raise gr.Error("Please upload at least one PDF or image file.")

    if question is None or not str(question).strip():
        raise gr.Error("Please enter a question about the uploaded document.")

    validated_paths = []
    oversized = []
    unsupported = []
    empty_files = []
    missing = []

    for f in files:
        path = file_path(f)
        name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()

        if not os.path.exists(path):
            missing.append(name or path)
            continue

        try:
            size = os.path.getsize(path)
        except OSError:
            missing.append(name or path)
            continue

        if size == 0:
            empty_files.append(name)
            continue

        if size > MAX_FILE_SIZE_BYTES:
            oversized.append(f"{name} ({size / (1024 * 1024):.1f} MB)")
            continue

        if ext not in ALLOWED_EXTENSIONS:
            unsupported.append(name or "(no extension)")
            continue

        validated_paths.append(path)

    if missing:
        raise gr.Error(
            "Could not read the following upload(s): "
            + ", ".join(missing)
            + ". Please re-upload the file(s)."
        )
    if empty_files:
        raise gr.Error(
            "The following file(s) are empty: "
            + ", ".join(empty_files)
            + ". Upload a non-empty PDF or image."
        )
    if oversized:
        limit_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise gr.Error(
            f"File(s) exceed the {limit_mb:.0f} MB limit: "
            + ", ".join(oversized)
            + ". Compress the file or upload a smaller document."
        )
    if unsupported:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise gr.Error(
            "Unsupported file type(s): "
            + ", ".join(unsupported)
            + f". Allowed types: {allowed}."
        )
    if not validated_paths:
        raise gr.Error("No valid PDF or image files found in the upload.")

    return validated_paths


def pdf_to_images(pdf_path: str, max_pages: int = MAX_PAGES_DEFAULT, dpi: Optional[int] = None) -> List[Image.Image]:
    """Convert PDF pages to images (limited for free tier).\n\n    DPI is chosen adaptively from max_pages when not provided:
    150 for short docs (≤ DPI_PAGE_THRESHOLD pages), 120 for longer ones.
    """
    if dpi is None:
        dpi = adaptive_dpi(max_pages)
    try:
        images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=max_pages)
    except Exception as e:
        raise gr.Error(f"Failed to process PDF '{os.path.basename(pdf_path)}': {str(e)}")

    if not images:
        raise gr.Error(
            f"PDF '{os.path.basename(pdf_path)}' has no readable pages. "
            "Upload a PDF that contains at least one page."
        )
    return [prepare_image(img) for img in images]


def load_pages_from_paths(
    paths: List[str], max_pages: int
) -> Tuple[List[Image.Image], List[str], Optional[str]]:
    """Load images from uploads under a single global page budget.\n\n    ``max_pages`` is the total number of pages/images kept across all files,
    not a per-PDF cap. Loading stops once the budget is filled.
    """
    all_images: List[Image.Image] = []
    all_sources: List[str] = []
    skipped_files: List[str] = []
    truncated_pdfs: List[str] = []
    budget = max(1, int(max_pages))
    dpi = adaptive_dpi(budget)

    for path in paths:
        name = os.path.basename(path)
        if budget <= 0:
            skipped_files.append(name)
            continue

        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            requested = budget
            imgs = pdf_to_images(path, max_pages=requested, dpi=dpi)
            if len(imgs) == requested:
                truncated_pdfs.append(name)
            all_images.extend(imgs)
            all_sources.extend([f"{name} — page {i+1}" for i in range(len(imgs))])
            budget -= len(imgs)
        else:
            try:
                img = prepare_image(Image.open(path))
            except Exception as e:
                raise gr.Error(f"Failed to open image '{name}': {str(e)}")
            if img.size[0] == 0 or img.size[1] == 0:
                raise gr.Error(f"Image '{name}' has zero width or height.")
            all_images.append(img)
            all_sources.append(name)
            budget -= 1

    if not all_images:
        raise gr.Error("No pages could be extracted from the uploaded files.")

    note_parts: List[str] = []
    if truncated_pdfs or skipped_files:
        note_parts.append(
            f"Page budget reached ({len(all_images)} of {max_pages} allowed). "
            "Further pages were not loaded."
        )
        if truncated_pdfs:
            note_parts.append(
                "Possibly truncated PDF(s): " + ", ".join(truncated_pdfs) + "."
            )
        if skipped_files:
            note_parts.append(
                "Not loaded: " + ", ".join(skipped_files) + "."
            )
    truncation_note = " ".join(note_parts) if note_parts else None
    return all_images, all_sources, truncation_note
