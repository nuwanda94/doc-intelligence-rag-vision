"""CPU-only tests for ranking / validation helpers in app.py."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest

from tests.conftest import GradioError


def test_tokenize_query_strips_stopwords_and_short_tokens(app_module):
    tokens = app_module.tokenize_query("What is the revenue on page 2 of the document?")
    assert "revenue" in tokens
    assert "page" not in tokens
    assert "the" not in tokens
    assert "is" not in tokens
    assert "2" not in tokens
    assert app_module.tokenize_query("") == set()
    assert app_module.tokenize_query(None) == set()


def test_tokenize_query_lowercases_and_splits_alnum(app_module):
    tokens = app_module.tokenize_query("Q3-FY2024 Invoice #42")
    assert "fy2024" in tokens
    assert "invoice" in tokens
    assert "42" in tokens
    assert "q3" in tokens


def test_adaptive_dpi_threshold(app_module):
    assert app_module.adaptive_dpi(app_module.DPI_PAGE_THRESHOLD) == app_module.DPI_HIGH
    assert app_module.adaptive_dpi(app_module.DPI_PAGE_THRESHOLD + 1) == app_module.DPI_LOW
    assert app_module.adaptive_dpi(1) == app_module.DPI_HIGH
    assert app_module.adaptive_dpi(10) == app_module.DPI_LOW


def test_page_relevance_score_boosts_explicit_page_refs(app_module):
    label = "report.pdf — page 3"
    with_ref = app_module.page_relevance_score(label, "summarize page 3", page_index=2)
    without_ref = app_module.page_relevance_score(label, "summarize the totals", page_index=2)
    assert with_ref >= without_ref + 5.0


def test_page_relevance_score_filename_overlap(app_module):
    invoice_label = "acme-invoice-2024.pdf — page 1"
    other_label = "notes.png"
    q = "What is the invoice total for Acme?"
    assert app_module.page_relevance_score(invoice_label, q, 0) > app_module.page_relevance_score(
        other_label, q, 1
    )


def test_page_relevance_score_earlier_pages_tiebreak_slightly(app_module):
    label = "doc.pdf — page 1"
    first = app_module.page_relevance_score(label, "hello", 0)
    later = app_module.page_relevance_score(label, "hello", 9)
    assert first > later


def test_rank_pages_empty(app_module):
    images, labels, scores, omitted, mode = app_module.rank_pages([], [], "anything")
    assert images == []
    assert labels == []
    assert scores == []
    assert omitted == []
    assert mode == "label"


def test_rank_pages_keeps_top_k_by_score(app_module):
    pages = [object(), object(), object()]
    sources = [
        "notes.pdf — page 1",
        "acme-invoice.pdf — page 2",
        "random.png",
    ]
    images, labels, scores, omitted, mode = app_module.rank_pages(
        pages, sources, "What is on the acme invoice?", top_k=1, ranking_mode="label"
    )
    assert len(images) == 1
    assert labels == ["acme-invoice.pdf — page 2"]
    assert set(omitted) == {"notes.pdf — page 1", "random.png"}
    assert scores[0] >= 1.0
    assert mode == "label"


def test_rank_pages_top_k_clamped(app_module):
    pages = [object(), object()]
    sources = ["a.pdf — page 1", "b.pdf — page 2"]
    images, labels, _, omitted, _ = app_module.rank_pages(
        pages, sources, "x", top_k=99, ranking_mode="label"
    )
    assert len(images) == 2
    assert omitted == []
    images, labels, _, omitted, _ = app_module.rank_pages(
        pages, sources, "x", top_k=0, ranking_mode="label"
    )
    assert len(images) == 1


def test_labels_are_informative(app_module):
    assert app_module.labels_are_informative(
        ["acme-invoice.pdf — page 1"], "What is the invoice total?"
    )
    assert not app_module.labels_are_informative(
        ["scan.pdf — page 1", "scan.pdf — page 2"], "What is the revenue?"
    )


def test_page_text_relevance_score_uses_ocr_overlap(app_module):
    label = "scan.pdf — page 1"
    q = "What is the revenue figure?"
    label_only = app_module.page_relevance_score(label, q, 0)
    with_text = app_module.page_text_relevance_score(label, "Annual revenue was $12M", q, 0)
    assert with_text > label_only


def test_page_text_relevance_score_length_normalizes(app_module):
    label = "scan.pdf — page 1"
    q = "What is the revenue figure?"
    short = app_module.page_text_relevance_score(label, "Revenue figure twelve million", q, 0)
    wordy = app_module.page_text_relevance_score(
        label,
        "Revenue figure twelve million "
        + "appendix notes glossary index acknowledgements bibliography "
        + "cover preface contents photos captions footnotes references",
        q,
        0,
    )
    assert short > wordy


def test_rank_pages_ocr_prefers_short_relevant_over_wordy(app_module):
    pages = [object(), object()]
    sources = ["scan.pdf — page 1", "scan.pdf — page 2"]
    texts = {
        0: (
            "revenue figure twelve million plus a long appendix of notes glossary "
            "index acknowledgements bibliography cover preface contents photos "
            "captions footnotes references extras padding filler paragraphs"
        ),
        1: "revenue figure twelve million",
    }

    def fake_ocr(img):
        return texts[pages.index(img)]

    images, labels, scores, omitted, mode = app_module.rank_pages(
        pages,
        sources,
        "What is the revenue figure?",
        top_k=1,
        ranking_mode="ocr",
        ocr_fn=fake_ocr,
    )
    assert mode == "ocr"
    assert labels == ["scan.pdf — page 2"]
    assert omitted == ["scan.pdf — page 1"]
    assert scores[0] > 0.05


def test_rank_pages_label_mode_ignores_page_text_length(app_module):
    pages = [object(), object()]
    sources = ["scan.pdf — page 1", "scan.pdf — page 2"]
    called = {"n": 0}

    def fake_ocr(_img):
        called["n"] += 1
        return "revenue figure twelve million"

    images, labels, _, omitted, mode = app_module.rank_pages(
        pages,
        sources,
        "What is the revenue figure?",
        top_k=1,
        ranking_mode="label",
        ocr_fn=fake_ocr,
    )
    assert called["n"] == 0
    assert mode == "label"
    assert labels == ["scan.pdf — page 1"]
    assert omitted == ["scan.pdf — page 2"]


def test_rank_pages_ocr_mode_prefers_text_overlap(app_module):
    pages = [object(), object(), object()]
    sources = [
        "scan.pdf — page 1",
        "scan.pdf — page 2",
        "scan.pdf — page 3",
    ]
    texts = {
        0: "introduction and table of contents",
        1: "annual revenue and operating margin",
        2: "appendix photos",
    }

    def fake_ocr(img):
        # identity of page is the object identity order in pages
        return texts[pages.index(img)]

    images, labels, scores, omitted, mode = app_module.rank_pages(
        pages,
        sources,
        "What is the revenue?",
        top_k=1,
        ranking_mode="ocr",
        ocr_fn=fake_ocr,
    )
    assert mode == "ocr"
    assert labels == ["scan.pdf — page 2"]
    assert set(omitted) == {"scan.pdf — page 1", "scan.pdf — page 3"}
    assert scores[0] > 0.05


def test_rank_pages_auto_stays_label_when_filename_helps(app_module):
    pages = [object(), object()]
    sources = ["notes.pdf — page 1", "acme-invoice.pdf — page 1"]
    called = {"n": 0}

    def fake_ocr(_img):
        called["n"] += 1
        return "revenue total"

    images, labels, _, _, mode = app_module.rank_pages(
        pages,
        sources,
        "What is on the acme invoice?",
        top_k=1,
        ranking_mode="auto",
        ocr_fn=fake_ocr,
    )
    assert called["n"] == 0
    assert mode == "label"
    assert labels == ["acme-invoice.pdf — page 1"]


def test_rank_pages_ocr_falls_back_when_no_text(app_module):
    pages = [object(), object()]
    sources = ["scan.pdf — page 1", "scan.pdf — page 2"]
    images, labels, _, omitted, mode = app_module.rank_pages(
        pages,
        sources,
        "What is the revenue?",
        top_k=1,
        ranking_mode="ocr",
        ocr_fn=lambda _img: "",
    )
    assert mode == "label"
    assert len(images) == 1
    assert labels[0] == "scan.pdf — page 1"


def test_rank_pages_native_pdf_text_skips_ocr(app_module):
    pages = [object(), object()]
    sources = ["report.pdf — page 1", "report.pdf — page 2"]
    origins = [
        {"kind": "pdf", "path": "/tmp/report.pdf", "page": 1},
        {"kind": "pdf", "path": "/tmp/report.pdf", "page": 2},
    ]
    ocr_calls = {"n": 0}
    native_calls = []

    def fake_ocr(_img):
        ocr_calls["n"] += 1
        return "should not be used"

    def fake_native(path, page):
        native_calls.append((path, page))
        if page == 2:
            return "annual revenue twelve million"
        return "table of contents"

    images, labels, _, omitted, mode = app_module.rank_pages(
        pages,
        sources,
        "What is the revenue?",
        top_k=1,
        ranking_mode="ocr",
        ocr_fn=fake_ocr,
        native_text_fn=fake_native,
        page_origins=origins,
    )
    assert mode == "ocr"
    assert labels == ["report.pdf — page 2"]
    assert ocr_calls["n"] == 0
    assert native_calls == [("/tmp/report.pdf", 1), ("/tmp/report.pdf", 2)]
    assert set(omitted) == {"report.pdf — page 1"}


def test_rank_pages_ocr_when_native_text_empty(app_module):
    pages = [object(), object()]
    sources = ["scan.pdf — page 1", "scan.pdf — page 2"]
    origins = [
        {"kind": "pdf", "path": "/tmp/scan.pdf", "page": 1},
        {"kind": "pdf", "path": "/tmp/scan.pdf", "page": 2},
    ]

    def fake_native(_path, _page):
        return ""

    def fake_ocr(img):
        return {
            0: "cover photo",
            1: "operating revenue details",
        }[pages.index(img)]

    images, labels, _, _, mode = app_module.rank_pages(
        pages,
        sources,
        "What is the revenue?",
        top_k=1,
        ranking_mode="ocr",
        ocr_fn=fake_ocr,
        native_text_fn=fake_native,
        page_origins=origins,
    )
    assert mode == "ocr"
    assert labels == ["scan.pdf — page 2"]


def test_rank_pages_reuses_text_cache(app_module):
    pages = [object()]
    sources = ["scan.pdf — page 1"]
    origins = [{"kind": "pdf", "path": "/tmp/scan.pdf", "page": 1}]
    cache = {0: "cached revenue figure"}
    native_calls = {"n": 0}
    ocr_calls = {"n": 0}

    images, labels, scores, _, mode = app_module.rank_pages(
        pages,
        sources,
        "What is the revenue?",
        top_k=1,
        ranking_mode="ocr",
        ocr_fn=lambda _img: ocr_calls.__setitem__("n", ocr_calls["n"] + 1) or "",
        native_text_fn=lambda *_a: native_calls.__setitem__("n", native_calls["n"] + 1) or "",
        page_origins=origins,
        ocr_cache=cache,
    )
    assert mode == "ocr"
    assert native_calls["n"] == 0
    assert ocr_calls["n"] == 0
    assert scores[0] > 0.05


def test_extract_native_pdf_page_text_invalid(app_module):
    assert app_module.extract_native_pdf_page_text("", 1) == ""
    assert app_module.extract_native_pdf_page_text("/tmp/x.pdf", 0) == ""


def test_normalize_ranking_mode_accepts_ui_labels(app_module):
    assert app_module.normalize_ranking_mode("Label keywords only") == "label"
    assert app_module.normalize_ranking_mode("OCR / page text") == "ocr"
    assert app_module.normalize_ranking_mode("Auto (OCR if labels uninformative)") == "auto"
    assert app_module.normalize_ranking_mode("label") == "label"
    assert app_module.normalize_ranking_mode(None) == app_module.DEFAULT_RANKING_MODE
    assert app_module.normalize_ranking_mode("mystery") == app_module.DEFAULT_RANKING_MODE


def test_build_gallery_and_sources_reports_requested_mode_and_topk(app_module):
    pages = [object()]
    sources = ["scan.pdf — page 1"]
    _, text = app_module.build_gallery_and_sources(
        pages,
        sources,
        scores=[1.25],
        omitted_labels=["scan.pdf — page 2"],
        total_extracted=2,
        ranking_mode="label",
        requested_mode="Auto (OCR if labels uninformative)",
        top_k=4,
    )
    assert "Ranking requested: Auto (OCR if labels uninformative)." in text
    assert "Ranking used: keyword overlap on file/page labels." in text
    assert "Top-k pages sent: 1 (budget 4)." in text
    assert "relevance 1.25" in text
    assert "Not sent to the model (1):" in text


def test_validate_uploads_requires_files_and_question(app_module):
    with pytest.raises(GradioError, match="at least one PDF"):
        app_module.validate_uploads(None, "what?")
    with pytest.raises(GradioError, match="enter a question"):
        app_module.validate_uploads(["/tmp/x.pdf"], "   ")


def test_validate_uploads_unsupported_and_empty(app_module, tmp_path):
    empty = tmp_path / "blank.pdf"
    empty.write_bytes(b"")
    with pytest.raises(GradioError, match="empty"):
        app_module.validate_uploads([str(empty)], "what is this?")

    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(GradioError, match="Unsupported"):
        app_module.validate_uploads([str(bad)], "what is this?")


def test_validate_uploads_missing_and_ok_image(app_module, tmp_path):
    with pytest.raises(GradioError, match="Could not read"):
        app_module.validate_uploads(["/no/such/file.pdf"], "question")

    ok = tmp_path / "page.png"
    ok.write_bytes(b"\x89PNG\r\n\x1anot-a-real-png-but-nonempty")
    paths = app_module.validate_uploads([str(ok)], "describe this")
    assert paths == [str(ok)]


def test_validate_uploads_oversized(app_module, tmp_path, monkeypatch):
    import docintel.ingest as ingest

    big = tmp_path / "huge.pdf"
    big.write_bytes(b"%PDF-1.4 placeholder")
    monkeypatch.setattr(app_module, "MAX_FILE_SIZE_BYTES", 1)
    monkeypatch.setattr(ingest, "MAX_FILE_SIZE_BYTES", 1)
    with pytest.raises(GradioError, match="exceed"):
        app_module.validate_uploads([str(big)], "summarize")


def test_file_signature_empty(app_module):
    assert app_module.file_signature(None) == ()
    assert app_module.file_signature([]) == ()


def test_file_signature_same_bytes_different_paths(app_module, tmp_path):
    payload = b"%PDF-1.4 identical-bytes"
    a = tmp_path / "upload-a" / "doc.pdf"
    b = tmp_path / "upload-b" / "doc.pdf"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_bytes(payload)
    b.write_bytes(payload)
    expected = (f"{len(payload)}:{hashlib.sha256(payload).hexdigest()}",)
    assert app_module.file_signature([str(a)]) == expected
    assert app_module.file_signature([str(b)]) == expected
    assert app_module.file_signature([str(a)]) == app_module.file_signature([str(b)])


def test_file_signature_changes_when_bytes_differ(app_module, tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert app_module.file_signature([str(a)]) != app_module.file_signature([str(b)])


def test_file_signature_missing_path_is_stable_and_distinct(app_module):
    missing = "/no/such/upload.pdf"
    sig = app_module.file_signature([missing])
    assert sig == (f"missing:{missing}",)
    assert sig != app_module.file_signature(["/also/missing.pdf"])


def _fake_pdf_pages(total_pages: int):
    """Return a convert_from_path stand-in that honors last_page vs a fixed length."""

    def _convert(path, dpi=None, first_page=1, last_page=None):
        end = total_pages if last_page is None else min(total_pages, last_page)
        start = max(1, first_page)
        count = max(0, end - start + 1)
        pages = []
        for _ in range(count):
            img = MagicMock()
            img.convert.return_value = img
            img.size = (10, 10)
            img.resize.return_value = img
            pages.append(img)
        return pages

    return _convert


def test_load_pages_exact_length_pdf_is_not_truncated(app_module, tmp_path, monkeypatch):
    import docintel.ingest as ingest

    pdf = tmp_path / "exact-six.pdf"
    pdf.write_bytes(b"%PDF-1.4 six-pages")
    monkeypatch.setattr(ingest, "convert_from_path", _fake_pdf_pages(6))

    images, sources, note, origins = app_module.load_pages_from_paths([str(pdf)], max_pages=6)
    assert len(images) == 6
    assert len(sources) == 6
    assert note is None
    assert "truncated" not in (note or "").lower()
    assert origins[0]["kind"] == "pdf"
    assert origins[0]["path"] == str(pdf)
    assert origins[-1]["page"] == 6


def test_load_pages_over_budget_pdf_is_truncated(app_module, tmp_path, monkeypatch):
    import docintel.ingest as ingest

    pdf = tmp_path / "seven.pdf"
    pdf.write_bytes(b"%PDF-1.4 seven-pages")
    monkeypatch.setattr(ingest, "convert_from_path", _fake_pdf_pages(7))

    images, sources, note, origins = app_module.load_pages_from_paths([str(pdf)], max_pages=6)
    assert len(images) == 6
    assert len(origins) == 6
    assert sources[-1].endswith("page 6")
    assert note is not None
    assert "Possibly truncated" in note
    assert "seven.pdf" in note
