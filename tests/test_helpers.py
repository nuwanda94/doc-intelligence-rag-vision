"""CPU-only tests for ranking / validation helpers in app.py."""

from __future__ import annotations

import hashlib

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
