"""CPU-only tests for ranking / validation helpers in app.py."""

from __future__ import annotations

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
    images, labels, scores, omitted = app_module.rank_pages([], [], "anything")
    assert images == []
    assert labels == []
    assert scores == []
    assert omitted == []


def test_rank_pages_keeps_top_k_by_score(app_module):
    pages = [object(), object(), object()]
    sources = [
        "notes.pdf — page 1",
        "acme-invoice.pdf — page 2",
        "random.png",
    ]
    images, labels, scores, omitted = app_module.rank_pages(
        pages, sources, "What is on the acme invoice?", top_k=1
    )
    assert len(images) == 1
    assert labels == ["acme-invoice.pdf — page 2"]
    assert set(omitted) == {"notes.pdf — page 1", "random.png"}
    assert scores[0] >= 1.0


def test_rank_pages_top_k_clamped(app_module):
    pages = [object(), object()]
    sources = ["a.pdf — page 1", "b.pdf — page 2"]
    images, labels, _, omitted = app_module.rank_pages(pages, sources, "x", top_k=99)
    assert len(images) == 2
    assert omitted == []
    images, labels, _, omitted = app_module.rank_pages(pages, sources, "x", top_k=0)
    assert len(images) == 1


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
    big = tmp_path / "huge.pdf"
    big.write_bytes(b"%PDF-1.4 placeholder")
    monkeypatch.setattr(app_module, "MAX_FILE_SIZE_BYTES", 1)
    with pytest.raises(GradioError, match="exceed"):
        app_module.validate_uploads([str(big)], "summarize")
