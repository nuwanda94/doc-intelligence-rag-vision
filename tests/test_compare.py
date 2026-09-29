"""CPU-only tests for multi-document compare ranking and prompts."""

from __future__ import annotations

from docintel.constants import COMPARE_SYSTEM_PROMPT, SYSTEM_PROMPT
from docintel.ranking import (
    build_gallery_and_sources,
    document_key_from_label,
    per_document_quotas,
    rank_pages,
)
from docintel.vlm import build_vlm_messages


def test_document_key_from_label():
    assert document_key_from_label("report.pdf — page 2") == "report.pdf"
    assert document_key_from_label("photo.png") == "photo.png"
    assert document_key_from_label("") == "document"


def test_per_document_quotas_even_split():
    quotas = per_document_quotas({"a.pdf": 4, "b.pdf": 4}, top_k=4)
    assert quotas == {"a.pdf": 2, "b.pdf": 2}


def test_per_document_quotas_more_docs_than_k():
    quotas = per_document_quotas({"a": 3, "b": 3, "c": 3}, top_k=2)
    assert sum(quotas.values()) == 2
    assert quotas["a"] == 1
    assert quotas["b"] == 1
    assert quotas["c"] == 0


def test_per_document_quotas_respects_small_docs():
    quotas = per_document_quotas({"tiny.pdf": 1, "big.pdf": 8}, top_k=4)
    assert quotas["tiny.pdf"] == 1
    assert quotas["big.pdf"] == 3
    assert sum(quotas.values()) == 4


def test_rank_pages_compare_splits_across_files():
    pages = [object(), object(), object(), object()]
    sources = [
        "alpha.pdf — page 1",
        "alpha.pdf — page 2",
        "beta.pdf — page 1",
        "beta.pdf — page 2",
    ]
    images, labels, scores, omitted, mode = rank_pages(
        pages,
        sources,
        "compare the totals",
        top_k=2,
        ranking_mode="label",
        compare=True,
    )
    assert len(images) == 2
    files = {lbl.split(" — ")[0] for lbl in labels}
    assert files == {"alpha.pdf", "beta.pdf"}
    assert mode == "label"
    assert len(omitted) == 2


def test_rank_pages_compare_falls_back_for_single_file():
    pages = [object(), object(), object()]
    sources = [
        "only.pdf — page 1",
        "only.pdf — page 2",
        "only.pdf — page 3",
    ]
    images, labels, _, omitted, _ = rank_pages(
        pages,
        sources,
        "summarize",
        top_k=2,
        ranking_mode="label",
        compare=True,
    )
    assert len(images) == 2
    assert all(lbl.startswith("only.pdf") for lbl in labels)
    assert len(omitted) == 1


def test_rank_pages_compare_respects_global_topk_cap():
    pages = [object()] * 6
    sources = [
        "a.pdf — page 1",
        "a.pdf — page 2",
        "a.pdf — page 3",
        "b.pdf — page 1",
        "b.pdf — page 2",
        "b.pdf — page 3",
    ]
    images, labels, _, omitted, _ = rank_pages(
        pages,
        sources,
        "compare",
        top_k=3,
        ranking_mode="label",
        compare=True,
    )
    assert len(images) == 3
    assert len(labels) == 3
    assert len(omitted) == 3
    files = {lbl.split(" — ")[0] for lbl in labels}
    assert files == {"a.pdf", "b.pdf"}


def test_build_gallery_compare_note():
    pages = [object(), object()]
    sources = ["a.pdf — page 1", "b.pdf — page 1"]
    _, text = build_gallery_and_sources(
        pages,
        sources,
        scores=[1.0, 0.5],
        compare=True,
        top_k=4,
    )
    assert "Compare mode: top-k budget split across documents" in text
    assert "a.pdf" in text and "b.pdf" in text


def test_build_vlm_messages_compare_prompt():
    messages = build_vlm_messages(
        [],
        [object()],
        ["a.pdf — page 1"],
        "How do they differ?",
        compare=True,
    )
    assert messages[0]["content"] == COMPARE_SYSTEM_PROMPT
    assert messages[0]["content"] != SYSTEM_PROMPT
    user_bits = messages[-1]["content"]
    texts = [p["text"] for p in user_bits if p.get("type") == "text"]
    assert any("Compare the uploaded documents" in t for t in texts)
    assert any("How do they differ?" in t for t in texts)
