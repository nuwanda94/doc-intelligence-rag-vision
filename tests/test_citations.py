"""CPU-only tests for citation matching and gallery badges."""

from __future__ import annotations

from docintel.citations import (
    apply_citation_highlights,
    cited_source_indices,
    extract_cited_page_numbers,
    source_is_cited,
)


def test_extract_cited_page_numbers():
    nums = extract_cited_page_numbers(
        "Revenue is $12M [page 2] and the chart on page 4 agrees."
    )
    assert nums == {2, 4}
    assert extract_cited_page_numbers("no citations here") == set()


def test_source_is_cited_by_page_number():
    label = "report.pdf — page 3"
    assert source_is_cited(label, "See the table on [page 3].")
    assert not source_is_cited(label, "See the table on [page 1].")


def test_source_is_cited_by_full_label():
    label = "acme-invoice.pdf — page 1"
    assert source_is_cited(label, f"The total is $40 [{label}].")
    assert not source_is_cited(label, "No page mention at all.")


def test_source_is_cited_filename_plus_page():
    label = "acme.pdf — page 2"
    sources = ["notes.pdf — page 2", label]
    assert source_is_cited(
        label, "See the table in acme.pdf [page 2].", all_labels=sources
    )
    assert not source_is_cited(
        "notes.pdf — page 2", "See the table in acme.pdf [page 2].", all_labels=sources
    )


def test_source_is_cited_bare_page_does_not_cross_files():
    sources = ["alpha.pdf — page 2", "beta.pdf — page 2"]
    answer = "Both reports mention the same figure on [page 2]."
    assert cited_source_indices(sources, answer) == []
    assert not source_is_cited(sources[0], answer, all_labels=sources)
    assert not source_is_cited(sources[1], answer, all_labels=sources)


def test_source_is_cited_bare_page_single_file_session():
    sources = ["report.pdf — page 1", "report.pdf — page 2"]
    idxs = cited_source_indices(sources, "See [page 2].")
    assert idxs == [1]


def test_cited_source_indices():
    sources = ["notes.pdf — page 1", "acme.pdf — page 2", "chart.png"]
    idxs = cited_source_indices(sources, "Answer cites acme.pdf [page 2] only.")
    assert idxs == [1]


def test_apply_citation_highlights_badges_and_sources():
    gallery = [
        (object(), "Sent to model · 1/2 · notes.pdf — page 1"),
        (object(), "Sent to model · 2/2 · acme.pdf — page 2"),
    ]
    sources = ["notes.pdf — page 1", "acme.pdf — page 2"]
    out, text = apply_citation_highlights(
        gallery, sources, "Total is 12 [acme.pdf — page 2].", "Pages sent to the model"
    )
    assert out[0][1].startswith("Sent to model")
    assert out[1][1].startswith("Cited · ")
    assert "Cited in answer: acme.pdf — page 2" in text


def test_apply_citation_highlights_noop_without_cites():
    gallery = [(object(), "Sent to model · 1/1 · notes.pdf — page 1")]
    out, text = apply_citation_highlights(
        gallery, ["notes.pdf — page 1"], "No page refs.", "orig"
    )
    assert out is gallery
    assert text == "orig"
