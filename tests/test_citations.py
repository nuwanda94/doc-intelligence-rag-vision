"""CPU-only tests for citation matching and gallery badges."""

from __future__ import annotations


def test_extract_cited_page_numbers(app_module):
    nums = app_module.extract_cited_page_numbers(
        "Revenue is $12M [page 2] and the chart on page 4 agrees."
    )
    assert nums == {2, 4}
    assert app_module.extract_cited_page_numbers("no citations here") == set()


def test_source_is_cited_by_page_number(app_module):
    label = "report.pdf — page 3"
    assert app_module.source_is_cited(label, "See the table on [page 3].")
    assert not app_module.source_is_cited(label, "See the table on [page 1].")


def test_source_is_cited_by_full_label(app_module):
    label = "acme-invoice.pdf — page 1"
    assert app_module.source_is_cited(
        label, f"The total is $40 [{label}]."
    )
    assert not app_module.source_is_cited(label, "No page mention at all.")


def test_cited_source_indices(app_module):
    sources = ["notes.pdf — page 1", "acme.pdf — page 2", "chart.png"]
    idxs = app_module.cited_source_indices(sources, "Answer cites [page 2] only.")
    assert idxs == [1]


def test_apply_citation_highlights_badges_and_sources(app_module):
    gallery = [
        (object(), "Sent to model · 1/2 · notes.pdf — page 1"),
        (object(), "Sent to model · 2/2 · acme.pdf — page 2"),
    ]
    sources = ["notes.pdf — page 1", "acme.pdf — page 2"]
    out, text = app_module.apply_citation_highlights(
        gallery, sources, "Total is 12 [page 2].", "Pages sent to the model"
    )
    assert out[0][1].startswith("Sent to model")
    assert out[1][1].startswith("Cited · ")
    assert "Cited in answer: acme.pdf — page 2" in text


def test_apply_citation_highlights_noop_without_cites(app_module):
    gallery = [(object(), "Sent to model · 1/1 · notes.pdf — page 1")]
    out, text = app_module.apply_citation_highlights(
        gallery, ["notes.pdf — page 1"], "No page refs.", "orig"
    )
    assert out is gallery
    assert text == "orig"
