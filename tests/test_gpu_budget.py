"""CPU-only tests for GPU budget status and generate stop criteria."""

from __future__ import annotations

from threading import Event

from docintel.constants import GPU_DURATION_SECONDS, format_gpu_budget_status
from docintel.pipeline import EventStoppingCriteria


def test_format_gpu_budget_status_elapsed_and_remaining():
    text = format_gpu_budget_status(15, budget_s=120)
    assert "15s elapsed" in text
    assert "105s remaining" in text
    assert "120s ZeroGPU budget" in text


def test_format_gpu_budget_status_clamps_over_budget():
    text = format_gpu_budget_status(200, budget_s=120)
    assert "200s elapsed" in text
    assert "0s remaining" in text


def test_format_gpu_budget_status_defaults_and_bad_values():
    text = format_gpu_budget_status(0)
    assert f"{GPU_DURATION_SECONDS}s ZeroGPU budget" in text
    assert "0s elapsed" in text
    assert "~120s remaining" in format_gpu_budget_status("nope", budget_s=None)


def test_event_stopping_criteria_follows_event():
    flag = Event()
    criteria = EventStoppingCriteria(flag)
    assert criteria() is False
    flag.set()
    assert criteria() is True
