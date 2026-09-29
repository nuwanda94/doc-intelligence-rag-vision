"""Import app.py without loading the VLM / GPU stack."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


class GradioError(Exception):
    """Stand-in for gradio.Error so validation tests can catch it."""


def _install_heavy_stubs() -> None:
    # Always (re)install stubs so missing attributes like themes stay available
    # even if a previous partial import left incomplete modules in sys.modules.

    spaces = types.ModuleType("spaces")

    def _gpu(*_args, **_kwargs):
        def decorator(fn):
            return fn

        return decorator

    spaces.GPU = _gpu
    sys.modules["spaces"] = spaces

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    sys.modules["torch"] = torch

    transformers = types.ModuleType("transformers")
    processor = MagicMock(name="AutoProcessor")
    processor.from_pretrained.return_value = MagicMock(name="processor")
    model_cls = MagicMock(name="Qwen2_5_VLForConditionalGeneration")
    model_cls.from_pretrained.return_value.eval.return_value = MagicMock(name="model")
    transformers.Qwen2_5_VLForConditionalGeneration = model_cls
    transformers.AutoProcessor = processor
    transformers.TextIteratorStreamer = MagicMock(name="TextIteratorStreamer")
    sys.modules["transformers"] = transformers

    qwen = types.ModuleType("qwen_vl_utils")
    qwen.process_vision_info = MagicMock(name="process_vision_info")
    sys.modules["qwen_vl_utils"] = qwen

    pdf2image = types.ModuleType("pdf2image")
    pdf2image.convert_from_path = MagicMock(name="convert_from_path")
    sys.modules["pdf2image"] = pdf2image

    gradio = types.ModuleType("gradio")
    gradio.Error = GradioError
    gradio.update = MagicMock(name="gr.update")

    themes = types.ModuleType("gradio.themes")
    themes.Soft = MagicMock(name="gr.themes.Soft")
    gradio.themes = themes
    sys.modules["gradio.themes"] = themes

    def _blocks_factory(*_args, **_kwargs):
        ctx = MagicMock(name="gr.Blocks.instance")
        ctx.__enter__.return_value = ctx
        ctx.__exit__.return_value = False
        return ctx

    gradio.Blocks = MagicMock(name="gr.Blocks", side_effect=_blocks_factory)

    for attr in (
        "Markdown",
        "Row",
        "Column",
        "File",
        "Slider",
        "Accordion",
        "Checkbox",
        "Chatbot",
        "Textbox",
        "Button",
        "Gallery",
        "State",
        "Examples",
        "Radio",
    ):
        setattr(gradio, attr, MagicMock(name=f"gr.{attr}"))
    sys.modules["gradio"] = gradio

    pil = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")

    class _Resampling:
        LANCZOS = 1

    class _Image:
        Resampling = _Resampling

        def convert(self, *_a, **_k):
            return self

        @property
        def size(self):
            return (1, 1)

        def resize(self, *_a, **_k):
            return self

    pil_image.Image = _Image
    pil_image.Resampling = _Resampling
    pil_image.open = MagicMock(name="Image.open")
    sys.modules["PIL"] = pil
    sys.modules["PIL.Image"] = pil_image


@pytest.fixture(scope="session")
def app_module():
    _install_heavy_stubs()
    # Drop a failed partial import so a clean re-import can succeed.
    for name in list(sys.modules):
        if name == "app" or name.startswith("docintel"):
            del sys.modules[name]
    import app as app_mod

    return app_mod
