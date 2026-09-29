from __future__ import annotations

from threading import Event, Thread
from time import monotonic
from typing import Any, Dict, Iterator, List, Optional

import gradio as gr
from transformers import TextIteratorStreamer
from qwen_vl_utils import process_vision_info

from docintel.citations import apply_citation_highlights
from docintel.constants import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_RANKING_MODE,
    DEFAULT_TEMPERATURE,
    GPU_DURATION_SECONDS,
    MAX_PAGES_DEFAULT,
    STATUS_DONE,
    STATUS_PREPARING,
    TOP_K_PAGES,
    TOP_P,
    format_gpu_budget_status,
)
from docintel.ingest import file_signature, load_pages_from_paths, validate_uploads
from docintel.ranking import build_gallery_and_sources, normalize_ranking_mode, rank_pages
from docintel.structured import build_structured_exports, parse_structured_output
from docintel.vlm import build_vlm_messages

try:
    from transformers import StoppingCriteria, StoppingCriteriaList
except ImportError:  # pragma: no cover - tests stub transformers
    StoppingCriteria = object  # type: ignore[misc,assignment]
    StoppingCriteriaList = list  # type: ignore[misc,assignment]


class EventStoppingCriteria(StoppingCriteria):
    """Halt ``model.generate`` when a threading Event is set (Stop / cancel)."""

    def __init__(self, stop_event: Event):
        try:
            super().__init__()
        except TypeError:
            pass
        self.stop_event = stop_event

    def __call__(self, input_ids=None, scores=None, **kwargs) -> bool:
        return bool(self.stop_event.is_set())


def make_chat_analyze(spaces, processor, model):
    """Bind ZeroGPU decorator + loaded model/processor to the chat pipeline.

    CPU work (validation, hashing, PDF/image ingest, text/OCR ranking, message
    construction, structured parse) stays on the outer function. Only tensor
    prep and ``model.generate`` run under ``@spaces.GPU`` so the 120s quota
    is not spent on pdf2image / pdftotext / Tesseract.
    """

    @spaces.GPU(duration=GPU_DURATION_SECONDS)
    def generate_on_gpu(
        messages: List[Dict[str, Any]],
        temperature: float,
        max_new_tokens: int,
    ) -> Iterator[str]:
        """Tokenize vision messages and stream ``model.generate`` on ZeroGPU.

        Create the stop Event *inside* this function. ZeroGPU pickles args into a
        worker process; ``threading.Event`` (and its lock) is not picklable, so it
        must never be passed across the decorator boundary. Gradio cancel closes
        this generator (GeneratorExit at yield); the finally block signals the
        generate thread to exit.
        """
        stop_event = Event()
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=TOP_P,
            stopping_criteria=StoppingCriteriaList([EventStoppingCriteria(stop_event)]),
        )
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        try:
            for token in streamer:
                if stop_event.is_set():
                    break
                yield token
        finally:
            stop_event.set()
            thread.join(timeout=5)

    def chat_analyze(
        message: str,
        history: Optional[List[Dict[str, Any]]],
        files: Optional[List],
        doc_state: Optional[Dict[str, Any]],
        max_pages: int = MAX_PAGES_DEFAULT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        structured_output: bool = False,
        top_k: int = TOP_K_PAGES,
        ranking_mode: str = DEFAULT_RANKING_MODE,
        compare_mode: bool = False,
    ):
        """One chat turn: reuse cached pages when possible, re-rank, stream the reply."""
        history = list(history or [])
        question = (message or "").strip()
        pending = history + [{"role": "user", "content": question}]
        requested_mode = normalize_ranking_mode(ranking_mode)
        compare = bool(compare_mode)
        try:
            top_k_budget = int(top_k) if top_k is not None else TOP_K_PAGES
        except (TypeError, ValueError):
            top_k_budget = TOP_K_PAGES

        empty_exports = (None, None)
        yield (
            STATUS_PREPARING,
            pending + [{"role": "assistant", "content": "Preparing document\u2026"}],
            doc_state,
            None,
            "",
            gr.update(value=""),
            *empty_exports,
        )

        sig = file_signature(files)
        cached_sig = (doc_state or {}).get("file_sig")
        cached_budget = (doc_state or {}).get("max_pages")
        if (
            doc_state is None
            or not doc_state.get("images")
            or cached_sig != sig
            or cached_budget != max_pages
        ):
            paths = validate_uploads(files, question)
            all_images, all_sources, truncation_note, origins = load_pages_from_paths(paths, max_pages)
            doc_state = {
                "images": all_images,
                "sources": all_sources,
                "origins": origins,
                "file_sig": sig,
                "max_pages": max_pages,
                "truncation_note": truncation_note,
                "ocr_text": {},
            }
        elif not question:
            raise gr.Error("Please enter a question about the uploaded document.")

        all_images = doc_state["images"]
        all_sources = doc_state["sources"]
        truncation_note = doc_state.get("truncation_note")
        ocr_cache = doc_state.setdefault("ocr_text", {})
        page_origins = doc_state.get("origins") or []

        page_images, source_info, scores, omitted_labels, used_mode = rank_pages(
            all_images,
            all_sources,
            question,
            top_k=top_k_budget,
            ranking_mode=requested_mode,
            ocr_cache=ocr_cache,
            page_origins=page_origins,
            compare=compare,
        )

        messages = build_vlm_messages(
            history,
            page_images,
            source_info,
            question,
            structured=bool(structured_output),
            compare=compare,
        )

        gallery, sources_text = build_gallery_and_sources(
            page_images,
            source_info,
            scores=scores,
            omitted_labels=omitted_labels,
            total_extracted=len(all_images),
            truncation_note=truncation_note,
            ranking_mode=used_mode,
            requested_mode=requested_mode,
            top_k=top_k_budget,
            compare=compare,
        )

        partial = ""
        streamed_history = pending + [{"role": "assistant", "content": ""}]
        gpu_started = monotonic()
        yield (
            format_gpu_budget_status(0),
            streamed_history,
            doc_state,
            gallery,
            sources_text,
            gr.update(value=""),
            *empty_exports,
        )
        for token in generate_on_gpu(messages, temperature, max_new_tokens):
            partial += token
            streamed_history = pending + [{"role": "assistant", "content": partial}]
            yield (
                format_gpu_budget_status(monotonic() - gpu_started),
                streamed_history,
                doc_state,
                gallery,
                sources_text,
                gr.update(),
                *empty_exports,
            )

        if not partial:
            partial = "(No answer generated.)"

        parse_note = ""
        parse_ok = False
        parsed = None
        if structured_output:
            parse_ok, display, parsed = parse_structured_output(partial)
            partial = display
            if not parse_ok:
                parse_note = " Structured JSON was invalid; showing raw output."

        csv_path, json_path = build_structured_exports(
            bool(structured_output), parse_ok, parsed
        )

        gallery, sources_text = apply_citation_highlights(
            gallery, source_info, partial, sources_text
        )

        final_history = pending + [{"role": "assistant", "content": partial}]
        done_status = STATUS_DONE + parse_note
        if truncation_note:
            done_status = done_status + " " + truncation_note
        yield done_status, final_history, doc_state, gallery, sources_text, gr.update(value=""), csv_path, json_path

    chat_analyze.generate_on_gpu = generate_on_gpu
    chat_analyze.EventStoppingCriteria = EventStoppingCriteria
    return chat_analyze
