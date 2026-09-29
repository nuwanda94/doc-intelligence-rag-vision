from __future__ import annotations

from threading import Thread
from typing import Any, Dict, List, Optional

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
    STATUS_GENERATING,
    STATUS_PREPARING,
    TOP_K_PAGES,
    TOP_P,
)
from docintel.ingest import file_signature, load_pages_from_paths, validate_uploads
from docintel.ranking import build_gallery_and_sources, normalize_ranking_mode, rank_pages
from docintel.structured import parse_structured_output
from docintel.vlm import build_vlm_messages


def make_chat_analyze(spaces, processor, model):
    """Bind ZeroGPU decorator + loaded model/processor to the chat pipeline."""

    @spaces.GPU(duration=GPU_DURATION_SECONDS)
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
    ):
        """One chat turn: reuse cached pages when possible, re-rank, stream the reply."""
        history = list(history or [])
        question = (message or "").strip()
        pending = history + [{"role": "user", "content": question}]
        requested_mode = normalize_ranking_mode(ranking_mode)
        try:
            top_k_budget = int(top_k) if top_k is not None else TOP_K_PAGES
        except (TypeError, ValueError):
            top_k_budget = TOP_K_PAGES

        yield STATUS_PREPARING, pending + [{"role": "assistant", "content": "Preparing document\u2026"}], doc_state, None, "", gr.update(value="")

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
            all_images, all_sources, truncation_note = load_pages_from_paths(paths, max_pages)
            doc_state = {
                "images": all_images,
                "sources": all_sources,
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

        page_images, source_info, scores, omitted_labels, used_mode = rank_pages(
            all_images,
            all_sources,
            question,
            top_k=top_k_budget,
            ranking_mode=requested_mode,
            ocr_cache=ocr_cache,
        )

        messages = build_vlm_messages(
            history, page_images, source_info, question, structured=bool(structured_output)
        )

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

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
        )

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
        )
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        partial = ""
        streamed_history = pending + [{"role": "assistant", "content": ""}]
        yield STATUS_GENERATING, streamed_history, doc_state, gallery, sources_text, gr.update(value="")
        for token in streamer:
            partial += token
            streamed_history = pending + [{"role": "assistant", "content": partial}]
            yield STATUS_GENERATING, streamed_history, doc_state, gallery, sources_text, gr.update()
        thread.join()
        if not partial:
            partial = "(No answer generated.)"

        parse_note = ""
        if structured_output:
            ok, display, _parsed = parse_structured_output(partial)
            partial = display
            if not ok:
                parse_note = " Structured JSON was invalid; showing raw output."

        gallery, sources_text = apply_citation_highlights(
            gallery, source_info, partial, sources_text
        )

        final_history = pending + [{"role": "assistant", "content": partial}]
        done_status = STATUS_DONE + parse_note
        if truncation_note:
            done_status = done_status + " " + truncation_note
        yield done_status, final_history, doc_state, gallery, sources_text, gr.update(value="")

    return chat_analyze
