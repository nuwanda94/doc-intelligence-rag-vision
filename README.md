---
title: Multimodal Document Intelligence
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.27.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: VLM Q&A over PDFs/images with page ranking and chat
---

# Multimodal Document Intelligence

A Gradio app that answers questions about uploaded PDFs and images using **Qwen2.5-VL-3B-Instruct** on Hugging Face ZeroGPU.

This is a **vision-language document Q&A** tool with lightweight page selection—not a classical embedding/vector RAG system.

## What it does

- **Upload** one or more PDFs or images (PNG, JPEG, WebP, BMP)
- **Render** PDF pages to images (adaptive DPI; a global page budget caps total pages across all files)
- **Rank** pages and send only the top-k to the VLM. Default ranking is keyword overlap on filename/page labels; optional **OCR / page-text** ranking (Tesseract on downscaled rasters) can be forced or used automatically when labels are uninformative. Advanced settings expose top-k and ranking mode.
- **Chat** multi-turn: follow-ups reuse cached page images (content-hash signatures) and conversation history
- **Stream** answers token-by-token
- **Show** which pages were sent (with relevance scores and ranking mode) and which were omitted
- **Citation badges** on gallery captions when the answer mentions matching source labels (caption emphasis, not pixel bounding boxes)
- **Optional structured mode**: request JSON with `answer`, `key_values`, `tables`, and `citations`; parsed output can be downloaded as CSV (tables) and JSON (key-values)

## Architecture

```
Upload → validate & load pages → rank top-k (labels and/or OCR text) → Qwen2.5-VL-3B → streamed answer + source gallery + optional export
```

There is **no** vector store, FAISS index, or page-level embedding retrieval. Ranking is keyword overlap on source labels, optionally plus OCR text from a bounded candidate set—not semantic embeddings.

## Tech stack

| Piece | Choice |
|--------|--------|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| UI | Gradio (chat + gallery) |
| Runtime | Hugging Face ZeroGPU |
| PDF | pdf2image |
| OCR ranking (optional) | pytesseract / Tesseract |
| Vision helpers | qwen-vl-utils |
| Layout | `app.py` entrypoint; logic in `docintel/` |

## Limitations

- **Free ZeroGPU** has time and concurrency limits; cold starts can be slow.
- **Page budget**: default max pages is small (≈6; slider up to ~10). Long documents are truncated.
- **Ranking is still shallow**: label keywords and/or token overlap on OCR text—not embeddings. Irrelevant pages can still be selected; relevant ones can be dropped. OCR is bounded (candidate cap + downscale) and falls back to labels if Tesseract yields no text.
- **Citation highlights** match source labels mentioned in the answer; they are caption badges, not region overlays, and can false-positive across files that share a page number.
- **No persistent multi-document index**: each session works from the current upload cache only. Multi-file uploads share one ranked pool (no dedicated compare mode yet).
- **Structured JSON** is best-effort; parse failures show raw text and skip export.
- Tables and charts depend on image quality and model vision; results are not guaranteed exact.

## Future ideas

- Page-level embeddings or native PDF text-layer ranking before OCR
- Multi-document comparison with per-file top-k
- Length-normalized ranking scores and tighter citation matching across files

## Local / Space notes

See `docintel/constants.py` for constants (`MAX_PAGES_*`, `TOP_K_PAGES`, DPI, file size limits, ranking modes). `app.py` is the Hugging Face Space entrypoint. System packages for PDF rendering (and Tesseract, when OCR ranking is used) are listed in `packages.txt`.
