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
- **Render** PDF pages to images (adaptive DPI; page count is capped for free-tier GPU limits)
- **Rank** pages with a cheap keyword-overlap score against the question and send only the top-k pages to the VLM
- **Chat** multi-turn: follow-ups reuse cached page images and conversation history
- **Stream** answers token-by-token
- **Show** which pages were sent (with relevance scores) and which were omitted
- **Optional structured mode**: request JSON with `answer`, `key_values`, `tables`, and `citations`

## Architecture

```
Upload → validate & load pages → keyword rank top-k → Qwen2.5-VL-3B → streamed answer + source gallery
```

There is **no** vector store, FAISS index, or page-level embedding retrieval. Ranking uses filename/page labels and question tokens only (no OCR of page pixels for ranking).

## Tech stack

| Piece | Choice |
|--------|--------|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| UI | Gradio (chat + gallery) |
| Runtime | Hugging Face ZeroGPU |
| PDF | pdf2image |
| Vision helpers | qwen-vl-utils |

## Limitations

- **Free ZeroGPU** has time and concurrency limits; cold starts can be slow.
- **Page budget**: default max pages is small (≈6; slider up to ~10). Long documents are truncated.
- **Ranking is shallow**: keyword overlap on source labels, not semantic embeddings or OCR text. Irrelevant pages can still be selected; relevant ones can be dropped.
- **No persistent multi-document index**: each session works from the current upload cache only.
- **Structured JSON** is best-effort; the model may still emit imperfect JSON.
- Tables and charts depend on image quality and model vision; results are not guaranteed exact.

## Future ideas

- Page-level embeddings or OCR-based retrieval (closer to classical RAG)
- Citation overlays on original page images
- Multi-document comparison
- Export answers / tables to CSV or PDF

## Local / Space notes

See `app.py` for constants (`MAX_PAGES_*`, `TOP_K_PAGES`, DPI, file size limits). System packages for PDF rendering are listed in `packages.txt`.
