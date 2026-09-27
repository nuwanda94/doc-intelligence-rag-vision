---
title: Multimodal Document Intelligence Tool RAG Vision
emoji: 🔥
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.27.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Modern VLMs, practical document understanding, RAG thinking.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Multimodal Document Intelligence Tool

**Live demo of a production-style document understanding system** built with Qwen2.5-VL.

## Problem
Traditional RAG struggles with scanned PDFs, tables, charts, and diagrams. Pure VLMs are powerful but need good UX and page management.

## Solution
- Upload PDFs or images
- Automatic page rendering
- Multimodal Q&A with strong document understanding (tables, charts, OCR-like reading)
- Source page visualization
- Clean, shareable Gradio interface

## Architecture
Upload → PDF→Images (pdf2image) → Qwen2.5-VL-3B → Answer + Source Gallery

## Tech Stack
- **Model**: Qwen/Qwen2.5-VL-3B-Instruct
- **UI**: Gradio
- **Hardware**: Hugging Face ZeroGPU
- **Extras**: pdf2image, qwen-vl-utils

## Limitations (be honest – recruiters love this)

- Free ZeroGPU has time/page limits
- Best with ≤ 6–8 pages
- No persistent multi-document vector store yet (easy future upgrade)

## Future Improvements

- Full RAG with FAISS + page-level embeddings
- Table extraction to structured JSON/CSV
- Multi-document comparison
- Citation highlighting on original pages
- Export report as PDF
