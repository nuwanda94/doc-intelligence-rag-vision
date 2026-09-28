# Implementation Plan – Multimodal Document Intelligence Tool

Ordered backlog of improvements. Each item is designed to be implemented in a single focused PR.
Automation picks the next incomplete item (status: TODO), implements it, opens a PR, and merges to main.

## Status Legend
- TODO
- IN_PROGRESS
- DONE

---

## Phase 1 – Robustness & Quick Wins (High Priority)

### 1. [DONE] fix: remove unused dependencies
- Remove `faiss-cpu` and `sentence-transformers` from requirements.txt (they are not used).
- Keep only what is actually imported/used.
- Type: chore

### 2. [DONE] fix: add processor pixel limits to prevent OOM
- When creating AutoProcessor, set:
  min_pixels=256*28*28
  max_pixels=1280*28*28
- Type: fix

### 3. [DONE] feat: resize and compress images before inference
- After loading each page/image, resize so max side ≤ 1280 px.
- Convert to RGB and optionally compress.
- Type: feat

### 4. [DONE] fix: improve input validation and error messages
- Validate file types explicitly and return clear Gradio errors for unsupported files.
- Guard against empty uploads, zero pages, oversized files.
- Type: fix

### 5. [DONE] feat: adaptive DPI for PDF conversion
- Lower DPI for higher page counts (e.g. 120 for >4 pages, 150 otherwise).
- Expose or keep simple heuristic.
- Type: feat

### 6. [DONE] feat: stream the model answer
- Use Gradio streaming / yield partial tokens from model.generate.
- Type: feat

### 7. [DONE] fix: improve system prompt for citations and tables
- Strengthen system prompt to require page citations and precise table/chart extraction.
- Type: fix

### 8. [DONE] feat: highlight only relevant source pages in gallery
- Ideally rank or at least document which pages were sent; improve gallery labeling.
- Type: feat

### 9. [DONE] chore: clean dead code and magic numbers
- Remove unused `tempfile` import.
- Extract constants (MAX_PAGES_DEFAULT, DPI, etc.).
- Type: chore

### 10. [DONE] feat: better loading states and Clear button
- Add clear button and improve progress/loading UX.
- Type: feat

---

## Phase 2 – Real Multimodal RAG (Core Differentiator)

### 11. [DONE] feat: simple page relevance ranking
- After converting pages to images, compute a cheap relevance score (keyword overlap or simple embedding) against the question and keep top-k pages.
- Type: feat

### 12. [DONE] feat: integrate page ranking into the VLM pipeline
- Only send the top-k ranked pages to the VLM.
- Update sources display accordingly.
- Type: feat

### 13. [DONE] feat: multi-turn conversation support
- Convert UI to chat-style with state for previous images + history.
- Type: feat

---

## Phase 3 – Production Polish

### 14. [DONE] feat: structured output mode (JSON tables / key-value)
- Optional mode that asks the model for structured JSON.
- Type: feat

### 15. [DONE] docs: update README to accurately reflect current capabilities
- Remove over-claim of "RAG" until Phase 2 is complete; document limitations honestly.
- Type: docs

### 16. [DONE] chore: add basic smoke-test GitHub Action
- Simple workflow that checks app.py imports and syntax.
- Type: chore

---

## Phase 4 – Retrieval Depth, Hygiene & Trust

### 17. [DONE] docs: align UI copy with honest non-RAG framing
- Update Gradio header/markdown that still says "RAG + Vision" to match README (vision-language Q&A + lightweight page selection).
- Keep Space `short_description` accurate; avoid implying vector/embedding RAG.
- Type: docs

### 18. [DONE] fix: enforce a global max-page budget across all uploads
- Today `max_pages` is applied per PDF then pages are concatenated; multiple files can exceed the intended GPU budget.
- Cap total extracted pages globally (slider still controls the budget); stop loading once the budget is reached and note truncation in sources/status.
- Type: fix

### 19. [DONE] chore: add MIT LICENSE and pin critical dependency versions
- Add a root `LICENSE` file if missing (README claims MIT).
- Pin or tightly bound versions for `transformers`, `torch`/`torchvision`, `gradio`, `qwen-vl-utils`, `pdf2image`, `Pillow` for reproducible Space builds; leave a short comment on intentional floats if any remain.
- Type: chore

### 20. [DONE] chore: align smoke-test Python version with Space metadata
- README / Space metadata uses Python 3.12; smoke workflow uses 3.11 — make them match.
- Type: chore

### 21. [DONE] fix: soften Hugging Face Space sync (avoid blind force-push)
- Review `.github/workflows/sync-to-hf.yml`; prefer a non-destructive push when possible, or document why `--force` is required and fail clearly if `HF_TOKEN` is missing.
- Do not change Space app behavior—ops only.
- Type: fix

### 22. [IN_PROGRESS] test: unit tests for pure helpers (no GPU)
- Add lightweight tests (e.g. `pytest`) for `tokenize_query`, `page_relevance_score`, `rank_pages`, `adaptive_dpi`, and validation edge cases that can run without loading the model.
- Wire tests into smoke CI (or a dedicated job) without installing the full torch/VLM stack if feasible.
- Type: chore

### 23. [TODO] refactor: split `app.py` into modules with thin Space entrypoint
- Extract e.g. constants/prompts, ingest (PDF/image load + resize), ranking, VLM message build + generate, and Gradio UI wiring.
- Keep `app.py` as the HF `app_file` entrypoint; preserve behavior and public UI.
- Type: chore

### 24. [TODO] feat: optional OCR-based page ranking
- Add an optional ranking path that OCRs downscaled page images (or extracts text cheaply) and scores question overlap against page text—not only filename/page labels.
- Keep keyword-label ranking as default/fallback for speed; gate OCR behind advanced setting or auto-use when labels are uninformative.
- Stay within ZeroGPU memory/time limits (OCR only candidates or top-N pages before final top-k).
- Type: feat

### 25. [TODO] feat: expose top-k and ranking mode in advanced settings
- UI controls for `TOP_K_PAGES` and ranking mode (label-only vs OCR/text when available).
- Reflect chosen mode and scores in the sources panel.
- Type: feat

### 26. [TODO] feat: validate and surface structured JSON output
- When structured mode is on, attempt to parse model output as JSON; on failure show a clear error and optional raw text.
- Optionally light repair (strip markdown fences) before parse; do not invent fields.
- Type: feat

### 27. [TODO] feat: content-hash file signature for page cache
- Replace path-only `file_signature` with a stable signature (e.g. size + hash of file bytes) so re-uploads of the same bytes reuse cache when paths change.
- Type: fix

### 28. [TODO] feat: citation highlight overlays on gallery pages
- When the answer cites page labels, visually emphasize matching gallery items (caption badge or border)—no requirement for pixel-level bbox overlays in this item.
- Type: feat

### 29. [TODO] feat: export structured tables / key-values
- When structured JSON parse succeeds, offer download of tables as CSV and/or key-values as JSON.
- No-op when structured mode is off or parse fails.
- Type: feat

---

## Rules for the automation
1. Always work on the latest `main`.
2. Create a branch named `<type>/<short-description>` (e.g. `fix/processor-pixel-limits`).
3. Implement **exactly one** incomplete item (lowest number first).
4. Update this PLAN.md: mark the item IN_PROGRESS then DONE after merge.
5. Open a PR with conventional commit title: `feat: ...`, `fix: ...`, `chore: ...`, `docs: ...`
6. Merge the PR (squash preferred) into `main`.
7. If no TODO items remain, exit gracefully and report completion.
