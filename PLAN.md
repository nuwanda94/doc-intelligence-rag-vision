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

### 7. [DONE] feat: improve system prompt for citations and tables
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
- Optional mode that asks the model for JSON.
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

### 22. [DONE] test: unit tests for pure helpers (no GPU)
- Add lightweight tests (e.g. `pytest`) for `tokenize_query`, `page_relevance_score`, `rank_pages`, `adaptive_dpi`, and validation edge cases that can run without loading the model.
- Wire tests into smoke CI (or a dedicated job) without installing the full torch/VLM stack if feasible.
- Type: chore

### 23. [DONE] refactor: split `app.py` into modules with thin Space entrypoint
- Extract e.g. constants/prompts, ingest (PDF/image load + resize), ranking, VLM message build + generate, and Gradio UI wiring.
- Keep `app.py` as the HF `app_file` entrypoint; preserve behavior and public UI.
- Type: chore

### 24. [DONE] feat: optional OCR-based page ranking
- Add an optional ranking path that OCRs downscaled page images (or extracts text cheaply) and scores question overlap against page text—not only filename/page labels.
- Keep keyword-label ranking as default/fallback for speed; gate OCR behind advanced setting or auto-use when labels are uninformative.
- Stay within ZeroGPU memory/time limits (OCR only candidates or top-N pages before final top-k).
- Type: feat

### 25. [DONE] feat: expose top-k and ranking mode in advanced settings
- UI controls for `TOP_K_PAGES` and ranking mode (label-only vs OCR/text when available).
- Reflect chosen mode and scores in the sources panel.
- Type: feat

### 26. [DONE] feat: validate and surface structured JSON output
- When structured mode is on, attempt to parse model output as JSON; on failure show a clear error and optional raw text.
- Optionally light repair (strip markdown fences) before parse; do not invent fields.
- Type: feat

### 27. [DONE] feat: content-hash file signature for page cache
- Replace path-only `file_signature` with a stable signature (e.g. size + hash of file bytes) so re-uploads of the same bytes reuse cache when paths change.
- Type: fix

### 28. [DONE] feat: citation highlight overlays on gallery pages
- When the answer cites page labels, visually emphasize matching gallery items (caption badge or border)—no requirement for pixel-level bbox overlays in this item.
- Type: feat

### 29. [DONE] feat: export structured tables / key-values
- When structured JSON parse succeeds, offer download of tables as CSV and/or key-values as JSON.
- No-op when structured mode is off or parse fails.
- Type: feat

---

## Phase 5 – Correctness, Quota & Honest Docs (Staff review, 2026-09-29)

Review of `main` after items 1–29. Highest-leverage remaining work is **accuracy of user-facing docs**, **not billing CPU work to ZeroGPU**, and **false positives** in truncation/citations—not more UI chrome.

### 30. [DONE] docs: refresh README to match shipped behavior
- README still claims ranking is filename/page labels only and that there is no OCR of page pixels; OCR ranking (item 24), citation gallery badges (item 28), and CSV/JSON export (item 29) already ship.
- Point constants at `docintel/constants.py` (not `app.py`).
- Remove or rewrite "Future ideas" that are already implemented (citation overlays, CSV export).
- Type: docs

### 31. [DONE] fix: keep ingest/ranking off the ZeroGPU decorator
- `@spaces.GPU` wraps all of `chat_analyze`, including validation, SHA hashing, pdf2image, Tesseract, and ranking.
- Split CPU prepare (pages + rank + messages) from GPU generate/stream so quota and the 120s duration cover only `model.generate`.
- Preserve streaming yields and `doc_state` cache behavior.
- Type: fix

### 32. [DONE] fix: stop flagging exact-length PDFs as truncated
- `load_pages_from_paths` treats `len(imgs) == requested` as truncation, so a 6-page PDF with max_pages=6 is labeled "Possibly truncated".
- Detect remaining pages (pdfinfo / page count, or convert `requested+1` and drop the extra) and only note truncation when pages were actually omitted.
- Add a unit test for equal-length vs over-budget PDFs.
- Type: fix

### 33. [DONE] fix: citation highlights must not cross files on page number alone
- `source_is_cited` treats any answer mention of `page N` as a hit for every source whose label contains that number, so two PDFs both get page 2 highlighted.
- Prefer full source-label / filename match; use bare page-number fallback only for a single-file session or when the filename token is also present.
- Extend `tests/test_citations.py`.
- Type: fix

### 34. [DONE] feat: rank digital PDFs with the native text layer before OCR
- Tesseract on downscaled rasters is slow and lossy when `pdftotext` / pypdf already has a text layer.
- Extract per-page PDF text when present; score that like OCR text; keep Tesseract for image-only / empty text-layer pages.
- Bound extraction to the same candidate limit; cache on `doc_state`; no extra GPU use.
- Type: feat

### 35. [IN_PROGRESS] fix: length-normalize text ranking scores
- `page_text_relevance_score` adds raw overlapping token counts, so wordy pages beat short relevant pages.
- Use Jaccard (or overlap / sqrt(|page tokens|)) plus the existing label/page-ref bonuses.
- Keep label-only ranking unchanged; update OCR/text-mode tests.
- Type: fix

### 36. [TODO] chore: compile the whole package in CI and test modules directly
- Smoke `py_compile` lists files by hand and omits `docintel/citations.py`.
- Compile `app.py` + `docintel/` recursively; prefer `from docintel.*` in tests over importing `app.py` (stubs can stay for anything that still needs the entrypoint).
- Type: chore

### 37. [TODO] feat: allow stopping generation and surface GPU time budget
- Users cannot cancel a hung stream; ZeroGPU `duration=120` is invisible until the decorator kills the run.
- Wire Gradio cancel / a Stop control on the generate thread, and show remaining/elapsed budget in status while streaming.
- Type: feat

### 38. [TODO] feat: compare two documents in one question
- Multi-file upload concatenates pages into one ranked pool with no compare prompt.
- Add an optional compare mode (or a dedicated example path) that labels sources by document, sends top-k per file, and asks the VLM to contrast them with per-file citations.
- Stay within the global page budget and top-k cap.
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
