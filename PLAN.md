# Implementation Plan – Multimodal Document Intelligence Tool

Ordered backlog of improvements. Each item is designed to be implemented in a single focused PR.
Automation picks the next incomplete item (status: TODO), implements it, opens a PR, and merges to main.

## Status Legend
- TODO
- IN_PROGRESS
- DONE

---

## Phase 1 – Robustness & Quick Wins (High Priority)

### 1. [TODO] fix: remove unused dependencies
- Remove `faiss-cpu` and `sentence-transformers` from requirements.txt (they are not used).
- Keep only what is actually imported/used.
- Type: chore

### 2. [TODO] fix: add processor pixel limits to prevent OOM
- When creating AutoProcessor, set:
  min_pixels=256*28*28
  max_pixels=1280*28*28
- Type: fix

### 3. [TODO] feat: resize and compress images before inference
- After loading each page/image, resize so max side ≤ 1280 px.
- Convert to RGB and optionally compress.
- Type: feat

### 4. [TODO] fix: improve input validation and error messages
- Validate file types explicitly and return clear Gradio errors for unsupported files.
- Guard against empty uploads, zero pages, oversized files.
- Type: fix

### 5. [TODO] feat: adaptive DPI for PDF conversion
- Lower DPI for higher page counts (e.g. 120 for >4 pages, 150 otherwise).
- Expose or keep simple heuristic.
- Type: feat

### 6. [TODO] feat: stream the model answer
- Use Gradio streaming / yield partial tokens from model.generate.
- Type: feat

### 7. [TODO] fix: improve system prompt for citations and tables
- Strengthen system prompt to require page citations and precise table/chart extraction.
- Type: fix

### 8. [TODO] feat: highlight only relevant source pages in gallery
- Ideally rank or at least document which pages were sent; improve gallery labeling.
- Type: feat

### 9. [TODO] chore: clean dead code and magic numbers
- Remove unused `tempfile` import.
- Extract constants (MAX_PAGES_DEFAULT, DPI, etc.).
- Type: chore

### 10. [TODO] feat: better loading states and Clear button
- Add clear button and improve progress/loading UX.
- Type: feat

---

## Phase 2 – Real Multimodal RAG (Core Differentiator)

### 11. [TODO] feat: simple page relevance ranking
- After converting pages to images, compute a cheap relevance score (keyword overlap or simple embedding) against the question and keep top-k pages.
- Type: feat

### 12. [TODO] feat: integrate page ranking into the VLM pipeline
- Only send the top-k ranked pages to the VLM.
- Update sources display accordingly.
- Type: feat

### 13. [TODO] feat: multi-turn conversation support
- Convert UI to chat-style with state for previous images + history.
- Type: feat

---

## Phase 3 – Production Polish

### 14. [TODO] feat: structured output mode (JSON tables / key-value)
- Optional mode that asks the model for structured JSON.
- Type: feat

### 15. [TODO] docs: update README to accurately reflect current capabilities
- Remove over-claim of "RAG" until Phase 2 is complete; document limitations honestly.
- Type: docs

### 16. [TODO] chore: add basic smoke-test GitHub Action
- Simple workflow that checks app.py imports and syntax.
- Type: chore

---

## Rules for the automation
1. Always work on the latest `main`.
2. Create a branch named `<type>/<short-description>` (e.g. `fix/processor-pixel-limits`).
3. Implement **exactly one** incomplete item (lowest number first).
4. Update this PLAN.md: mark the item IN_PROGRESS then DONE after merge.
5. Open a PR with conventional commit title: `feat: ...`, `fix: ...`, `chore: ...`, `docs: ...`
6. Merge the PR (squash preferred) into `main`.
7. If no TODO items remain, exit gracefully and report completion.
