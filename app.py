import spaces
import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path
from PIL import Image
import os
import re
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Model Loading (module level for ZeroGPU)
# -----------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
MAX_IMAGE_SIDE = 1280
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB per file
DPI_HIGH = 150  # used when processing few pages
DPI_LOW = 120   # used when processing many pages (reduces memory / latency)
DPI_PAGE_THRESHOLD = 4  # switch to DPI_LOW when max_pages exceeds this
MAX_PAGES_DEFAULT = 6
MAX_PAGES_SLIDER_MAX = 10
TOP_K_PAGES = 4  # keep the highest-scoring pages after cheap ranking
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_NEW_TOKENS = 512
MIN_MAX_NEW_TOKENS = 128
MAX_MAX_NEW_TOKENS = 1024
TOP_P = 0.9
GPU_DURATION_SECONDS = 120
QUEUE_MAX_SIZE = 10
MAX_HISTORY_TURNS = 8  # prior user/assistant pairs kept in the VLM prompt
STATUS_IDLE = "Ready — upload a document and start a conversation."
STATUS_PREPARING = "Preparing pages and allocating GPU… this can take a minute on a cold start."
STATUS_GENERATING = "Generating answer…"
STATUS_DONE = "Done. Ask a follow-up or clear to start over."

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PAGE_REF_RE = re.compile(r"\bpage\s*(\d+)\b", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at", "to", "for",
    "from", "with", "by", "is", "are", "was", "were", "be", "been", "this", "that",
    "these", "those", "it", "its", "as", "about", "into", "over", "under", "what",
    "which", "who", "how", "why", "when", "where", "please", "show", "tell",
    "give", "me", "my", "your", "you", "we", "our", "document", "page", "pages",
    "pdf", "image", "file",
})

SYSTEM_PROMPT = (
    "You are an expert document intelligence assistant. "
    "Answer using only the provided document pages/images. Do not invent facts.\n\n"
    "Citations:\n"
    "- Cite the source page for every claim using the labels shown with the images "
    "(e.g. [page 2] or [filename — page 3]).\n"
    "- If several pages support a point, list each relevant page.\n"
    "- If the answer cannot be found in the provided pages, say so clearly and do not guess.\n\n"
    "Tables and charts:\n"
    "- Reproduce table values exactly (numbers, units, column/row headers). "
    "Prefer a Markdown table when extracting tabular data.\n"
    "- For charts and diagrams, state axes, units, legends, and the specific data points or trends asked about.\n"
    "- Preserve currency symbols, percentages, and significant figures as written.\n\n"
    "Multi-turn:\n"
    "- Use earlier questions and answers as context, but ground every new claim in the current pages.\n"
    "- If a follow-up refers to 'that', 'the table', or a previous figure, resolve it from history "
    "and still cite the page.\n\n"
    "Be concise but complete. Lead with the direct answer, then supporting evidence with citations."
)

STRUCTURED_SYSTEM_PROMPT = (
    "You are an expert document intelligence assistant. "
    "Answer using only the provided document pages/images. Do not invent facts.\n\n"
    "Output format (mandatory):\n"
    "Respond with a single JSON object only. No markdown fences, no prose outside JSON.\n"
    "Use this schema:\n"
    "{\n"
    '  "answer": "short direct answer string",\n'
    '  "key_values": [{"key": "string", "value": "string", "page": "source label"}],\n'
    '  "tables": [{\n'
    '    "title": "string",\n'
    '    "page": "source label",\n'
    '    "headers": ["col1", "col2"],\n'
    '    "rows": [["cell", "cell"]]\n'
    "  }],\n"
    '  "citations": ["source labels that support the answer"]\n'
    "}\n"
    "Rules:\n"
    "- Reproduce table values exactly (numbers, units, headers).\n"
    "- If a field is unknown, use an empty list or empty string — do not guess.\n"
    "- Cite pages using the source labels shown with the images.\n"
    "- Use earlier questions and answers as context, but ground every new claim in the current pages.\n"
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

# -----------------------------
# Helpers
# -----------------------------
def prepare_image(img: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Convert to RGB and downscale so the longest side is at most max_side."""
    img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def adaptive_dpi(max_pages: int) -> int:
    """Choose PDF rasterization DPI from how many pages will be converted.\n\n    Higher page counts use a lower DPI to limit memory and conversion time\n    on the free tier, while short documents keep higher detail.\n    """
    if max_pages > DPI_PAGE_THRESHOLD:
        return DPI_LOW
    return DPI_HIGH


def _file_path(file_obj) -> str:
    path = file_obj.name if hasattr(file_obj, "name") else file_obj
    if not path or not isinstance(path, str):
        raise gr.Error("Could not read an uploaded file. Please try uploading again.")
    return path


def tokenize_query(text: str) -> set:
    """Lowercase alphanumeric tokens with stopwords removed."""
    tokens = set(_TOKEN_RE.findall((text or "").lower()))
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def page_relevance_score(source_label: str, question: str, page_index: int) -> float:
    """Cheap relevance: keyword overlap with the source label plus explicit page refs.\n\n    Images have no OCR yet, so the score uses the question text and the\n    human-readable source label (filename + page number).\n    """
    q = question or ""
    q_tokens = tokenize_query(q)
    label_tokens = tokenize_query(source_label.replace("\u2014", " ").replace("-", " ").replace(".", " "))

    overlap = len(q_tokens & label_tokens)
    score = float(overlap)

    mentioned_pages = {int(n) for n in _PAGE_REF_RE.findall(q)}
    page_nums = {int(n) for n in re.findall(r"\bpage\s*(\d+)\b", source_label, flags=re.IGNORECASE)}
    if mentioned_pages and page_nums & mentioned_pages:
        score += 5.0

    score += max(0.0, 0.05 * (1.0 / (page_index + 1)))
    return score


def rank_pages(
    page_images: List[Image.Image],
    source_info: List[str],
    question: str,
    top_k: int = TOP_K_PAGES,
) -> Tuple[List[Image.Image], List[str], List[float], List[str]]:
    """Split pages into top-k for the VLM and the remainder not sent.\n\n    Returns (selected_images, selected_labels, selected_scores, omitted_labels).\n    Selected lists are ordered by descending relevance score.\n    """
    if not page_images:
        return [], [], [], []

    scored = []
    for i, (img, src) in enumerate(zip(page_images, source_info)):
        scored.append((page_relevance_score(src, question, i), i, img, src))

    scored.sort(key=lambda row: (-row[0], row[1]))
    k = max(1, min(int(top_k), len(scored)))
    selected = scored[:k]
    omitted = scored[k:]
    images = [row[2] for row in selected]
    labels = [row[3] for row in selected]
    scores = [row[0] for row in selected]
    omitted_labels = [row[3] for row in omitted]
    return images, labels, scores, omitted_labels


def label_sent_page(src: str, index: int, total: int, score: Optional[float] = None) -> str:
    """Caption for a page that was actually sent to the VLM."""
    base = f"Sent to model \u00b7 {index}/{total} \u00b7 {src}"
    if score is None:
        return base
    return f"{base} \u00b7 score {score:.2f}"


def build_gallery_and_sources(
    page_images: List[Image.Image],
    source_info: List[str],
    scores: Optional[List[float]] = None,
    omitted_labels: Optional[List[str]] = None,
    total_extracted: Optional[int] = None,
) -> Tuple[List[Tuple[Image.Image, str]], str]:
    """Gallery of pages forwarded to the model, plus a ranking-aware source list."""
    sent = len(source_info)
    extracted = total_extracted if total_extracted is not None else sent
    gallery = []
    for i, (img, src) in enumerate(zip(page_images, source_info)):
        score = scores[i] if scores and i < len(scores) else None
        gallery.append((img, label_sent_page(src, i + 1, sent, score)))

    header = f"Pages sent to the model ({sent} of {extracted} extracted):"
    lines = [header]
    for i, src in enumerate(source_info):
        if scores and i < len(scores):
            lines.append(f"- {src} (relevance {scores[i]:.2f})")
        else:
            lines.append(f"- {src}")
    if omitted_labels:
        lines.append("")
        lines.append(f"Not sent to the model ({len(omitted_labels)}):")
        lines.extend([f"- {src}" for src in omitted_labels])
    return gallery, "\n".join(lines)


def file_signature(files: Optional[List]) -> Tuple[str, ...]:
    if not files:
        return ()
    return tuple(_file_path(f) for f in files)


def message_text(content: Any) -> str:
    """Normalize a Gradio chat message content field to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def build_vlm_messages(
    history: List[Dict[str, Any]],
    page_images: List[Image.Image],
    source_info: List[str],
    question: str,
    structured: bool = False,
) -> List[Dict[str, Any]]:
    """System + prior text turns + current user turn with ranked page images."""
    system = STRUCTURED_SYSTEM_PROMPT if structured else SYSTEM_PROMPT
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
    ]

    prior = history or []
    if MAX_HISTORY_TURNS > 0 and len(prior) > MAX_HISTORY_TURNS * 2:
        prior = prior[-(MAX_HISTORY_TURNS * 2):]

    for turn in prior:
        role = turn.get("role")
        text = message_text(turn.get("content")).strip()
        if role not in ("user", "assistant") or not text:
            continue
        messages.append({"role": role, "content": text})

    content: List[Dict[str, Any]] = []
    for img, src in zip(page_images, source_info):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[Source: {src}]"})
    user_text = question
    if structured:
        user_text = (
            question
            + "\n\nReturn only valid JSON matching the required schema "
            "(answer, key_values, tables, citations)."
        )
    content.append({"type": "text", "text": user_text})
    messages.append({"role": "user", "content": content})
    return messages


def clear_workspace():
    """Reset uploads, chat history, cached pages, outputs, and status."""
    return (
        None,
        [],
        None,
        MAX_PAGES_DEFAULT,
        DEFAULT_TEMPERATURE,
        DEFAULT_MAX_NEW_TOKENS,
        False,
        [],
        "",
        STATUS_IDLE,
        gr.update(value=""),
    )


def validate_uploads(files: Optional[List], question: str) -> List[str]:
    """Raise clear Gradio errors for empty, unsupported, empty-byte, or oversized uploads."""
    if not files:
        raise gr.Error("Please upload at least one PDF or image file.")

    if question is None or not str(question).strip():
        raise gr.Error("Please enter a question about the uploaded document.")

    validated_paths = []
    oversized = []
    unsupported = []
    empty_files = []
    missing = []

    for f in files:
        path = _file_path(f)
        name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()

        if not os.path.exists(path):
            missing.append(name or path)
            continue

        try:
            size = os.path.getsize(path)
        except OSError:
            missing.append(name or path)
            continue

        if size == 0:
            empty_files.append(name)
            continue

        if size > MAX_FILE_SIZE_BYTES:
            oversized.append(f"{name} ({size / (1024 * 1024):.1f} MB)")
            continue

        if ext not in ALLOWED_EXTENSIONS:
            unsupported.append(name or "(no extension)")
            continue

        validated_paths.append(path)

    if missing:
        raise gr.Error(
            "Could not read the following upload(s): "
            + ", ".join(missing)
            + ". Please re-upload the file(s)."
        )
    if empty_files:
        raise gr.Error(
            "The following file(s) are empty: "
            + ", ".join(empty_files)
            + ". Upload a non-empty PDF or image."
        )
    if oversized:
        limit_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise gr.Error(
            f"File(s) exceed the {limit_mb:.0f} MB limit: "
            + ", ".join(oversized)
            + ". Compress the file or upload a smaller document."
        )
    if unsupported:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise gr.Error(
            "Unsupported file type(s): "
            + ", ".join(unsupported)
            + f". Allowed types: {allowed}."
        )
    if not validated_paths:
        raise gr.Error("No valid PDF or image files found in the upload.")

    return validated_paths


def pdf_to_images(pdf_path: str, max_pages: int = MAX_PAGES_DEFAULT, dpi: Optional[int] = None) -> List[Image.Image]:
    """Convert PDF pages to images (limited for free tier).\n\n    DPI is chosen adaptively from max_pages when not provided:\n    150 for short docs (≤ DPI_PAGE_THRESHOLD pages), 120 for longer ones.\n    """
    if dpi is None:
        dpi = adaptive_dpi(max_pages)
    try:
        images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=max_pages)
    except Exception as e:
        raise gr.Error(f"Failed to process PDF '{os.path.basename(pdf_path)}': {str(e)}")

    if not images:
        raise gr.Error(
            f"PDF '{os.path.basename(pdf_path)}' has no readable pages. "
            "Upload a PDF that contains at least one page."
        )
    return [prepare_image(img) for img in images]


def load_pages_from_paths(paths: List[str], max_pages: int) -> Tuple[List[Image.Image], List[str]]:
    all_images: List[Image.Image] = []
    all_sources: List[str] = []
    dpi = adaptive_dpi(max_pages)

    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)

        if ext == ".pdf":
            imgs = pdf_to_images(path, max_pages=max_pages, dpi=dpi)
            all_images.extend(imgs)
            all_sources.extend([f"{name} — page {i+1}" for i in range(len(imgs))])
        else:
            try:
                img = prepare_image(Image.open(path))
            except Exception as e:
                raise gr.Error(f"Failed to open image '{name}': {str(e)}")
            if img.size[0] == 0 or img.size[1] == 0:
                raise gr.Error(f"Image '{name}' has zero width or height.")
            all_images.append(img)
            all_sources.append(name)

    if not all_images:
        raise gr.Error("No pages could be extracted from the uploaded files.")
    return all_images, all_sources


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
):
    """One chat turn: reuse cached pages when possible, re-rank, stream the reply."""
    history = list(history or [])
    question = (message or "").strip()
    pending = history + [{"role": "user", "content": question}]

    yield STATUS_PREPARING, pending + [{"role": "assistant", "content": "Preparing document\u2026"}], doc_state, None, "", gr.update(value="")

    sig = file_signature(files)
    cached_sig = (doc_state or {}).get("file_sig")
    if doc_state is None or not doc_state.get("images") or cached_sig != sig:
        paths = validate_uploads(files, question)
        all_images, all_sources = load_pages_from_paths(paths, max_pages)
        doc_state = {
            "images": all_images,
            "sources": all_sources,
            "file_sig": sig,
        }
    elif not question:
        raise gr.Error("Please enter a question about the uploaded document.")

    all_images = doc_state["images"]
    all_sources = doc_state["sources"]

    page_images, source_info, scores, omitted_labels = rank_pages(
        all_images, all_sources, question, top_k=TOP_K_PAGES
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
    final_history = pending + [{"role": "assistant", "content": partial}]
    yield STATUS_DONE, final_history, doc_state, gallery, sources_text, gr.update(value="")


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(
    title="Multimodal Document Intelligence",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css="""
    .gradio-container {max-width: 1100px !important}
    """
) as demo:
    gr.Markdown("""
    # 📄 Multimodal Document Intelligence
    **RAG + Vision powered by Qwen2.5-VL-3B**

    Upload PDFs or images, then chat about content, tables, charts, diagrams, or scanned text.
    Follow-up questions reuse the extracted pages and prior answers.
    """)

    status = gr.Markdown(STATUS_IDLE)
    doc_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(
                label="Upload PDF(s) or Images",
                file_count="multiple",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"]
            )
            chatbot = gr.Chatbot(
                label="Conversation",
                height=420,
                type="messages",
            )
            question = gr.Textbox(
                label="Your message",
                placeholder="Ask about the document, then follow up (e.g. 'now extract the table on page 2').",
                lines=3,
            )
            with gr.Accordion("Advanced Settings", open=False):
                max_pages = gr.Slider(1, MAX_PAGES_SLIDER_MAX, value=MAX_PAGES_DEFAULT, step=1, label="Max PDF pages to process")
                temperature = gr.Slider(0.0, 1.0, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature")
                max_tokens = gr.Slider(MIN_MAX_NEW_TOKENS, MAX_MAX_NEW_TOKENS, value=DEFAULT_MAX_NEW_TOKENS, step=64, label="Max new tokens")
                structured_output = gr.Checkbox(
                    label="Structured JSON output",
                    value=False,
                    info="Ask the model for JSON with answer, key-value pairs, tables, and citations.",
                )

            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", variant="secondary", size="lg")

        with gr.Column(scale=1):
            sources = gr.Textbox(label="Pages sent to the model", lines=6)
            gallery = gr.Gallery(
                label="Pages sent to the model",
                columns=2,
                height=400,
                object_fit="contain",
            )

    gr.Examples(
        examples=[
            [None, "Summarize the main points of this document."],
            [None, "Extract all numerical values and tables."],
            [None, "What does the chart/diagram show?"],
        ],
        inputs=[files, question],
        label="Example questions (upload your own files)"
    )

    submit_btn.click(
        fn=chat_analyze,
        inputs=[question, chatbot, files, doc_state, max_pages, temperature, max_tokens, structured_output],
        outputs=[status, chatbot, doc_state, gallery, sources, question],
        show_progress="full",
    )
    question.submit(
        fn=chat_analyze,
        inputs=[question, chatbot, files, doc_state, max_pages, temperature, max_tokens, structured_output],
        outputs=[status, chatbot, doc_state, gallery, sources, question],
        show_progress="full",
    )

    clear_btn.click(
        fn=clear_workspace,
        inputs=None,
        outputs=[files, chatbot, doc_state, max_pages, temperature, max_tokens, structured_output, gallery, sources, status, question],
    )

    gr.Markdown(f"""
    ---
    **Tech**: Qwen2.5-VL-3B-Instruct · Gradio · ZeroGPU  
    **Limitations**: Free tier processes up to ~{MAX_PAGES_SLIDER_MAX} pages. Follow-ups reuse cached pages from the current upload.
    """)

if __name__ == "__main__":
    demo.queue(max_size=QUEUE_MAX_SIZE).launch()
