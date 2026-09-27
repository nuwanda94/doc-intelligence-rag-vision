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
from typing import List, Optional, Tuple

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
STATUS_IDLE = "Ready — upload a document and ask a question."
STATUS_PREPARING = "Preparing pages and allocating GPU… this can take a minute on a cold start."
STATUS_GENERATING = "Generating answer…"
STATUS_DONE = "Done."

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
    "Be concise but complete. Lead with the direct answer, then supporting evidence with citations."
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
    """Choose PDF rasterization DPI from how many pages will be converted.

    Higher page counts use a lower DPI to limit memory and conversion time
    on the free tier, while short documents keep higher detail.
    """
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
    """Cheap relevance: keyword overlap with the source label plus explicit page refs.

    Images have no OCR yet, so the score uses the question text and the
    human-readable source label (filename + page number).
    """
    q = question or ""
    q_tokens = tokenize_query(q)
    label_tokens = tokenize_query(source_label.replace("—", " ").replace("-", " ").replace(".", " "))

    overlap = len(q_tokens & label_tokens)
    score = float(overlap)

    mentioned_pages = {int(n) for n in _PAGE_REF_RE.findall(q)}
    # source labels use 1-based page numbers: "report.pdf — page 3"
    page_nums = {int(n) for n in re.findall(r"\bpage\s*(\d+)\b", source_label, flags=re.IGNORECASE)}
    if mentioned_pages and page_nums & mentioned_pages:
        score += 5.0

    # Slight recency-of-order prior so ties keep earlier pages first after sort.
    score += max(0.0, 0.05 * (1.0 / (page_index + 1)))
    return score


def rank_pages(
    page_images: List[Image.Image],
    source_info: List[str],
    question: str,
    top_k: int = TOP_K_PAGES,
) -> Tuple[List[Image.Image], List[str], List[float], List[str]]:
    """Split pages into top-k for the VLM and the remainder not sent.

    Returns (selected_images, selected_labels, selected_scores, omitted_labels).
    Selected lists are ordered by descending relevance score.
    """
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
    base = f"Sent to model · {index}/{total} · {src}"
    if score is None:
        return base
    return f"{base} · score {score:.2f}"


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


def clear_workspace():
    """Reset uploads, question, outputs, and status to a clean idle state."""
    return (
        None,
        "",
        MAX_PAGES_DEFAULT,
        DEFAULT_TEMPERATURE,
        DEFAULT_MAX_NEW_TOKENS,
        "",
        [],
        "",
        STATUS_IDLE,
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
            unsupported.append(name or f"(no extension)")
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
    """Convert PDF pages to images (limited for free tier).

    DPI is chosen adaptively from max_pages when not provided:
    150 for short docs (≤ DPI_PAGE_THRESHOLD pages), 120 for longer ones.
    """
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

@spaces.GPU(duration=GPU_DURATION_SECONDS)
def analyze_document(
    files: Optional[List],
    question: str,
    max_pages: int = MAX_PAGES_DEFAULT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
):
    yield STATUS_PREPARING, "Preparing document…", None, ""

    paths = validate_uploads(files, question)

    all_images = []
    all_sources = []
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

    # Rank every extracted page, then forward only the top-k images to the VLM.
    page_images, source_info, scores, omitted_labels = rank_pages(
        all_images, all_sources, question, top_k=TOP_K_PAGES
    )

    # Build multimodal messages from ranked pages only (Qwen2.5-VL style)
    content = []
    for img, src in zip(page_images, source_info):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"[Source: {src}]"})
    content.append({"type": "text", "text": question})

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": content
        }
    ]

    # Prepare inputs
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

    # Show sources immediately, then stream tokens into the answer box.
    yield STATUS_GENERATING, "", gallery, sources_text
    partial = ""
    for token in streamer:
        partial += token
        yield STATUS_GENERATING, partial, gallery, sources_text
    thread.join()
    if not partial:
        yield STATUS_DONE, "(No answer generated.)", gallery, sources_text
    else:
        yield STATUS_DONE, partial, gallery, sources_text

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

    Upload PDFs or images → Ask questions about content, tables, charts, diagrams, or scanned text.
    """)

    status = gr.Markdown(STATUS_IDLE)

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(
                label="Upload PDF(s) or Images",
                file_count="multiple",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"]
            )
            question = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What is the total revenue in Q3? Extract the key findings from the chart. Summarize page 2.",
                lines=3
            )
            with gr.Accordion("Advanced Settings", open=False):
                max_pages = gr.Slider(1, MAX_PAGES_SLIDER_MAX, value=MAX_PAGES_DEFAULT, step=1, label="Max PDF pages to process")
                temperature = gr.Slider(0.0, 1.0, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature")
                max_tokens = gr.Slider(MIN_MAX_NEW_TOKENS, MAX_MAX_NEW_TOKENS, value=DEFAULT_MAX_NEW_TOKENS, step=64, label="Max new tokens")

            with gr.Row():
                submit_btn = gr.Button("Analyze Document", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", variant="secondary", size="lg")

        with gr.Column(scale=1):
            answer = gr.Textbox(
                label="Answer",
                lines=12,
                placeholder="The generated answer will stream here…",
            )
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
        fn=analyze_document,
        inputs=[files, question, max_pages, temperature, max_tokens],
        outputs=[status, answer, gallery, sources],
        show_progress="full",
    )

    clear_btn.click(
        fn=clear_workspace,
        inputs=None,
        outputs=[files, question, max_pages, temperature, max_tokens, answer, gallery, sources, status],
    )

    gr.Markdown(f"""
    ---
    **Tech**: Qwen2.5-VL-3B-Instruct · Gradio · ZeroGPU  
    **Limitations**: Free tier processes up to ~{MAX_PAGES_SLIDER_MAX} pages. Complex multi-document RAG can be added later.
    """)

if __name__ == "__main__":
    demo.queue(max_size=QUEUE_MAX_SIZE).launch()
