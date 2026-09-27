import spaces
import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import os
from typing import List, Optional

# -----------------------------
# Model Loading (module level for ZeroGPU)
# -----------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
MAX_IMAGE_SIDE = 1280
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB per file

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


def _file_path(file_obj) -> str:
    path = file_obj.name if hasattr(file_obj, "name") else file_obj
    if not path or not isinstance(path, str):
        raise gr.Error("Could not read an uploaded file. Please try uploading again.")
    return path


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


def pdf_to_images(pdf_path: str, max_pages: int = 8, dpi: int = 150) -> List[Image.Image]:
    """Convert PDF pages to images (limited for free tier)."""
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

@spaces.GPU(duration=120)
def analyze_document(
    files: Optional[List],
    question: str,
    max_pages: int = 6,
    temperature: float = 0.3,
    max_new_tokens: int = 512
):
    paths = validate_uploads(files, question)

    page_images = []
    source_info = []

    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)

        if ext == ".pdf":
            imgs = pdf_to_images(path, max_pages=max_pages)
            page_images.extend(imgs)
            source_info.extend([f"{name} — page {i+1}" for i in range(len(imgs))])
        else:
            try:
                img = prepare_image(Image.open(path))
            except Exception as e:
                raise gr.Error(f"Failed to open image '{name}': {str(e)}")
            if img.size[0] == 0 or img.size[1] == 0:
                raise gr.Error(f"Image '{name}' has zero width or height.")
            page_images.append(img)
            source_info.append(name)

    if not page_images:
        raise gr.Error("No pages could be extracted from the uploaded files.")

    # Build multimodal messages (Qwen2.5-VL style)
    content = []
    for img in page_images:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": question})

    messages = [
        {
            "role": "system",
            "content": "You are an expert document intelligence assistant. Answer questions accurately based only on the provided document pages/images. Cite page numbers when possible. Extract tables, numbers, and key entities precisely. If information is missing, say so clearly."
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

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Simple gallery of source pages
    gallery = [(img, src) for img, src in zip(page_images, source_info)]

    sources_text = "\n".join([f"- {s}" for s in source_info])
    return answer, gallery, sources_text

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
                max_pages = gr.Slider(1, 10, value=6, step=1, label="Max PDF pages to process")
                temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")
                max_tokens = gr.Slider(128, 1024, value=512, step=64, label="Max new tokens")

            submit_btn = gr.Button("Analyze Document", variant="primary", size="lg")

        with gr.Column(scale=1):
            answer = gr.Textbox(label="Answer", lines=12)          # ← fixed
            sources = gr.Textbox(label="Sources used", lines=4)
            gallery = gr.Gallery(label="Document Pages", columns=2, height=400, object_fit="contain")

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
        outputs=[answer, gallery, sources]
    )

    gr.Markdown("""
    ---
    **Tech**: Qwen2.5-VL-3B-Instruct · Gradio · ZeroGPU  
    **Limitations**: Free tier processes up to ~8 pages. Complex multi-document RAG can be added later.
    """)

if __name__ == "__main__":
    demo.queue(max_size=10).launch()
