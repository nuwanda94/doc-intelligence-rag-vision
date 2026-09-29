import spaces
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from docintel.citations import (
    apply_citation_highlights,
    cited_source_indices,
    extract_cited_page_numbers,
    source_is_cited,
)
from docintel.constants import (
    ALLOWED_EXTENSIONS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_RANKING_MODE,
    DEFAULT_TEMPERATURE,
    DPI_HIGH,
    DPI_LOW,
    DPI_PAGE_THRESHOLD,
    GPU_DURATION_SECONDS,
    MAX_FILE_SIZE_BYTES,
    MAX_IMAGE_SIDE,
    MAX_MAX_NEW_TOKENS,
    MAX_PAGES_DEFAULT,
    MAX_PAGES_SLIDER_MAX,
    MAX_PIXELS,
    MIN_MAX_NEW_TOKENS,
    MIN_PIXELS,
    MODEL_ID,
    OCR_CANDIDATE_LIMIT,
    QUEUE_MAX_SIZE,
    RANKING_MODE_AUTO,
    RANKING_MODE_LABEL,
    RANKING_MODE_OCR,
    RANKING_MODE_UI,
    STATUS_DONE,
    STATUS_GENERATING,
    STATUS_IDLE,
    STATUS_PREPARING,
    STRUCTURED_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TOP_K_PAGES,
    TOP_K_SLIDER_MAX,
    TOP_K_SLIDER_MIN,
    TOP_P,
)
from docintel.ingest import (
    adaptive_dpi,
    file_path as _file_path,
    file_signature,
    load_pages_from_paths,
    pdf_to_images,
    prepare_image,
    validate_uploads,
)
from docintel.pipeline import make_chat_analyze
from docintel.ranking import (
    build_gallery_and_sources,
    label_sent_page,
    labels_are_informative,
    normalize_ranking_mode,
    ocr_page_text,
    page_relevance_score,
    page_text_relevance_score,
    rank_pages,
    tokenize_query,
)
from docintel.ui import build_demo, clear_workspace, launch
from docintel.vlm import build_vlm_messages, message_text

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

chat_analyze = make_chat_analyze(spaces, processor, model)
demo = build_demo(chat_analyze)

if __name__ == "__main__":
    launch(demo)
