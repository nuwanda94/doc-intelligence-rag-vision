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
