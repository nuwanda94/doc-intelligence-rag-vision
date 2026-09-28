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
