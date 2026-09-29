from __future__ import annotations

import gradio as gr

from docintel.constants import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_RANKING_MODE,
    DEFAULT_TEMPERATURE,
    MAX_MAX_NEW_TOKENS,
    MAX_PAGES_DEFAULT,
    MAX_PAGES_SLIDER_MAX,
    MIN_MAX_NEW_TOKENS,
    QUEUE_MAX_SIZE,
    RANKING_MODE_AUTO,
    RANKING_MODE_LABEL,
    RANKING_MODE_OCR,
    RANKING_MODE_UI,
    STATUS_IDLE,
    TOP_K_PAGES,
    TOP_K_SLIDER_MAX,
    TOP_K_SLIDER_MIN,
)

_THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
_CSS = """
    .gradio-container {max-width: 1100px !important}
"""


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
        TOP_K_PAGES,
        RANKING_MODE_UI[DEFAULT_RANKING_MODE],
        [],
        "",
        STATUS_IDLE,
        gr.update(value=""),
        None,
        None,
    )


def build_demo(chat_analyze):
    # Gradio 6: theme/css belong on launch(), not Blocks(); Chatbot is messages-only.
    with gr.Blocks(title="Multimodal Document Intelligence") as demo:
        gr.Markdown("""
    # 📄 Multimodal Document Intelligence
    **Vision-language Q&A with lightweight page selection** (not embedding/vector RAG) · Qwen2.5-VL-3B

    Upload PDFs or images, then chat about content, tables, charts, diagrams, or scanned text.
    Pages are ranked with a cheap keyword score; only the top-k pages are sent to the model.
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
                )
                question = gr.Textbox(
                    label="Your message",
                    placeholder="Ask about the document, then follow up (e.g. 'now extract the table on page 2').",
                    lines=3,
                )
                with gr.Accordion("Advanced Settings", open=False):
                    max_pages = gr.Slider(1, MAX_PAGES_SLIDER_MAX, value=MAX_PAGES_DEFAULT, step=1, label="Max pages to process (all files)")
                    top_k = gr.Slider(
                        TOP_K_SLIDER_MIN,
                        TOP_K_SLIDER_MAX,
                        value=TOP_K_PAGES,
                        step=1,
                        label="Top-k pages to send to the model",
                        info="After ranking, only this many pages are forwarded to the VLM.",
                    )
                    ranking_mode = gr.Radio(
                        choices=[
                            RANKING_MODE_UI[RANKING_MODE_AUTO],
                            RANKING_MODE_UI[RANKING_MODE_LABEL],
                            RANKING_MODE_UI[RANKING_MODE_OCR],
                        ],
                        value=RANKING_MODE_UI[DEFAULT_RANKING_MODE],
                        label="Page ranking mode",
                        info="Label-only is fastest. OCR scores question overlap against page text when available.",
                    )
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
                sources = gr.Textbox(label="Pages sent to the model", lines=8)
                gallery = gr.Gallery(
                    label="Pages sent to the model",
                    columns=2,
                    height=400,
                    object_fit="contain",
                )
                tables_csv = gr.File(
                    label="Download tables (CSV)",
                    file_count="single",
                    interactive=False,
                )
                key_values_json = gr.File(
                    label="Download key-values (JSON)",
                    file_count="single",
                    interactive=False,
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

        analyze_inputs = [
            question,
            chatbot,
            files,
            doc_state,
            max_pages,
            temperature,
            max_tokens,
            structured_output,
            top_k,
            ranking_mode,
        ]
        analyze_outputs = [
            status,
            chatbot,
            doc_state,
            gallery,
            sources,
            question,
            tables_csv,
            key_values_json,
        ]

        submit_btn.click(
            fn=chat_analyze,
            inputs=analyze_inputs,
            outputs=analyze_outputs,
            show_progress="full",
        )
        question.submit(
            fn=chat_analyze,
            inputs=analyze_inputs,
            outputs=analyze_outputs,
            show_progress="full",
        )

        clear_btn.click(
            fn=clear_workspace,
            inputs=None,
            outputs=[
                files,
                chatbot,
                doc_state,
                max_pages,
                temperature,
                max_tokens,
                structured_output,
                top_k,
                ranking_mode,
                gallery,
                sources,
                status,
                question,
                tables_csv,
                key_values_json,
            ],
        )

        gr.Markdown(f"""
    ---
    **Tech**: Qwen2.5-VL-3B-Instruct · Gradio · ZeroGPU  
    **How pages are chosen**: keyword overlap on file/page labels, optional OCR text overlap — not a vector index.  
    **Limitations**: Free tier processes up to ~{MAX_PAGES_SLIDER_MAX} pages. Follow-ups reuse cached pages from the current upload.
    """)

    return demo


def launch(demo):
    demo.queue(max_size=QUEUE_MAX_SIZE).launch(theme=_THEME, css=_CSS)
