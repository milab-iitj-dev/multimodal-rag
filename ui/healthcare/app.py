"""
Healthcare MRAG — Professional Inference UI (Gradio 6.x).

Single-page browser-based inference interface that wraps the existing
RAG pipeline without modifying any backend code.

Architecture:
    Browser → Gradio → inference_fn() → RAGVQAPipeline.run_single() → display

The UI collects inputs, calls the pipeline, and formats outputs.
All retrieval, context building, and generation logic lives in the
existing src/ and pipelines/ packages — untouched.

Usage:
    # Standalone (for development)
    python -m ui.app --model-config configs/model_config.yaml ...

    # Via launcher (recommended)
    python scripts/launch_ui.py --share
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.theme import create_theme, CUSTOM_CSS
from ui.formatters import (
    detect_query_mode,
    format_answer,
    format_evidence,
    format_timing,
    format_error,
)


# ================================================================
# Header HTML — premium compact header
# ================================================================
HEADER_HTML = """
<div class="mrag-header">
    <div class="mrag-header-left">
        <div class="mrag-logo">M</div>
        <div class="mrag-header-text">
            <h1><span>MRAG</span></h1>
            <p>Multimodal Retrieval-Augmented Generation</p>
        </div>
    </div>
    <div class="mrag-status">
        <span class="mrag-status-dot"></span>
        Ready
    </div>
</div>
"""


# ================================================================
# Application Builder
# ================================================================

def create_app(pipeline=None, gpu_name: Optional[str] = None) -> tuple:
    """
    Build the Gradio Blocks application.

    Args:
        pipeline: Loaded RAGVQAPipeline instance (or None for UI-only testing).
        gpu_name: GPU device name for the timing footer.

    Returns:
        Tuple of (gr.Blocks app, dict of launch kwargs).
        Callers should do: app.launch(**launch_kwargs, ...)
    """

    theme = create_theme()

    # Gradio 6.x: theme and css are launch() parameters
    launch_kwargs = {
        "theme": theme,
        "css": CUSTOM_CSS,
    }

    with gr.Blocks(
        title="Healthcare MRAG — Chest X-Ray Analysis",
    ) as app:

        # ── Header ────────────────────────────────────────
        gr.HTML(HEADER_HTML)

        with gr.Row(elem_classes=["main-row"]):

            # ══════════════════════════════════════════════
            # LEFT COLUMN — Input Panel
            # ══════════════════════════════════════════════
            with gr.Column(scale=4, elem_classes=["input-panel"]):
                gr.HTML('<div class="section-hdr">Input</div>')

                # Image upload
                image_input = gr.Image(
                    label="Upload Chest X-Ray",
                    type="pil",
                    sources=["upload", "clipboard"],
                    elem_classes=["upload-area"],
                    height=200,
                )

                # Clinical question
                gr.HTML('<div class="section-hdr" style="margin-top: 6px;">Clinical Question</div>')
                question_input = gr.Textbox(
                    placeholder="e.g., Are there signs of pneumothorax on the right side?",
                    lines=3,
                    max_lines=5,
                    show_label=False,
                    elem_classes=["question-box"],
                )

                # Mode badge (auto-updated)
                mode_display = gr.HTML(
                    value=detect_query_mode(None, ""),
                    elem_id="mode-badge",
                )

                # Buttons
                with gr.Row(elem_classes=["btn-row"]):
                    analyze_btn = gr.Button(
                        "⚙ Analyze",
                        variant="primary",
                        scale=2,
                        elem_classes=["analyze-btn"],
                    )
                    clear_btn = gr.Button(
                        "↺ Clear",
                        variant="secondary",
                        scale=1,
                        elem_classes=["clear-btn"],
                    )

            # ══════════════════════════════════════════════
            # RIGHT COLUMN — Output Panel
            # ══════════════════════════════════════════════
            with gr.Column(scale=6, elem_classes=["output-panel"]):
                gr.HTML('<div class="section-hdr">Results</div>')

                # Answer card (most prominent)
                answer_output = gr.HTML(
                    value=_initial_placeholder(),
                    elem_id="answer-card",
                )

                # Retrieved evidence (collapsible)
                with gr.Accordion(
                    "Retrieved Evidence",
                    open=False,
                    elem_classes=["evidence-section"],
                ):
                    evidence_output = gr.HTML(
                        value="",
                        elem_id="evidence-panel",
                    )

                # Timing footer (subtle)
                timing_output = gr.HTML(
                    value="",
                    elem_id="timing-footer",
                )

        # ── Event Handlers ────────────────────────────────

        def on_input_change(image, question):
            """Update mode badge when inputs change."""
            return detect_query_mode(image, question)

        def on_analyze(image, question, progress=gr.Progress()):
            """
            Main inference callback.

            Thin wrapper: collects inputs → calls pipeline → formats output.
            """
            # Validate inputs
            has_image = image is not None
            has_text = bool(question and question.strip())

            if not has_image and not has_text:
                return (
                    format_error("Please provide an image, a question, or both."),
                    "",
                    "",
                )

            if pipeline is None:
                return (
                    format_error(
                        "Pipeline not loaded. Launch with: "
                        "python scripts/launch_ui.py --model-config configs/model_config.yaml"
                    ),
                    "",
                    "",
                )

            # Default question for image-only mode
            query = question.strip() if has_text else (
                "Describe the key findings visible in this chest X-ray."
            )

            try:
                progress(0.1, desc="Starting analysis...")
                progress(0.3, desc="Retrieving similar cases...")

                # ── Call the existing pipeline — ZERO backend changes ──
                output = pipeline.run_single(
                    query=query,
                    query_image=image if has_image else None,
                )

                progress(0.9, desc="Formatting results...")

                # Compute average relevance score from retrieved docs
                avg_score = 0.0
                if output.retrieved_docs:
                    avg_score = sum(d.score for d in output.retrieved_docs) / len(output.retrieved_docs)

                # ── Format for display ──
                answer_html = format_answer(output.answer, relevance_score=avg_score)
                evidence_html = format_evidence(output.retrieved_docs)
                timing_html = format_timing(
                    retrieval_time=output.retrieval_time_sec,
                    generation_time=output.generation_time_sec,
                    total_time=output.total_time_sec,
                    gpu_name=gpu_name,
                )

                progress(1.0, desc="Done")
                return answer_html, evidence_html, timing_html

            except Exception as e:
                tb = traceback.format_exc()
                print(f"[MRAG UI] Error during inference:\n{tb}")
                return (
                    format_error(f"{type(e).__name__}: {str(e)}"),
                    "",
                    "",
                )

        def on_clear():
            """Reset all inputs and outputs."""
            return (
                None,                            # image
                "",                              # question
                detect_query_mode(None, ""),     # mode badge
                _initial_placeholder(),          # answer
                "",                              # evidence
                "",                              # timing
            )

        # ── Wire events ──

        # Auto-update mode badge on input changes
        image_input.change(
            fn=on_input_change,
            inputs=[image_input, question_input],
            outputs=[mode_display],
        )
        question_input.blur(
            fn=on_input_change,
            inputs=[image_input, question_input],
            outputs=[mode_display],
        )

        # Analyze button
        analyze_btn.click(
            fn=on_analyze,
            inputs=[image_input, question_input],
            outputs=[answer_output, evidence_output, timing_output],
        )

        # Clear button
        clear_btn.click(
            fn=on_clear,
            inputs=[],
            outputs=[
                image_input, question_input, mode_display,
                answer_output, evidence_output, timing_output,
            ],
        )

    return app, launch_kwargs


# ================================================================
# Placeholder
# ================================================================

def _initial_placeholder() -> str:
    """Shown in the output panel before any inference."""
    return (
        '<div class="output-placeholder">'
        '  <div class="placeholder-icon-wrap">🩻</div>'
        '  <div class="placeholder-title">Ready for Analysis</div>'
        '  <div class="placeholder-sub">'
        '    Upload a chest X-ray and enter a clinical question,<br>'
        '    then click <strong>Analyze</strong> to begin.'
        '  </div>'
        '</div>'
    )


# ================================================================
# Standalone launch (for development / testing)
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Healthcare MRAG UI (dev mode)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Healthcare MRAG — UI (Development Mode)")
    print("  Pipeline NOT loaded — UI layout preview only.")
    print("=" * 60)

    app, launch_kwargs = create_app(pipeline=None)
    app.queue(max_size=5)
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        **launch_kwargs,
    )
