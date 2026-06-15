"""
MMRAG Unified — Combined Gradio UI for Healthcare + Scientific.

Uses the healthcare UI's premium dark theme as the base, adds a
domain selector tab so users can switch between Healthcare (chest
X-ray VQA) and Scientific (paper QA) from a single interface.

Architecture:
    Browser → Gradio → domain_tab_switch → correct pipeline → display

Usage:
    python ui/unified_app.py                # Dev mode (no pipelines)
    python ui/unified_app.py --share        # With public link
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

from ui.healthcare.theme import create_theme, CUSTOM_CSS
from ui.healthcare.formatters import (
    format_answer,
    format_evidence,
    format_timing,
    format_error,
)


# ================================================================
# Header HTML — unified version
# ================================================================
HEADER_HTML = """
<div class="mrag-header">
    <div class="mrag-header-left">
        <div class="mrag-logo">M</div>
        <div class="mrag-header-text">
            <h1><span>MMRAG</span> Unified</h1>
            <p>Healthcare + Scientific · Multimodal Retrieval-Augmented Generation</p>
        </div>
    </div>
    <div class="mrag-status">
        <span class="mrag-status-dot"></span>
        Ready
    </div>
</div>
"""


# ================================================================
# Healthcare Inference
# ================================================================

def healthcare_inference(image, question, pipeline=None, gpu_name=None):
    """Run healthcare pipeline inference."""
    has_image = image is not None
    has_text = bool(question and question.strip())

    if not has_image and not has_text:
        return (
            format_error("Please provide a chest X-ray image, a question, or both."),
            "",
            "",
        )

    if pipeline is None:
        return (
            format_error(
                "Healthcare pipeline not loaded. "
                "Launch with: python scripts/healthcare/launch_ui.py"
            ),
            "",
            "",
        )

    query = question.strip() if has_text else (
        "Describe the key findings visible in this chest X-ray."
    )

    try:
        output = pipeline.run_single(
            query=query,
            query_image=image if has_image else None,
        )

        avg_score = 0.0
        if output.retrieved_docs:
            avg_score = sum(d.score for d in output.retrieved_docs) / len(output.retrieved_docs)

        answer_html = format_answer(output.answer, relevance_score=avg_score)
        evidence_html = format_evidence(output.retrieved_docs)
        timing_html = format_timing(
            retrieval_time=output.retrieval_time_sec,
            generation_time=output.generation_time_sec,
            total_time=output.total_time_sec,
            gpu_name=gpu_name,
        )
        return answer_html, evidence_html, timing_html

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Unified UI] Healthcare error:\n{tb}")
        return format_error(f"{type(e).__name__}: {e}"), "", ""


# ================================================================
# Scientific Inference
# ================================================================

def scientific_inference(question, pipeline=None):
    """Run scientific pipeline inference."""
    if not question or not question.strip():
        return (
            _sci_card("⚠️ Please enter a question about scientific papers.", "warning"),
            _sci_confidence(0, False, False, 0, 0),
            _sci_sources([]),
        )

    if pipeline is None:
        return (
            _sci_card(
                "Scientific pipeline not loaded. "
                "Set RAG_CONFIG_PATH and ensure indexes are built.",
                "error",
            ),
            _sci_confidence(0, False, False, 0, 0),
            _sci_sources([]),
        )

    try:
        result = pipeline.query(question.strip())

        # Answer card
        if result.check_result.passed:
            ans_html = _sci_card(result.answer, "success")
        else:
            ans_html = _sci_card(result.answer, "warning")

        # Confidence
        conf_html = _sci_confidence(
            result.confidence,
            result.check_result.attribution_passed,
            result.check_result.faithfulness_passed,
            result.total_time,
            result.retries,
        )

        # Sources
        sources = []
        if hasattr(result, 'sources') and result.sources:
            for s in result.sources:
                sources.append({
                    "title": getattr(s, 'paper_title', 'Unknown'),
                    "score": getattr(s, 'relevance_score', 0),
                    "pages": getattr(s, 'page_numbers', []),
                    "arxiv_url": getattr(s, 'arxiv_url', ''),
                    "snippet": getattr(s, 'text_snippet', ''),
                })
        src_html = _sci_sources(sources)

        return ans_html, conf_html, src_html

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Unified UI] Scientific error:\n{tb}")
        return (
            _sci_card(f"Error: {e}", "error"),
            _sci_confidence(0, False, False, 0, 0),
            _sci_sources([]),
        )


def _sci_card(text: str, level: str = "success") -> str:
    """Format a scientific answer card."""
    colors = {
        "success": ("#2563eb", "#f0f9ff", "✅ Answer from Documents"),
        "warning": ("#d97706", "#fffbeb", "⚠️ Low Confidence Answer"),
        "error":   ("#dc2626", "#fef2f2", "❌ Error"),
    }
    color, bg, label = colors.get(level, colors["success"])
    safe = text.replace("\n", "<br>") if text else ""
    return (
        f'<div style="background:{bg}; border-left:4px solid {color}; '
        f'border-radius:8px; padding:16px; font-size:15px; line-height:1.6; color:#1e293b;">'
        f'<strong style="color:{color}; font-size:13px; text-transform:uppercase; '
        f'letter-spacing:0.05em;">{label}</strong>'
        f'<div style="margin-top:8px;">{safe}</div>'
        f'</div>'
    )


def _sci_confidence(confidence, attr_passed, faith_passed, total_time, retries):
    """Format scientific confidence display."""
    pct = int(confidence * 100) if confidence else 0
    col = "#16a34a" if pct > 60 else "#fb923c" if pct > 30 else "#dc2626"
    attr_pct = 100 if attr_passed else 20
    faith_pct = 100 if faith_passed else 20
    return (
        f'<div style="background:#f9fafb; border:1px solid #e5e7eb; '
        f'border-radius:10px; padding:16px;">'
        f'<div style="text-align:center; margin-bottom:12px;">'
        f'<div style="font-size:36px; font-weight:800; color:{col};">{pct}%</div>'
        f'<div style="color:#6b7280; font-size:13px;">overall confidence</div>'
        f'</div>'
        f'<div style="font-size:12px; color:#4b5563;">'
        f'<div style="margin-bottom:6px;">Attribution: '
        f'<strong style="color:{"#16a34a" if attr_pct == 100 else "#dc2626"}">{attr_pct}%</strong></div>'
        f'<div>Faithfulness: '
        f'<strong style="color:{"#16a34a" if faith_pct == 100 else "#dc2626"}">{faith_pct}%</strong></div>'
        f'</div>'
        f'<div style="color:#9ca3af; font-size:11px; margin-top:10px; '
        f'border-top:1px solid #f3f4f6; padding-top:6px; text-align:center;">'
        f'⏱ {total_time:.1f}s &nbsp;·&nbsp; 🔄 {retries} retries</div>'
        f'</div>'
    )


def _sci_sources(sources):
    """Format scientific sources panel."""
    if not sources:
        return (
            '<div style="background:#fef2f2; border:1px solid #fca5a5; '
            'border-radius:10px; padding:16px; color:#b91c1c; font-size:13px; '
            'text-align:center;">🚫 No sources found.</div>'
        )
    cards = ""
    for i, s in enumerate(sources, 1):
        score_pct = int(s["score"] * 100)
        score_col = "#16a34a" if score_pct >= 70 else "#fb923c" if score_pct >= 40 else "#2563eb"
        pages = ", ".join(str(p) for p in s["pages"])
        arxiv = (
            f'<a href="{s["arxiv_url"]}" target="_blank" '
            f'style="color:#2563eb;">🔗 arXiv</a>' if s["arxiv_url"] else ""
        )
        snippet = s["snippet"][:180].replace("\n", " ") if s["snippet"] else ""
        cards += (
            f'<div style="border:1px solid #e5e7eb; border-radius:8px; '
            f'padding:12px; margin-bottom:8px; background:white;">'
            f'<div style="display:flex; justify-content:space-between;">'
            f'<div style="font-weight:600; color:#1e3a8a; font-size:13px;">'
            f'[{i}] {s["title"][:55]}...</div>'
            f'<span style="background:{score_col}15; color:{score_col}; '
            f'padding:2px 8px; border-radius:12px; font-size:11px; '
            f'font-weight:700;">{score_pct}%</span></div>'
            f'<div style="color:#6b7280; font-size:12px; margin-top:4px;">'
            f'📖 Page {pages} &nbsp;|&nbsp; {arxiv}</div>'
            f'{f"""<div style="color:#4b5563; font-size:11px; margin-top:6px; font-style:italic;">"{snippet}..."</div>""" if snippet else ""}'
            f'</div>'
        )
    return (
        f'<div style="background:#f9fafb; border:1px solid #e5e7eb; '
        f'border-radius:10px; padding:16px;">'
        f'<div style="font-weight:700; color:#1e293b; margin-bottom:10px;">'
        f'📚 Sources ({len(sources)})</div>{cards}</div>'
    )


# ================================================================
# Application Builder
# ================================================================

def create_unified_app(
    healthcare_pipeline=None,
    scientific_pipeline=None,
    gpu_name: Optional[str] = None,
) -> tuple:
    """
    Build the unified Gradio Blocks application with tabbed domains.

    Args:
        healthcare_pipeline: Loaded healthcare RAGVQAPipeline (or None).
        scientific_pipeline: Loaded scientific OnlinePipeline (or None).
        gpu_name: GPU device name for display.

    Returns:
        Tuple of (gr.Blocks app, dict of launch kwargs).
    """
    theme = create_theme()
    launch_kwargs = {"theme": theme, "css": CUSTOM_CSS}

    with gr.Blocks(title="MMRAG Unified — Healthcare + Scientific") as app:
        gr.HTML(HEADER_HTML)

        with gr.Tabs(elem_classes=["domain-tabs"]):

            # ══════════════════════════════════════════════
            # TAB 1: Healthcare (Chest X-Ray VQA)
            # ══════════════════════════════════════════════
            with gr.Tab("🏥 Healthcare", id="healthcare"):
                with gr.Row(elem_classes=["main-row"]):

                    # Left: Input
                    with gr.Column(scale=4, elem_classes=["input-panel"]):
                        gr.HTML('<div class="section-hdr">Chest X-Ray Input</div>')

                        h_image = gr.Image(
                            label="Upload Chest X-Ray",
                            type="pil",
                            sources=["upload", "clipboard"],
                            elem_classes=["upload-area"],
                            height=200,
                        )
                        gr.HTML('<div class="section-hdr" style="margin-top:6px;">Clinical Question</div>')
                        h_question = gr.Textbox(
                            placeholder="e.g., Are there signs of pleural effusion?",
                            lines=3,
                            max_lines=5,
                            show_label=False,
                            elem_classes=["question-box"],
                        )
                        with gr.Row(elem_classes=["btn-row"]):
                            h_analyze = gr.Button("⚙ Analyze", variant="primary", scale=2, elem_classes=["analyze-btn"])
                            h_clear = gr.Button("↺ Clear", variant="secondary", scale=1, elem_classes=["clear-btn"])

                    # Right: Output
                    with gr.Column(scale=6, elem_classes=["output-panel"]):
                        gr.HTML('<div class="section-hdr">Results</div>')
                        h_answer = gr.HTML(value=_healthcare_placeholder(), elem_id="h-answer-card")
                        with gr.Accordion("Retrieved Evidence", open=False, elem_classes=["evidence-section"]):
                            h_evidence = gr.HTML(value="", elem_id="h-evidence-panel")
                        h_timing = gr.HTML(value="", elem_id="h-timing-footer")

                # Healthcare events
                h_analyze.click(
                    fn=lambda img, q: healthcare_inference(img, q, healthcare_pipeline, gpu_name),
                    inputs=[h_image, h_question],
                    outputs=[h_answer, h_evidence, h_timing],
                )
                h_clear.click(
                    fn=lambda: (None, "", _healthcare_placeholder(), "", ""),
                    outputs=[h_image, h_question, h_answer, h_evidence, h_timing],
                )

            # ══════════════════════════════════════════════
            # TAB 2: Scientific (Paper QA)
            # ══════════════════════════════════════════════
            with gr.Tab("🔬 Scientific", id="scientific"):
                with gr.Row(elem_classes=["main-row"]):

                    # Left: Input
                    with gr.Column(scale=5, elem_classes=["input-panel"]):
                        gr.HTML('<div class="section-hdr">Scientific Question</div>')
                        s_question = gr.Textbox(
                            placeholder="e.g., How does patch embedding work in Vision Transformer?",
                            lines=3,
                            max_lines=5,
                            show_label=False,
                            elem_classes=["question-box"],
                        )
                        gr.Examples(
                            examples=[
                                ["How does patch embedding work in Vision Transformer?"],
                                ["What datasets were used to evaluate Swin Transformer?"],
                                ["How does Swin Transformer handle variable-resolution images?"],
                            ],
                            inputs=s_question,
                            label="💡 Example Questions",
                        )
                        with gr.Row(elem_classes=["btn-row"]):
                            s_ask = gr.Button("🔍 Ask", variant="primary", scale=2, elem_classes=["analyze-btn"])
                            s_clear = gr.Button("↺ Clear", variant="secondary", scale=1, elem_classes=["clear-btn"])

                    # Right: Output
                    with gr.Column(scale=5, elem_classes=["output-panel"]):
                        gr.HTML('<div class="section-hdr">Answer</div>')
                        s_answer = gr.HTML(value=_scientific_placeholder())
                        with gr.Row():
                            with gr.Column(scale=1):
                                s_conf = gr.HTML(value="")
                            with gr.Column(scale=1):
                                s_sources = gr.HTML(value="")

                # Scientific events
                s_ask.click(
                    fn=lambda q: scientific_inference(q, scientific_pipeline),
                    inputs=[s_question],
                    outputs=[s_answer, s_conf, s_sources],
                )
                s_question.submit(
                    fn=lambda q: scientific_inference(q, scientific_pipeline),
                    inputs=[s_question],
                    outputs=[s_answer, s_conf, s_sources],
                )
                s_clear.click(
                    fn=lambda: ("", _scientific_placeholder(), "", ""),
                    outputs=[s_question, s_answer, s_conf, s_sources],
                )

    return app, launch_kwargs


# ================================================================
# Placeholders
# ================================================================

def _healthcare_placeholder() -> str:
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


def _scientific_placeholder() -> str:
    return (
        '<div class="output-placeholder">'
        '  <div class="placeholder-icon-wrap">🔬</div>'
        '  <div class="placeholder-title">Ready for Paper QA</div>'
        '  <div class="placeholder-sub">'
        '    Enter a question about Vision Transformer research,<br>'
        '    then click <strong>Ask</strong> to search through papers.'
        '  </div>'
        '</div>'
    )


# ================================================================
# Standalone launch
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MMRAG Unified UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  MMRAG Unified — Combined UI (Development Mode)")
    print("  Pipelines NOT loaded — UI layout preview only.")
    print("=" * 60)

    app, launch_kwargs = create_unified_app()
    app.queue(max_size=5)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        **launch_kwargs,
    )
