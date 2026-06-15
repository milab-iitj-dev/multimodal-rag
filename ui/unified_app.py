"""
MMRAG Unified — Combined Gradio UI consuming UnifiedResponse.

Both tabs (Healthcare + Scientific) consume the SAME schema:
  response.answer, response.confidence, response.sources, response.metadata

No domain-specific parsing. The UI only knows about UnifiedResponse.

Usage:
    python ui/unified_app.py                # Dev mode (no pipelines)
    python ui/unified_app.py --share        # With public link
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.healthcare.theme import create_theme, CUSTOM_CSS
from src.shared.schemas.response import UnifiedResponse, SourceItem


# ================================================================
# Shared formatters (work on UnifiedResponse — domain-agnostic)
# ================================================================

def format_answer_card(response: UnifiedResponse) -> str:
    """Format an answer card from UnifiedResponse."""
    conf = response.confidence
    pct = int(conf * 100)

    if pct >= 70:
        badge_text, badge_color = "HIGH CONFIDENCE", "#10b981"
    elif pct >= 40:
        badge_text, badge_color = "MEDIUM CONFIDENCE", "#f59e0b"
    else:
        badge_text, badge_color = "LOW CONFIDENCE", "#f43f5e"

    # Escape HTML
    answer = (
        response.answer
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )

    return f'''
    <div class="answer-card">
      <div class="ans-hdr">
        <span class="ans-label">{response.domain.upper()} · Analysis Result</span>
        <span class="ans-badge" style="color:{badge_color};background:rgba({_hex_to_rgb(badge_color)},0.12);">{badge_text} ({pct}%)</span>
      </div>
      <div class="ans-body">{answer}</div>
      <div class="relevance-bar">
        <span class="rlabel">Confidence</span>
        <div class="rtrack"><div class="rfill" style="width:{pct}%"></div></div>
        <span class="rval">{conf:.2f}</span>
      </div>
    </div>
    '''


def format_sources_card(response: UnifiedResponse) -> str:
    """Format sources from UnifiedResponse."""
    if not response.sources:
        return (
            '<div style="padding:16px;text-align:center;color:#64748b;'
            'font-size:0.82rem;">No sources retrieved.</div>'
        )

    cards = []
    for i, src in enumerate(response.sources, 1):
        score_pct = int(src.score * 100)
        if score_pct >= 70:
            score_class = "ev-score-high"
        elif score_pct >= 40:
            score_class = "ev-score-med"
        else:
            score_class = "ev-score-low"

        snippet_html = ""
        if src.snippet:
            safe_snippet = src.snippet[:300].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            snippet_html = f'<div class="ev-field-label">Evidence</div><div class="ev-field-text">{safe_snippet}</div>'

        url_html = ""
        if src.url:
            url_html = f' · <a href="{src.url}" target="_blank" style="color:#10b981;">🔗 Link</a>'

        pages_html = ""
        if src.page_numbers:
            pages_html = f' · Pages: {", ".join(str(p) for p in src.page_numbers)}'

        cards.append(f'''
        <div class="ev-card">
          <div class="ev-card-hdr">
            <span class="ev-score {score_class}">{src.score:.2f}</span>
            <span class="ev-title">[{i}] {_escape(src.title)}</span>
          </div>
          <div class="ev-card-body">
            {snippet_html}
            <div class="ev-field-label">Info</div>
            <div class="ev-field-text">Score: {score_pct}%{pages_html}{url_html}</div>
          </div>
        </div>
        ''')

    return "".join(cards)


def format_metadata_bar(response: UnifiedResponse) -> str:
    """Format timing/metadata bar from UnifiedResponse."""
    total = response.metadata.get("total_time_sec", 0)
    retrieval = response.metadata.get("retrieval_time_sec", 0)
    generation = response.metadata.get("generation_time_sec", 0)
    n_sources = response.metadata.get("num_retrieved", len(response.sources))

    return f'''
    <div class="timing-bar">
      <div class="timing-left">
        <div class="t-item"><span class="t-icon">⏱</span> Total <span class="t-val">{total:.2f}s</span></div>
        <div class="t-item"><span class="t-icon">◎</span> Retrieval <span class="t-val">{retrieval:.1f}s</span></div>
        <div class="t-item"><span class="t-icon">◉</span> Generation <span class="t-val">{generation:.1f}s</span></div>
      </div>
      <div class="timing-right">
        <div class="t-item"><span class="t-icon">📄</span> Sources <span class="t-val">{n_sources}</span></div>
      </div>
    </div>
    '''


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


# ================================================================
# Router-backed inference (domain-agnostic)
# ================================================================

def _get_router():
    """Lazy-init the DomainRouter with registered pipelines."""
    if not hasattr(_get_router, "_instance"):
        from src.router.domain_router import DomainRouter
        from pipelines.healthcare.adapter import HealthcarePipeline
        from pipelines.scientific.adapter import ScientificPipeline

        router = DomainRouter()
        router.register("healthcare", HealthcarePipeline(inner_pipeline=None))
        router.register("scientific", ScientificPipeline(inner_pipeline=None))
        _get_router._instance = router
    return _get_router._instance


def run_healthcare(image, question) -> tuple:
    """Healthcare tab callback — uses router for domain-agnostic flow."""
    try:
        router = _get_router()
        response = router.route(
            query=question.strip() if question else "Describe the key findings in this chest X-ray.",
            domain_hint="healthcare",
            image=image,
        )
        return (
            format_answer_card(response),
            format_sources_card(response),
            format_metadata_bar(response),
        )
    except Exception as e:
        traceback.print_exc()
        err_resp = UnifiedResponse(domain="healthcare", answer=f"Error: {e}")
        return format_answer_card(err_resp), "", ""


def run_scientific(question) -> tuple:
    """Scientific tab callback — uses router for domain-agnostic flow."""
    if not question or not question.strip():
        err_resp = UnifiedResponse(domain="scientific", answer="Please enter a question.")
        return format_answer_card(err_resp), "", ""
    try:
        router = _get_router()
        response = router.route(
            query=question.strip(),
            domain_hint="scientific",
        )
        return (
            format_answer_card(response),
            format_sources_card(response),
            format_metadata_bar(response),
        )
    except Exception as e:
        traceback.print_exc()
        err_resp = UnifiedResponse(domain="scientific", answer=f"Error: {e}")
        return format_answer_card(err_resp), "", ""


# ================================================================
# Header
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
        '    Enter a question about scientific papers,<br>'
        '    then click <strong>Ask</strong> to search.'
        '  </div>'
        '</div>'
    )


# ================================================================
# Application Builder
# ================================================================

def create_unified_app() -> tuple:
    """Build the unified Gradio Blocks application with tabbed domains."""
    theme = create_theme()
    launch_kwargs = {"theme": theme, "css": CUSTOM_CSS}

    with gr.Blocks(title="MMRAG Unified — Healthcare + Scientific") as app:
        gr.HTML(HEADER_HTML)

        with gr.Tabs():

            # ── Healthcare Tab ──
            with gr.Tab("🏥 Healthcare", id="healthcare"):
                with gr.Row(elem_classes=["main-row"]):
                    with gr.Column(scale=4, elem_classes=["input-panel"]):
                        gr.HTML('<div class="section-hdr">Chest X-Ray Input</div>')
                        h_image = gr.Image(label="Upload Chest X-Ray", type="pil",
                                           sources=["upload", "clipboard"],
                                           elem_classes=["upload-area"], height=200)
                        gr.HTML('<div class="section-hdr" style="margin-top:6px;">Clinical Question</div>')
                        h_question = gr.Textbox(placeholder="e.g., Are there signs of pleural effusion?",
                                                lines=3, show_label=False, elem_classes=["question-box"])
                        with gr.Row(elem_classes=["btn-row"]):
                            h_analyze = gr.Button("⚙ Analyze", variant="primary", scale=2, elem_classes=["analyze-btn"])
                            h_clear = gr.Button("↺ Clear", variant="secondary", scale=1, elem_classes=["clear-btn"])

                    with gr.Column(scale=6, elem_classes=["output-panel"]):
                        gr.HTML('<div class="section-hdr">Results</div>')
                        h_answer = gr.HTML(value=_healthcare_placeholder())
                        with gr.Accordion("Retrieved Evidence", open=False, elem_classes=["evidence-section"]):
                            h_evidence = gr.HTML(value="")
                        h_timing = gr.HTML(value="")

                h_analyze.click(fn=run_healthcare, inputs=[h_image, h_question],
                                outputs=[h_answer, h_evidence, h_timing])
                h_clear.click(fn=lambda: (None, "", _healthcare_placeholder(), "", ""),
                              outputs=[h_image, h_question, h_answer, h_evidence, h_timing])

            # ── Scientific Tab ──
            with gr.Tab("🔬 Scientific", id="scientific"):
                with gr.Row(elem_classes=["main-row"]):
                    with gr.Column(scale=5, elem_classes=["input-panel"]):
                        gr.HTML('<div class="section-hdr">Scientific Question</div>')
                        s_question = gr.Textbox(placeholder="e.g., How does patch embedding work in Vision Transformer?",
                                                lines=3, show_label=False, elem_classes=["question-box"])
                        gr.Examples(
                            examples=[
                                ["How does patch embedding work in Vision Transformer?"],
                                ["What datasets were used to evaluate Swin Transformer?"],
                            ],
                            inputs=s_question, label="💡 Examples",
                        )
                        with gr.Row(elem_classes=["btn-row"]):
                            s_ask = gr.Button("🔍 Ask", variant="primary", scale=2, elem_classes=["analyze-btn"])
                            s_clear = gr.Button("↺ Clear", variant="secondary", scale=1, elem_classes=["clear-btn"])

                    with gr.Column(scale=5, elem_classes=["output-panel"]):
                        gr.HTML('<div class="section-hdr">Answer</div>')
                        s_answer = gr.HTML(value=_scientific_placeholder())
                        with gr.Accordion("Retrieved Sources", open=False, elem_classes=["evidence-section"]):
                            s_sources = gr.HTML(value="")
                        s_timing = gr.HTML(value="")

                s_ask.click(fn=run_scientific, inputs=[s_question],
                            outputs=[s_answer, s_sources, s_timing])
                s_question.submit(fn=run_scientific, inputs=[s_question],
                                  outputs=[s_answer, s_sources, s_timing])
                s_clear.click(fn=lambda: ("", _scientific_placeholder(), "", ""),
                              outputs=[s_question, s_answer, s_sources, s_timing])

    return app, launch_kwargs


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
    print("  MMRAG Unified — Combined UI")
    print("  Pipelines: placeholder mode (no GPU)")
    print("=" * 60)

    app, launch_kwargs = create_unified_app()
    app.queue(max_size=5)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        **launch_kwargs,
    )
