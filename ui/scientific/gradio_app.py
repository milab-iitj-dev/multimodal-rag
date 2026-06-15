"""
Scientific Multimodal RAG — Gradio Q&A Application
==================================================

Launch via:
    gradio app/gradio_app.py
    # or directly:
    python app/gradio_app.py
"""

from __future__ import annotations

import os
import sys
import yaml
from pathlib import Path
import gradio as gr

# ── Path setup: allow running from any directory ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from pipelines.online_pipeline import OnlinePipeline

# Global pipeline instance
pipeline = None
cfg = None

def init_pipeline():
    global pipeline, cfg
    config_path = os.getenv("RAG_CONFIG_PATH", "configs/config.yaml")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply RAG_BASE_DIR overrides
    base = os.getenv("RAG_BASE_DIR", "")
    if base:
        for key, val in cfg.get("paths", {}).items():
            if not os.path.isabs(val):
                cfg["paths"][key] = os.path.join(base, val)

    pipeline = OnlinePipeline(cfg)

def gradio_rag(query: str):
    if not query or not query.strip():
        return (
            "<p style='color:#ef4444;padding:12px'>⚠️ Please enter a question.</p>",
            "<p style='color:#9ca3af;padding:12px'>Confidence...</p>",
            "<p style='color:#9ca3af;padding:12px'>Sources...</p>"
        )
    
    try:
        result = pipeline.query(query.strip())
    except Exception as e:
        return (
            f"<p style='color:#ef4444;padding:12px'>❌ Query failed: {e}</p>",
            "<p style='color:#9ca3af;padding:12px'>Confidence...</p>",
            "<p style='color:#9ca3af;padding:12px'>Sources...</p>"
        )

    # Answer HTML card (Premium Sleek Design)
    if result.check_result.passed:
        ans_html = f"""
        <div style="background:#f0f9ff; border-left:4px solid #2563eb;
                    border-radius:8px; padding:16px; font-size:15px; line-height:1.6; color:#1e293b;">
            <strong style="color:#2563eb; font-size:13px; text-transform:uppercase; letter-spacing:0.05em;">
                ✅ Answer from Documents
            </strong>
            <div style="margin-top:8px;">{result.answer.replace(chr(10), '<br>')}</div>
        </div>"""
    else:
        ans_html = f"""
        <div style="background:#fffbeb; border-left:4px solid #d97706;
                    border-radius:8px; padding:16px; font-size:15px; line-height:1.6; color:#1e293b;">
            <strong style="color:#d97706; font-size:13px; text-transform:uppercase; letter-spacing:0.05em;">
                ⚠️ Low Confidence Answer
            </strong>
            <div style="margin-top:8px; color:#b45309; font-size:13px;">
                This question might be outside the scope of the ingested papers, or retrieval confidence is low.
            </div>
            <div style="margin-top:8px; background:rgba(217, 119, 6, 0.05); padding:10px; border-radius:6px;">
                {result.answer.replace(chr(10), '<br>')}
            </div>
        </div>"""

    # Confidence HTML card
    pct = int(result.confidence * 100)
    col = "#16a34a" if pct > 60 else "#fb923c" if pct > 30 else "#dc2626"
    
    attribution_pct = 100 if result.check_result.attribution_passed else 20
    faithfulness_pct = 100 if result.check_result.faithfulness_passed else 20
    
    conf_html = f"""
    <div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:16px;">
        <div style="text-align:center; margin-bottom:12px;">
            <div style="font-size:36px; font-weight:800; color:{col}; line-height:1;">{pct}%</div>
            <div style="color:#6b7280; font-size:13px; margin-top:4px;">overall confidence</div>
        </div>
        <div style="font-size:12px; color:#4b5563;">
            <div style="margin-bottom:6px;">
                Attribution: <strong style="color:{'#16a34a' if attribution_pct == 100 else '#dc2626'}">{attribution_pct}%</strong>
            </div>
            <div>
                Faithfulness: <strong style="color:{'#16a34a' if faithfulness_pct == 100 else '#dc2626'}">{faithfulness_pct}%</strong>
            </div>
        </div>
        <div style="color:#9ca3af; font-size:11px; margin-top:10px; border-top:1px solid #f3f4f6; padding-top:6px; text-align:center;">
            ⏱ {result.total_time:.1f}s &nbsp;·&nbsp; 🔄 {result.retries} retries
        </div>
    </div>"""

    # Sources HTML panel
    if result.sources:
        cards = ""
        for i, s in enumerate(result.sources, 1):
            score_pct = int(s.relevance_score * 100)
            score_col = "#16a34a" if score_pct >= 70 else "#fb923c" if score_pct >= 40 else "#2563eb"
            pages_str = ", ".join(str(p) for p in s.page_numbers)
            arxiv = (f'<a href="{s.arxiv_url}" target="_blank" '
                     f'style="color:#2563eb; text-decoration:none;">🔗 arXiv</a>' if s.arxiv_url else "")
            snippet = s.text_snippet[:180].replace("\n", " ") if s.text_snippet else ""
            
            cards += f"""
            <div style="border:1px solid #e5e7eb; border-radius:8px;
                        padding:12px; margin-bottom:8px; background:white;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="font-weight:600; color:#1e3a8a; font-size:13px;">
                        [{i}] {s.paper_title[:55]}...
                    </div>
                    <span style="background:{score_col}15; color:{score_col}; border:1px solid {score_col}33;
                                 padding:2px 8px; border-radius:12px; font-size:11px; font-weight:700; white-space:nowrap;">
                        {score_pct}%
                    </span>
                </div>
                <div style="color:#6b7280; font-size:12px; margin-top:4px;">
                    📖 Page {pages_str} &nbsp;|&nbsp; {arxiv}
                </div>
                {f'<div style="color:#4b5563; font-size:11px; margin-top:6px; border-top:1px solid #f3f4f6; padding-top:6px; font-style:italic;">"{snippet}..."</div>' if snippet else ''}
            </div>"""
        
        src_html = f"""
        <div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:16px;">
            <div style="font-weight:700; color:#1e293b; margin-bottom:10px; font-size:14px;">
                📚 Sources ({len(result.sources)})
            </div>
            {cards}
        </div>"""
    else:
        src_html = """
        <div style="background:#fef2f2; border:1px solid #fca5a5; border-radius:10px; padding:16px; color:#b91c1c; font-size:13px; text-align:center;">
            🚫 No sources found.
        </div>"""

    return ans_html, conf_html, src_html

# Initialize pipeline
init_pipeline()

# Launch Gradio Block interface
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    title="Scientific RAG — Vision Transformer Papers"
) as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <h1 style="font-size:30px; font-weight:800; color:#1e3a8a; margin-bottom:6px;">
            🔬 Scientific RAG System
        </h1>
        <p style="color:#6b7280; font-size:15px;">
            ColPali + SciNCL + Qwen2-VL &nbsp;|&nbsp; Multi-Vector Fusion Retrieval
        </p>
    </div>""")

    with gr.Row():
        with gr.Column(scale=5):
            q_box = gr.Textbox(
                label="Question",
                placeholder="Ask something about Vision Transformers...",
                lines=2,
            )
        with gr.Column(scale=1, min_width=120):
            ask_btn   = gr.Button("🔍 Ask",   variant="primary")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

    gr.Examples(
        examples=[
            ["How does patch embedding work in Vision Transformer?"],
            ["How does Swin Transformer handle variable-resolution images?"],
            ["What datasets were used to evaluate Swin Transformer?"],
            ["What is the capital of France?"],   # out-of-scope test
        ],
        inputs=q_box,
        label="💡 Example Questions"
    )

    with gr.Row():
        with gr.Column(scale=3):
            ans_out  = gr.HTML("<p style='color:#9ca3af; padding:20px; text-align:center;'>Answer will appear here...</p>")
        with gr.Column(scale=2):
            conf_out = gr.HTML("<p style='color:#9ca3af; padding:12px; text-align:center;'>Confidence level...</p>")
            src_out  = gr.HTML("<p style='color:#9ca3af; padding:12px; text-align:center;'>Retrieved sources...</p>")

    # Set actions
    ask_btn.click(fn=gradio_rag, inputs=[q_box], outputs=[ans_out, conf_out, src_out])
    q_box.submit(fn=gradio_rag,  inputs=[q_box], outputs=[ans_out, conf_out, src_out])
    clear_btn.click(
        fn=lambda: (
            "<p style='color:#9ca3af; padding:20px; text-align:center;'>Answer will appear here...</p>",
            "<p style='color:#9ca3af; padding:12px; text-align:center;'>Confidence level...</p>",
            "<p style='color:#9ca3af; padding:12px; text-align:center;'>Retrieved sources...</p>",
            "",
        ),
        outputs=[ans_out, conf_out, src_out, q_box]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
    )
