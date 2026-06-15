"""
Scientific Multimodal RAG — Streamlit Q&A Application
======================================================

Launch via:
    python main.py --mode online
    # or directly:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st
import yaml

# ── Path setup: allow running from any directory ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scientific RAG — Vision Transformer Papers",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Premium Dark Theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f1419 100%);
    min-height: 100vh;
}

/* ── Header ── */
.rag-header {
    background: linear-gradient(90deg, #1e2a4a, #2d1b4e, #1a2a3e);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border: 1px solid rgba(99, 102, 241, 0.3);
    box-shadow: 0 4px 32px rgba(99, 102, 241, 0.15);
}

.rag-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.rag-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 400;
}

.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 8px;
    margin-top: 8px;
}
.badge-blue   { background: rgba(56,189,248,0.15);  color: #38bdf8;  border: 1px solid rgba(56,189,248,0.3);  }
.badge-purple { background: rgba(167,139,250,0.15); color: #a78bfa;  border: 1px solid rgba(167,139,250,0.3); }
.badge-green  { background: rgba(52,211,153,0.15);  color: #34d399;  border: 1px solid rgba(52,211,153,0.3);  }
.badge-orange { background: rgba(251,146,60,0.15);  color: #fb923c;  border: 1px solid rgba(251,146,60,0.3);  }

/* ── Answer card ── */
.answer-card {
    background: linear-gradient(135deg, rgba(30,40,70,0.8), rgba(20,30,50,0.9));
    border-radius: 14px;
    padding: 22px 26px;
    border: 1px solid rgba(99,102,241,0.4);
    box-shadow: 0 2px 20px rgba(99,102,241,0.1);
    margin: 12px 0;
    color: #e2e8f0;
    font-size: 0.97rem;
    line-height: 1.75;
}

.answer-label {
    font-size: 0.82rem;
    font-weight: 700;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}

/* ── Source card ── */
.source-card {
    background: rgba(15, 20, 35, 0.7);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    border-left: 4px solid #818cf8;
    transition: border-color 0.2s;
}
.source-card:hover { border-left-color: #a78bfa; }

.source-title {
    font-weight: 600;
    color: #c7d2fe;
    font-size: 0.9rem;
    margin-bottom: 4px;
}

.source-meta {
    color: #64748b;
    font-size: 0.8rem;
}

.source-snippet {
    color: #94a3b8;
    font-size: 0.82rem;
    font-style: italic;
    margin-top: 8px;
    border-top: 1px solid rgba(100,116,139,0.2);
    padding-top: 8px;
}

/* ── Confidence bar ── */
.conf-label {
    font-size: 0.82rem;
    color: #94a3b8;
    font-weight: 500;
    margin-bottom: 4px;
}

.conf-bar-bg {
    background: rgba(30,41,59,0.6);
    border-radius: 999px;
    height: 8px;
    margin-bottom: 12px;
}

/* ── Warning card (out-of-scope) ── */
.oos-card {
    background: rgba(120, 53, 15, 0.25);
    border: 1px solid rgba(251,146,60,0.4);
    border-radius: 14px;
    padding: 20px 24px;
    color: #fed7aa;
    margin: 12px 0;
}

/* ── Status steps ── */
.step-done   { color: #34d399; }
.step-active { color: #818cf8; animation: pulse 1.5s infinite; }
.step-wait   { color: #475569; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Metric boxes ── */
.metric-box {
    background: rgba(20,25,45,0.6);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid rgba(99,102,241,0.2);
}
.metric-val  { font-size: 1.6rem; font-weight: 700; color: #818cf8; }
.metric-label{ font-size: 0.8rem; color: #64748b; margin-top: 4px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1320 0%, #111827 100%);
    border-right: 1px solid rgba(99,102,241,0.15);
}

/* ── Input box ── */
.stTextArea textarea {
    background: rgba(15,20,40,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: rgba(129,140,248,0.6) !important;
    box-shadow: 0 0 0 2px rgba(129,140,248,0.15) !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
}

footer { display: none !important; }
#MainMenu { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Config + Pipeline loader (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_pipeline():
    """Load config and initialise the OnlinePipeline (cached across sessions)."""
    config_path = os.getenv("RAG_CONFIG_PATH", "configs/config.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply RAG_BASE_DIR overrides
    base = os.getenv("RAG_BASE_DIR", "")
    if base:
        for key, val in cfg.get("paths", {}).items():
            if not os.path.isabs(val):
                cfg["paths"][key] = os.path.join(base, val)

    from pipelines.online_pipeline import OnlinePipeline
    return cfg, OnlinePipeline(cfg)


@st.cache_data
def load_index_stats(indices_dir: str) -> dict:
    """Load index summary for sidebar display."""
    summary_path = os.path.join(indices_dir, "summary.json")
    if os.path.exists(summary_path):
        import json
        with open(summary_path) as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

def render_header() -> None:
    st.markdown("""
    <div class="rag-header">
        <div class="rag-title">🔬 Scientific RAG System</div>
        <div class="rag-subtitle">
            Ask questions about Vision Transformer research papers.
            Answers are grounded in retrieved pages — with source citations.
        </div>
        <div style="margin-top: 10px;">
            <span class="badge badge-blue">🖼️ ColPali Vision</span>
            <span class="badge badge-green">📝 SciNCL Text</span>
            <span class="badge badge-purple">🤖 Qwen2-VL</span>
            <span class="badge badge-orange">⚡ Score Fusion</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(cfg: dict) -> dict:
    """Render sidebar with index stats and example queries. Returns UI settings."""
    paths = cfg.get("paths", {})

    with st.sidebar:
        st.markdown("### ⚙️ System Status")

        stats = load_index_stats(paths.get("indices", "data/indices"))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{stats.get('num_papers', '?')}</div>
                <div class="metric-label">Papers</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{stats.get('total_pages', '?')}</div>
                <div class="metric-label">Pages</div>
            </div>""", unsafe_allow_html=True)

        npy_dir   = paths.get("multivectors", "data/indices/multivectors")
        chroma_dir = paths.get("chroma_index", "data/indices/chroma_index")
        npy_count = len(list(Path(npy_dir).glob("*.npy"))) if Path(npy_dir).exists() else 0

        st.markdown(f"""
        <div style="margin-top:12px;">
            <div style="color:#64748b; font-size:0.8rem; margin-bottom:6px;">INDEX STATUS</div>
            <div style="color:{'#34d399' if npy_count > 0 else '#ef4444'}; font-size:0.85rem;">
                {'✅' if npy_count > 0 else '❌'} ColPali: {npy_count} embeddings
            </div>
            <div style="color:{'#34d399' if Path(chroma_dir).exists() else '#ef4444'}; font-size:0.85rem; margin-top:4px;">
                {'✅' if Path(chroma_dir).exists() else '❌'} ChromaDB: {'ready' if Path(chroma_dir).exists() else 'not found'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("### 💡 Example Questions")
        examples = [
            "How does patch embedding work in Vision Transformer?",
            "What is the difference between ViT and CNN for image classification?",
            "How does Swin Transformer handle variable-resolution images?",
            "What are the main results of EfficientFormer?",
            "How many parameters does the 22B ViT model have?",
            "What attention mechanism does DeepViT use?",
            "What datasets were used to train the original ViT?",
        ]

        selected_example = None
        for ex in examples:
            if st.button(ex[:55] + "..." if len(ex) > 55 else ex,
                         key=f"ex_{hash(ex)}", use_container_width=True):
                selected_example = ex

        st.divider()
        st.markdown("### 🔧 Settings")
        top_k = st.slider("Top-K retrieved pages", 1, 7, 3)

        return {"selected_example": selected_example, "top_k": top_k}


# ══════════════════════════════════════════════════════════════════════════════
# Confidence renderer
# ══════════════════════════════════════════════════════════════════════════════

def render_confidence(result) -> None:
    """Render confidence bars for overall, domain, retrieval."""
    conf = result.confidence
    check = result.check_result

    def bar(label: str, value: float) -> str:
        pct   = int(value * 100)
        color = "#34d399" if pct >= 70 else "#fb923c" if pct >= 40 else "#ef4444"
        return f"""
        <div class="conf-label">{label} &nbsp; <b style="color:{color}">{pct}%</b></div>
        <div class="conf-bar-bg">
            <div style="background:{color}; width:{pct}%; height:8px; border-radius:999px;
                        transition: width 0.6s ease;"></div>
        </div>"""

    st.markdown("**📊 Confidence**")
    st.markdown(
        bar("Overall Confidence",   conf) +
        bar("Attribution",  1.0 if check.attribution_passed  else 0.2) +
        bar("Faithfulness", 1.0 if check.faithfulness_passed else 0.2),
        unsafe_allow_html=True,
    )

    checks_passed = sum([check.attribution_passed, check.faithfulness_passed,
                         check.confidence_passed])
    status_color  = "#34d399" if checks_passed == 3 else "#fb923c" if checks_passed >= 2 else "#ef4444"
    st.markdown(
        f"""<div style="margin-top:10px; padding:8px 12px; background:rgba(15,20,35,0.5);
            border-radius:8px; font-size:0.82rem; color:{status_color};">
            {"✅ Answered from documents" if check.passed else "⚠️ Low confidence answer"}
            &nbsp;·&nbsp; {result.total_time:.1f}s
            &nbsp;·&nbsp; {result.retries} retries
        </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Source renderer
# ══════════════════════════════════════════════════════════════════════════════

def render_sources(result) -> None:
    """Render retrieved source cards."""
    if not result.sources:
        st.markdown("""
        <div style="background:rgba(127,29,29,0.2); border:1px solid rgba(239,68,68,0.3);
             border-radius:10px; padding:14px; color:#fca5a5; font-size:0.85rem;">
            🚫 No relevant pages found in the documents.
        </div>""", unsafe_allow_html=True)
        return

    st.markdown(f"**📚 Sources ({len(result.sources)} retrieved)**")

    for i, src in enumerate(result.sources):
        score_pct = int(src.relevance_score * 100)
        color     = "#34d399" if score_pct >= 70 else "#fb923c" if score_pct >= 40 else "#818cf8"
        pages_str = ", ".join(str(p) for p in src.page_numbers)
        snippet   = (src.text_snippet or "")[:200].replace("\n", " ")
        arxiv_url = src.arxiv_url or f"https://arxiv.org/abs/{src.paper_id}"

        st.markdown(f"""
        <div class="source-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div class="source-title">📄 [{i+1}] {src.paper_title[:55]}</div>
                <span style="background:{color}22; color:{color}; border:1px solid {color}44;
                      padding:2px 10px; border-radius:999px; font-size:0.75rem;
                      font-weight:700; white-space:nowrap; margin-left:8px;">
                    {score_pct}%
                </span>
            </div>
            <div class="source-meta">
                📖 Page {pages_str} &nbsp;|&nbsp;
                🔗 <a href="{arxiv_url}" target="_blank"
                      style="color:#818cf8; text-decoration:none;">arXiv:{src.paper_id}</a>
            </div>
            {f'<div class="source-snippet">"{snippet}..."</div>' if snippet else ''}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Load pipeline ──
    try:
        cfg, pipeline = load_pipeline()
    except FileNotFoundError as e:
        st.error(f"❌ Config not found: {e}\nRun from project root: `python main.py --mode online`")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load pipeline: {e}")
        st.stop()

    # ── Render header ──
    render_header()

    # ── Sidebar ──
    sidebar = render_sidebar(cfg)

    # ── Session state ──
    if "history" not in st.session_state:
        st.session_state.history = []
    if "query_text" not in st.session_state:
        st.session_state.query_text = ""

    # ── Query input ──
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        # Pre-fill with selected example from sidebar
        if sidebar.get("selected_example"):
            st.session_state.query_text = sidebar["selected_example"]

        question = st.text_area(
            "Ask a question about the research papers",
            key="query_text",
            placeholder="e.g. How does patch embedding work in Vision Transformer?",
            height=80,
            label_visibility="collapsed",
        )

    def clear_callback():
        st.session_state.history = []
        st.session_state.query_text = ""

    with col_btn:
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        ask_clicked = st.button("🔍 Ask", use_container_width=True)
        st.button("🗑️ Clear", use_container_width=True, on_click=clear_callback)

    # ── Run query ──
    if ask_clicked and question.strip():
        with st.spinner(""):
            # Progress display
            progress_placeholder = st.empty()

            steps = {
                "colpali_encode": ("🔵 Encoding query with ColPali...",   20),
                "scincl_encode":  ("🟢 Encoding query with SciNCL...",    35),
                "retrieval":      ("🔍 Retrieving relevant pages...",      50),
                "fusion":         ("⚡ Fusing ColPali + SciNCL scores...", 65),
                "context":        ("📄 Building context from pages...",    75),
                "qwen_generate":  ("🤖 Generating answer with Qwen2-VL...",85),
                "self_check":     ("✅ Running self-check...",             95),
            }

            def status_callback(step_name: str, message: str, pct: int) -> None:
                lines = []
                reached = False
                for k, (label, p) in steps.items():
                    if k == step_name:
                        reached = True
                        lines.append(f"<span class='step-active'>⟳ {label}</span>")
                    elif not reached:
                        lines.append(f"<span class='step-done'>✓ {label}</span>")
                    else:
                        lines.append(f"<span class='step-wait'>○ {label}</span>")

                progress_placeholder.markdown(
                    "<div style='background:rgba(15,20,40,0.7); border-radius:12px; "
                    "padding:16px; border:1px solid rgba(99,102,241,0.2);'>"
                    + "<br>".join(lines)
                    + "</div>",
                    unsafe_allow_html=True,
                )

            try:
                # Override top_k from sidebar
                cfg["retrieval"]["top_k"] = sidebar.get("top_k", 3)

                result = pipeline.query(question.strip(), status_callback=status_callback)

                progress_placeholder.empty()

                # Save to history
                st.session_state.history.insert(0, {
                    "question": question.strip(),
                    "result":   result,
                })

            except Exception as e:
                progress_placeholder.empty()
                st.error(f"❌ Query failed: {e}")
                st.exception(e)
                return

    # ── Display results ──
    if st.session_state.history:
        latest = st.session_state.history[0]
        result = latest["result"]
        q      = latest["question"]

        st.markdown(f"""
        <div style="margin: 20px 0 8px 0; padding: 12px 18px;
             background: rgba(30,40,70,0.5); border-radius: 10px;
             border-left: 3px solid #818cf8; color:#c7d2fe; font-weight:600;">
            ❓ {q}
        </div>""", unsafe_allow_html=True)

        # Two-column layout: Answer + (Confidence + Sources)
        col_ans, col_right = st.columns([3, 2])

        with col_ans:
            if result.check_result.passed:
                st.markdown(f"""
                <div class="answer-card">
                    <div class="answer-label">✅ Answer from Documents</div>
                    {result.answer}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="oos-card">
                    <div style="font-size:1rem; font-weight:700; margin-bottom:10px;">
                        ⚠️ Low Confidence Answer
                    </div>
                    <div style="font-size:0.9rem; margin-bottom:12px; opacity:0.8;">
                        The answer below has low retrieval confidence.
                        Try rephrasing your question.
                    </div>
                    <div style="background:rgba(0,0,0,0.2); border-radius:8px; padding:14px;">
                        {result.answer or "No answer could be generated."}
                    </div>
                </div>""", unsafe_allow_html=True)

        with col_right:
            render_confidence(result)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            render_sources(result)

        # ── History ──
        if len(st.session_state.history) > 1:
            with st.expander(f"📜 Query History ({len(st.session_state.history)} questions)"):
                for i, item in enumerate(st.session_state.history[1:], 1):
                    r = item["result"]
                    score_pct = int(r.confidence * 100)
                    color = "#34d399" if score_pct >= 70 else "#fb923c" if score_pct >= 40 else "#ef4444"
                    st.markdown(
                        f"**[{i}]** {item['question'][:80]}..."
                        f"&nbsp;&nbsp;<span style='color:{color};font-size:0.8rem'>{score_pct}% conf</span>",
                        unsafe_allow_html=True,
                    )

    else:
        # Empty state
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color:#475569;">
            <div style="font-size:3rem; margin-bottom:16px;">🔬</div>
            <div style="font-size:1.1rem; font-weight:600; color:#64748b;">
                Ready to answer questions about Vision Transformer papers
            </div>
            <div style="font-size:0.9rem; margin-top:8px; color:#475569;">
                Type a question above or pick an example from the sidebar
            </div>
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
