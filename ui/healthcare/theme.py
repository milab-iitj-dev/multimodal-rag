"""
Custom Gradio theme and CSS for the Healthcare MRAG inference UI.

Design principles:
  - Premium dark medical workstation aesthetic
  - Refined teal accent with subtle gradients
  - Answer card is the dominant visual element
  - Evidence is elegant and organized
  - Compact, balanced layout with no wasted space
  - Professional typography with clear visual hierarchy
"""

import gradio as gr


# ================================================================
# Color Palette — refined for premium medical aesthetic
# ================================================================
COLORS = {
    # Backgrounds — layered depth
    "bg_page":      "#080c14",      # deepest page background
    "bg_app":       "#0c1220",      # application container
    "bg_panel":     "#111827",      # panel / column backgrounds
    "bg_card":      "#151e2d",      # card surfaces
    "bg_elevated":  "#1a2540",      # hover / elevated states
    "bg_input":     "#0f1829",      # input fields

    # Borders — subtle hierarchy
    "border_subtle":  "#1c2a3f",    # faintest borders
    "border_default": "#243044",    # standard borders
    "border_strong":  "#2d3d56",    # emphasized borders

    # Accent — clinical teal
    "accent":        "#10b981",     # primary accent (emerald-500)
    "accent_bright": "#34d399",     # bright highlights
    "accent_dim":    "#059669",     # dark accent
    "accent_glow":   "rgba(16, 185, 129, 0.12)",
    "accent_glow_strong": "rgba(16, 185, 129, 0.25)",

    # Text — refined hierarchy
    "text_bright":  "#f1f5f9",      # headings, emphasis
    "text_primary": "#e2e8f0",      # primary body text
    "text_muted":   "#94a3b8",      # secondary text
    "text_dim":     "#64748b",      # labels, timestamps
    "text_faint":   "#475569",      # decorative text

    # Semantic
    "success":    "#10b981",
    "warning":    "#f59e0b",
    "error":      "#f43f5e",

    # Scores
    "score_high": "#10b981",
    "score_med":  "#f59e0b",
    "score_low":  "#6b7280",
}


# ================================================================
# Custom Gradio Theme
# ================================================================
def create_theme() -> gr.themes.Base:
    """Build a premium dark theme for the medical MRAG interface."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#ecfdf5", c100="#d1fae5", c200="#a7f3d0",
            c300="#6ee7b7", c400="#34d399", c500="#10b981",
            c600="#059669", c700="#047857", c800="#065f46",
            c900="#064e3b", c950="#022c22",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0",
            c300="#cbd5e1", c400="#94a3b8", c500="#64748b",
            c600="#475569", c700="#334155", c800="#1e293b",
            c900="#0f172a", c950="#020617",
        ),
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        # Global
        body_background_fill=COLORS["bg_page"],
        body_text_color=COLORS["text_primary"],
        body_text_color_subdued=COLORS["text_muted"],

        # Blocks
        block_background_fill=COLORS["bg_panel"],
        block_border_color=COLORS["border_subtle"],
        block_label_text_color=COLORS["text_dim"],
        block_title_text_color=COLORS["text_primary"],

        # Inputs
        input_background_fill=COLORS["bg_input"],
        input_border_color=COLORS["border_default"],
        input_placeholder_color=COLORS["text_faint"],

        # Buttons
        button_primary_background_fill=COLORS["accent"],
        button_primary_background_fill_hover=COLORS["accent_dim"],
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="transparent",
        button_secondary_border_color=COLORS["border_default"],
        button_secondary_text_color=COLORS["text_muted"],

        # Borders & shadows
        block_border_width="1px",
        block_shadow="none",
        input_border_width="1px",
        button_border_width="1px",

        # Spacing
        block_padding="12px",
        layout_gap="12px",
    )
    return theme


# ================================================================
# Custom CSS — premium medical UI
# ================================================================
CUSTOM_CSS = """
/* ── IMPORTS ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── GLOBAL RESET & CONTAINER ──────────────────────────── */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
    background: %(bg_page)s !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.gradio-container .main {
    gap: 0 !important;
}

/* Clean up default Gradio block chrome */
.gradio-container .block {
    border: none !important;
    box-shadow: none !important;
}

/* ── HEADER ────────────────────────────────────────────── */
.mrag-header {
    background: %(bg_app)s;
    border: 1px solid %(border_subtle)s;
    border-radius: 14px;
    padding: 14px 22px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    backdrop-filter: blur(12px);
}
.mrag-header-left {
    display: flex;
    align-items: center;
    gap: 14px;
}
.mrag-logo {
    width: 36px;
    height: 36px;
    border-radius: 9px;
    background: linear-gradient(135deg, %(accent)s, %(accent_dim)s);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    color: #fff;
    font-weight: 700;
    flex-shrink: 0;
    box-shadow: 0 2px 8px %(accent_glow)s;
}
.mrag-header-text h1 {
    font-size: 1.05rem;
    font-weight: 700;
    color: %(text_bright)s;
    margin: 0;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.mrag-header-text h1 span {
    color: %(accent_bright)s;
}
.mrag-header-text p {
    font-size: 0.7rem;
    color: %(text_dim)s;
    margin: 2px 0 0 0;
    letter-spacing: 0.03em;
    font-weight: 400;
}
.mrag-status {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.72rem;
    color: %(text_muted)s;
    font-weight: 500;
}
.mrag-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%%;
    background: %(success)s;
    box-shadow: 0 0 8px %(success)s;
    animation: pulse-dot 2.5s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%%, 100%% { opacity: 1; box-shadow: 0 0 8px %(success)s; }
    50%% { opacity: 0.6; box-shadow: 0 0 4px %(success)s; }
}

/* ── MAIN CONTENT WRAPPER ──────────────────────────────── */
.main-row {
    gap: 14px !important;
}

/* ── PANEL WRAPPER (shared for left & right) ───────────── */
.input-panel, .output-panel {
    background: %(bg_app)s !important;
    border: 1px solid %(border_subtle)s !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

/* ── SECTION HEADER ────────────────────────────────────── */
.section-hdr {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: %(text_dim)s;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid %(border_subtle)s;
}

/* ── IMAGE UPLOAD ──────────────────────────────────────── */
.upload-area {
    margin-bottom: 14px !important;
}
.upload-area > div {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}
.upload-area .image-container,
.upload-area .upload-button,
.upload-area [data-testid="image"] {
    border: 1.5px dashed %(border_default)s !important;
    border-radius: 10px !important;
    background: %(bg_card)s !important;
    transition: all 0.25s ease !important;
    min-height: 160px !important;
}
.upload-area .image-container:hover,
.upload-area .upload-button:hover,
.upload-area [data-testid="image"]:hover {
    border-color: %(accent)s !important;
    background: %(bg_elevated)s !important;
    box-shadow: 0 0 16px %(accent_glow)s !important;
}
/* Style the uploaded image container */
.upload-area img {
    border-radius: 8px !important;
}

/* ── QUESTION TEXTBOX ──────────────────────────────────── */
.question-box {
    margin-bottom: 10px !important;
}
.question-box textarea {
    background: %(bg_card)s !important;
    border: 1.5px solid %(border_default)s !important;
    border-radius: 10px !important;
    color: %(text_primary)s !important;
    font-size: 0.88rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 12px 14px !important;
    line-height: 1.5 !important;
    transition: all 0.25s ease !important;
    resize: none !important;
}
.question-box textarea::placeholder {
    color: %(text_faint)s !important;
    font-style: italic;
}
.question-box textarea:focus {
    border-color: %(accent)s !important;
    box-shadow: 0 0 0 3px %(accent_glow)s !important;
    outline: none !important;
}

/* ── MODE BADGE ────────────────────────────────────────── */
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    background: linear-gradient(135deg, %(bg_card)s, %(bg_elevated)s);
    border: 1px solid %(border_default)s;
    color: %(text_muted)s;
    letter-spacing: 0.02em;
}
.mode-badge .mi { font-size: 0.82rem; }

/* ── BUTTONS ───────────────────────────────────────────── */
.btn-row {
    gap: 8px !important;
    margin-top: 4px !important;
}
.analyze-btn {
    background: linear-gradient(135deg, %(accent)s 0%%, %(accent_dim)s 100%%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 0 !important;
    color: #fff !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 3px 14px %(accent_glow_strong)s !important;
    letter-spacing: 0.01em;
}
.analyze-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px %(accent_glow_strong)s !important;
    filter: brightness(1.08) !important;
}
.analyze-btn:active {
    transform: translateY(0) !important;
}
.clear-btn {
    border: 1px solid %(border_default)s !important;
    border-radius: 10px !important;
    background: %(bg_card)s !important;
    color: %(text_muted)s !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}
.clear-btn:hover {
    border-color: %(text_dim)s !important;
    background: %(bg_elevated)s !important;
    color: %(text_primary)s !important;
}

/* ── ANSWER CARD (most prominent element) ──────────────── */
.answer-card {
    position: relative;
    background: %(bg_card)s;
    border: 1px solid %(border_default)s;
    border-left: 3px solid %(accent)s;
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2),
                0 0 24px %(accent_glow)s;
    margin-bottom: 14px;
}
.answer-card .ans-hdr {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid %(border_subtle)s;
}
.answer-card .ans-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: %(accent_bright)s;
}
.answer-card .ans-badge {
    font-size: 0.62rem;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 10px;
    background: %(accent_glow)s;
    color: %(accent_bright)s;
    letter-spacing: 0.04em;
}
.answer-card .ans-body {
    font-size: 0.92rem;
    line-height: 1.75;
    color: %(text_primary)s;
    font-weight: 400;
}
/* Relevance bar */
.relevance-bar {
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid %(border_subtle)s;
    display: flex;
    align-items: center;
    gap: 10px;
}
.relevance-bar .rlabel {
    font-size: 0.7rem;
    color: %(text_dim)s;
    font-weight: 500;
    min-width: 70px;
}
.relevance-bar .rtrack {
    flex: 1;
    height: 4px;
    background: %(border_subtle)s;
    border-radius: 2px;
    overflow: hidden;
}
.relevance-bar .rfill {
    height: 100%%;
    border-radius: 2px;
    background: linear-gradient(90deg, %(accent_dim)s, %(accent_bright)s);
    transition: width 0.6s ease;
}
.relevance-bar .rval {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: %(accent_bright)s;
    min-width: 36px;
    text-align: right;
}

/* ── EVIDENCE SECTION ──────────────────────────────────── */
.evidence-section {
    border: 1px solid %(border_subtle)s !important;
    border-radius: 12px !important;
    background: %(bg_app)s !important;
    overflow: hidden !important;
    margin-bottom: 10px !important;
}
.evidence-section > .label-wrap {
    background: %(bg_card)s !important;
    border: none !important;
    border-bottom: 1px solid %(border_subtle)s !important;
    border-radius: 0 !important;
    padding: 10px 16px !important;
    cursor: pointer;
}
.evidence-section > .label-wrap span {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: %(text_dim)s !important;
}
.evidence-section > .label-wrap:hover {
    background: %(bg_elevated)s !important;
}

/* Evidence card */
.ev-card {
    background: %(bg_card)s;
    border: 1px solid %(border_subtle)s;
    border-radius: 10px;
    margin: 10px 12px;
    overflow: hidden;
    transition: border-color 0.2s ease;
}
.ev-card:hover {
    border-color: %(border_strong)s;
}
.ev-card-hdr {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: %(bg_elevated)s;
    border-bottom: 1px solid %(border_subtle)s;
}
.ev-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 5px;
    line-height: 1.4;
}
.ev-score-high { background: rgba(16,185,129,0.2); color: %(score_high)s; border: 1px solid rgba(16,185,129,0.3); }
.ev-score-med  { background: rgba(245,158,11,0.15); color: %(score_med)s; border: 1px solid rgba(245,158,11,0.25); }
.ev-score-low  { background: rgba(107,114,128,0.15); color: %(score_low)s; border: 1px solid rgba(107,114,128,0.25); }

.ev-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: %(text_primary)s;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.ev-card-body {
    padding: 12px 14px;
}
.ev-field-label {
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: %(text_dim)s;
    margin-bottom: 3px;
}
.ev-field-label:not(:first-child) {
    margin-top: 10px;
}
.ev-field-text {
    font-size: 0.82rem;
    line-height: 1.6;
    color: %(text_muted)s;
}

/* ── TIMING FOOTER ─────────────────────────────────────── */
.timing-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: %(bg_card)s;
    border: 1px solid %(border_subtle)s;
    border-radius: 10px;
}
.timing-left {
    display: flex;
    align-items: center;
    gap: 18px;
}
.timing-right {
    display: flex;
    align-items: center;
    gap: 6px;
}
.t-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.7rem;
    color: %(text_dim)s;
}
.t-item .t-icon {
    font-size: 0.72rem;
    opacity: 0.7;
}
.t-item .t-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: %(text_muted)s;
    font-size: 0.72rem;
}
.t-item-gpu {
    font-size: 0.68rem;
    color: %(text_faint)s;
    font-weight: 500;
}
.t-item-gpu .gpu-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 5px;
    background: %(bg_elevated)s;
    border: 1px solid %(border_subtle)s;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: %(text_dim)s;
}

/* ── PLACEHOLDER (pre-inference) ───────────────────────── */
.output-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 50px 20px;
    text-align: center;
    min-height: 220px;
}
.placeholder-icon-wrap {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: %(bg_card)s;
    border: 1px solid %(border_default)s;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-bottom: 14px;
}
.placeholder-title {
    font-size: 0.88rem;
    font-weight: 500;
    color: %(text_muted)s;
    margin-bottom: 6px;
}
.placeholder-sub {
    font-size: 0.78rem;
    color: %(text_dim)s;
    line-height: 1.5;
}

/* ── ERROR CARD ────────────────────────────────────────── */
.error-card {
    background: %(bg_card)s;
    border: 1px solid rgba(244, 63, 94, 0.3);
    border-left: 3px solid %(error)s;
    border-radius: 12px;
    padding: 18px 20px;
}
.error-card .err-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: %(error)s;
    margin-bottom: 8px;
}
.error-card .err-text {
    font-size: 0.85rem;
    line-height: 1.6;
    color: %(text_muted)s;
}

/* ── SCROLLBAR ─────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: %(border_default)s; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: %(text_faint)s; }

/* ── GRADIO OVERRIDES ──────────────────────────────────── */
/* Remove Gradio default footer */
footer { display: none !important; }

/* Clean up accordion internals */
.gradio-accordion {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.gradio-accordion > .label-wrap {
    border-radius: 10px !important;
}

/* Remove visible label from image component we handle manually */
.upload-area > label {
    display: none !important;
}

/* Responsive */
@media (max-width: 768px) {
    .mrag-header { padding: 10px 14px; }
    .main-row { flex-direction: column !important; }
    .input-panel, .output-panel { border-radius: 10px !important; }
}
""" % COLORS
