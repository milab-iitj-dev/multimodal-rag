"""
Output formatters for the Healthcare MRAG inference UI.

Converts RAGOutput and RetrievedDocument dataclasses into
display-ready HTML strings for the Gradio components.

These are pure functions — no Gradio imports, no backend imports,
no side effects. They receive plain data and return strings.
"""

from typing import List, Optional


# ================================================================
# Query Mode Detection
# ================================================================

def detect_query_mode(image, question: str) -> str:
    """
    Auto-detect the query mode from user inputs.

    Args:
        image:    PIL image or None.
        question: Question string (may be empty).

    Returns:
        HTML badge string for the detected mode.
    """
    has_image = image is not None
    has_text = bool(question and question.strip())

    if has_image and has_text:
        icon, label = "⚡", "Image + Text"
    elif has_image:
        icon, label = "🩻", "Image Only"
    elif has_text:
        icon, label = "💬", "Text Only"
    else:
        icon, label = "⏳", "Waiting for input"

    return (
        f'<div class="mode-badge">'
        f'<span class="mi">{icon}</span> '
        f'{label} · Auto-detected'
        f'</div>'
    )


# ================================================================
# Answer Card — the most prominent UI element
# ================================================================

def format_answer(answer: str, relevance_score: float = 0.0) -> str:
    """
    Format the clinical answer as a premium HTML card.

    Features:
      - Left accent border (clinical green)
      - Header with label + confidence badge
      - Clean body typography
      - Optional relevance score bar

    Args:
        answer: The generated answer string from the RAG pipeline.
        relevance_score: Optional relevance/confidence score (0-1).

    Returns:
        HTML string for the answer card.
    """
    if not answer:
        return _empty_placeholder()

    # Build relevance bar if score available
    relevance_html = ""
    if relevance_score and relevance_score > 0:
        pct = min(relevance_score * 100, 100)
        relevance_html = (
            f'<div class="relevance-bar">'
            f'  <span class="rlabel">Relevance</span>'
            f'  <div class="rtrack"><div class="rfill" style="width: {pct:.0f}%"></div></div>'
            f'  <span class="rval">{relevance_score:.2f}</span>'
            f'</div>'
        )

    return (
        f'<div class="answer-card">'
        f'  <div class="ans-hdr">'
        f'    <span class="ans-label">Analysis Result</span>'
        f'    <span class="ans-badge">RAG-Enhanced</span>'
        f'  </div>'
        f'  <div class="ans-body">{_escape_html(answer)}</div>'
        f'  {relevance_html}'
        f'</div>'
    )


# ================================================================
# Retrieved Evidence Cards
# ================================================================

def format_evidence(retrieved_docs) -> str:
    """
    Format retrieved evidence documents as elegant HTML cards.

    Each card has a header (score + case ID) and body (findings, impression).

    Args:
        retrieved_docs: List of RetrievedDocument from the pipeline.

    Returns:
        HTML string with all evidence cards.
    """
    if not retrieved_docs:
        return (
            '<div style="padding: 16px; text-align: center; '
            'color: #64748b; font-size: 0.82rem;">'
            'No evidence retrieved.</div>'
        )

    cards = []
    for doc in retrieved_docs:
        rank = doc.metadata.get("rank", "?")
        score = doc.score
        case_id = doc.doc_id
        findings = doc.metadata.get("findings", "")
        impression = doc.metadata.get("impression", "")

        # Score badge class
        if score >= 0.8:
            score_class = "ev-score-high"
        elif score >= 0.5:
            score_class = "ev-score-med"
        else:
            score_class = "ev-score-low"

        # Build body fields
        body_parts = []
        if findings:
            body_parts.append(
                f'<div class="ev-field-label">Findings</div>'
                f'<div class="ev-field-text">{_escape_html(_truncate(findings, 350))}</div>'
            )
        if impression:
            body_parts.append(
                f'<div class="ev-field-label">Impression</div>'
                f'<div class="ev-field-text">{_escape_html(_truncate(impression, 350))}</div>'
            )
        # Fallback to full report text
        if not findings and not impression and doc.text:
            body_parts.append(
                f'<div class="ev-field-label">Report</div>'
                f'<div class="ev-field-text">{_escape_html(_truncate(doc.text, 450))}</div>'
            )

        body_html = "".join(body_parts) if body_parts else (
            '<div class="ev-field-text" style="color: #475569;">No report data available.</div>'
        )

        card = (
            f'<div class="ev-card">'
            f'  <div class="ev-card-hdr">'
            f'    <span class="ev-score {score_class}">{score:.2f}</span>'
            f'    <span class="ev-title">Case {_escape_html(str(case_id))}</span>'
            f'  </div>'
            f'  <div class="ev-card-body">{body_html}</div>'
            f'</div>'
        )
        cards.append(card)

    return "".join(cards)


# ================================================================
# Timing Footer
# ================================================================

def format_timing(
    retrieval_time: float = 0.0,
    generation_time: float = 0.0,
    total_time: float = 0.0,
    gpu_name: Optional[str] = None,
) -> str:
    """
    Format pipeline timing as a compact status bar.

    Args:
        retrieval_time:  Seconds for retrieval.
        generation_time: Seconds for generation.
        total_time:      Total wall-clock time.
        gpu_name:        GPU name (optional).

    Returns:
        HTML string for the timing bar.
    """
    gpu_html = ""
    if gpu_name:
        gpu_html = (
            f'<div class="t-item-gpu">'
            f'  <span class="gpu-chip">⬡ {_escape_html(gpu_name)}</span>'
            f'</div>'
        )

    return (
        f'<div class="timing-bar">'
        f'  <div class="timing-left">'
        f'    {_t("⏱", "Total", f"{total_time:.2f}s")}'
        f'    {_t("◎", "Retrieval", f"{retrieval_time:.2f}s")}'
        f'    {_t("◉", "Inference", f"{generation_time:.2f}s")}'
        f'  </div>'
        f'  <div class="timing-right">{gpu_html}</div>'
        f'</div>'
    )


def _t(icon: str, label: str, value: str) -> str:
    """Build a single timing item."""
    return (
        f'<div class="t-item">'
        f'  <span class="t-icon">{icon}</span>'
        f'  {label}'
        f'  <span class="t-val">{value}</span>'
        f'</div>'
    )


# ================================================================
# Placeholder & Error States
# ================================================================

def _empty_placeholder() -> str:
    """Placeholder shown before inference runs."""
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


def format_error(error_msg: str) -> str:
    """Format an error message for display."""
    return (
        f'<div class="error-card">'
        f'  <div class="err-label">Error</div>'
        f'  <div class="err-text">{_escape_html(error_msg)}</div>'
        f'</div>'
    )


# ================================================================
# Helpers
# ================================================================

def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text with word-boundary ellipsis."""
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"
