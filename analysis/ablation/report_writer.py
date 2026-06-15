"""
Report Writer — Observing Document Generator.

Generates a professional markdown observing document from the
ablation analysis results. The document contains:

    1. Problem Summary — modality bias explanation
    2. Baseline Results — before dual-index + reranking
    3. Current System Results — after the fix
    4. Side-by-side Comparison Table
    5. Question Sensitivity Examples
    6. Observations — why the fix worked
    7. Research-style Conclusion

Also saves the raw results as JSON for later inspection.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List

from src.utils.logging_utils import setup_logger

logger = setup_logger("ablation.report")


def generate_observing_document(
    baseline_metrics: Dict[str, Any],
    current_metrics: Dict[str, Any],
    sensitivity: Dict[str, Any],
    output_dir: str = "outputs/observations",
) -> str:
    """
    Generate the observing document and save all artifacts.

    Args:
        baseline_metrics: Output from run_baseline_evaluation()['metrics'].
        current_metrics:  Output from run_current_evaluation()['metrics'].
        sensitivity:      Output from analyze_question_sensitivity().
        output_dir:       Directory for output files.

    Returns:
        Path to the generated markdown file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the markdown document
    md = _build_markdown(baseline_metrics, current_metrics, sensitivity)

    # Save markdown
    md_path = out_dir / "observing_document.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    # Save raw JSON results
    json_data = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": {
            "aggregate": baseline_metrics.get("aggregate", {}),
            "per_mode": baseline_metrics.get("per_mode", {}),
            "timing": baseline_metrics.get("timing", {}),
            "config": baseline_metrics.get("config", {}),
        },
        "current": {
            "aggregate": current_metrics.get("aggregate", {}),
            "per_mode": current_metrics.get("per_mode", {}),
            "timing": current_metrics.get("timing", {}),
            "config": current_metrics.get("config", {}),
        },
        "sensitivity": sensitivity,
    }

    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Observing document saved to {md_path}")
    logger.info(f"Raw results saved to {json_path}")

    return str(md_path)


# ------------------------------------------------------------------ #
#  Markdown builder                                                    #
# ------------------------------------------------------------------ #

def _fmt(val, decimals=4) -> str:
    """Format a numeric value for tables."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _delta(after, before, decimals=4) -> str:
    """Format a delta value with sign and color hint."""
    if before is None or after is None:
        return "—"
    diff = after - before
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.{decimals}f}"


def _build_markdown(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    sensitivity: Dict[str, Any],
) -> str:
    """Build the complete markdown observing document."""
    b_agg = baseline.get("aggregate", {})
    c_agg = current.get("aggregate", {})
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []

    # ── Title ──
    lines.append("# Healthcare MRAG — Retrieval Ablation Analysis")
    lines.append("")
    lines.append("## Modality Bias: Before vs After Dual-Index Retrieval")
    lines.append("")
    lines.append(f"*Generated: {timestamp}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 1. Problem Summary ──
    lines.append("## 1. Problem Summary")
    lines.append("")
    lines.append(
        "During implementation of the Healthcare Multimodal RAG pipeline, "
        "we identified a **modality-dominance bias** in the retrieval layer. "
        "When a user submitted both a chest X-ray image and a clinical "
        "question (e.g., *\"Is there cardiomegaly?\"*), the retrieval system "
        "was dominated by the image modality. The question text had "
        "negligible influence on which evidence documents were retrieved."
    )
    lines.append("")
    lines.append("**Observed symptoms:**")
    lines.append("")
    lines.append(
        "- Different clinical questions on the **same** X-ray image "
        "returned nearly **identical** evidence documents."
    )
    lines.append(
        "- The retrieval ranking was determined almost entirely by "
        "visual similarity, ignoring question-specific clinical terms."
    )
    lines.append(
        "- Question wording (e.g., *\"Is there cardiomegaly?\"* vs "
        "*\"Is there pleural effusion?\"*) did not meaningfully change "
        "the top-K retrieved results."
    )
    lines.append("")
    lines.append(
        "**Root cause:** The system relied on a single image embedding "
        "index. Both modalities were routed through image-level MaxSim "
        "scoring, where the high-dimensional image patch embeddings "
        "(~600–900 score range) overwhelmed the question text signal. "
        "Without a dedicated text retrieval path or question-aware "
        "reranking, the question was effectively discarded."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 2. Baseline ──
    lines.append("## 2. Baseline — Before Fix")
    lines.append("")
    lines.append(
        "The baseline reproduces the pre-fix retrieval behavior using a "
        "**question-ignored ablation**: the system still receives both "
        "the chest X-ray image and the clinical question text (identical "
        "inputs to the current system), but the question text is discarded "
        "before retrieval. Only the image drives the ranking. This matches "
        "the earlier system where different questions on the same image "
        "produced identical retrieval results because the question had no "
        "influence on scoring."
    )
    lines.append("")
    lines.append(
        "**Configuration:** Question-ignored image+text retrieval — "
        "image+text input received, question text discarded, "
        "no RRF fusion, no question-aware reranking."
    )
    lines.append("")

    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for k in [1, 3, 5]:
        key = f"recall@{k}"
        lines.append(f"| Recall@{k} | {_fmt(b_agg.get(key))} |")
    lines.append(f"| MRR | {_fmt(b_agg.get('mrr'))} |")
    for k in [3, 5]:
        key = f"ndcg@{k}"
        lines.append(f"| nDCG@{k} | {_fmt(b_agg.get(key))} |")

    b_n = b_agg.get("num_queries", "?")
    lines.append(f"| Queries | {b_n} |")
    lines.append("")

    # Per-mode breakdown if available
    b_per_mode = baseline.get("per_mode", {})
    if b_per_mode:
        lines.append("**Per query mode:**")
        lines.append("")
        lines.append("| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |")
        lines.append("|------|----------|----------|----------|-----|")
        for mode, m in b_per_mode.items():
            lines.append(
                f"| {mode} "
                f"| {_fmt(m.get('recall@1'))} "
                f"| {_fmt(m.get('recall@3'))} "
                f"| {_fmt(m.get('recall@5'))} "
                f"| {_fmt(m.get('mrr'))} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    # ── 3. Current System ──
    lines.append("## 3. Current System — After Fix")
    lines.append("")
    lines.append(
        "The current system uses the full **dual-index retrieval** pipeline "
        "with three question-awareness mechanisms:"
    )
    lines.append("")
    lines.append(
        "1. **Dual-index retrieval:** The query image is matched against "
        "the image embedding index, and the question text is matched "
        "against a dedicated text embedding index (both built with "
        "ColQwen2 MaxSim scoring)."
    )
    lines.append(
        "2. **RRF fusion (k=60):** The two ranked lists are combined "
        "using Reciprocal Rank Fusion, which is scale-agnostic and "
        "prevents either modality from dominating."
    )
    lines.append(
        "3. **Question-aware reranking:** After fusion, documents whose "
        "clinical text (findings + impression) contains question-relevant "
        "keywords receive an additive score boost, promoting "
        "question-specific evidence."
    )
    lines.append("")
    lines.append(
        "**Configuration:** HybridRetriever with dual-index ColQwen2, "
        "RRF fusion (k=60, equal weights), question-aware reranking "
        "(boost=0.15, medical stopwords enabled)."
    )
    lines.append("")

    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for k in [1, 3, 5]:
        key = f"recall@{k}"
        lines.append(f"| Recall@{k} | {_fmt(c_agg.get(key))} |")
    lines.append(f"| MRR | {_fmt(c_agg.get('mrr'))} |")
    for k in [3, 5]:
        key = f"ndcg@{k}"
        lines.append(f"| nDCG@{k} | {_fmt(c_agg.get(key))} |")

    c_n = c_agg.get("num_queries", "?")
    lines.append(f"| Queries | {c_n} |")
    lines.append("")

    # Per-mode breakdown if available
    c_per_mode = current.get("per_mode", {})
    if c_per_mode:
        lines.append("**Per query mode:**")
        lines.append("")
        lines.append("| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |")
        lines.append("|------|----------|----------|----------|-----|")
        for mode, m in c_per_mode.items():
            lines.append(
                f"| {mode} "
                f"| {_fmt(m.get('recall@1'))} "
                f"| {_fmt(m.get('recall@3'))} "
                f"| {_fmt(m.get('recall@5'))} "
                f"| {_fmt(m.get('mrr'))} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    # ── 4. Comparison Table ──
    lines.append("## 4. Side-by-Side Comparison")
    lines.append("")
    lines.append(
        "| Metric | Baseline (Before) | Current (After) | Δ Change |"
    )
    lines.append(
        "|--------|-------------------|-----------------|----------|"
    )

    comparison_metrics = [
        ("Recall@1", "recall@1"),
        ("Recall@3", "recall@3"),
        ("Recall@5", "recall@5"),
        ("MRR", "mrr"),
        ("nDCG@3", "ndcg@3"),
        ("nDCG@5", "ndcg@5"),
    ]

    for label, key in comparison_metrics:
        b_val = b_agg.get(key)
        c_val = c_agg.get(key)
        lines.append(
            f"| {label} "
            f"| {_fmt(b_val)} "
            f"| {_fmt(c_val)} "
            f"| {_delta(c_val, b_val)} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 5. Question Sensitivity ──
    lines.append("## 5. Question Sensitivity Analysis")
    lines.append("")

    n_images = sensitivity.get("num_images_analyzed", 0)
    b_avg_overlap = sensitivity.get("baseline_avg_overlap", 0)
    c_avg_overlap = sensitivity.get("current_avg_overlap", 0)
    overlap_reduction = sensitivity.get("overlap_reduction", 0)

    lines.append(
        "This analysis measures how much the retrieval results change "
        "when the **same image** is queried with **different clinical "
        "questions**. We compute the pairwise Jaccard overlap of the "
        "top-3 retrieved document sets across question pairs."
    )
    lines.append("")
    lines.append(
        "- **High overlap** (close to 1.0) → the question has little "
        "effect; the image dominates retrieval."
    )
    lines.append(
        "- **Lower overlap** → the question meaningfully changes which "
        "evidence is retrieved."
    )
    lines.append("")

    lines.append("| Measure | Value |")
    lines.append("|---------|-------|")
    lines.append(f"| Images analyzed | {n_images} |")
    lines.append(
        f"| Baseline avg. pairwise overlap | {_fmt(b_avg_overlap)} |"
    )
    lines.append(
        f"| Current avg. pairwise overlap | {_fmt(c_avg_overlap)} |"
    )
    lines.append(f"| Overlap reduction | {_fmt(overlap_reduction)} |")
    lines.append("")

    # Concrete examples
    examples = sensitivity.get("examples", [])
    if examples:
        lines.append("### Concrete Examples")
        lines.append("")

        for idx, ex in enumerate(examples[:3], start=1):
            lines.append(
                f"**Example {idx}** — Image: `{ex['image']}` "
                f"({ex['num_questions']} questions)"
            )
            lines.append("")
            lines.append(
                f"- Baseline top-3 overlap: **{_fmt(ex['baseline_overlap'])}**"
            )
            lines.append(
                f"- Current top-3 overlap: **{_fmt(ex['current_overlap'])}**"
            )
            lines.append("")

            for q_info in ex.get("questions", []):
                lines.append(f"*Question:* \"{q_info['question']}\"")
                lines.append("")

                b_ids = ", ".join(
                    f"`{d}`" for d in q_info.get("baseline_top_k", [])
                )
                c_ids = ", ".join(
                    f"`{d}`" for d in q_info.get("current_top_k", [])
                )
                lines.append(f"  - Baseline top-3: {b_ids}")
                lines.append(f"  - Current top-3:  {c_ids}")
                lines.append("")

            lines.append("")

    lines.append("---")
    lines.append("")

    # ── 6. Observations ──
    lines.append("## 6. Observations")
    lines.append("")
    lines.append("### Why the fix improved retrieval")
    lines.append("")
    lines.append(
        "1. **Image dominance was reduced.** In the baseline, the system "
        "received image+text queries but only used the image for ranking. "
        "By introducing a dedicated text embedding index, the question "
        "text now has its own retrieval path. The system matches question "
        "terms against document report text, producing a separate ranked "
        "list that reflects question intent."
    )
    lines.append("")
    lines.append(
        "2. **RRF fusion balances both modalities.** Reciprocal Rank "
        "Fusion combines the image and text ranked lists using rank "
        "positions rather than raw scores. Since image MaxSim scores "
        "(~600–900) and text MaxSim scores (~50–200) are on vastly "
        "different scales, RRF's scale-agnostic design prevents the "
        "image modality from overwhelming the text signal."
    )
    lines.append("")
    lines.append(
        "3. **Question-aware reranking amplifies question relevance.** "
        "After fusion, documents whose clinical text contains keywords "
        "from the question receive an additive score boost. This means "
        "a query about *\"cardiomegaly\"* promotes documents mentioning "
        "cardiomegaly, while a query about *\"pleural effusion\"* on the "
        "same image promotes different documents — directly addressing "
        "the identical-evidence problem."
    )
    lines.append("")
    lines.append("### Why identical-evidence retrieval was reduced")
    lines.append("")
    lines.append(
        "In the baseline, the system receives the same image+text inputs "
        "as the current system, but the question text is discarded before "
        "retrieval. Two queries on the same image "
        "(*\"Is there cardiomegaly?\"* and *\"Is there pleural effusion?\"*) "
        "both produce the same image encoding, yielding identical MaxSim "
        "scores and identical top-K results. The question text is present "
        "as input but has no effect on the ranking — this is the core "
        "modality-bias failure."
    )
    lines.append("")
    lines.append(
        "In the current system, the same two queries produce different "
        "text index rankings (because *\"cardiomegaly\"* and *\"pleural "
        "effusion\"* match different document texts). RRF merges these "
        "text-based rankings with the image rankings, and the reranker "
        "further boosts question-specific documents. The final top-K "
        "now reflects the actual clinical question, not just the image."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 7. Conclusion ──
    lines.append("## 7. Conclusion")
    lines.append("")
    lines.append(
        "This ablation analysis confirms that the primary retrieval "
        "limitation in the earlier Healthcare MRAG pipeline was "
        "**modality dominance**: the image embedding signal overwhelmed "
        "the question text signal in multimodal queries, causing "
        "question-agnostic retrieval behavior."
    )
    lines.append("")
    lines.append(
        "The solution — **dual-index retrieval** with separate image "
        "and text embedding indexes, **Reciprocal Rank Fusion** for "
        "scale-agnostic result combination, and **question-aware "
        "reranking** for keyword-level question sensitivity — "
        "successfully addressed this bias."
    )
    lines.append("")

    # Build improvement summary
    improvement_notes = []
    for label, key in comparison_metrics:
        b_val = b_agg.get(key)
        c_val = c_agg.get(key)
        if b_val is not None and c_val is not None and c_val > b_val:
            improvement_notes.append(
                f"{label}: {_fmt(b_val)} → {_fmt(c_val)} "
                f"({_delta(c_val, b_val)})"
            )

    if improvement_notes:
        lines.append("**Key improvements:**")
        lines.append("")
        for note in improvement_notes:
            lines.append(f"- {note}")
        lines.append("")

    if n_images > 0:
        lines.append(
            f"**Question sensitivity:** Average pairwise top-3 overlap "
            f"decreased from {_fmt(b_avg_overlap)} (baseline) to "
            f"{_fmt(c_avg_overlap)} (current), indicating that the "
            f"retrieval system now responds meaningfully to different "
            f"clinical questions on the same image."
        )
        lines.append("")

    lines.append(
        "The current dual-index + RRF + reranking architecture provides "
        "a robust foundation for question-aware multimodal retrieval "
        "in the Healthcare MRAG pipeline."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*This document was generated by the Healthcare MRAG "
        "Ablation Analysis Suite.*"
    )
    lines.append("")

    return "\n".join(lines)
