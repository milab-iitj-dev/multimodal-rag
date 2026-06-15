"""
Question Sensitivity Analyzer.

Measures how much the question text influences retrieval results
when the same image is queried with different clinical questions.

Core comparison:
    Baseline (image-only):
        Same image + different question → almost identical top-K
        → high pairwise overlap (the modality-bias problem)

    Current system (dual-index + RRF + reranking):
        Same image + different question → meaningfully different top-K
        → lower pairwise overlap (question now matters)

The metric used is pairwise Jaccard similarity of top-K retrieved
document sets. High Jaccard means results don't change with the
question; low Jaccard means the question influences retrieval.
"""

from typing import List, Dict, Any
from collections import defaultdict

from src.utils.logging_utils import setup_logger

logger = setup_logger("ablation.sensitivity")


def analyze_question_sensitivity(
    baseline_results: List[Dict[str, Any]],
    current_results: List[Dict[str, Any]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Analyze how question variation affects retrieval for the same image.

    Groups query results by source image, then for each image that was
    queried with at least two different questions, computes pairwise
    Jaccard overlap of the top-K retrieved document sets.

    Args:
        baseline_results: Per-query results from the baseline evaluator.
        current_results:  Per-query results from the current evaluator.
        top_k:            Number of top results to compare (default 3).

    Returns:
        Dict with aggregate overlap stats and concrete examples.
    """
    # Group by source image
    baseline_by_image = _group_by_image(baseline_results)
    current_by_image = _group_by_image(current_results)

    examples = []
    baseline_overlaps = []
    current_overlaps = []

    # Analyze images that have multiple different questions
    for image_key, b_queries in baseline_by_image.items():
        c_queries = current_by_image.get(image_key, [])

        # Need at least 2 different questions on the same image
        unique_b_questions = {q["query_text"] for q in b_queries}
        if len(unique_b_questions) < 2:
            continue

        # Compute pairwise top-K overlap for baseline and current
        b_overlap = _compute_avg_overlap(b_queries, top_k)
        c_overlap = _compute_avg_overlap(c_queries, top_k) if c_queries else 0.0

        baseline_overlaps.append(b_overlap)
        current_overlaps.append(c_overlap)

        # Collect concrete examples for the report (up to 5)
        if len(examples) < 5:
            example = _build_example(
                image_key, b_queries, c_queries, top_k,
                b_overlap, c_overlap,
            )
            if example:
                examples.append(example)

    # Sort examples by largest overlap reduction (most dramatic improvement)
    examples.sort(
        key=lambda e: e["baseline_overlap"] - e["current_overlap"],
        reverse=True,
    )

    # Aggregate statistics
    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    avg_baseline = _mean(baseline_overlaps)
    avg_current = _mean(current_overlaps)

    analysis = {
        "num_images_analyzed": len(baseline_overlaps),
        "baseline_avg_overlap": avg_baseline,
        "current_avg_overlap": avg_current,
        "overlap_reduction": round(avg_baseline - avg_current, 4),
        "examples": examples,
    }

    logger.info(
        f"Sensitivity analysis complete: {len(baseline_overlaps)} images — "
        f"baseline overlap={avg_baseline:.4f}, "
        f"current overlap={avg_current:.4f}, "
        f"reduction={analysis['overlap_reduction']:.4f}"
    )

    return analysis


# ------------------------------------------------------------------ #
#  Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _group_by_image(
    results: List[Dict[str, Any]],
) -> Dict[str, List[Dict]]:
    """
    Group query results by source image.

    Uses source_doc_id as the grouping key (same document = same
    image). Falls back to query_image_path if source_doc_id is
    not available.
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        key = r.get("source_doc_id") or r.get("query_image_path", "")
        if key:
            groups[key].append(r)
    return dict(groups)


def _compute_avg_overlap(
    queries: List[Dict[str, Any]],
    top_k: int,
) -> float:
    """
    Compute average pairwise Jaccard overlap of top-K result sets.

    Returns a value in [0.0, 1.0]:
        1.0 = all query pairs return identical top-K (bad: question ignored)
        0.0 = no overlap at all between any pair
    """
    if len(queries) < 2:
        return 0.0

    overlaps = []
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            set_i = set(queries[i]["retrieved_ids"][:top_k])
            set_j = set(queries[j]["retrieved_ids"][:top_k])

            if not set_i or not set_j:
                continue

            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            jaccard = intersection / union if union > 0 else 0.0
            overlaps.append(jaccard)

    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def _build_example(
    image_key: str,
    b_queries: List[Dict],
    c_queries: List[Dict],
    top_k: int,
    b_overlap: float,
    c_overlap: float,
) -> Dict[str, Any]:
    """Build a concrete example for the report."""
    # Build a lookup from query_text → current result
    c_lookup = {q["query_text"]: q for q in c_queries}

    questions = []
    for bq in b_queries[:3]:  # Show up to 3 questions per image
        cq = c_lookup.get(bq["query_text"])

        questions.append({
            "question": bq["query_text"],
            "finding": bq["finding"],
            "baseline_top_k": bq["retrieved_ids"][:top_k],
            "baseline_scores": [
                round(s, 4) for s in bq.get("retrieved_scores", [])[:top_k]
            ],
            "current_top_k": cq["retrieved_ids"][:top_k] if cq else [],
            "current_scores": [
                round(s, 4) for s in cq.get("retrieved_scores", [])[:top_k]
            ] if cq else [],
        })

    if len(questions) < 2:
        return None

    return {
        "image": image_key,
        "num_questions": len(b_queries),
        "baseline_overlap": round(b_overlap, 4),
        "current_overlap": round(c_overlap, 4),
        "questions": questions,
    }
