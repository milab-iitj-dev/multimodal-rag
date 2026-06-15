"""
Grounding Metrics — supported / contradicted / unsupported rates.

Evaluates whether generated answers are consistent with the
retrieved evidence, using the GroundingResult from the pipeline.
"""

from typing import List, Dict, Any


def classify_grounding(grounding_result) -> str:
    """
    Classify a single grounding result into one of three categories.

    Args:
        grounding_result: GroundingResult from the pipeline.

    Returns:
        "supported", "contradicted", or "unsupported".
    """
    if grounding_result is None:
        return "unsupported"

    if grounding_result.contradiction_detected:
        return "contradicted"

    if grounding_result.is_grounded:
        return "supported"

    return "unsupported"


def compute_grounding_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute aggregate grounding metrics.

    Args:
        results: List of dicts, each with:
            - "grounding_label": str ("supported"/"contradicted"/"unsupported")
            - "query_type": str (for per-type breakdown)
            - "was_corrected": bool
            - "confidence_level": str

    Returns:
        Aggregate and per-query-type grounding metrics.
    """
    if not results:
        return {"aggregate": {}, "per_query_type": {}}

    # Aggregate counts
    total = len(results)
    supported = sum(1 for r in results if r["grounding_label"] == "supported")
    contradicted = sum(1 for r in results if r["grounding_label"] == "contradicted")
    unsupported = sum(1 for r in results if r["grounding_label"] == "unsupported")
    corrected = sum(1 for r in results if r.get("was_corrected", False))

    aggregate = {
        "total": total,
        "supported": supported,
        "contradicted": contradicted,
        "unsupported": unsupported,
        "corrected": corrected,
        "supported_rate": round(supported / total, 4),
        "contradicted_rate": round(contradicted / total, 4),
        "unsupported_rate": round(unsupported / total, 4),
        "correction_rate": round(corrected / total, 4),
    }

    # Per query type
    type_groups: Dict[str, List] = {}
    for r in results:
        qt = r.get("query_type", "unknown")
        type_groups.setdefault(qt, []).append(r)

    per_type = {}
    for qt, group in type_groups.items():
        n = len(group)
        per_type[qt] = {
            "total": n,
            "supported": sum(1 for r in group if r["grounding_label"] == "supported"),
            "contradicted": sum(1 for r in group if r["grounding_label"] == "contradicted"),
            "unsupported": sum(1 for r in group if r["grounding_label"] == "unsupported"),
            "supported_rate": round(
                sum(1 for r in group if r["grounding_label"] == "supported") / n, 4
            ),
            "contradicted_rate": round(
                sum(1 for r in group if r["grounding_label"] == "contradicted") / n, 4
            ),
        }

    # Confidence distribution
    confidence_dist = {}
    for r in results:
        cl = r.get("confidence_level", "unknown")
        confidence_dist[cl] = confidence_dist.get(cl, 0) + 1

    return {
        "aggregate": aggregate,
        "per_query_type": per_type,
        "confidence_distribution": confidence_dist,
    }
