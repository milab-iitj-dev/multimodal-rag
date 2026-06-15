"""
Retrieval Metrics -- Recall@k (Hit@k), MRR, nDCG, Precision@k.

Evaluates retrieval quality by comparing retrieved document IDs
against gold-standard relevant document IDs.

IMPORTANT: Recall@k here is binary Hit@k (standard for RAG):
    Recall@k(query) = 1 if ANY relevant document appears in top-k, else 0
    Then averaged across all queries.

This is NOT set-overlap recall (|retrieved & gold| / |gold|) which
would give near-zero scores when |gold| >> k (e.g., 349 cardiomegaly
docs but only retrieving top-3).
"""

import math
from typing import List, Dict, Any


def hit_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int,
) -> float:
    """
    Hit@k (binary recall): 1.0 if ANY gold document appears in top-k.

    This is the standard RAG retrieval metric. For each query, we ask:
    "Did the retriever find at least one relevant document in the top-k?"

    Args:
        retrieved_ids: Ranked list of retrieved doc IDs (best first).
        gold_ids:      List of relevant document IDs.
        k:             Cutoff rank.

    Returns:
        1.0 if hit, 0.0 if miss.
    """
    if not gold_ids:
        return 0.0
    gold_set = set(gold_ids)
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in gold_set for doc_id in top_k) else 0.0


# Alias for backward compatibility and clarity in reports
recall_at_k = hit_at_k


def precision_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int,
) -> float:
    """
    Precision@k: fraction of top-k documents that are relevant.

    Args:
        retrieved_ids: Ranked list of retrieved doc IDs (best first).
        gold_ids:      List of relevant document IDs.
        k:             Cutoff rank.

    Returns:
        Precision score in [0.0, 1.0].
    """
    if not retrieved_ids or k == 0:
        return 0.0
    gold_set = set(gold_ids)
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for d in top_k if d in gold_set)
    return relevant_in_top_k / len(top_k)


def reciprocal_rank(
    retrieved_ids: List[str],
    gold_ids: List[str],
) -> float:
    """
    Reciprocal Rank: 1/rank of the first relevant document.

    Returns 0.0 if no relevant document is found.
    """
    gold_set = set(gold_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    Uses binary relevance: 1 if doc is in gold set, 0 otherwise.
    IDCG is computed with min(|gold|, k) relevant docs at top.
    """
    gold_set = set(gold_ids)

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in gold_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG (all relevant docs at the top)
    ideal_rels = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def relevance_vector(
    retrieved_ids: List[str],
    gold_ids: List[str],
) -> List[int]:
    """
    Binary relevance vector for a ranked list.

    Returns a list of 0/1 values, one per retrieved doc.
    Useful for per-query diagnostics.
    """
    gold_set = set(gold_ids)
    return [1 if doc_id in gold_set else 0 for doc_id in retrieved_ids]


def per_query_diagnostics(
    result: Dict[str, Any],
    k_values: List[int] = None,
) -> Dict[str, Any]:
    """
    Compute per-query diagnostic info for a single retrieval result.

    Returns a dict with all individual metrics and the relevance vector.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    retrieved = result["retrieved_ids"]
    gold = result["gold_ids"]

    rel_vec = relevance_vector(retrieved, gold)
    rr = reciprocal_rank(retrieved, gold)

    diag = {
        "query_id": result.get("query_id", "?"),
        "query_text": result.get("query_text", "?"),
        "query_mode": result.get("query_mode", "?"),
        "finding": result.get("finding", "?"),
        "num_gold_docs": len(gold),
        "retrieved_ids": retrieved,
        "relevance_vector": rel_vec,
        "reciprocal_rank": round(rr, 4),
    }

    for k in k_values:
        diag[f"hit@{k}"] = hit_at_k(retrieved, gold, k)
        diag[f"precision@{k}"] = round(precision_at_k(retrieved, gold, k), 4)
        diag[f"ndcg@{k}"] = round(ndcg_at_k(retrieved, gold, k), 4)

    return diag


def compute_retrieval_metrics(
    results: List[Dict[str, Any]],
    k_values: List[int] = None,
) -> Dict[str, Any]:
    """
    Compute aggregate retrieval metrics over a list of query results.

    Uses Hit@k (binary recall) -- the standard for RAG evaluation.

    Args:
        results: List of dicts, each with:
            - "retrieved_ids": List[str]
            - "gold_ids": List[str]
            - "query_mode": str (optional, for grouping)
        k_values: List of k values for Hit@k and nDCG@k.

    Returns:
        Dict with aggregate metrics, per-mode breakdowns, and diagnostics.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    # Per-query diagnostics
    diagnostics = [per_query_diagnostics(r, k_values) for r in results]

    # Aggregate accumulators
    all_hits = {k: [] for k in k_values}
    all_precision = {k: [] for k in k_values}
    all_rr = []
    all_ndcg = {k: [] for k in k_values}

    # Per query mode
    mode_results: Dict[str, Dict] = {}

    for diag in diagnostics:
        mode = diag["query_mode"]

        rr = diag["reciprocal_rank"]
        all_rr.append(rr)

        for k in k_values:
            all_hits[k].append(diag[f"hit@{k}"])
            all_precision[k].append(diag[f"precision@{k}"])
            all_ndcg[k].append(diag[f"ndcg@{k}"])

        # Per-mode accumulation
        if mode not in mode_results:
            mode_results[mode] = {
                "rr": [],
                **{f"hit@{k}": [] for k in k_values},
                **{f"precision@{k}": [] for k in k_values},
                **{f"ndcg@{k}": [] for k in k_values},
            }
        mode_results[mode]["rr"].append(rr)
        for k in k_values:
            mode_results[mode][f"hit@{k}"].append(diag[f"hit@{k}"])
            mode_results[mode][f"precision@{k}"].append(diag[f"precision@{k}"])
            mode_results[mode][f"ndcg@{k}"].append(diag[f"ndcg@{k}"])

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    # Aggregate metrics
    aggregate = {
        "mrr": _mean(all_rr),
        "num_queries": len(results),
    }
    for k in k_values:
        aggregate[f"recall@{k}"] = _mean(all_hits[k])
        aggregate[f"precision@{k}"] = _mean(all_precision[k])
        aggregate[f"ndcg@{k}"] = _mean(all_ndcg[k])

    # Per-mode metrics
    per_mode = {}
    for mode, data in mode_results.items():
        per_mode[mode] = {
            "mrr": _mean(data["rr"]),
            "num_queries": len(data["rr"]),
        }
        for k in k_values:
            per_mode[mode][f"recall@{k}"] = _mean(data[f"hit@{k}"])
            per_mode[mode][f"precision@{k}"] = _mean(data[f"precision@{k}"])
            per_mode[mode][f"ndcg@{k}"] = _mean(data[f"ndcg@{k}"])

    return {
        "aggregate": aggregate,
        "per_mode": per_mode,
        "diagnostics": diagnostics,
    }

