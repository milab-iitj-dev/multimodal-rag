"""
Current System Evaluator — Full Dual-Index + RRF + Reranking.

Evaluates the current working retrieval system using the exact
same query set as the baseline evaluator. The current system uses:

    1. Dual-index retrieval (image + text indexes)
    2. Reciprocal Rank Fusion (RRF) to merge both ranked lists
    3. Question-aware reranking (keyword overlap boost)

This matches the production HybridRetriever behavior exactly.
No production code is modified — the existing HybridRetriever is
simply called through its public API.

Current path:
    Image+text query → image index + text index → RRF → reranking
    Text-only query  → text index → reranking
    Image-only query → image index only
"""

import time
from typing import List, Dict, Any

from PIL import Image

from src.retrieval.hybrid_retriever import HybridRetriever
from evaluation.metrics.retrieval_metrics import compute_retrieval_metrics
from src.utils.logging_utils import setup_logger

logger = setup_logger("ablation.current")


def run_current_evaluation(
    retriever: HybridRetriever,
    queries: List[Dict[str, Any]],
    top_k_values: List[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate retrieval using the current working system.

    Uses HybridRetriever with dual-index + RRF + reranking.
    The question text now influences retrieval through three
    mechanisms:

      1. Text index: the question is encoded and matched against
         document text embeddings via MaxSim scoring.
      2. RRF fusion: image and text rankings are combined using
         Reciprocal Rank Fusion, balancing both modalities.
      3. Question-aware reranking: keyword overlap between the
         question and document clinical text boosts the score
         of question-relevant documents.

    Args:
        retriever:    HybridRetriever with loaded dual index.
        queries:      List of query dicts from OpenITestBuilder
                      (same queries as baseline).
        top_k_values: k values for Recall@k / nDCG@k (default [1,3,5]).

    Returns:
        Dict with 'metrics' (aggregate, per_mode, diagnostics) and
        'eval_results' (per-query raw results).
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5]

    max_k = max(top_k_values)
    eval_results = []
    start_time = time.time()

    logger.info(
        f"Running CURRENT SYSTEM evaluation: {len(queries)} queries "
        f"(dual-index + RRF + reranking)"
    )

    for i, q in enumerate(queries):
        query_text = q["query_text"]
        query_image = None

        # Load query image if available
        if q.get("query_image_path") and q["query_mode"] != "text_only":
            try:
                query_image = Image.open(
                    q["query_image_path"]
                ).convert("RGB")
            except Exception as e:
                logger.warning(f"Cannot load image {q['query_image_path']}: {e}")

        # Skip image modes without an available image
        if q["query_mode"] in ("image_only", "hybrid") and query_image is None:
            continue

        # For image_only mode, use a generic query text
        if q["query_mode"] == "image_only":
            query_text = "What does this image show?"

        try:
            # ── CURRENT SYSTEM BEHAVIOR ──
            # Full HybridRetriever with dual-index + RRF + reranking.
            # The question text influences retrieval through all three
            # mechanisms (text index, RRF fusion, reranking).
            retrieved = retriever.retrieve(
                query=query_text,
                query_image=query_image,
                top_k=max_k,
            )

            retrieved_ids = [doc.doc_id for doc in retrieved]

            eval_results.append({
                "query_id": q["query_id"],
                "query_text": query_text,
                "query_mode": q["query_mode"],
                "finding": q["finding"],
                "retrieved_ids": retrieved_ids,
                "gold_ids": q["gold_ids"],
                "retrieved_scores": [doc.score for doc in retrieved],
                "source_doc_id": q.get("source_doc_id", ""),
                "query_image_path": q.get("query_image_path", ""),
            })

            if (i + 1) % 10 == 0:
                logger.info(f"  Current progress: {i + 1}/{len(queries)}")

        except Exception as e:
            logger.error(f"Current error on query {q['query_id']}: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"Current system complete: {len(eval_results)} queries "
        f"in {elapsed:.1f}s"
    )

    # Compute metrics using the shared metric functions
    metrics = compute_retrieval_metrics(eval_results, k_values=top_k_values)
    metrics["timing"] = {
        "total_seconds": round(elapsed, 2),
        "avg_per_query": round(
            elapsed / max(len(eval_results), 1), 3
        ),
    }
    metrics["config"] = {
        "mode": "current",
        "description": (
            "Dual-index (image + text) retrieval with "
            "RRF fusion and question-aware reranking"
        ),
        "num_queries": len(eval_results),
        "top_k_values": top_k_values,
    }

    return {
        "metrics": metrics,
        "eval_results": eval_results,
    }
