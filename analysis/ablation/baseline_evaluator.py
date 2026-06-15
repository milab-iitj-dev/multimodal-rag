"""
Baseline Evaluator — Question-Ignored Image+Text Retrieval.

Reproduces the pre-fix modality-biased retrieval behavior where
image+text queries are received but the question text has NO
influence on the retrieval ranking.

The input is still image + question (identical to the current system),
but the retrieval path ignores the question and routes everything
through image-level similarity only. This makes different questions
on the same image return identical evidence — exactly the
modality-bias problem that existed before dual-index retrieval
and question-aware reranking were introduced.

This is a READ-ONLY ablation. It does NOT modify any production code.
It calls the existing ColQwen2Retriever in a restricted mode that
reproduces the earlier failure.

Ablation path:
    Image+text query → question discarded → image index only
    Text-only query  → cross-modal (text → image index, no text index)
    Image-only query → image index (same as current)
"""

import time
from typing import List, Dict, Any, Optional

from PIL import Image

from src.retrieval.colqwen2_retriever import ColQwen2Retriever
from evaluation.metrics.retrieval_metrics import compute_retrieval_metrics
from src.utils.logging_utils import setup_logger

logger = setup_logger("ablation.baseline")


def run_baseline_evaluation(
    retriever: ColQwen2Retriever,
    queries: List[Dict[str, Any]],
    top_k_values: List[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate retrieval in the degraded question-ignored baseline mode.

    The system still receives image+text queries (same inputs as the
    current system), but the question text is discarded before
    retrieval. Only the image drives the ranking. This reproduces
    the pre-fix behavior where:

      - The system received image + question inputs,
      - but the question text had no influence on retrieval ranking,
      - different questions on the same image produced identical
        retrieved documents,
      - retrieval was dominated entirely by image-level similarity.

    For text-only queries, uses cross-modal matching (text tokens
    scored against image patch embeddings) without a dedicated text
    index — the behavior before Phase 3 added text embeddings.

    Args:
        retriever:    ColQwen2Retriever with loaded index (used directly,
                      bypassing HybridRetriever).
        queries:      List of query dicts from OpenITestBuilder.
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
        f"Running BASELINE evaluation: {len(queries)} queries "
        f"(question-ignored image+text retrieval)"
    )

    for i, q in enumerate(queries):
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

        try:
            if query_image is not None:
                # ── BASELINE: QUESTION-IGNORED RETRIEVAL ──
                # The system receives image + question (same as current),
                # but the question is NOT used for ranking. A generic
                # string replaces the actual question, so retrieval is
                # driven entirely by image similarity. This is the core
                # modality-bias failure: the same image always produces
                # the same results regardless of what question was asked.
                retrieved = retriever.retrieve_by_image(
                    query_image,
                    query="What does this image show?",
                    top_k=max_k,
                )
            else:
                # Text-only: cross-modal fallback (text → image index)
                # Before Phase 3, there was no dedicated text index.
                # Use the main retrieve() which falls back to cross-modal
                # if no text embeddings are available, or uses them if
                # present — but without reranking (which only
                # HybridRetriever applies).
                retrieved = retriever.retrieve(
                    query=q["query_text"],
                    top_k=max_k,
                )

            retrieved_ids = [doc.doc_id for doc in retrieved]

            eval_results.append({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "query_mode": q["query_mode"],
                "finding": q["finding"],
                "retrieved_ids": retrieved_ids,
                "gold_ids": q["gold_ids"],
                "retrieved_scores": [doc.score for doc in retrieved],
                "source_doc_id": q.get("source_doc_id", ""),
                "query_image_path": q.get("query_image_path", ""),
            })

            if (i + 1) % 10 == 0:
                logger.info(f"  Baseline progress: {i + 1}/{len(queries)}")

        except Exception as e:
            logger.error(f"Baseline error on query {q['query_id']}: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"Baseline complete: {len(eval_results)} queries in {elapsed:.1f}s"
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
        "mode": "baseline",
        "description": (
            "Question-ignored image+text retrieval — "
            "image+text input received but question text discarded, "
            "no RRF fusion, no question-aware reranking"
        ),
        "num_queries": len(eval_results),
        "top_k_values": top_k_values,
    }

    return {
        "metrics": metrics,
        "eval_results": eval_results,
    }
