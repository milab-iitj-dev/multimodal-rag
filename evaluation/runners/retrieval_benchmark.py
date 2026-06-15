"""
Retrieval Benchmark Runner -- evaluate retrieval quality on OpenI.

Runs the current retriever (HybridRetriever with ColQwen2 + RRF)
against gold-labeled queries and computes Recall@k, MRR, nDCG.

Usage:
    python -m evaluation.runners.retrieval_benchmark \\
        --retrieval-config configs/retrieval_config.yaml \\
        --data-config configs/data_config.yaml \\
        --index-dir data/indexes/colqwen2_index \\
        --max-queries 50 \\
        --output-dir outputs/benchmarks/retrieval
"""

import json
import time
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image

# ── Clean __pycache__ to prevent stale bytecode on HPC ──
for _cache_dir in Path("evaluation").rglob("__pycache__"):
    shutil.rmtree(_cache_dir, ignore_errors=True)

from src.embeddings.colqwen2_embedder import ColQwen2Embedder
from src.retrieval.colqwen2_retriever import ColQwen2Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from evaluation.datasets.openi_test_builder import OpenITestBuilder
from evaluation.metrics.retrieval_metrics import (
    compute_retrieval_metrics,
    hit_at_k,
)
from src.utils.logging_utils import setup_logger

logger = setup_logger("benchmark.retrieval")

# ── Sanity check: verify hit_at_k is binary, not set-overlap ──
_test_hit = hit_at_k(["a"], ["a", "b", "c"], k=1)
assert _test_hit == 1.0, (
    f"FATAL: hit_at_k returned {_test_hit}, expected 1.0. "
    f"Stale __pycache__ may be loaded. Delete all __pycache__ dirs and retry."
)
logger.info("Metric sanity check passed: hit_at_k is binary Hit@k")



def run_retrieval_benchmark(
    retrieval_config: dict,
    data_config: dict,
    index_dir: str = "data/indexes/colqwen2_index",
    max_queries: int = 50,
    top_k_values: List[int] = None,
    query_modes: List[str] = None,
    output_dir: str = "outputs/benchmarks/retrieval",
) -> Dict[str, Any]:
    """
    Run the retrieval benchmark end-to-end.

    Steps:
        1. Load ColQwen2 index
        2. Build gold-labeled test queries from OpenI
        3. Run retriever on each query
        4. Compute Recall@k, MRR, nDCG
        5. Save results

    Returns:
        Dict with aggregate and per-mode metrics.
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5]
    if query_modes is None:
        query_modes = ["text_only", "image_only", "hybrid"]

    max_k = max(top_k_values)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Initialize retriever ──
    logger.info("Initializing ColQwen2 retriever...")
    embedder = ColQwen2Embedder()
    embedder.load(retrieval_config)

    colqwen2 = ColQwen2Retriever(embedder, config=retrieval_config)
    colqwen2.load_index(index_dir)
    logger.info(f"Index loaded: {colqwen2.num_indexed} documents")

    retrieval_method = (
        retrieval_config.get("retrieval", {}).get("method", "colqwen2")
    )
    if retrieval_method == "hybrid" and colqwen2.has_text_index:
        retriever = HybridRetriever(
            colqwen2_retriever=colqwen2, config=retrieval_config
        )
        logger.info("Using HybridRetriever (dual-index + RRF)")
    else:
        retriever = colqwen2
        logger.info("Using ColQwen2Retriever (image-only)")

    # ── Step 2: Build gold-labeled queries ──
    ds_cfg = data_config.get("dataset", {})
    reports_dir = Path(ds_cfg.get("reports_dir", "data/openi/reports"))
    images_dir = Path(ds_cfg.get("images_dir", "data/openi/images"))

    reports_csv = reports_dir / "indiana_reports.csv"
    projections_csv = reports_dir / "indiana_projections.csv"

    builder = OpenITestBuilder(
        reports_csv=str(reports_csv),
        projections_csv=str(projections_csv),
        images_dir=str(images_dir),
    )
    builder.load()

    queries = builder.build_test_queries(
        max_queries_per_finding=max_queries // len(query_modes),
        query_modes=query_modes,
    )

    if max_queries and len(queries) > max_queries:
        queries = queries[:max_queries]

    logger.info(f"Running {len(queries)} retrieval queries (top_k={max_k})")

    # ── Step 3: Run retriever ──
    eval_results = []
    start_time = time.time()

    for i, q in enumerate(queries):
        query_text = q["query_text"]
        query_image = None

        # Load image for image-based modes
        if q["query_image_path"] and q["query_mode"] != "text_only":
            try:
                query_image = Image.open(q["query_image_path"]).convert("RGB")
            except Exception as e:
                logger.warning(f"Cannot load image: {e}")

        # Skip image modes if no image available
        if q["query_mode"] in ("image_only", "hybrid") and query_image is None:
            continue

        # For image_only, use a minimal text query
        if q["query_mode"] == "image_only":
            query_text = "What does this image show?"

        try:
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
            })

            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(queries)}")

        except Exception as e:
            logger.error(f"Error on query {q['query_id']}: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"Retrieval complete: {len(eval_results)} queries in {elapsed:.1f}s"
    )

    # ── Step 4: Compute metrics ──
    metrics = compute_retrieval_metrics(eval_results, k_values=top_k_values)
    metrics["timing"] = {
        "total_seconds": round(elapsed, 2),
        "avg_seconds_per_query": round(elapsed / max(len(eval_results), 1), 3),
    }
    metrics["config"] = {
        "index_dir": index_dir,
        "retriever": type(retriever).__name__,
        "num_queries": len(eval_results),
        "top_k_values": top_k_values,
        "query_modes": query_modes,
    }

    # ── Step 5: Print per-query diagnostics ──
    diagnostics = metrics.get("diagnostics", [])

    print("\n" + "=" * 70)
    print("PER-QUERY DIAGNOSTICS")
    print("=" * 70)
    for diag in diagnostics:
        print(
            f"\n  Query:    {diag['query_text']}"
            f"\n  Finding:  {diag['finding']}"
            f"\n  Mode:     {diag['query_mode']}"
            f"\n  Gold:     {diag['num_gold_docs']} relevant docs"
            f"\n  Retrieved:{diag['retrieved_ids']}"
            f"\n  Rel vec:  {diag['relevance_vector']}"
            f"\n  Hit@1={diag.get('hit@1', '?')} "
            f"Hit@3={diag.get('hit@3', '?')} "
            f"Hit@5={diag.get('hit@5', '?')} "
            f"RR={diag['reciprocal_rank']}"
        )
    print("=" * 70)

    # ── Step 6: Save results ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = out_dir / f"retrieval_{timestamp}.json"

    output = {
        "benchmark": "retrieval",
        "timestamp": timestamp,
        "metrics": {
            "aggregate": metrics["aggregate"],
            "per_mode": metrics["per_mode"],
            "timing": metrics["timing"],
            "config": metrics["config"],
        },
        "diagnostics": diagnostics,
        "per_sample": eval_results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Results saved to {results_path}")

    # ── Step 7: Print aggregate summary ──
    agg = metrics["aggregate"]
    print("\n" + "=" * 60)
    print("RETRIEVAL BENCHMARK RESULTS")
    print("=" * 60)
    print("  Recall@k (Hit@k = 1 if any gold doc in top-k, else 0):")
    for k in top_k_values:
        print(f"    Recall@{k}:    {agg.get(f'recall@{k}', 0):.4f}")
    print(f"  MRR:           {agg.get('mrr', 0):.4f}")
    print("  Precision@k (fraction of top-k that are relevant):")
    for k in top_k_values:
        print(f"    Precision@{k}: {agg.get(f'precision@{k}', 0):.4f}")
    print("  nDCG@k:")
    for k in top_k_values:
        print(f"    nDCG@{k}:     {agg.get(f'ndcg@{k}', 0):.4f}")
    print(f"\n  Queries:       {agg.get('num_queries', 0)}")
    print(f"  Time:          {elapsed:.1f}s")

    if metrics.get("per_mode"):
        print("\nPer Query Mode:")
        for mode, m in metrics["per_mode"].items():
            print(f"  {mode:12s} | R@1={m.get('recall@1', 0):.4f} "
                  f"| R@3={m.get('recall@3', 0):.4f} "
                  f"| R@5={m.get('recall@5', 0):.4f} "
                  f"| MRR={m.get('mrr', 0):.4f} "
                  f"| n={m.get('num_queries', 0)}")
    print("=" * 60)

    return metrics


# ── CLI ──

def main():
    import yaml

    parser = argparse.ArgumentParser(
        description="Retrieval Benchmark on OpenI"
    )
    parser.add_argument(
        "--retrieval-config",
        default="configs/retrieval_config.yaml",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data_config.yaml",
    )
    parser.add_argument("--index-dir", default="data/indexes/colqwen2_index")
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/benchmarks/retrieval")
    args = parser.parse_args()

    with open(args.retrieval_config) as f:
        retrieval_config = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)

    run_retrieval_benchmark(
        retrieval_config=retrieval_config,
        data_config=data_config,
        index_dir=args.index_dir,
        max_queries=args.max_queries,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
