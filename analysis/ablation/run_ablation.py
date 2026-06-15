"""
Ablation Runner — Modality-Bias Ablation Analysis Entry Point.

Orchestrates the full ablation workflow:

    1. Load ColQwen2 embedder and dual index (read-only)
    2. Build gold-labeled test queries from OpenI
    3. Run BASELINE evaluation (question-ignored image+text retrieval)
    4. Run CURRENT SYSTEM evaluation (dual-index + RRF + reranking)
    5. Run question sensitivity analysis
    6. Generate the observing document and raw JSON results

Output:
    outputs/observations/observing_document.md
    outputs/observations/ablation_results.json

Usage:
    python -m analysis.ablation.run_ablation \\
        --retrieval-config configs/retrieval_config.yaml \\
        --data-config configs/data_config.yaml \\
        --index-dir data/indexes/colqwen2_index \\
        --max-queries 50 \\
        --output-dir outputs/observations

IMPORTANT:
    This script does NOT modify any production code or pipeline.
    It uses the existing index and retrievers in READ-ONLY mode.
    All output is written to a separate observations directory.
"""

import shutil
import argparse
from pathlib import Path

import yaml

# ── Clean __pycache__ to prevent stale bytecode ──
for _cache_dir in Path("analysis").rglob("__pycache__"):
    shutil.rmtree(_cache_dir, ignore_errors=True)

from src.embeddings.colqwen2_embedder import ColQwen2Embedder
from src.retrieval.colqwen2_retriever import ColQwen2Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from evaluation.datasets.openi_test_builder import OpenITestBuilder
from src.utils.logging_utils import setup_logger

from analysis.ablation.baseline_evaluator import run_baseline_evaluation
from analysis.ablation.current_evaluator import run_current_evaluation
from analysis.ablation.sensitivity_analyzer import analyze_question_sensitivity
from analysis.ablation.report_writer import generate_observing_document

logger = setup_logger("ablation.main")


def run_ablation(
    retrieval_config: dict,
    data_config: dict,
    index_dir: str = "data/indexes/colqwen2_index",
    max_queries: int = 50,
    top_k_values: list = None,
    output_dir: str = "outputs/observations",
) -> str:
    """
    Run the full modality-bias ablation analysis.

    Steps:
        1. Load the ColQwen2 index (shared, read-only)
        2. Create both retriever instances (baseline + current)
        3. Build gold-labeled test queries from OpenI
        4. Run baseline evaluation (question-ignored path)
        5. Run current system evaluation (full hybrid path)
        6. Analyze question sensitivity across both runs
        7. Generate the observing document

    Args:
        retrieval_config: Parsed retrieval YAML config.
        data_config:      Parsed data YAML config.
        index_dir:        Path to the ColQwen2 index directory.
        max_queries:      Maximum number of test queries to run.
        top_k_values:     k values for metrics (default [1, 3, 5]).
        output_dir:       Directory for output files.

    Returns:
        Path to the generated observing document.
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5]

    # ── Step 1: Load ColQwen2 embedder and index ──
    logger.info("=" * 60)
    logger.info("STEP 1: Loading ColQwen2 embedder and index")
    logger.info("=" * 60)

    embedder = ColQwen2Embedder()
    embedder.load(retrieval_config)

    colqwen2 = ColQwen2Retriever(embedder, config=retrieval_config)
    colqwen2.load_index(index_dir)
    logger.info(
        f"Index loaded: {colqwen2.num_indexed} documents, "
        f"text index: {'yes' if colqwen2.has_text_index else 'no'}"
    )

    # ── Step 2: Create HybridRetriever (current system) ──
    hybrid = HybridRetriever(
        colqwen2_retriever=colqwen2, config=retrieval_config
    )
    logger.info("HybridRetriever initialized for current system evaluation")

    # ── Step 3: Build gold-labeled test queries ──
    logger.info("=" * 60)
    logger.info("STEP 2: Building gold-labeled test queries")
    logger.info("=" * 60)

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

    # Build queries for all modes
    # Focus on hybrid queries (image+text) since that is where
    # the modality bias problem manifests. Also include text_only
    # and image_only for completeness.
    query_modes = ["hybrid", "text_only", "image_only"]
    queries = builder.build_test_queries(
        max_queries_per_finding=max(max_queries // len(query_modes), 5),
        query_modes=query_modes,
    )

    if max_queries and len(queries) > max_queries:
        queries = queries[:max_queries]

    # Count by mode
    mode_counts = {}
    for q in queries:
        mode = q["query_mode"]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    logger.info(
        f"Built {len(queries)} test queries: "
        + ", ".join(f"{m}={c}" for m, c in sorted(mode_counts.items()))
    )

    # ── Step 4: Run baseline evaluation ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: BASELINE — Question-ignored image+text retrieval")
    logger.info("=" * 60)

    baseline_output = run_baseline_evaluation(
        retriever=colqwen2,
        queries=queries,
        top_k_values=top_k_values,
    )

    b_agg = baseline_output["metrics"]["aggregate"]
    logger.info(
        f"Baseline results: "
        f"R@1={b_agg.get('recall@1', 0):.4f}, "
        f"R@3={b_agg.get('recall@3', 0):.4f}, "
        f"R@5={b_agg.get('recall@5', 0):.4f}, "
        f"MRR={b_agg.get('mrr', 0):.4f}"
    )

    # ── Step 5: Run current system evaluation ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: CURRENT — Dual-index + RRF + reranking")
    logger.info("=" * 60)

    current_output = run_current_evaluation(
        retriever=hybrid,
        queries=queries,
        top_k_values=top_k_values,
    )

    c_agg = current_output["metrics"]["aggregate"]
    logger.info(
        f"Current results: "
        f"R@1={c_agg.get('recall@1', 0):.4f}, "
        f"R@3={c_agg.get('recall@3', 0):.4f}, "
        f"R@5={c_agg.get('recall@5', 0):.4f}, "
        f"MRR={c_agg.get('mrr', 0):.4f}"
    )

    # ── Step 6: Question sensitivity analysis ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: Question Sensitivity Analysis")
    logger.info("=" * 60)

    sensitivity = analyze_question_sensitivity(
        baseline_results=baseline_output["eval_results"],
        current_results=current_output["eval_results"],
        top_k=3,
    )

    # ── Step 7: Generate observing document ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6: Generating Observing Document")
    logger.info("=" * 60)

    report_path = generate_observing_document(
        baseline_metrics=baseline_output["metrics"],
        current_metrics=current_output["metrics"],
        sensitivity=sensitivity,
        output_dir=output_dir,
    )

    # ── Final summary ──
    print()
    print("=" * 60)
    print("  ABLATION ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Observing document : {report_path}")
    print(f"  Raw JSON results   : {output_dir}/ablation_results.json")
    print()
    print("  BASELINE (before fix — question-ignored image+text):")
    print(f"    Recall@1 = {b_agg.get('recall@1', 0):.4f}")
    print(f"    Recall@3 = {b_agg.get('recall@3', 0):.4f}")
    print(f"    Recall@5 = {b_agg.get('recall@5', 0):.4f}")
    print(f"    MRR      = {b_agg.get('mrr', 0):.4f}")
    print()
    print("  CURRENT SYSTEM (after fix):")
    print(f"    Recall@1 = {c_agg.get('recall@1', 0):.4f}")
    print(f"    Recall@3 = {c_agg.get('recall@3', 0):.4f}")
    print(f"    Recall@5 = {c_agg.get('recall@5', 0):.4f}")
    print(f"    MRR      = {c_agg.get('mrr', 0):.4f}")
    print()
    print("  QUESTION SENSITIVITY:")
    print(
        f"    Baseline avg overlap = "
        f"{sensitivity.get('baseline_avg_overlap', 0):.4f}"
    )
    print(
        f"    Current avg overlap  = "
        f"{sensitivity.get('current_avg_overlap', 0):.4f}"
    )
    print(
        f"    Overlap reduction    = "
        f"{sensitivity.get('overlap_reduction', 0):.4f}"
    )
    print("=" * 60)

    return report_path


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Modality-Bias Ablation Analysis — "
            "Compare baseline (question-ignored image+text) vs current "
            "system (dual-index + RRF + reranking)"
        ),
    )
    parser.add_argument(
        "--retrieval-config",
        default="configs/retrieval_config.yaml",
        help="Path to retrieval config YAML",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data_config.yaml",
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--index-dir",
        default="data/indexes/colqwen2_index",
        help="Path to ColQwen2 index directory",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=50,
        help="Maximum number of test queries to run",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/observations",
        help="Directory for output files",
    )
    args = parser.parse_args()

    # Load configs
    with open(args.retrieval_config, "r", encoding="utf-8") as f:
        retrieval_config = yaml.safe_load(f)
    with open(args.data_config, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    # Run ablation
    run_ablation(
        retrieval_config=retrieval_config,
        data_config=data_config,
        index_dir=args.index_dir,
        max_queries=args.max_queries,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
