"""
Generation Benchmark Runner -- evaluate answer quality on VQA-RAD.

Runs the full MRAG pipeline on VQA-RAD test questions and computes
generation metrics (EM, F1, BLEU, ROUGE-L, BERTScore).

Usage:
    python -m evaluation.runners.generation_benchmark \\
        --model-config configs/model_config.yaml \\
        --retrieval-config configs/retrieval_config.yaml \\
        --vqa-rad-dir data/vqa_rad \\
        --max-samples 50 \\
        --output-dir outputs/benchmarks/generation
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image

from evaluation.datasets.vqa_rad_loader import VQARADLoader
from evaluation.metrics.generation_metrics import compute_generation_metrics
from src.utils.logging_utils import setup_logger

logger = setup_logger("benchmark.generation")


def run_generation_benchmark(
    model_config: dict,
    retrieval_config: dict,
    vqa_rad_dir: str,
    index_dir: str = "data/indexes/colqwen2_index",
    max_samples: int = 50,
    compute_bert: bool = True,
    output_dir: str = "outputs/benchmarks/generation",
) -> Dict[str, Any]:
    """
    Run the generation benchmark on VQA-RAD.

    Steps:
        1. Load VQA-RAD test split
        2. Initialize the full RAG pipeline
        3. Run pipeline on each question
        4. Compare predictions against ground truth
        5. Compute EM, F1, BLEU, ROUGE-L, BERTScore
        6. Save results
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load VQA-RAD ──
    logger.info(f"Loading VQA-RAD from {vqa_rad_dir}")
    loader = VQARADLoader(data_dir=vqa_rad_dir)
    loader.load()

    test_samples = loader.get_test_split()
    if max_samples and len(test_samples) > max_samples:
        test_samples = test_samples[:max_samples]

    logger.info(
        f"Test set: {len(test_samples)} samples "
        f"({sum(1 for s in test_samples if s['question_type'] == 'closed')} closed, "
        f"{sum(1 for s in test_samples if s['question_type'] == 'open')} open)"
    )

    # ── Step 2: Initialize pipeline ──
    from src.generation.model_factory import create_model
    from pipelines.rag_vqa import RAGVQAPipeline

    logger.info("Loading VLM...")
    model = create_model(model_config)
    model.load(model_config)

    pipeline = RAGVQAPipeline(
        vlm=model,
        retrieval_config=retrieval_config,
        index_dir=index_dir,
        top_k=3,
        output_dir=str(out_dir),
    )

    # ── Step 3: Run pipeline on test samples ──
    eval_results = []
    skipped = 0
    start_time = time.time()

    for i, sample in enumerate(test_samples):
        question = sample["question"]
        reference = sample["answer"]
        question_type = sample["question_type"]

        # Load image
        query_image = None
        if sample.get("image_path"):
            try:
                query_image = Image.open(sample["image_path"]).convert("RGB")
            except Exception as e:
                logger.warning(f"Cannot load image: {e}")

        if query_image is None:
            skipped += 1
            continue

        try:
            output = pipeline.run_single(
                query=question,
                query_image=query_image,
            )

            prediction = output.answer

            eval_results.append({
                "qid": sample["qid"],
                "question": question,
                "question_type": question_type,
                "prediction": prediction,
                "reference": reference,
                "confidence": (
                    output.confidence.score if output.confidence else 0.0
                ),
                "consensus": (
                    output.evidence_summary.consensus
                    if output.evidence_summary else "unknown"
                ),
            })

            if (i + 1) % 10 == 0:
                logger.info(
                    f"  [{i+1}/{len(test_samples)}] "
                    f"Q: '{question[:40]}' "
                    f"Pred: '{prediction[:40]}' "
                    f"Ref: '{reference[:40]}'"
                )

        except Exception as e:
            logger.error(f"Error on sample {sample['qid']}: {e}")
            skipped += 1

    elapsed = time.time() - start_time
    logger.info(
        f"Generation complete: {len(eval_results)} samples, "
        f"{skipped} skipped, {elapsed:.1f}s"
    )

    # ── Step 4: Compute metrics ──
    metrics = compute_generation_metrics(
        eval_results, compute_bert=compute_bert
    )
    metrics["timing"] = {
        "total_seconds": round(elapsed, 2),
        "avg_seconds_per_sample": round(
            elapsed / max(len(eval_results), 1), 3
        ),
    }
    metrics["config"] = {
        "dataset": "VQA-RAD",
        "total_samples": len(eval_results),
        "skipped": skipped,
    }

    # ── Step 5: Save results ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = out_dir / f"generation_{timestamp}.json"

    output_data = {
        "benchmark": "generation",
        "timestamp": timestamp,
        "metrics": metrics,
        "per_sample": eval_results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Results saved to {results_path}")

    # Print summary
    agg = metrics.get("aggregate", {})
    print("\n" + "=" * 60)
    print("GENERATION BENCHMARK RESULTS (VQA-RAD)")
    print("=" * 60)
    print(f"  Exact Match:  {agg.get('exact_match', 0):.4f}")
    print(f"  Token F1:     {agg.get('f1', 0):.4f}")
    print(f"  BLEU-1:       {agg.get('bleu_1', 0):.4f}")
    print(f"  ROUGE-L:      {agg.get('rouge_l', 0):.4f}")
    if agg.get("bertscore_f1") is not None:
        print(f"  BERTScore F1: {agg.get('bertscore_f1', 0):.4f}")
    else:
        print(f"  BERTScore:    (not available)")
    print(f"\n  Samples:      {agg.get('num_samples', 0)}")
    print(f"  Time:         {elapsed:.1f}s")

    print("\nPer Question Type:")
    for qt, m in metrics.get("per_type", {}).items():
        print(
            f"  {qt:8s} | EM={m.get('exact_match', 0):.4f} "
            f"| F1={m.get('f1', 0):.4f} "
            f"| ROUGE-L={m.get('rouge_l', 0):.4f} "
            f"| n={m.get('num_samples', 0)}"
        )
    print("=" * 60)

    return metrics


# ── CLI ──

def main():
    import yaml

    parser = argparse.ArgumentParser(
        description="Generation Benchmark on VQA-RAD"
    )
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval_config.yaml")
    parser.add_argument("--vqa-rad-dir", default="data/vqa_rad")
    parser.add_argument("--index-dir", default="data/indexes/colqwen2_index")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument(
        "--no-bertscore", action="store_true",
        help="Skip BERTScore (faster, no dependency)",
    )
    parser.add_argument("--output-dir", default="outputs/benchmarks/generation")
    args = parser.parse_args()

    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.retrieval_config) as f:
        retrieval_config = yaml.safe_load(f)

    run_generation_benchmark(
        model_config=model_config,
        retrieval_config=retrieval_config,
        vqa_rad_dir=args.vqa_rad_dir,
        index_dir=args.index_dir,
        max_samples=args.max_samples,
        compute_bert=not args.no_bertscore,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
