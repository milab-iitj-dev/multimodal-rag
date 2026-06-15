"""
Grounding Benchmark Runner -- evaluate answer-evidence consistency.

Runs the full MRAG pipeline on OpenI test queries and measures
how often answers are supported, contradicted, or unsupported
by the retrieved evidence.

Usage:
    python -m evaluation.runners.grounding_benchmark \\
        --model-config configs/model_config.yaml \\
        --retrieval-config configs/retrieval_config.yaml \\
        --data-config configs/data_config.yaml \\
        --max-samples 50 \\
        --output-dir outputs/benchmarks/grounding
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image

from evaluation.metrics.grounding_metrics import (
    classify_grounding,
    compute_grounding_metrics,
)
from src.utils.logging_utils import setup_logger

logger = setup_logger("benchmark.grounding")


# Test queries covering all four query types
GROUNDING_QUERIES = [
    # BINARY CLINICAL
    {"query": "Is there cardiomegaly?", "type": "binary_clinical", "needs_image": True},
    {"query": "Is there pleural effusion?", "type": "binary_clinical", "needs_image": True},
    {"query": "Is there pneumothorax?", "type": "binary_clinical", "needs_image": True},
    {"query": "Is there atelectasis?", "type": "binary_clinical", "needs_image": True},
    {"query": "Is there consolidation?", "type": "binary_clinical", "needs_image": True},
    {"query": "Is there pulmonary edema?", "type": "binary_clinical", "needs_image": True},
    {"query": "Is there pneumonia?", "type": "binary_clinical", "needs_image": True},
    {"query": "Is there emphysema?", "type": "binary_clinical", "needs_image": True},
    {"query": "Are there nodules?", "type": "binary_clinical", "needs_image": True},
    {"query": "Is there scoliosis?", "type": "binary_clinical", "needs_image": True},
    # DESCRIPTIVE IMAGE
    {"query": "Describe the findings", "type": "descriptive_image", "needs_image": True},
    {"query": "What does this chest X-ray show?", "type": "descriptive_image", "needs_image": True},
    {"query": "Summarize the image", "type": "descriptive_image", "needs_image": True},
    {"query": "Describe all abnormalities visible", "type": "descriptive_image", "needs_image": True},
    {"query": "What are the key findings?", "type": "descriptive_image", "needs_image": True},
    # TEXT ONLY (no image)
    {"query": "Is there cardiomegaly?", "type": "text_only", "needs_image": False},
    {"query": "Is there pleural effusion?", "type": "text_only", "needs_image": False},
    {"query": "What are signs of pneumothorax?", "type": "text_only", "needs_image": False},
    {"query": "Is there atelectasis?", "type": "text_only", "needs_image": False},
    {"query": "Is there consolidation?", "type": "text_only", "needs_image": False},
    # MIXED IMAGE + TEXT
    {"query": "What are signs of cardiomegaly?", "type": "mixed_image_text", "needs_image": True},
    {"query": "What are signs of pleural effusion?", "type": "mixed_image_text", "needs_image": True},
    {"query": "What are signs of pneumothorax?", "type": "mixed_image_text", "needs_image": True},
    {"query": "What are signs of atelectasis?", "type": "mixed_image_text", "needs_image": True},
    {"query": "What are signs of consolidation?", "type": "mixed_image_text", "needs_image": True},
]


def run_grounding_benchmark(
    model_config: dict,
    retrieval_config: dict,
    data_config: dict,
    index_dir: str = "data/indexes/colqwen2_index",
    max_samples: int = 50,
    output_dir: str = "outputs/benchmarks/grounding",
) -> Dict[str, Any]:
    """
    Run the grounding benchmark end-to-end.

    Steps:
        1. Initialize the full RAG pipeline
        2. Run test queries covering all query types
        3. Extract grounding results
        4. Compute supported/contradicted/unsupported rates
        5. Save results
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Initialize pipeline ──
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

    # ── Step 2: Load test images from dataset ──
    ds_cfg = data_config.get("dataset", {})
    images_dir = Path(ds_cfg.get("images_dir", "data/openi/images"))

    # Collect available images
    test_images = sorted(images_dir.glob("*.png"))[:max_samples]
    logger.info(f"Found {len(test_images)} test images")

    # ── Step 3: Run queries ──
    eval_results = []
    start_time = time.time()

    query_list = GROUNDING_QUERIES[:max_samples]
    img_idx = 0

    for i, qdef in enumerate(query_list):
        query_text = qdef["query"]
        query_type = qdef["type"]
        query_image = None
        image_path = None

        if qdef["needs_image"] and img_idx < len(test_images):
            try:
                image_path = str(test_images[img_idx])
                query_image = Image.open(image_path).convert("RGB")
                img_idx += 1
            except Exception as e:
                logger.warning(f"Cannot load image: {e}")

        # Skip image-requiring queries if no image
        if qdef["needs_image"] and query_image is None:
            continue

        try:
            logger.info(
                f"  [{i+1}/{len(query_list)}] {query_type}: '{query_text[:50]}'"
            )

            output = pipeline.run_single(
                query=query_text,
                query_image=query_image,
            )

            grounding_label = classify_grounding(output.grounding_result)

            result = {
                "query_id": f"ground_{i:04d}",
                "query": query_text,
                "query_type": query_type,
                "answer": output.answer[:500],
                "grounding_label": grounding_label,
                "was_corrected": (
                    output.grounding_result.was_corrected
                    if output.grounding_result else False
                ),
                "confidence_level": (
                    output.confidence.level
                    if output.confidence else "unknown"
                ),
                "confidence_score": (
                    output.confidence.score
                    if output.confidence else 0.0
                ),
                "consensus": (
                    output.evidence_summary.consensus
                    if output.evidence_summary else "unknown"
                ),
                "num_findings": (
                    len(output.evidence_summary.relevant_findings)
                    if output.evidence_summary else 0
                ),
                "image_path": image_path,
            }
            eval_results.append(result)

        except Exception as e:
            logger.error(f"Error on query '{query_text}': {e}")

    elapsed = time.time() - start_time

    # ── Step 4: Compute metrics ──
    metrics = compute_grounding_metrics(eval_results)
    metrics["timing"] = {
        "total_seconds": round(elapsed, 2),
        "avg_seconds_per_query": round(elapsed / max(len(eval_results), 1), 3),
    }

    # ── Step 5: Save results ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = out_dir / f"grounding_{timestamp}.json"

    output_data = {
        "benchmark": "grounding",
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
    print("GROUNDING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Total queries:      {agg.get('total', 0)}")
    print(f"  Supported rate:     {agg.get('supported_rate', 0):.4f}")
    print(f"  Contradicted rate:  {agg.get('contradicted_rate', 0):.4f}")
    print(f"  Unsupported rate:   {agg.get('unsupported_rate', 0):.4f}")
    print(f"  Correction rate:    {agg.get('correction_rate', 0):.4f}")

    print("\nPer Query Type:")
    for qt, m in metrics.get("per_query_type", {}).items():
        print(
            f"  {qt:20s} | supported={m.get('supported_rate', 0):.4f} "
            f"| contradicted={m.get('contradicted_rate', 0):.4f} "
            f"| n={m.get('total', 0)}"
        )

    if metrics.get("confidence_distribution"):
        print("\nConfidence Distribution:")
        for level, count in metrics["confidence_distribution"].items():
            print(f"  {level}: {count}")
    print("=" * 60)

    return metrics


# ── CLI ──

def main():
    import yaml

    parser = argparse.ArgumentParser(
        description="Grounding Benchmark on OpenI"
    )
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--retrieval-config", default="configs/retrieval_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--index-dir", default="data/indexes/colqwen2_index")
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--output-dir", default="outputs/benchmarks/grounding")
    args = parser.parse_args()

    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.retrieval_config) as f:
        retrieval_config = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)

    run_grounding_benchmark(
        model_config=model_config,
        retrieval_config=retrieval_config,
        data_config=data_config,
        index_dir=args.index_dir,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
