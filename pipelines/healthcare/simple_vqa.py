"""
Simple VQA Pipeline — Phase 1.

The minimal end-to-end pipeline: Image → Question → Answer.
No retrieval, no grounding, no safety — just pure multimodal inference.

This pipeline:
  1. Loads a dataset (OpenI)
  2. Loads a VLM (LLaVA-1.5-7B 4-bit)
  3. Runs inference on selected samples
  4. Saves results to JSON

Usage:
    python -m pipelines.simple_vqa --config configs/model_config.yaml
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from src.domains.healthcare.ingestion.base_loader import BaseDataset, MedicalSample
from src.domains.healthcare.ingestion.preprocessing import MedicalImagePreprocessor
from src.domains.healthcare.generation.base_generator import BaseVLM, VLMOutput
from src.shared.logging_utils import setup_logger

logger = setup_logger("pipeline.vqa")


class SimpleVQAPipeline:
    """
    Phase 1 pipeline: Image → Question → Answer.

    Orchestrates dataset loading, preprocessing, and model inference.
    Keeps each concern in its own module — the pipeline only wires them.
    """

    def __init__(
        self,
        model: BaseVLM,
        dataset: BaseDataset,
        preprocessor: Optional[MedicalImagePreprocessor] = None,
        output_dir: str = "outputs/vqa_results",
    ):
        self.model = model
        self.dataset = dataset
        self.preprocessor = preprocessor or MedicalImagePreprocessor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        sample_indices: Optional[List[int]] = None,
        max_samples: int = 10,
        custom_question: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run VQA inference on selected samples.

        Args:
            sample_indices: Specific indices to run (None = first max_samples).
            max_samples:    Max samples if sample_indices not provided.
            custom_question: Override question for all samples (None = use dataset Q).

        Returns:
            List of result dicts with input/output data.
        """
        if sample_indices is None:
            sample_indices = list(range(min(max_samples, len(self.dataset))))

        logger.info(f"Running VQA pipeline on {len(sample_indices)} samples")
        results = []

        for i, idx in enumerate(sample_indices):
            sample = self.dataset[idx]
            logger.info(f"  [{i+1}/{len(sample_indices)}] Sample: {sample.sample_id}")

            # Skip samples without images
            if sample.image is None:
                logger.warning(f"  Skipping {sample.sample_id}: no image")
                continue

            # Preprocess image
            image = self.preprocessor(sample.image)

            # Use custom question or dataset question
            question = custom_question or sample.question

            # Run inference
            try:
                output = self.model.generate(
                    image=image,
                    question=question,
                )

                result = {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "question": question,
                    "generated_answer": output.answer,
                    "ground_truth": sample.answer,
                    "generation_time_sec": output.generation_time_sec,
                    "input_tokens": output.input_token_count,
                    "output_tokens": output.output_token_count,
                }
                results.append(result)

                logger.info(f"  Question:  {question}")
                logger.info(f"  Answer:    {output.answer[:200]}...")
                logger.info(f"  Time:      {output.generation_time_sec}s")

            except Exception as e:
                logger.error(f"  Error on {sample.sample_id}: {e}")
                results.append({
                    "sample_id": sample.sample_id,
                    "error": str(e),
                })

        # Save results
        self._save_results(results)
        return results

    def run_single(
        self,
        image_path: str,
        question: str,
    ) -> VLMOutput:
        """
        Run inference on a single image + question.

        Useful for interactive debugging and demos.
        """
        from src.shared.image_utils import load_image

        image = load_image(image_path)
        image = self.preprocessor(image)
        return self.model.generate(image=image, question=question)

    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save results to a timestamped JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"vqa_results_{timestamp}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {out_path}")


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #

def main():
    """Run the VQA pipeline from command line."""
    import argparse
    import yaml
    from src.domains.healthcare.ingestion.dicom_loader import OpenIDataset
    from src.domains.healthcare.generation.model_factory import create_model
    from src.shared.device import print_gpu_status

    parser = argparse.ArgumentParser(description="Phase 1: Simple VQA Pipeline")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--question", type=str, default=None,
                        help="Custom question for all samples")
    parser.add_argument("--output-dir", default="outputs/vqa_results")
    args = parser.parse_args()

    # Print GPU status
    print_gpu_status()

    # Load configs
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)

    # Resolve relative data paths to project root
    from src.shared.config_loader import resolve_data_paths
    data_config = resolve_data_paths(data_config)

    # Load dataset
    ds_cfg = data_config["dataset"]
    dataset = OpenIDataset(
        images_dir=ds_cfg["images_dir"],
        reports_dir=ds_cfg["reports_dir"],
        max_samples=ds_cfg.get("max_samples"),
    )
    dataset.load()
    logger.info(f"Dataset summary: {dataset.summary()}")

    # Load model
    model = create_model(model_config)
    model.load(model_config)

    # Run pipeline
    pipeline = SimpleVQAPipeline(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
    )
    results = pipeline.run(
        max_samples=args.max_samples,
        custom_question=args.question,
    )

    logger.info(f"Pipeline complete. {len(results)} results generated.")


if __name__ == "__main__":
    main()
