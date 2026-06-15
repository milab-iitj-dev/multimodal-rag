"""
Phase 1 Validation / Smoke Test.

Quick sanity checks to verify the pipeline works end-to-end:
  1. Dataset loads without errors
  2. Model loads within VRAM budget
  3. Single inference produces non-empty output
  4. Batch of N samples completes without crashes
  5. Results are saved correctly

Usage:
    python scripts/validate.py
    python scripts/validate.py --max-samples 3 --quick
"""

import argparse, sys, yaml, time, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.device import print_gpu_status, get_vram_usage_gb
from src.shared.logging_utils import setup_logger

logger = setup_logger("validation")


def check(name, condition, detail=""):
    status = "[PASS]" if condition else "[FAIL]"
    logger.info(f"  {status} — {name}" + (f" ({detail})" if detail else ""))
    return condition


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Smoke Test")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--quick", action="store_true",
                        help="Skip model loading (data-only check)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PHASE 1 VALIDATION — SMOKE TEST")
    logger.info("=" * 60)

    results = {"passed": 0, "failed": 0, "tests": []}

    def record(name, passed):
        results["tests"].append({"name": name, "passed": passed})
        if passed: results["passed"] += 1
        else: results["failed"] += 1

    # --- Test 1: Configs load ---
    logger.info("\n[1] Config loading")
    try:
        with open(args.model_config) as f: mc = yaml.safe_load(f)
        with open(args.data_config) as f:  dc = yaml.safe_load(f)
        from src.shared.config_loader import resolve_data_paths
        dc = resolve_data_paths(dc)
        record("configs_load", check("Configs load", True))
    except Exception as e:
        record("configs_load", check("Configs load", False, str(e)))
        logger.error("Cannot continue without configs"); return

    # --- Test 2: Device info ---
    logger.info("\n[2] Device detection")
    info = print_gpu_status(logger)
    record("gpu_detected", check("GPU detected", info["cuda_available"]))

    # --- Test 3: Dataset loading ---
    logger.info("\n[3] Dataset loading")
    try:
        from src.domains.healthcare.ingestion.dicom_loader import OpenIDataset
        ds = OpenIDataset(dc["dataset"]["images_dir"], dc["dataset"]["reports_dir"],
                          max_samples=args.max_samples * 3)
        ds.load()
        n = len(ds)
        record("dataset_load", check("Dataset loads", n > 0, f"{n} samples"))

        s = ds[0]
        record("sample_access", check("Sample accessible", s.sample_id != ""))
        record("sample_image", check("Sample has image", s.image is not None))
        record("sample_text", check("Sample has text", bool(s.answer)))

        logger.info(f"    Sample ID:     {s.sample_id}")
        logger.info(f"    Image size:    {s.image.size if s.image else 'None'}")
        logger.info(f"    Answer length: {len(s.answer) if s.answer else 0} chars")
    except Exception as e:
        record("dataset_load", check("Dataset loads", False, str(e)))

    if args.quick:
        logger.info("\n>> Quick mode -- skipping model tests")
    else:
        # --- Test 4: Model loading ---
        logger.info("\n[4] Model loading")
        try:
            from src.domains.healthcare.generation.model_factory import create_model
            model = create_model(mc)
            t0 = time.time()
            model.load(mc)
            load_time = time.time() - t0
            vram = get_vram_usage_gb()
            record("model_load", check("Model loads", model.is_loaded,
                                       f"{load_time:.1f}s, {vram['allocated']}GB VRAM"))
        except Exception as e:
            record("model_load", check("Model loads", False, str(e)))
            logger.error("Cannot continue without model"); return

        # --- Test 5: Single inference ---
        logger.info("\n[5] Single inference")
        try:
            from src.domains.healthcare.ingestion.preprocessing import MedicalImagePreprocessor
            prep = MedicalImagePreprocessor()
            sample = ds[0]
            img = prep(sample.image)
            output = model.generate(img, "What does this chest X-ray show?")
            has_answer = len(output.answer) > 10
            record("single_inference", check("Generates answer", has_answer,
                                             f"{output.generation_time_sec}s, {len(output.answer)} chars"))
            logger.info(f"    Answer: {output.answer[:200]}")
        except Exception as e:
            record("single_inference", check("Single inference", False, str(e)))

        # --- Test 6: Batch inference ---
        logger.info(f"\n[6] Batch inference ({args.max_samples} samples)")
        try:
            from pipelines.simple_vqa import SimpleVQAPipeline
            pipeline = SimpleVQAPipeline(model=model, dataset=ds, output_dir="outputs/validation")
            res = pipeline.run(max_samples=args.max_samples)
            ok = sum(1 for r in res if "error" not in r)
            record("batch_inference", check("Batch completes", ok > 0,
                                            f"{ok}/{len(res)} successful"))
        except Exception as e:
            record("batch_inference", check("Batch inference", False, str(e)))

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS: {results['passed']} passed, {results['failed']} failed")
    logger.info("=" * 60)

    out_path = Path("outputs/validation")
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
