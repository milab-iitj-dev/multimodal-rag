"""
Phase 1 Inference — Local & Evaluation Pipeline.

Supports:
  - Single image:    python scripts/inference.py --image path/to/xray.png
  - Interactive:     python scripts/inference.py
  - Batch eval:      python scripts/inference.py --batch-eval
  - With adapter:    python scripts/inference.py --adapter checkpoints/llava-medical-vqa/final_adapter ...
  - CPU mode:        python scripts/inference.py --cpu ...

Batch evaluation mode (--batch-eval):
  - Selects 5 OpenI samples (mix of normal/abnormal)
  - Asks different clinical questions
  - Measures inference time per sample
  - Saves JSON, CSV, and Markdown report to outputs/evaluation/
"""

import argparse, csv, json, os, re, sys, time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.logging_utils import setup_logger
logger = setup_logger("inference")

EVAL_QUESTIONS = [
    "Briefly describe the key findings in this chest X-ray.",
    "Is there cardiomegaly? Answer briefly.",
    "Are the lungs clear or abnormal? State findings only.",
    "Is there pleural effusion? Answer in one sentence.",
    "What is the radiological impression? Be concise.",
]


def clean_answer(text):
    """Clean generated text: remove repetition, trailing filler, hallucinated extras."""
    text = text.strip()
    text = re.sub(r'(\. ){2,}', '. ', text)
    text = re.sub(r'(\.\s*){3,}', '.', text)
    if '.' in text:
        last_period = text.rfind('.')
        if last_period < len(text) - 1:
            text = text[:last_period + 1]
    filler = [
        r'\s*If you need further.*$',
        r'\s*Clinical correlation is recommended.*$',
        r'\s*Recommend followup.*$',
        r'\s*Follow up.*$',
        r'\s*please schedule.*$',
        r'\s*please contact.*$',
    ]
    for pattern in filler:
        text = re.sub(pattern, '.', text, flags=re.IGNORECASE)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def run_single(model_wrapper, image_path, question):
    """Run inference on a single image."""
    from src.shared.image_utils import load_image
    from src.domains.healthcare.ingestion.preprocessing import MedicalImagePreprocessor

    prep = MedicalImagePreprocessor()
    img = prep(load_image(image_path))

    t0 = time.time()
    output = model_wrapper.generate(image=img, question=question)
    elapsed = time.time() - t0

    answer = clean_answer(output.answer)

    print(f"\n{'='*60}")
    print(f"  Image:    {image_path}")
    print(f"  Question: {question}")
    print(f"  Answer:   {answer}")
    print(f"  Time:     {elapsed:.1f}s")
    print(f"  Tokens:   {output.input_token_count} → {output.output_token_count}")
    print(f"{'='*60}\n")

    return {
        "answer": answer,
        "time_sec": round(elapsed, 2),
        "input_tokens": output.input_token_count,
        "output_tokens": output.output_token_count,
        "question": question,
        "image_path": str(image_path),
    }


def run_batch_eval(model_wrapper, data_config, max_samples=5):
    """
    Run batch evaluation on OpenI samples.

    Selects diverse samples, asks different clinical questions,
    generates a professional report.
    """
    import yaml
    from src.domains.healthcare.ingestion.dicom_loader import OpenIDataset

    ds = data_config["dataset"]
    dataset = OpenIDataset(ds["images_dir"], ds["reports_dir"],
                           ds.get("max_samples", 100))
    dataset.load()

    # Select samples with ground truth
    from src.domains.healthcare.ingestion.preprocessing import MedicalImagePreprocessor, clean_report_text
    prep = MedicalImagePreprocessor()

    samples = []
    for idx in range(min(len(dataset), 50)):
        s = dataset[idx]
        if s.image is None or not s.answer:
            continue
        report = clean_report_text(s.answer)
        if len(report) < 5:
            continue
        samples.append({
            "image": prep(s.image),
            "report": report,
            "uid": getattr(s, "uid", f"sample_{idx}"),
            "image_path": getattr(s, "image_path", f"sample_{idx}"),
        })
        if len(samples) >= max_samples:
            break

    logger.info(f"Running evaluation on {len(samples)} samples ...")

    results = []
    total_time = 0

    for i, s in enumerate(samples):
        question = EVAL_QUESTIONS[i % len(EVAL_QUESTIONS)]
        logger.info(f"  Sample {i+1}/{len(samples)} — {s['uid']}")

        try:
            t0 = time.time()
            output = model_wrapper.generate(image=s["image"], question=question)
            elapsed = time.time() - t0
            answer = clean_answer(output.answer)
            total_time += elapsed

            result = {
                "sample_id": i + 1,
                "uid": s["uid"],
                "image_path": str(s["image_path"]),
                "question": question,
                "answer": answer,
                "ground_truth": s["report"],
                "time_sec": round(elapsed, 2),
                "input_tokens": output.input_token_count,
                "output_tokens": output.output_token_count,
            }
            results.append(result)

            print(f"\n--- Sample {i+1} (UID: {s['uid']}) ---")
            print(f"  Q:  {question}")
            print(f"  A:  {answer[:300]}")
            print(f"  GT: {s['report'][:300]}")
            print(f"  Time: {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"  Failed: {e}")

    # Summary
    import torch
    avg_time = total_time / len(results) if results else 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    summary = {
        "total_samples": len(results),
        "avg_time_sec": round(avg_time, 2),
        "total_time_sec": round(total_time, 2),
        "device": device,
        "gpu": gpu_name,
        "vram_gb": round(vram, 2),
        "model": "llava-hf/llava-1.5-7b-hf",
        "timestamp": datetime.now().isoformat(),
    }

    # Save
    out_dir = Path("outputs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = out_dir / f"eval_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    # CSV
    csv_path = out_dir / f"eval_{ts}.csv"
    fields = ["sample_id", "uid", "question", "answer", "ground_truth", "time_sec"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Markdown report
    md_path = out_dir / f"eval_report_{ts}.md"
    md_lines = [
        "# Phase 1 — Medical VQA Evaluation Report", "",
        f"**Date:** {summary['timestamp']}", f"**Device:** {summary['gpu'] or summary['device']}",
        f"**VRAM:** {summary['vram_gb']} GB", "",
        "| Metric | Value |", "|---|---|",
        f"| Samples | {summary['total_samples']} |",
        f"| Avg Time | {summary['avg_time_sec']}s |",
        f"| Total Time | {summary['total_time_sec']}s |", "",
        "---", "", "## Results", "",
    ]
    for r in results:
        gt = r.get("ground_truth", "N/A")
        md_lines.extend([
            f"### Sample {r['sample_id']} — {r.get('uid', '')}",
            f"**Q:** {r['question']}", f"**A:** {r['answer']}", f"**GT:** {gt[:300]}",
            f"*{r['time_sec']}s*", "", "---", "",
        ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n{'='*60}")
    print(f"  Evaluation complete: {len(results)} samples")
    print(f"  Avg time: {avg_time:.1f}s per image")
    print(f"  JSON:   {json_path}")
    print(f"  CSV:    {csv_path}")
    print(f"  Report: {md_path}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Inference & Evaluation")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--question", type=str, default="What does this chest X-ray show?")
    parser.add_argument("--batch-eval", action="store_true", help="Run 5-sample evaluation")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    import yaml
    from src.shared.device import print_gpu_status
    print_gpu_status(logger)

    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    if args.cpu:
        logger.info("CPU mode: disabling quantization")
        model_config["model"]["quantization"]["enabled"] = False
        model_config["model"]["device"] = "cpu"

    if args.adapter:
        adapter_path = str(Path(args.adapter).resolve())
        model_config["model"]["adapter_path"] = adapter_path
        logger.info(f"Adapter: {adapter_path}")

    from src.domains.healthcare.generation.model_factory import create_model
    logger.info("Loading model ...")
    model_wrapper = create_model(model_config)
    model_wrapper.load(model_config)
    logger.info("Model ready.")

    if args.image:
        run_single(model_wrapper, args.image, args.question)
    elif args.batch_eval:
        with open(args.data_config) as f:
            data_config = yaml.safe_load(f)
        from src.shared.config_loader import resolve_data_paths
        data_config = resolve_data_paths(data_config)
        run_batch_eval(model_wrapper, data_config, args.max_samples)
    else:
        # Interactive
        logger.info("Interactive mode. Type 'quit' to exit.")
        while True:
            img_path = input("\nImage path (or 'quit'): ").strip()
            if img_path.lower() in ("quit", "exit", "q"):
                break
            q = input("Question [What does this chest X-ray show?]: ").strip()
            if not q:
                q = "What does this chest X-ray show?"
            try:
                run_single(model_wrapper, img_path, q)
            except Exception as e:
                logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
