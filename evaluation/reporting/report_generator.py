"""
Benchmark Report Generator -- create markdown summary from results.

Reads JSON result files from all three benchmarks and generates
a clean markdown report suitable for a professor demo.

Usage:
    python -m evaluation.reporting.report_generator \\
        --results-dir outputs/benchmarks \\
        --output outputs/benchmarks/BENCHMARK_REPORT.md
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.utils.logging_utils import setup_logger

logger = setup_logger("benchmark.report")


def _find_latest_json(directory: Path, prefix: str) -> Optional[Path]:
    """Find the most recent JSON file with a given prefix."""
    if not directory.exists():
        return None
    files = sorted(directory.glob(f"{prefix}_*.json"), reverse=True)
    return files[0] if files else None


def _load_json(path: Path) -> Optional[Dict]:
    """Load a JSON file, return None on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Cannot load {path}: {e}")
        return None


def _fmt(val, decimals=4) -> str:
    """Format a numeric value for the table."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def generate_report(
    results_dir: str = "outputs/benchmarks",
    output_path: str = "outputs/benchmarks/BENCHMARK_REPORT.md",
) -> str:
    """
    Generate a markdown benchmark report from saved JSON results.

    Searches for the latest result files in:
        results_dir/retrieval/retrieval_*.json
        results_dir/grounding/grounding_*.json
        results_dir/generation/generation_*.json

    Returns:
        The generated markdown string.
    """
    base = Path(results_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Find latest results
    retrieval_file = _find_latest_json(base / "retrieval", "retrieval")
    grounding_file = _find_latest_json(base / "grounding", "grounding")
    generation_file = _find_latest_json(base / "generation", "generation")

    retrieval_data = _load_json(retrieval_file) if retrieval_file else None
    grounding_data = _load_json(grounding_file) if grounding_file else None
    generation_data = _load_json(generation_file) if generation_file else None

    # Build report
    lines = []
    lines.append("# Healthcare MRAG -- Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## System Overview")
    lines.append("")
    lines.append("| Component | Implementation |")
    lines.append("|-----------|---------------|")
    lines.append("| Retrieval | ColQwen2 dual-index + RRF fusion + question-aware reranking |")
    lines.append("| Evidence | EvidenceAggregator (consensus + full-scan modes) |")
    lines.append("| Generation | Qwen2-VL 7B (BF16) |")
    lines.append("| Grounding | GroundingVerifier (direction check + correction) |")
    lines.append("| Routing | QueryClassifier (binary / descriptive / text-only / mixed) |")
    lines.append("")

    # ── Retrieval ──
    lines.append("---")
    lines.append("")
    lines.append("## 1. Retrieval Benchmark")
    lines.append("")

    if retrieval_data:
        metrics = retrieval_data.get("metrics", {})
        agg = metrics.get("aggregate", {})

        lines.append(f"**Dataset:** OpenI test split  ")
        lines.append(f"**Queries:** {agg.get('num_queries', '?')}  ")
        timing = metrics.get("timing", {})
        lines.append(f"**Time:** {timing.get('total_seconds', '?')}s  ")
        lines.append("")

        # Aggregate table
        lines.append("### Aggregate Results")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for k in [1, 3, 5]:
            key = f"recall@{k}"
            lines.append(f"| Recall@{k} | {_fmt(agg.get(key))} |")
        lines.append(f"| MRR | {_fmt(agg.get('mrr'))} |")
        for k in [1, 3, 5]:
            key = f"ndcg@{k}"
            if key in agg:
                lines.append(f"| nDCG@{k} | {_fmt(agg.get(key))} |")
        lines.append("")

        # Per-mode table
        per_mode = metrics.get("per_mode", {})
        if per_mode:
            lines.append("### Per Query Mode")
            lines.append("")
            lines.append("| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Queries |")
            lines.append("|------|----------|----------|----------|-----|---------|")
            for mode, m in per_mode.items():
                lines.append(
                    f"| {mode} | {_fmt(m.get('recall@1'))} "
                    f"| {_fmt(m.get('recall@3'))} "
                    f"| {_fmt(m.get('recall@5'))} "
                    f"| {_fmt(m.get('mrr'))} "
                    f"| {m.get('num_queries', '?')} |"
                )
            lines.append("")
    else:
        lines.append("*Retrieval benchmark not yet run.*")
        lines.append("")

    # ── Grounding ──
    lines.append("---")
    lines.append("")
    lines.append("## 2. Grounding Benchmark")
    lines.append("")

    if grounding_data:
        metrics = grounding_data.get("metrics", {})
        agg = metrics.get("aggregate", {})

        lines.append(f"**Dataset:** OpenI (evidence source)  ")
        lines.append(f"**Queries:** {agg.get('total', '?')}  ")
        lines.append("")

        lines.append("### Aggregate Results")
        lines.append("")
        lines.append("| Metric | Rate |")
        lines.append("|--------|------|")
        lines.append(f"| Supported | {_fmt(agg.get('supported_rate'))} ({agg.get('supported', '?')}/{agg.get('total', '?')}) |")
        lines.append(f"| Contradicted | {_fmt(agg.get('contradicted_rate'))} ({agg.get('contradicted', '?')}/{agg.get('total', '?')}) |")
        lines.append(f"| Unsupported | {_fmt(agg.get('unsupported_rate'))} ({agg.get('unsupported', '?')}/{agg.get('total', '?')}) |")
        lines.append(f"| Corrected | {_fmt(agg.get('correction_rate'))} ({agg.get('corrected', '?')}/{agg.get('total', '?')}) |")
        lines.append("")

        # Per query type
        per_type = metrics.get("per_query_type", {})
        if per_type:
            lines.append("### Per Query Type")
            lines.append("")
            lines.append("| Query Type | Supported | Contradicted | Total |")
            lines.append("|------------|-----------|--------------|-------|")
            for qt, m in per_type.items():
                lines.append(
                    f"| {qt} | {_fmt(m.get('supported_rate'))} "
                    f"| {_fmt(m.get('contradicted_rate'))} "
                    f"| {m.get('total', '?')} |"
                )
            lines.append("")

        # Confidence distribution
        conf = metrics.get("confidence_distribution", {})
        if conf:
            lines.append("### Confidence Distribution")
            lines.append("")
            lines.append("| Level | Count |")
            lines.append("|-------|-------|")
            for level, count in sorted(conf.items()):
                lines.append(f"| {level} | {count} |")
            lines.append("")
    else:
        lines.append("*Grounding benchmark not yet run.*")
        lines.append("")

    # ── Generation ──
    lines.append("---")
    lines.append("")
    lines.append("## 3. Generation Benchmark")
    lines.append("")

    if generation_data:
        metrics = generation_data.get("metrics", {})
        agg = metrics.get("aggregate", {})
        cfg = metrics.get("config", {})

        lines.append(f"**Dataset:** {cfg.get('dataset', 'VQA-RAD')}  ")
        lines.append(f"**Samples:** {agg.get('num_samples', '?')}  ")
        lines.append("")

        lines.append("### Aggregate Results")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        lines.append(f"| Exact Match | {_fmt(agg.get('exact_match'))} |")
        lines.append(f"| Token F1 | {_fmt(agg.get('f1'))} |")
        lines.append(f"| BLEU-1 | {_fmt(agg.get('bleu_1'))} |")
        lines.append(f"| ROUGE-L | {_fmt(agg.get('rouge_l'))} |")
        bs = agg.get("bertscore_f1")
        lines.append(f"| BERTScore F1 | {_fmt(bs)} |")
        lines.append("")

        # Per type
        per_type = metrics.get("per_type", {})
        if per_type:
            lines.append("### Per Question Type")
            lines.append("")
            lines.append("| Type | EM | F1 | BLEU-1 | ROUGE-L | Samples |")
            lines.append("|------|----|----|--------|---------|---------|")
            for qt, m in per_type.items():
                lines.append(
                    f"| {qt} | {_fmt(m.get('exact_match'))} "
                    f"| {_fmt(m.get('f1'))} "
                    f"| {_fmt(m.get('bleu_1'))} "
                    f"| {_fmt(m.get('rouge_l'))} "
                    f"| {m.get('num_samples', '?')} |"
                )
            lines.append("")
    else:
        lines.append("*Generation benchmark not yet run.*")
        lines.append("")

    # ── Summary ──
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    benchmarks_run = sum(
        1 for d in [retrieval_data, grounding_data, generation_data]
        if d is not None
    )
    lines.append(f"**Benchmarks completed:** {benchmarks_run}/3")
    lines.append("")

    if retrieval_data and grounding_data and generation_data:
        r_agg = retrieval_data["metrics"]["aggregate"]
        g_agg = grounding_data["metrics"]["aggregate"]
        gen_agg = generation_data["metrics"]["aggregate"]

        lines.append("### Key Numbers")
        lines.append("")
        lines.append("| Layer | Key Metric | Score |")
        lines.append("|-------|-----------|-------|")
        lines.append(f"| Retrieval | Recall@3 | {_fmt(r_agg.get('recall@3'))} |")
        lines.append(f"| Grounding | Supported Rate | {_fmt(g_agg.get('supported_rate'))} |")
        lines.append(f"| Generation | Token F1 | {_fmt(gen_agg.get('f1'))} |")
        if gen_agg.get("bertscore_f1") is not None:
            lines.append(f"| Generation | BERTScore F1 | {_fmt(gen_agg.get('bertscore_f1'))} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Report generated by Healthcare MRAG Benchmark Suite*")
    lines.append("")

    # Write report
    report_text = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Report saved to {out_path}")
    print(f"\nReport saved to: {out_path}")

    return report_text


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Generate Benchmark Report"
    )
    parser.add_argument(
        "--results-dir", default="outputs/benchmarks",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output", default="outputs/benchmarks/BENCHMARK_REPORT.md",
        help="Path for the output markdown report",
    )
    args = parser.parse_args()

    report = generate_report(
        results_dir=args.results_dir,
        output_path=args.output,
    )

    # Print preview
    print("\n" + "=" * 60)
    print("REPORT PREVIEW")
    print("=" * 60)
    # Print first 40 lines
    for line in report.split("\n")[:40]:
        print(line)
    print("...")
    print("=" * 60)


if __name__ == "__main__":
    main()
