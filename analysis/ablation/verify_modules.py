"""Quick verification of the ablation analysis modules with mock data."""

from analysis.ablation.sensitivity_analyzer import analyze_question_sensitivity
from analysis.ablation.report_writer import generate_observing_document
import os

# ── Mock baseline results: same image, different questions -> SAME retrieved docs ──
baseline_results = [
    {
        "query_id": "q1", "query_text": "Is there cardiomegaly?",
        "query_mode": "hybrid", "finding": "cardiomegaly",
        "retrieved_ids": ["d1", "d2", "d3", "d4", "d5"],
        "gold_ids": ["d1", "d10", "d20"],
        "retrieved_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
        "source_doc_id": "img_001", "query_image_path": "img_001.png",
    },
    {
        "query_id": "q2", "query_text": "Is there pleural effusion?",
        "query_mode": "hybrid", "finding": "pleural effusion",
        "retrieved_ids": ["d1", "d2", "d3", "d4", "d5"],
        "gold_ids": ["d5", "d15", "d25"],
        "retrieved_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
        "source_doc_id": "img_001", "query_image_path": "img_001.png",
    },
    {
        "query_id": "q3", "query_text": "Is there pneumonia?",
        "query_mode": "hybrid", "finding": "pneumonia",
        "retrieved_ids": ["d1", "d2", "d3", "d4", "d5"],
        "gold_ids": ["d3", "d30"],
        "retrieved_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
        "source_doc_id": "img_001", "query_image_path": "img_001.png",
    },
]

# ── Mock current results: same image, different questions -> DIFFERENT retrieved docs ──
current_results = [
    {
        "query_id": "q1", "query_text": "Is there cardiomegaly?",
        "query_mode": "hybrid", "finding": "cardiomegaly",
        "retrieved_ids": ["d1", "d10", "d20", "d2", "d3"],
        "gold_ids": ["d1", "d10", "d20"],
        "retrieved_scores": [0.95, 0.85, 0.75, 0.65, 0.55],
        "source_doc_id": "img_001", "query_image_path": "img_001.png",
    },
    {
        "query_id": "q2", "query_text": "Is there pleural effusion?",
        "query_mode": "hybrid", "finding": "pleural effusion",
        "retrieved_ids": ["d5", "d15", "d25", "d1", "d2"],
        "gold_ids": ["d5", "d15", "d25"],
        "retrieved_scores": [0.92, 0.82, 0.72, 0.62, 0.52],
        "source_doc_id": "img_001", "query_image_path": "img_001.png",
    },
    {
        "query_id": "q3", "query_text": "Is there pneumonia?",
        "query_mode": "hybrid", "finding": "pneumonia",
        "retrieved_ids": ["d3", "d30", "d1", "d2", "d4"],
        "gold_ids": ["d3", "d30"],
        "retrieved_scores": [0.93, 0.83, 0.73, 0.63, 0.53],
        "source_doc_id": "img_001", "query_image_path": "img_001.png",
    },
]

# ── Test sensitivity analysis ──
print("Testing sensitivity analysis...")
sens = analyze_question_sensitivity(baseline_results, current_results, top_k=3)
print(f"  Baseline avg overlap: {sens['baseline_avg_overlap']}")
print(f"  Current avg overlap:  {sens['current_avg_overlap']}")
print(f"  Overlap reduction:    {sens['overlap_reduction']}")

assert sens["baseline_avg_overlap"] == 1.0, (
    f"Expected 1.0 baseline overlap, got {sens['baseline_avg_overlap']}"
)
assert sens["current_avg_overlap"] < 1.0, (
    f"Expected <1.0 current overlap, got {sens['current_avg_overlap']}"
)
print("  Sensitivity analysis: PASSED")

# ── Test report generation ──
print("\nTesting report generation...")
baseline_metrics = {
    "aggregate": {
        "recall@1": 0.3333, "recall@3": 0.6667, "recall@5": 1.0,
        "mrr": 0.5, "ndcg@3": 0.45, "ndcg@5": 0.55, "num_queries": 3,
    },
    "per_mode": {
        "hybrid": {
            "recall@1": 0.3333, "recall@3": 0.6667, "recall@5": 1.0,
            "mrr": 0.5, "num_queries": 3,
        },
    },
    "timing": {"total_seconds": 1.5},
    "config": {"mode": "baseline"},
}
current_metrics = {
    "aggregate": {
        "recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0,
        "mrr": 1.0, "ndcg@3": 1.0, "ndcg@5": 1.0, "num_queries": 3,
    },
    "per_mode": {
        "hybrid": {
            "recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0,
            "mrr": 1.0, "num_queries": 3,
        },
    },
    "timing": {"total_seconds": 2.0},
    "config": {"mode": "current"},
}

report_path = generate_observing_document(
    baseline_metrics, current_metrics, sens,
    output_dir="outputs/observations",
)
print(f"  Report generated: {report_path}")

# ── Verify output files ──
assert os.path.exists("outputs/observations/observing_document.md"), "Markdown not found"
assert os.path.exists("outputs/observations/ablation_results.json"), "JSON not found"
print("  Output files: FOUND")

# ── Verify document sections ──
with open("outputs/observations/observing_document.md", encoding="utf-8") as f:
    content = f.read()

required_sections = [
    "## 1. Problem Summary",
    "## 2. Baseline",
    "## 3. Current System",
    "## 4. Side-by-Side Comparison",
    "## 5. Question Sensitivity",
    "## 6. Observations",
    "## 7. Conclusion",
]
for section in required_sections:
    assert section in content, f"Missing section: {section}"
    print(f"  Section found: {section}")

# ── Verify comparison table ──
assert "Baseline (Before)" in content, "Missing comparison table header"
assert "Current (After)" in content, "Missing comparison table header"
assert "Δ Change" in content, "Missing delta column"
print("  Comparison table: FOUND")

# ── Verify JSON ──
import json
with open("outputs/observations/ablation_results.json", encoding="utf-8") as f:
    data = json.load(f)

assert "baseline" in data, "Missing baseline in JSON"
assert "current" in data, "Missing current in JSON"
assert "sensitivity" in data, "Missing sensitivity in JSON"
print("  JSON structure: VALID")

print("\n" + "=" * 50)
print("ALL VERIFICATION TESTS PASSED")
print("=" * 50)
