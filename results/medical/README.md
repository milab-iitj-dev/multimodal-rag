# Medical Domain — Benchmark Results

**System:** Healthcare MRAG · ColQwen2 dual-index retrieval · RRF fusion · Qwen2-VL generation  
**Dataset:** OpenI Chest X-ray (Indiana University) · 120 evaluation queries

---

## Contents

### Retrieval Benchmark

| File | Description |
|------|-------------|
| [RETRIEVAL_REPORT.md](retrieval/RETRIEVAL_REPORT.md) | Full retrieval benchmark report with aggregate metrics, per-mode breakdown, and per-query diagnostics |
| [retrieval_metrics.json](retrieval/retrieval_metrics.json) | Raw benchmark output (metrics, per-sample results, system configuration) |

### Ablation Study

| File | Description |
|------|-------------|
| [observing_document.md](ablation/observing_document.md) | Complete modality-dominance analysis: problem identification, experimental setup, before/after comparison, question sensitivity measurement, and conclusion |
| [ablation_results.json](ablation/ablation_results.json) | Raw ablation data: baseline vs current metrics, per-mode breakdown, question sensitivity overlap measurements |

### Figures

Generated visualizations will be placed here as benchmarks produce them.

---

## Key Metrics (Current System)

| Metric | Score |
|--------|-------|
| Recall@1 | 0.6750 |
| Recall@3 | 0.7750 |
| Recall@5 | 0.8083 |
| MRR | 0.7256 |
| nDCG@3 | 0.5757 |
| nDCG@5 | 0.5626 |

## Improvement Over Baseline

| Metric | Baseline | Current | Δ |
|--------|----------|---------|---|
| Recall@1 | 0.5167 | 0.6750 | +0.1583 |
| MRR | 0.5976 | 0.7256 | +0.1280 |
| nDCG@5 | 0.3880 | 0.5626 | +0.1746 |

The improvement is attributed to the dual-index + RRF fusion architecture that resolves the modality-dominance problem. See [Finding Report](../../docs/findings/modality_dominance.md) for details.
