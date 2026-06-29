# Hybrid / Cross-Domain — Benchmark Results

**System:** MMRAG Unified API with automatic domain routing  
**Status:** 🔲 Comparison pending

---

## Planned Contents

### Cross-Domain Comparison

Side-by-side comparison of medical and scientific retrieval performance under the unified API.

| Comparison | Description |
|-----------|-------------|
| Latency | End-to-end response time per domain |
| Retrieval quality | Recall@k and MRR per domain |
| Auto-routing accuracy | Domain classification accuracy on mixed query sets |
| API consistency | Response format validation across domains |

### Figures

Comparative visualizations across domains.

---

## Notes

Hybrid benchmarks will be run once both medical and scientific pipelines are fully operational under the unified API. The auto-routing evaluation can be performed using `tests/api/test_app.py` which validates domain detection with 55 test cases.
