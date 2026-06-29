# Scientific Domain — Benchmark Results

**System:** Scientific Paper QA · ColPali + SciNCL retrieval · Qwen2-VL generation  
**Status:** 🔲 Benchmarks pending

---

## Planned Contents

### Retrieval Benchmark

Evaluation of the scientific retrieval pipeline using paper corpora.

| Metric | Target |
|--------|--------|
| Recall@k | Measure hit rate on curated question sets |
| MRR | Mean reciprocal rank of first relevant document |
| nDCG@k | Graded relevance ranking quality |

### Generation Benchmark

Evaluation of answer generation quality using scientific QA datasets.

| Metric | Target |
|--------|--------|
| ROUGE-L | Surface-level text overlap |
| BERTScore | Semantic similarity |
| Faithfulness | Answer grounded in retrieved evidence |

### Figures

Visualizations comparing retrieval modes and generation quality.

---

## Notes

Scientific pipeline benchmarks will be run once the pipeline is fully integrated with the unified API. The evaluation runners are available at `evaluation/runners/`.
