# Benchmark Results

Organized benchmark outputs for reviewer reference.

## Directory Structure

```
results/
├── medical/          Healthcare MRAG (OpenI Chest X-ray)
│   ├── retrieval/    Retrieval benchmark metrics and report
│   ├── ablation/     Modality-dominance ablation study
│   ├── figures/      Generated visualizations
│   └── README.md
│
├── scientific/       Scientific Paper QA (placeholder)
│   ├── retrieval/    Retrieval benchmark metrics
│   ├── generation/   Generation quality evaluation
│   ├── figures/      Generated visualizations
│   └── README.md
│
└── hybrid/           Cross-domain comparison
    ├── comparison/   Domain comparison analysis
    ├── figures/      Generated visualizations
    └── README.md
```

## Quick Links

| Domain | Key Result | Status |
|--------|-----------|--------|
| [Medical](medical/) | R@5 = 0.8083, MRR = 0.7256, nDCG@5 = 0.5626 | ✅ Complete |
| [Scientific](scientific/) | — | 🔲 Pending |
| [Hybrid](hybrid/) | — | 🔲 Pending |

## Medical Highlights

- **120 queries** evaluated across 3 retrieval modes (text-only, image-only, hybrid)
- **Modality-dominance fix** improved Recall@1 from 0.5167 → 0.6750 (+30.6%)
- **Question sensitivity** restored: top-3 overlap reduced from 1.0 → 0.27
- Full analysis: [Modality Dominance Finding](../docs/findings/modality_dominance.md)
