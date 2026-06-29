# Retrieval Ablation — Modality Bias Analysis

**Healthcare MRAG · OpenI Chest X-ray Dataset**

*Generated: 2026-06-22 22:41:30 · 120 queries evaluated*

---

## 1. The Modality-Dominance Problem

When a user submits a chest X-ray **and** a clinical question (e.g., *"Is there cardiomegaly?"*), the retrieval system should find evidence documents relevant to **both** the image and the question. In the original system, the image modality dominated: the question text had no measurable influence on retrieval ranking.

**Symptom:** Different questions on the same X-ray → identical top-K results.

**Root cause:** Single image embedding index; image MaxSim scores (~600–900) overwhelmed text signal. Question text was present as input but effectively discarded during ranking.

---

## 2. Experimental Setup

| | Baseline (Before) | Current System (After) |
|---|---|---|
| **Retrieval** | Image index only; question discarded | Dual-index (image + text) |
| **Fusion** | None | RRF (k=60, equal weights) |
| **Reranking** | None | Question-aware keyword boost (w=0.15) |
| **Queries** | 120 (same set) | 120 (same set) |
| **Gold labels** | OpenI MeSH + Problems + non-negated text | Same |

---

## 3. Results

### Aggregate Comparison

| Metric | Baseline | Current | Δ Change |
|--------|----------|---------|----------|
| **Recall@1** | 0.5167 | 0.6750 | +0.1583 |
| **Recall@3** | 0.6583 | 0.7750 | +0.1167 |
| **Recall@5** | 0.7333 | 0.8083 | +0.0750 |
| **MRR** | 0.5976 | 0.7256 | +0.1280 |
| **nDCG@3** | 0.4061 | 0.5757 | +0.1696 |
| **nDCG@5** | 0.3880 | 0.5626 | +0.1746 |

### Visual Comparison

```
  Baseline vs Current System — Key Metrics
  ─────────────────────────────────────────

    Recall@1
  Baseline  |███████████████               | 0.5167
  Current   |████████████████████          | 0.6750

    Recall@3
  Baseline  |███████████████████           | 0.6583
  Current   |███████████████████████       | 0.7750

    Recall@5
  Baseline  |█████████████████████         | 0.7333
  Current   |████████████████████████      | 0.8083

         MRR
  Baseline  |█████████████████             | 0.5976
  Current   |█████████████████████         | 0.7256

      nDCG@3
  Baseline  |████████████                  | 0.4061
  Current   |█████████████████             | 0.5757

      nDCG@5
  Baseline  |███████████                   | 0.3880
  Current   |████████████████              | 0.5626
```

### Per Query Mode

| Mode | Metric | Baseline | Current | Δ |
|------|--------|----------|---------|---|
| **text_only** | R@1 | 0.7500 | 0.7500 | +0.0000 |
|  | R@3 | 0.8500 | 0.8500 | +0.0000 |
|  | R@5 | 0.8500 | 0.8500 | +0.0000 |
|  | MRR | 0.7875 | 0.7875 | +0.0000 |
| **image_only** | R@1 | 0.4000 | 0.4000 | +0.0000 |
|  | R@3 | 0.5750 | 0.5750 | +0.0000 |
|  | R@5 | 0.6750 | 0.6750 | +0.0000 |
|  | MRR | 0.5058 | 0.5058 | +0.0000 |
| **hybrid** | R@1 | 0.4000 | 0.8750 | +0.4750 |
|  | R@3 | 0.5500 | 0.9000 | +0.3500 |
|  | R@5 | 0.6750 | 0.9000 | +0.2250 |
|  | MRR | 0.4996 | 0.8833 | +0.3837 |

---

## 4. Question Sensitivity

We measure how much retrieval results change when the **same image** is queried with **different questions**. Pairwise Jaccard overlap of top-3 retrieved document sets is computed across question pairs.

- **High overlap (→1.0):** question ignored; image dominates.
- **Lower overlap:** question meaningfully influences retrieval.

| Measure | Value |
|---------|-------|
| Images analyzed | 10 |
| Baseline avg. overlap | **1.0000** |
| Current avg. overlap | **0.2735** |
| Overlap reduction | **0.7265** |

### Concrete Example

**Image `1000`** — 8 different questions

Baseline top-3 overlap: **1.0000** · Current top-3 overlap: **0.2214**

> *"Is there atelectasis?"*
> Baseline: `2816`, `2535`, `1705`
> Current:  `304`, `1063`, `2961`

> *"Is there mass?"*
> Baseline: `2816`, `2535`, `1705`
> Current:  `1000`, `1081`, `1303`

---

## 5. Why the Fix Works

| Component | Effect |
|-----------|--------|
| **Text index** | Question text gets its own retrieval path (text→text MaxSim), producing a separate ranking that reflects question intent |
| **RRF fusion** | Combines image and text rankings by rank position (not raw score), preventing image scores from overwhelming text signal |
| **Reranking** | Keyword overlap between question and document clinical text boosts question-relevant documents after fusion |

The three mechanisms are complementary: the text index provides question-aware retrieval candidates, RRF balances both modalities, and reranking fine-tunes the final ranking for question specificity.

---

## 6. Conclusion

The ablation confirms that **modality dominance** was the primary retrieval limitation. The dual-index + RRF + reranking architecture restores question sensitivity while preserving visual retrieval quality.

**Key metric improvements:**

- **Recall@1:** 0.5167 → 0.6750 (+0.1583)
- **Recall@3:** 0.6583 → 0.7750 (+0.1167)
- **Recall@5:** 0.7333 → 0.8083 (+0.0750)
- **MRR:** 0.5976 → 0.7256 (+0.1280)
- **nDCG@3:** 0.4061 → 0.5757 (+0.1696)
- **nDCG@5:** 0.3880 → 0.5626 (+0.1746)

**Question sensitivity:** Top-3 overlap reduced from 1.0000 → 0.2735, confirming that the system now responds to different clinical questions on the same image.

---

*Healthcare MRAG Ablation Analysis Suite · OpenI Chest X-ray Dataset · ColQwen2 + RRF*
