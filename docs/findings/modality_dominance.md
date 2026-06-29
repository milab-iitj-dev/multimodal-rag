# Modality Dominance in Multimodal Retrieval

**Healthcare MRAG · OpenI Chest X-ray Dataset · ColQwen2 + RRF**

---

## 1. Problem

When a user submits a chest X-ray **and** a clinical question (e.g., *"Is there cardiomegaly?"*), the retrieval system should find evidence relevant to **both** the image and the question. In the original system, the image modality dominated retrieval completely — different questions on the same X-ray returned **identical** top-K results.

**Symptom:** Question text had zero influence on retrieval ranking.

---

## 2. Root Cause

The original retrieval architecture used a **single image embedding index**. Query images were encoded via ColQwen2 and matched against document image embeddings using MaxSim scoring. While the question text was accepted as input, it was not used in any ranking computation.

Image MaxSim scores (600–900 range) existed in a single-modality space with no competing signal, making it impossible for question intent to affect the result.

---

## 3. Proposed Solution

A three-component fix targeting complementary aspects of the problem:

| Component | Role |
|-----------|------|
| **Dual index** | Separate image and text embedding indexes, each producing an independent ranking |
| **RRF fusion** | Reciprocal Rank Fusion (k=60) combines rankings by **position**, not raw score — preventing image scores from overwhelming text signal |
| **Keyword reranking** | Post-fusion boost (w=0.15) for documents whose clinical text overlaps with question keywords |

---

## 4. Experimental Method

- **Dataset:** OpenI Chest X-ray (Indiana University)
- **Queries:** 120 evaluation queries across 3 modes:
  - `text_only` (40) — question text without image
  - `image_only` (40) — image without question text
  - `hybrid` (40) — image + question together
- **Gold labels:** OpenI MeSH terms, Problems fields, and non-negated text mentions
- **Metrics:** Recall@k, MRR, nDCG@k, question sensitivity (Jaccard overlap)

Both baseline and current systems were evaluated on the identical query set.

---

## 5. Results

### Aggregate Improvement

| Metric | Baseline | Current | Δ Change |
|--------|----------|---------|----------|
| **Recall@1** | 0.5167 | **0.6750** | +0.1583 |
| **Recall@3** | 0.6583 | **0.7750** | +0.1167 |
| **Recall@5** | 0.7333 | **0.8083** | +0.0750 |
| **MRR** | 0.5976 | **0.7256** | +0.1280 |
| **nDCG@3** | 0.4061 | **0.5757** | +0.1696 |
| **nDCG@5** | 0.3880 | **0.5626** | +0.1746 |

### Per Query Mode

| Mode | Metric | Baseline | Current | Δ |
|------|--------|----------|---------|---|
| text_only | R@1 / MRR | 0.7500 / 0.7875 | 0.7500 / 0.7875 | 0.0 |
| image_only | R@1 / MRR | 0.4000 / 0.5058 | 0.4000 / 0.5058 | 0.0 |
| **hybrid** | **R@1 / MRR** | **0.4000 / 0.4996** | **0.8750 / 0.8833** | **+0.4750 / +0.3837** |

The improvement is concentrated entirely in the **hybrid** mode — exactly where the modality-dominance problem manifested. Text-only and image-only modes are unaffected, confirming the fix is targeted.

---

## 6. Question Sensitivity

We measure whether different questions on the **same image** produce different retrieval results, using pairwise Jaccard overlap of top-3 retrieved document sets.

| Measure | Baseline | Current |
|---------|----------|---------|
| Avg. top-3 overlap | **1.0000** | **0.2735** |
| Overlap reduction | — | **72.65%** |
| Images analyzed | 10 | 10 |

**Interpretation:** An overlap of 1.0 means the question is completely ignored. The reduction to 0.27 confirms that the system now produces different retrieval results for different clinical questions on the same X-ray.

**Concrete example** — Image `1000` with 8 different questions:

> *"Is there atelectasis?"* → Retrieved: `304`, `1063`, `2961`  
> *"Is there mass?"* → Retrieved: `1000`, `1081`, `1303`  
> Baseline retrieved `2816`, `2535`, `1705` for **all 8 questions**.

---

## 7. Key Takeaways

1. **Modality dominance was the primary retrieval limitation.** The image signal completely suppressed question intent in the original system.

2. **RRF fusion is critical.** Combining by rank position (not raw score) is what prevents one modality from dominating — score-level fusion would preserve the imbalance.

3. **The fix is targeted.** Text-only and image-only modes are unchanged; the improvement is localized to multimodal (hybrid) queries where the problem existed.

4. **Question sensitivity is restored.** Top-3 overlap dropped from 1.0 → 0.27, meaning the system now responds meaningfully to different clinical questions.

---

## 8. Conclusion

The dual-index + RRF + reranking architecture eliminates the modality-dominance problem. Hybrid retrieval Recall@1 improved from 0.40 → 0.88 (+120%), and question sensitivity was restored from zero to 72.65% differentiation. The fix is targeted, preserving unimodal retrieval quality while enabling genuine multimodal evidence retrieval.

---

*Source data: [Ablation Results](../../results/medical/ablation/ablation_results.json) · [Observing Document](../../results/medical/ablation/observing_document.md) · [Retrieval Report](../../results/medical/retrieval/RETRIEVAL_REPORT.md)*
