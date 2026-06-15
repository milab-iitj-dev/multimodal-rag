# Hybrid Extension Guide

Future plan for merging the Scientific Multimodal RAG system with Gokul's medical domain RAG, creating a hybrid system that routes queries between scientific and medical retrieval pipelines.

---

## Overview

The hybrid extension adds a **query router** that classifies incoming queries as either **scientific** or **medical**, then routes them to the appropriate retrieval and generation pipeline. This enables a single system to answer questions across both domains.

---

## Architecture

```
[User Query]
    │
    ▼
[Query Router] ──→ Scientific? ──→ [Scientific RAG Pipeline]
    │                                  (ColPali + SciNCL + Qwen2-VL)
    │
    └──→ Medical? ──→ [Medical RAG Pipeline]
                        (PubMed Embeddings + Medical VLM)
```

---

## Components to Implement

### 1. Query Router (`future/hybrid/query_router.py`)

A keyword-based and ML-based router that classifies queries:

- **Keyword rules**: Medical terms (diagnosis, treatment, patient, clinical, drug, dosage) route to medical; scientific terms (architecture, model, training, benchmark, SOTA) route to scientific.
- **Fallback**: If no clear signal, route to both and merge results.
- **Confidence threshold**: If routing confidence < 0.5, run both pipelines and merge.

### 2. Medical RAG Pipeline

- **Embedding**: PubMedBERT or BioLORD for medical text
- **Retrieval**: ChromaDB with medical corpus (PubMed abstracts, clinical guidelines)
- **Generation**: Medically-finetuned VLM or general VLM with medical system prompt
- **Self-check**: Medical-specific checks (dosage verification, contraindication flags)

### 3. Result Merger

- Combine results from both pipelines when the query is ambiguous
- Weight by routing confidence: `merged = α * scientific + (1-α) * medical`
- Deduplicate overlapping sources

---

## Integration Points

| Component | Scientific | Medical | Hybrid |
|---|---|---|---|
| Query validation | Shared | Shared | Shared |
| Query routing | N/A | N/A | **New** |
| Embedding | ColPali + SciNCL | PubMedBERT | Both |
| Retrieval | MaxSim + ChromaDB | ChromaDB | Merged |
| Generation | Qwen2-VL | Medical VLM | Routed |
| Self-check | 3-level | 3-level + medical | Merged |

---

## Implementation Steps

1. **Implement QueryRouter** with keyword rules and optional ML classifier
2. **Create MedicalEmbedder** following the BaseEmbedder interface
3. **Create MedicalRetriever** following the BaseRetriever interface
4. **Build HybridPipeline** that orchestrates routing and merging
5. **Add medical self-check rules** (dosage verification, contraindication)
6. **Test with mixed queries** spanning both domains
7. **Deploy on Kaggle** with both corpora indexed

---

## Configuration

Add a `hybrid_config.yaml`:

```yaml
hybrid:
  router:
    type: "keyword"  # "keyword" | "ml" | "ensemble"
    confidence_threshold: 0.5
    medical_keywords:
      - "diagnosis"
      - "treatment"
      - "patient"
      - "clinical"
      - "drug"
      - "dosage"
    scientific_keywords:
      - "architecture"
      - "model"
      - "training"
      - "benchmark"
  medical:
    embedder: "PubMedBERT"
    corpus_path: "data/medical/"
  fusion:
    method: "weighted"  # "weighted" | "reciprocal_rank"
    scientific_weight: 0.6
    medical_weight: 0.4
```

---

## Testing Strategy

- Unit tests for QueryRouter classification accuracy
- Integration tests with mixed scientific/medical queries
- A/B comparison: scientific-only vs. hybrid on scientific queries
- Latency benchmarking for routing overhead (should be < 100ms)
