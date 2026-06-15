# Unified MMRAG Architecture

## System Overview

MMRAG Unified combines two domain-specific Multimodal RAG pipelines
into a single project with shared utilities and a unified API layer.

```
                    ┌──────────────────────┐
                    │     User Query       │
                    │  (text ± image)      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │    Domain Router     │
                    │  domain_router.py    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                  ▼
    ┌─────────────────┐              ┌──────────────────┐
    │   Healthcare    │              │   Scientific     │
    │   Pipeline      │              │   Pipeline       │
    └─────────────────┘              └──────────────────┘
```

## Healthcare Pipeline

```
QueryClassifier → HybridRetriever (ColQwen2 RRF) →
EvidenceAggregator → Qwen2-VL → GroundingVerifier → ConfidenceEstimator
```

- **Data**: OpenI chest X-ray dataset (images + radiology reports)
- **Embeddings**: ColQwen2 multi-vector (per-patch image + per-token text)
- **Retrieval**: Dual-index MaxSim + Reciprocal Rank Fusion + question-aware reranking
- **Generation**: Qwen2-VL-7B (4-bit quantized)
- **Verification**: Evidence-based grounding check + confidence scoring (0.0–1.0)

## Scientific Pipeline

```
ColPali + SciNCL → Weighted Score Fusion →
ContextBuilder → Qwen2-VL → SelfCheck (3-level)
```

- **Data**: arXiv papers (PDF → page images + extracted text)
- **Embeddings**: ColPali (vision, multi-vector) + SciNCL (text, 768-d dense)
- **Retrieval**: ColPali MaxSim (.npy) + ChromaDB ANN → weighted score fusion (0.7/0.3)
- **Generation**: Qwen2-VL (4-bit quantized)
- **Verification**: Self-check with attribution, faithfulness, and relevance scoring

## Shared Layer

Both pipelines share:
- `logging_utils.py` — Structured file + console logging
- `device.py` — GPU detection and VRAM management
- `image_utils.py` — Image loading and preprocessing
- `config_loader.py` — YAML config loading + path resolution

## API Layer

A single FastAPI endpoint (`src/api/app.py`) that:
- Accepts `POST /query` with `{query, domain, image_path}`
- Auto-detects domain via keyword analysis or uses explicit parameter
- Routes to the correct pipeline
- Returns a unified response format
