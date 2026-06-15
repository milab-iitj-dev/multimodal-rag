# Architecture — Scientific Multimodal RAG

## System Overview

The Scientific Multimodal RAG system is a three-stage pipeline that answers questions about scientific papers by combining **visual** (ColPali) and **textual** (SciNCL) retrieval with **multi-modal generation** (Qwen2-VL). The design prioritises GPU memory efficiency through staggered model loading, where only one model is loaded at a time.

---

## Component Details

### 1. Embedding Layer

#### ColPali (Vision Embedder)

- **Model**: `vidore/colpali-v1.2` (Gemma-2B backbone)
- **Output**: Multi-vector embedding per page — shape `(num_tokens, 128)`
- **VRAM**: ~2.5 GB (float16)
- **Storage**: `.npy` files on disk (one per page)
- **Why multi-vector**: Preserves fine-grained alignment between visual patch tokens and query tokens for MaxSim late-interaction scoring
- **Input**: 448×448 RGB page images (resize + pad with white pixels)

#### SciNCL (Text Embedder)

- **Model**: `malteos/scincl` (SciBERT-base backbone)
- **Output**: Single 768-dim dense vector per text chunk
- **VRAM**: ~0.6 GB (float16)
- **Storage**: ChromaDB collection with cosine HNSW index
- **Processing**: Mean pooling over last hidden state → L2 normalisation
- **Max tokens**: 512 per input text

### 2. Retrieval Layer

#### ColPali Retriever (MaxSim)

- **Algorithm**: MaxSim (maximum-similarity aggregation)
  1. Compute similarity matrix: `S = query @ page.T` → shape `(Q, P)`
  2. Per query token, take max across page tokens: `max_per_query = S.max(dim=1)`
  3. Sum maxima: `score = max_per_query.sum()`
- **Why MaxSim**: Late-interaction mechanism captures fine-grained alignment; each query term independently finds its best match among visual patches
- **Why .npy**: ChromaDB stores one vector per document; ColPali produces N vectors per page

#### Text Retriever (ChromaDB ANN)

- **Backend**: ChromaDB PersistentClient with HNSW index
- **Distance**: Cosine distance (converted to similarity: `1 - distance`)
- **Normalisation**: Min-max to `[0, 1]` after retrieval

#### Fusion Retriever

- **Strategy**: Weighted score fusion with min-max normalisation
- **Weights**: 0.7 ColPali + 0.3 SciNCL (default)
- **Rationale**: Scientific papers convey critical information through figures, tables, and layout. ColPali captures these visual elements natively; SciNCL provides a safety net for purely textual queries.
- **Pages in both sets**: `fused = 0.7 * colpali + 0.3 * scincl`
- **Pages in one set**: `fused = weight * score` (not penalised to zero)

### 3. Generation Layer

#### RAG Generator

- **VLM**: Qwen2-VL-2B-Instruct (4-bit NF4 quantization)
- **VRAM**: ~1.5 GB (quantized)
- **Context**: System prompt + user prompt + page images (up to 3) + text context
- **Max new tokens**: 512
- **Temperature**: 0.3 (low for factual answers)
- **Retry logic**: If confidence < threshold (0.6), retry up to 2 more times

#### Self-Checker

Three-level verification before returning answers:

1. **Attribution**: Regex scan for `[Source: ...]` citation markers
2. **Faithfulness**: Keyword overlap between answer and context (excluding stop words)
3. **Confidence**: Threshold check on model's self-reported confidence

---

## Data Flow

```
[PDF Files]
    │
    ▼
[DualPDFParser] ──── Page Images (.png) ──→ [ColPali Embed] ──→ .npy files
    │                                              │ UNLOAD
    └── Markdown Text ──→ [SciNCL Embed] ──→ ChromaDB
                              │ UNLOAD
                              
[User Query]
    │
    ▼
[Query Validation]
    │
    ├─→ [ColPali Encode Query] ──→ Multi-vector ──→ [MaxSim Retrieve] ──→ ColPali Results
    │         │ UNLOAD
    └─→ [SciNCL Encode Query] ──→ 768-d Vector ──→ [ChromaDB ANN] ──→ SciNCL Results
              │ UNLOAD
              
[ColPali Results] + [SciNCL Results]
    │
    ▼
[Score Fusion] ──→ Normalise ──→ Weighted Sum ──→ Fused Results
    │
    ▼
[Context Builder] ──→ System Prompt + User Prompt + Images + Text
    │
    ▼
[Qwen2-VL Generate] ──→ Answer + Confidence
    │         │ UNLOAD
    ▼
[Self-Check] ──→ Attribution + Faithfulness + Confidence
    │
    ▼
[RAGResult] ──→ Answer, Sources, Scores, Check, Timing
```

---

## Design Decisions

| Decision | Rationale | Alternative Considered |
|---|---|---|
| Staggered model loading | Peak VRAM = max(single model) ~2.5 GB instead of sum ~4.6 GB | Load all models simultaneously (requires T4 or better) |
| 0.7/0.3 ColPali/SciNCL weight | Layout, figures, and tables are critical in scientific papers; text alone misses visual structure | 0.5/0.5 equal weighting |
| .npy for ColPali | ChromaDB stores one vector per document; ColPali produces N vectors per page | Custom ChromaDB multi-vector adapter |
| MaxSim scoring | Late-interaction preserves fine-grained alignment between query terms and visual regions | Single-vector pooling (loses spatial information) |
| 4-bit quantization for Qwen2-VL | Reduces VRAM from ~4 GB to ~1.5 GB | float16 (requires more VRAM) |
| Mean pooling for SciNCL | Better sentence-level representations than [CLS] token | [CLS] token pooling |
| Self-check pipeline | Catches hallucination, missing citations, and low confidence before returning answers | No verification (faster but less reliable) |
| ChromaDB for SciNCL | Fits the single-vector paradigm; HNSW index provides fast ANN search | FAISS (requires more setup) |

---

## GPU Memory Management

### Staggered Loading Strategy

The pipeline loads models one at a time and unloads each immediately after use:

```
Step 1: Load ColPali (~2.5 GB) → Encode Query → Unload ColPali
Step 2: Load SciNCL (~0.6 GB) → Encode Query → Unload SciNCL
Step 3: MaxSim Retrieve (no GPU) + ChromaDB Query (no GPU)
Step 4: Load Qwen2-VL (~1.5 GB) → Generate Answer → Unload Qwen2-VL
Step 5: Self-Check (no GPU)
```

**Peak VRAM**: ~2.5 GB (only ColPali needs to be loaded at once)

### VRAM Breakdown

| Model | VRAM (float16) | VRAM (quantized) | Load Order |
|---|---|---|---|
| ColPali | ~2.5 GB | N/A | 1st |
| SciNCL | ~0.6 GB | N/A | 2nd |
| Qwen2-VL | ~4.0 GB | ~1.5 GB (4-bit) | 3rd |

### OOM Recovery

- **ColPali batch**: Auto-reduces batch size (8 → 4 → 2 → 1 → CPU fallback)
- **SciNCL batch**: Falls back to one-by-one embedding
- **Qwen2-VL**: Falls back to sources-only mode (no generated answer)

---

## Kaggle Sessions

### Session 1: Offline Pipeline (Index Building)

**Duration**: ~30-40 minutes for 10 PDFs (~100 pages)

| Step | Time | GPU | Output |
|---|---|---|---|
| Download PDFs | 2-3 min | No | `data/raw/*.pdf` |
| Parse PDFs | 5-8 min | No | `data/parsed/pages/`, `data/parsed/markdown/` |
| ColPali embed | 10-15 min | Yes | `data/indices/multivectors/*.npy` |
| SciNCL embed | 3-5 min | Yes | `data/indices/chroma_index/` |
| Save metadata | <1 min | No | `data/indices/doc_mapping.json`, `page_metadata.json` |

### Session 2: Online Pipeline (Query Demo)

**Duration**: ~5-12 seconds per query

| Step | Time | GPU | Notes |
|---|---|---|---|
| ColPali encode query | 1-2s | Yes | Load → Encode → Unload |
| SciNCL encode query | 0.3-0.5s | Yes | Load → Encode → Unload |
| MaxSim retrieval | ~0.5s | No | CPU computation over .npy |
| ChromaDB retrieval | ~0.1s | No | HNSW ANN search |
| Context building | ~0.05s | No | String assembly |
| Qwen2-VL generation | 3-8s | Yes | Load → Generate → Unload |
| Self-check | ~0.01s | No | Regex + keyword overlap |

---

## Fallback Chains

The pipeline degrades gracefully when components fail:

| Scenario | Fallback |
|---|---|
| ColPali fails | SciNCL only (weight = 1.0) |
| SciNCL fails | ColPali only (weight = 1.0) |
| Both fail | TF-IDF keyword search over ChromaDB |
| Qwen2-VL fails | Return retrieved sources only (no answer) |
| Confidence < threshold | Retry up to 2 more times |
