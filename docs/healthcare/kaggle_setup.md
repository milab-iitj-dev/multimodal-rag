# Kaggle Setup Guide

This project uses Kaggle for GPU-dependent validation since a local GPU is not available.

## GPU Requirements

| Phase | Component | VRAM | Recommended GPU |
|-------|-----------|------|-----------------|
| Phase 1 | LLaVA-1.5-7B (4-bit) | ~4–5 GB | T4 16 GB |
| Phase 2 | ColQwen2 (indexing) | ~6.5 GB | T4 16 GB |
| Phase 2 | LLaVA (generation) | ~4–5 GB | T4 16 GB |
| Full RAG | Sequential loading | < 16 GB | T4 16 GB |

## VRAM Management

When running both ColQwen2 and LLaVA on a single GPU:
1. Load ColQwen2, build/load index
2. Unload ColQwen2 (`embedder.unload()`)
3. Load LLaVA for generation
4. Peak VRAM stays under 16 GB

## Quick Setup (Using Kaggle Scripts)

The simplest way to run on Kaggle is to use the self-contained scripts in the `kaggle/` directory. Each script installs its own dependencies and finds datasets automatically.

### Step 1: Create Kaggle Notebook

1. Go to [kaggle.com](https://www.kaggle.com) → "New Notebook"
2. **Settings** → GPU: T4 x2, Internet: ON
3. Add dataset: search "OpenI Chest X-rays Indiana University"

### Step 2: Set HuggingFace Token

Go to "Add-ons" → "Secrets" → Add new secret:
- **Label**: `HF_TOKEN`
- **Value**: Your HuggingFace token

### Step 3: Run

Paste the entire contents of the desired script into one cell and run:

| Script | Purpose | Runtime |
|--------|---------|---------|
| `kaggle/train_kaggle.py` | QLoRA training | ~2–3 hours |
| `kaggle/kaggle_inference.py` | Phase 1 evaluation | ~5 minutes |
| `kaggle/kaggle_rag.py` | Phase 2 full validation | ~4–5 hours |

### Step 4: Download Results

After the run completes, go to the "Output" tab and download:
- `results/` — JSON, CSV, and Markdown reports
- `final_adapter/` — Trained LoRA adapter (training only)

## Advanced Setup (Using Modular Code)

For development or custom experiments, you can use the modular codebase:

```python
# Install dependencies
!pip install -q transformers>=4.46.0 accelerate peft bitsandbytes colpali-engine

# Add your repo as a Kaggle dataset, then:
import sys
sys.path.insert(0, "/kaggle/input/healthcare-mrag/")

# Phase 1
from src.generation.model_factory import create_model
from src.ingestion.dicom_loader import OpenIDataset

# Phase 2
from src.embeddings.colqwen2_embedder import ColQwen2Embedder
from src.retrieval.colqwen2_retriever import ColQwen2Retriever
from src.context.context_builder import ContextBuilder
from src.generation.rag_generator import RAGGenerator
```

## Validation Results

See [`outputs/kaggle_validation/`](../outputs/kaggle_validation/) for saved outputs from successful runs. Detailed analysis is in [`docs/kaggle_validation.md`](kaggle_validation.md).
