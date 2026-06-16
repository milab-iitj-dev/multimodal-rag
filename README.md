 🔬🏥 MMRAG Unified — Multimodal Retrieval-Augmented Generation

**A unified system combining Healthcare and Scientific Multimodal RAG pipelines.**

Ask questions about chest X-rays **or** scientific research papers, and get accurate, evidence-grounded answers powered by state-of-the-art vision-language models.

---

## Architecture

```
                         ┌──────────────────────────────────────┐
                         │            USER QUERY                │
                         │    "Is there pleural effusion?"      │
                         │    "What is the Vision Transformer?" │
                         └────────────────┬─────────────────────┘
                                          │
                              ┌───────────▼───────────┐
                              │    Domain Router      │
                              │   (auto-detection)    │
                              └───────────┬───────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                            ▼
        ┌───────────────────┐                       ┌───────────────────┐
        │  🏥 Healthcare    │                       │  🔬 Scientific    │
        │     Pipeline      │                       │     Pipeline      │
        ├───────────────────┤                       ├───────────────────┤
        │ ColQwen2 Embedder │                       │ ColPali Embedder  │
        │ Dual-Index MaxSim │                       │ SciNCL Embedder   │
        │ RRF Fusion        │                       │ ChromaDB ANN      │
        │ Evidence Aggreg.  │                       │ Score Fusion      │
        │ Qwen2-VL Generate │                       │ Qwen2-VL Generate │
        │ Grounding Verify  │                       │ Self-Check (3lvl) │
        │ Confidence Score  │                       │ Citation Attrib.  │
        └───────────────────┘                       └───────────────────┘
```

---

## Domains

### 🏥 Healthcare — Chest X-ray VQA

- **Dataset**: OpenI (3,826 chest X-ray image-report pairs)
- **Embeddings**: ColQwen2 (multi-vector, per-patch + per-token)
- **Retrieval**: Dual-index MaxSim + Reciprocal Rank Fusion + question-aware reranking
- **Generation**: Qwen2-VL-7B-Instruct (4-bit quantized)
- **Verification**: Evidence-based grounding + confidence scoring (0.0–1.0)
- **Query types**: Binary clinical, descriptive, text-only, mixed

### 🔬 Scientific — Research Paper QA

- **Dataset**: arXiv Vision Transformer papers (PDF → page images + text)
- **Embeddings**: ColPali (vision, multi-vector) + SciNCL (text, 768-d dense)
- **Retrieval**: ColPali MaxSim + ChromaDB ANN → weighted score fusion (0.7 / 0.3)
- **Generation**: Qwen2-VL (4-bit quantized)
- **Verification**: Self-check with attribution, faithfulness, and relevance

---

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/milab-iitj-dev/mmrag-unified.git
cd mmrag-unified

# Install all dependencies
pip install -r requirements.txt

# Or install per-domain
pip install -e ".[healthcare]"   # Healthcare only
pip install -e ".[scientific]"   # Scientific only
pip install -e ".[all]"          # Everything
```

### Healthcare Pipeline

```bash
# Step 1: Build the ColQwen2 index (run once)
python -m pipelines.healthcare.offline_indexing

# Step 2: Run a query
python -m pipelines.healthcare.rag_vqa --query "Is there pleural effusion?" --image path/to/xray.png

# Step 3: Launch the Gradio UI
python -m scripts.healthcare.launch_ui
```

### Scientific Pipeline

```bash
# Step 1: Download and parse papers + build index (run once)
python -m pipelines.scientific.offline_pipeline

# Step 2: Run a query
python -m pipelines.scientific.online_pipeline --query "What is the Vision Transformer?"

# Step 3: Launch the Gradio UI
python ui/scientific/gradio_app.py
```

### Unified API

```bash
# Start the FastAPI server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Query (auto-detects domain)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Is there cardiomegaly?", "domain": "healthcare"}'
```

---

## Project Structure

```
mmrag_unified/
├── configs/                        # Configuration files
│   ├── unified_config.yaml         # Router + API settings
│   ├── healthcare/                 # Healthcare-specific configs
│   └── scientific/                 # Scientific-specific configs
│
├── src/                            # Core library
│   ├── shared/                     # Shared utilities (logging, device, image, config)
│   ├── router/                     # Domain detection + routing
│   │   └── domain_router.py
│   ├── api/                        # Unified FastAPI endpoint
│   │   └── app.py
│   └── domains/                    # Domain-specific implementations
│       ├── healthcare/             # Healthcare MRAG pipeline
│       │   ├── ingestion/          # OpenI dataset loading
│       │   ├── embeddings/         # ColQwen2 multi-vector embedder
│       │   ├── indexing/           # Dual index builder + document store
│       │   ├── retrieval/          # MaxSim + RRF hybrid retriever
│       │   ├── context/            # Query classifier + evidence aggregator
│       │   └── generation/         # Qwen2-VL + grounding + confidence
│       └── scientific/             # Scientific MRAG pipeline
│           ├── embeddings/         # ColPali + SciNCL embedders
│           ├── retrieval/          # MaxSim + ChromaDB + fusion
│           ├── context/            # Context builder + prompt templates
│           ├── generation/         # RAG generator + self-check
│           └── models/             # VLM wrappers + model factory
│
├── pipelines/                      # End-to-end pipelines
│   ├── healthcare/                 # offline_indexing, rag_vqa, simple_vqa
│   └── scientific/                 # offline_pipeline, online_pipeline
│
├── scripts/                        # CLI entry points
│   ├── healthcare/                 # inference, validate, train, launch_ui
│   └── scientific/                 # HPC scripts, SLURM jobs
│
├── ui/                             # Web interfaces
│   ├── healthcare/                 # Gradio medical-themed UI
│   └── scientific/                 # Gradio + Streamlit UIs
│
├── evaluation/                     # Benchmark framework
├── analysis/ablation/              # Ablation study tools
├── tests/                          # Test suites per domain
├── docs/                           # Documentation per domain
│
├── main.py                         # Scientific entry point
├── requirements.txt                # Merged dependencies
├── setup.py                        # Package installation
└── README.md                       # This file
```

---

## Key Differences Between Domains

| Feature | Healthcare | Scientific |
|---------|-----------|------------|
| **Data source** | OpenI chest X-rays + reports | arXiv PDF papers |
| **Image embedder** | ColQwen2 (multi-vector) | ColPali (multi-vector) |
| **Text embedder** | ColQwen2 (shared model) | SciNCL (separate model) |
| **Text index** | `.pt` tensor files | ChromaDB vector DB |
| **Fusion** | RRF (rank-based) | Weighted score (0.7/0.3) |
| **Verification** | Grounding + Confidence | Self-check (3-level) |
| **VLM** | Qwen2-VL-7B | Qwen2-VL |
| **Unique feature** | Evidence consensus + auto-correction | Citation attribution |

---

## Documentation

- [Unified Architecture](docs/architecture.md)
- **Healthcare**: [Domain Guide](docs/healthcare/healthcare_domain.md) · [Kaggle Setup](docs/healthcare/kaggle_setup.md)
- **Scientific**: [Architecture](docs/scientific/architecture.md) · [Run Local](docs/scientific/RUN_LOCAL.md) · [Run HPC](docs/scientific/RUN_HPC.md) · [Evaluation](docs/scientific/evaluation_guide.md)

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

## Citation

If you use this work, please cite:

```bibtex
@software{mmrag_unified,
  title={MMRAG Unified: Multimodal RAG for Healthcare and Scientific Domains},
  author={MILab IITJ},
  year={2025},
}
```
