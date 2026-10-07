# Scientific MRAG Domain — HPC Setup & Asset Verification Guide

This document outlines the required data indices, models, and configuration files for the **Scientific MRAG Pipeline** (ColPali + SciNCL + ChromaDB + Qwen2-VL).

---

## 1. Required Asset Checklist

To enable full online inference for the Scientific pipeline, the following files must be present on the HPC under `configs/scientific/` and the indexed path (`data/indices` or `$RAG_BASE_DIR/data/indices`):

| Asset Path | Required Files / Directories | Purpose |
| :--- | :--- | :--- |
| `configs/scientific/config.yaml` | `config.yaml` | Scientific pipeline hyperparameters & model names |
| `data/indices/page_metadata.json` | `page_metadata.json` | Document page metadata, titles, and arXiv IDs |
| `data/indices/chroma_index/` | ChromaDB collection files | SciNCL text vector embeddings for paragraph retrieval |
| `data/indices/multivectors/` | `*.npy` multi-vector files | ColPali visual page embeddings for patch retrieval |

---

## 2. Canonical Data Sync (Rsync Commands)

If the Scientific data indices are located on a local machine or release repository, transfer them using rsync:

```bash
# 1. Sync Scientific Config
rsync -avzP configs/scientific/ gokulfaleja@10.6.7.112:/data/gokulfaleja/projects/mmrag_unified/configs/scientific/

# 2. Sync Scientific Multi-vector Indices (ColPali & SciNCL ChromaDB)
rsync -avzP data/indices/ gokulfaleja@10.6.7.112:/data/gokulfaleja/data/indices/
```

---

## 3. HPC Verification Commands

After transferring assets, run the following verification command inside the `mmrag` conda environment:

```bash
python -c "
import os, json, yaml, torch

config_path = 'configs/scientific/config.yaml'
index_dir = 'data/indices'

print('1. Checking config:', os.path.exists(config_path))
if os.path.exists(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        print('   Loaded config successfully.')

print('2. Checking page_metadata.json:', os.path.exists(f'{index_dir}/page_metadata.json'))
print('3. Checking chroma_index:', os.path.exists(f'{index_dir}/chroma_index'))
print('4. Checking multivectors:', os.path.exists(f'{index_dir}/multivectors'))

# Imports
try:
    import chromadb
    import colpali_engine
    print('5. Imports verified: chromadb and colpali_engine loaded successfully.')
except ImportError as e:
    print('5. Import Error:', e)
"
```

---

## 4. Scientific Query Execution Test

To test the scientific pipeline independently once loaded:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the Vision Transformer architecture and patch tokenization.",
    "domain": "scientific",
    "top_k": 3
  }'
```
