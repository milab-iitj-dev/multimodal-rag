# Kaggle Validation — Evidence of Working Pipelines

## Context

This project does **not** have access to a local GPU. All GPU-dependent validation (model loading, inference, index building, retrieval) was performed on **Kaggle** using Tesla T4 GPUs.

Both Phase 1 and Phase 2 pipelines have been validated end-to-end on Kaggle and are confirmed working.

---

## Phase 1 — Direct VQA (LLaVA-1.5-7B + QLoRA)

### What Was Validated

- LLaVA-1.5-7B loaded in 4-bit NF4 quantization (~4 GB VRAM)
- QLoRA adapter (r=16, α=32) fine-tuned on OpenI chest X-rays
- Single-image inference with clinical questions
- Evaluation on 5 diverse OpenI samples

### Results Summary

| Metric | Value |
|--------|-------|
| GPU | Tesla T4 (16 GB) |
| VRAM Usage | 4.15 GB |
| Samples Evaluated | 5/5 |
| Avg Inference Time | 7.26s |
| Quantization | 4-bit NF4 |
| Adapter | LoRA r=16, α=32 |

### Sample Outputs

| UID | Question | Model Answer |
|-----|----------|--------------|
| 1 | Key findings? | No acute cardiopulmonary abnormality identified |
| 2 | Cardiomegaly? | No acute abnormality, heart size normal |
| 4 | Lungs clear? | No acute cardiopulmonary disease identified |
| 7 | Pleural effusion? | No acute cardiopulmonary abnormality |
| 9 | Radiological impression? | No acute cardiopulmonary abnormality identified |

### Evidence Files

All Phase 1 outputs are in [`outputs/kaggle_validation/phase1_kaggle_results/`](../outputs/kaggle_validation/phase1_kaggle_results/):
- `phase1_evaluation_report.md` — Full markdown report
- `phase1_evaluation_results.json` — Structured results
- `phase1_evaluation_results.csv` — CSV format
- `phase1_console_output.txt` — Complete console log
- `screenshots/` — Kaggle notebook screenshots

---

## Phase 2 — RAG Pipeline (ColQwen2 + LLaVA)

### What Was Validated

- ColQwen2 (`vidore/colqwen2-v1.0-hf`) loaded for multi-vector document embedding
- Offline indexing of OpenI image-report pairs
- MaxSim late-interaction retrieval
- Context building from top-3 retrieved cases
- LLaVA generation conditioned on retrieved evidence
- Two complete runs: 50-sample quick validation + full 3826-document index

### Architecture Validated

```
OpenI Image-Report Pairs
  → ColQwen2 Encoding (multi-vector embeddings)
  → Saved Retrieval Index

User Query (text or image+text)
  → ColQwen2 Query Embedding
  → MaxSim Similarity Search
  → Top 3 Retrieved Cases
  → Context Builder (evidence formatting)
  → LLaVA-1.5-7B (4-bit) + LoRA
  → Grounded Medical Answer
```

### Run 1: Quick Validation (50 documents)

| Metric | Value |
|--------|-------|
| Documents Indexed | 50 |
| Embedding Dim | 128 |
| Index Build Time | 239.27s |
| Total Queries | 5 |
| Text-Only Queries | 3 |
| Image+Text Queries | 2 |
| Avg Retrieval Time | 2.71s |
| Avg Generation Time | 4.34s |
| Avg Total Time | 7.07s |

### Run 2: Full Validation (3826 documents)

| Metric | Value |
|--------|-------|
| Documents Indexed | 3826 |
| Embedding Dim | 128 |
| Index Build Time | 14734.65s (~4.1 hours) |
| Total Queries | 3 |
| Image+Text Queries | 3 |
| Avg Retrieval Time | 5.8s |
| Avg Generation Time | 7.53s |
| Avg Total Time | 13.33s |

### Sample RAG Outputs (Full Index)

**Query**: "What are the key findings in this chest X-ray?"
- **Retrieved Cases**: 1, 1785, 1738 (scores: 761.0, 661.5, 660.0)
- **Answer**: Clear lung fields without bibasilar airspace disease or nodules. Heart size appears normal for age...

**Query**: "Is there cardiomegaly or any cardiac abnormality?"
- **Retrieved Cases**: 4, 1689, 315 (scores: 761.0, 670.8, 670.7)
- **Answer**: Normal heart size and medial contours. Clear lungs without active airspace disease.

**Query**: "Are there signs of pleural effusion or pneumonia?"
- **Retrieved Cases**: 7, 2140, 3284 (scores: 740.0, 662.9, 662.0)
- **Answer**: No evidence suggestive of pleuritis, pneumonitis, or edema. Normal chest X-ray appearance.

### Evidence Files

All Phase 2 outputs are in [`outputs/kaggle_validation/phase2_kaggle_results/`](../outputs/kaggle_validation/phase2_kaggle_results/):
- `phase2_report_50samples.md` / `phase2_report_3826docs.md` — Reports
- `phase2_results_*.json` / `*.csv` — Structured results
- `phase2_console_*.txt` — Console logs

---

## How to Reproduce

### On Kaggle

1. Create a new Kaggle notebook with GPU T4 enabled
2. Enable Internet access
3. Add the "OpenI Chest X-rays Indiana University" dataset
4. Set your HuggingFace token as a Kaggle secret named `HF_TOKEN`
5. Use the scripts in `kaggle/`:
   - **Phase 1 Training**: Paste `kaggle/train_kaggle.py` → Run
   - **Phase 1 Evaluation**: Paste `kaggle/kaggle_inference.py` → Run
   - **Phase 2 Validation**: Paste `kaggle/kaggle_rag.py` → Run

### Locally (when GPU is available)

```bash
# Phase 1
python scripts/inference.py --batch-eval --max-samples 5

# Phase 2
python -m pipelines.offline_indexing --data-config configs/data_config.yaml --retrieval-config configs/retrieval_config.yaml
python -m pipelines.rag_vqa --eval --max-samples 5
```

---

## GPU Requirements

| Phase | Component | VRAM |
|-------|-----------|------|
| Phase 1 | LLaVA-1.5-7B (4-bit) | ~4–5 GB |
| Phase 2 | ColQwen2 (indexing) | ~6.5 GB |
| Phase 2 | LLaVA (generation) | ~4–5 GB |
| Phase 2 | Sequential loading | < 16 GB total |

T4 16 GB is sufficient for both phases when models are loaded sequentially (ColQwen2 → unload → LLaVA).
