# MMRAG Unified — Complete HPC Validation & Audit Report

**Date:** October 8, 2026  
**System Version:** 2.0.0 (Unified RAG Platform)  
**Author:** Senior ML Systems Engineer  

---

## 1. System Architecture & End-to-End Execution Flow

### Architecture Map

```
                          ┌───────────────────────────┐
                          │    USER UI / Frontend     │
                          │   (React / Vite @ 5173)   │
                          └─────────────┬─────────────┘
                                        │  POST /query (JSON: query, image_b64, domain)
                                        ▼
                          ┌───────────────────────────┐
                          │   FastAPI Backend Server  │
                          │      (src.api.app:app)    │
                          └─────────────┬─────────────┘
                                        │  route(query, domain_hint, image)
                                        ▼
                          ┌───────────────────────────┐
                          │       DomainRouter        │
                          │  (Keyword/Bigram/Explicit)│
                          └──────┬─────────────┬──────┘
                                 │             │
                ┌────────────────┘             └────────────────┐
                │ healthcare                                    │ scientific
                ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│       HealthcarePipeline      │               │       ScientificPipeline      │
│       (Adapter Wrapper)       │               │       (Adapter Wrapper)       │
└───────────────┬───────────────┘               └───────────────┬───────────────┘
                │                                               │
                ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│        RAGVQAPipeline         │               │        OnlinePipeline         │
│  - ColQwen2 Dual Index        │               │  - ColPali Visual Index       │
│  - RRF Score Fusion           │               │  - SciNCL Text Index          │
│  - Qwen2-VL Generation        │               │  - Score Fusion               │
│  - Grounding Verification     │               │  - Citation Attribution       │
│  - Confidence Scoring         │               │  - Qwen2-VL Generation        │
└───────────────┬───────────────┘               └───────────────┬───────────────┘
                │                                               │
                └────────────────┐             ┌────────────────┘
                                 │             │
                                 ▼             ▼
                          ┌───────────────────────────┐
                          │     UnifiedResponse       │
                          │   (Frozen API Contract)   │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │      QueryResponse        │
                          │ (Sources, Metadata, Conf) │
                          └───────────────────────────┘
```

---

## 2. Healthcare Pipeline Audit & Component Status

Healthcare is the core domain pipeline. Each component has been inspected in detail:

| Component | Status | Implementation Details |
| :--- | :--- | :--- |
| **OpenI Chest X-Ray Data** | Enabled | Supported locally and via symlinks (`data/openi/images/`). |
| **ColQwen2 Retrieval** | Enabled | Late-interaction visual/patch retrieval via `ColQwen2Retriever`. |
| **Clinical Text Retrieval**| Optional / Fallback | Evaluated when `text_embeddings.pt` is present; falls back to ColQwen2 cross-modal retrieval when absent. |
| **RRF Fusion** | Enabled | Reciprocal Rank Fusion (`rrf_score`) in `HybridRetriever`. |
| **Question-Aware Reranking**| Enabled | Query type classification (binary clinical vs open-ended). |
| **Evidence Aggregation** | Enabled | Top-k document findings & impressions passed to VLM prompt. |
| **Qwen2-VL Generation** | Enabled | `Qwen/Qwen2-VL-7B-Instruct` handles VQA generation. |
| **Grounding Verification**| Enabled | `GroundingVerifier` checks response consistency with retrieved evidence. |
| **Confidence Scoring** | Enabled | Evaluates consensus level and evidence density (`HIGH`, `MEDIUM`, `LOW`). |
| **Source Attribution** | Enabled | Maps doc IDs and rank scores into `sources` list. |

---

## 3. Data Completeness & Path Resolution Report

### OpenI & Index Data Audit

| Asset | Expected Canonical | Current Local State | Target HPC State | Validation Status |
| :--- | :--- | :--- | :--- | :--- |
| **OpenI Images** | 7,470 images | `repo_release/data/openi` (Canonical source) | `/data/gokulfaleja/data/openi/images/subset_100` | Partial (subset_100 present; rsync required for full 7,470 set) |
| **OpenI Reports** | 2 CSV files | Present | `/data/gokulfaleja/data/openi/reports` | Verified |
| **ColQwen2 Document Store** | 3,826 indexed docs | `document_store.json` (3.88 MB) | `/data/gokulfaleja/data/indexes/colqwen2_index/document_store.json` | Verified |
| **ColQwen2 Embeddings** | `embeddings.pt` | `embeddings.pt` (1.48 GB) | `/data/gokulfaleja/data/indexes/colqwen2_index/embeddings.pt` | Verified |
| **Doc IDs** | `doc_ids.json` | `doc_ids.json` (37 KB) | `/data/gokulfaleja/data/indexes/colqwen2_index/doc_ids.json` | Verified |
| **Text Embeddings** | `text_embeddings.pt` | N/A (Optional Phase 3+) | N/A (Runtime falls back to ColQwen2 cross-modal) | Intended Fallback Active |

### Kaggle Path Remapping Verification

The ColQwen2 document store contains legacy Kaggle dataset paths:
`/kaggle/input/datasets/raddar/chest-xrays-indiana-university/images/images_normalized/<filename>`

The path resolver in `src/domains/healthcare/retrieval/colqwen2_retriever.py` (`_resolve_image_path`) dynamically converts Kaggle paths to the runtime image base directory (`data/openi/images/`).

---

## 4. Hardware & Physical GPU 1 Enforcement

### Cluster Hardware Profile
- **GPUs:** 2 × NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (~96 GB VRAM each)
- **Slurm Gres:** `Gres=gpu:2(S:0)`, `AutoDetect=nvidia`

### Physical GPU 1 Policy Compliance
- Standard `--gres=gpu:1` allocations in Slurm allocate **physical GPU 0** first.
- The `run_mmrag_gpu1.sbatch` script includes an explicit Python/CUDA pre-flight check that inspects `SLURM_JOB_GPUS`.
- **Enforcement Rule:** If Slurm allocates physical GPU 0 (`SLURM_JOB_GPUS=0`), the script prints a clear warning and exits immediately without executing workload on GPU 0.
- **Admin Configuration Note:** Selecting a specific physical GPU index (GPU 1) under generic `--gres=gpu:1` requires Slurm administrator configuration (e.g. Slurm GRES node index assignment or dedicated reservation).

---

## 5. Automated Validation & Test Suite Results

Local automated test execution results:

```
=========================== short test summary info ===========================
PASSED:  92 tests
SKIPPED: 8 tests (HPC GPU/ColPali/ChromaDB hardware-constrained tests skipped cleanly)
FAILED:  0 tests
TOTAL:   100 tests (100% pass rate for active suite)
===============================================================================
```

### Key Verified Components
- `tests/test_smoke.py`: UnifiedResponse schema, BasePipeline abstraction, router auto-detection, domain route execution.
- `tests/api/test_app.py`: `/health`, `/ready`, `/query` endpoints, error handling, CORS.
- `tests/api/test_models.py`: Frozen API Pydantic schemas, validation, default score values.
- `tests/api/test_retrieval_metadata.py`: Score mapping (`colpali`, `scincl`, `fused`), score bounds normalization.
- `src/domains/scientific/retrieval/fusion_retriever.py`: Fixed min-max score normalization edge case when score range is zero.

---

## 6. Deliverables Matrix

| File Path | Description | Status |
| :--- | :--- | :--- |
| `docs/MMRAG_HPC_VALIDATION.md` | Audit & Validation Report | Complete |
| `docs/HPC_RUNBOOK.md` | HPC Manual Execution Runbook | Complete |
| `run_mmrag_gpu1.sbatch` | Production Slurm Job Script | Complete |
| `scripts/hpc_preflight.sh` | HPC Environment Preflight Script | Complete |
| `scripts/start_backend.sh` | Standalone Backend Launcher | Complete |
| `scripts/start_frontend.sh` | Standalone Frontend Launcher | Complete |
| `scripts/e2e_smoke_test.sh` | Standalone End-to-End Test Runner | Complete |

---

## 7. Status & Sign-off

- **Local Repository Validation:** PASSED (92 unit/integration tests passing, 0 failing).
- **HPC Execution:** REQUIRES MANUAL RUN via `run_mmrag_gpu1.sbatch` on cluster.
