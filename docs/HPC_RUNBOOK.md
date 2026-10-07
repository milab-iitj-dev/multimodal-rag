# MMRAG Unified — HPC Runbook & Manual Execution Guide

This document contains step-by-step copy-paste commands for deploying and executing the **MMRAG Unified System** on the HPC cluster.

---

## 1. SSH Connection to HPC (From Local Machine)

Execute from your Windows terminal:

```bash
ssh gokulfaleja@10.6.7.112
```

---

## 2. Environment & Directory Setup on HPC

Once logged into the HPC node:

```bash
# 1. Navigate to your workspace directory
cd /data/gokulfaleja

# 2. Ensure log and cache directories exist
mkdir -p /data/gokulfaleja/cache/huggingface
mkdir -p /data/gokulfaleja/cache/torch
mkdir -p /data/gokulfaleja/logs
mkdir -p /data/gokulfaleja/data/openi
mkdir -p /data/gokulfaleja/data/indexes/colqwen2_index

# 3. Activate conda environment
source activate mmrag
```

---

## 3. Canonical Data Sync (Rsync Commands)

If full OpenI image dataset (7,470 images) or ColQwen2 index need to be updated from canonical local storage:

### Transfer Canonical OpenI Data (7,470 images & report CSVs)
Run from **Windows PowerShell**:

```powershell
rsync -avzP "D:\Summer Internship\Project\repo_release\data\openi\" gokulfaleja@10.6.7.112:/data/gokulfaleja/data/openi/
```

### Transfer Canonical ColQwen2 Index
Run from **Windows PowerShell**:

```powershell
rsync -avzP "D:\Summer Internship\Project\healthcare_mrag\phase2_outputs\colqwen2_index\" gokulfaleja@10.6.7.112:/data/gokulfaleja/data/indexes/colqwen2_index/
```

---

## 4. Run Preflight Verification Script

Before submitting jobs, run the preflight script on the HPC:

```bash
cd /data/gokulfaleja/mmrag_unified  # Or your cloned repo directory
chmod +x scripts/hpc_preflight.sh
./scripts/hpc_preflight.sh
```

---

## 5. Submit Slurm Production Job (GPU 1 Enforced)

Submit the production Slurm job script:

```bash
sbatch run_mmrag_gpu1.sbatch
```

### Monitor Job Output

```bash
# View active job status
squeue -u gokulfaleja

# Tail job log output (replace <JOBID> with your actual Slurm job ID)
tail -f /data/gokulfaleja/logs/mmrag_gpu1_<JOBID>.log
```

---

## 6. Local SSH Port Forwarding (From Windows Terminal)

To access the interactive frontend UI and FastAPI Swagger docs on your local browser:

Run from **Windows Terminal**:

```cmd
ssh -N -L 8000:localhost:8000 -L 5173:localhost:5173 gokulfaleja@10.6.7.112
```

### Open Local Browser URLs:
- **React Frontend UI:** [http://localhost:5173](http://localhost:5173)
- **FastAPI OpenAPI Specs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Probe:** [http://localhost:8000/health](http://localhost:8000/health)
- **API Readiness Probe:** [http://localhost:8000/ready](http://localhost:8000/ready)

---

## 7. HPC Admin Configuration Request for GPU 1

If Slurm rejects the job because generic `--gres=gpu:1` allocated GPU 0:

**Request to HPC Administrator:**
> "Please configure Slurm node GPU allocation or provide a dedicated GRES flag to allow requesting physical GPU 1 explicitly (e.g. `--gres=gpu:1` mapped to PCI device index 1, or dedicated reservation on GPU 1) for user `gokulfaleja`."
