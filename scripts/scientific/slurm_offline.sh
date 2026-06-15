#!/bin/bash
#SBATCH --job-name=sci_rag_offline
#SBATCH --output=/scratch/data/divyasaxena_rs/%x_%j.out
#SBATCH --error=/scratch/data/divyasaxena_rs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL

# ═══════════════════════════════════════════════════════════════
#  IITJ HPC — Offline RAG Pipeline SLURM Job
#  Runs: PDF download → parsing → ColPali embedding → ChromaDB
#
#  Submit:  sbatch scripts/slurm_offline.sh
#  Monitor: squeue -u divyasaxena_rs
#           tail -f /scratch/data/divyasaxena_rs/sci_rag_offline_<ID>.out
# ═══════════════════════════════════════════════════════════════

set -e

INTERN_NAME="Vineet"
HPC_USER="divyasaxena_rs"
SCRATCH_ROOT="/scratch/data/${HPC_USER}"
WORK_DIR="${SCRATCH_ROOT}/${INTERN_NAME}_internship"
PROJECT_DIR="${WORK_DIR}/Scientific-Multimodal-RAG"
VENV_DIR="${WORK_DIR}/rag_venv"
CACHE_DIR="${WORK_DIR}/.cache/huggingface"

echo "JOB: ${SLURM_JOB_NAME} [ID: ${SLURM_JOB_ID}] NODE: ${SLURMD_NODENAME}"
echo "START: $(date)"

# Load modules
module purge
module load python/3.10 2>/dev/null || module load python3
module load cuda/12.1   2>/dev/null || module load cuda

# Activate env
source "${VENV_DIR}/bin/activate"
export HF_HOME="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${CACHE_DIR}/hub"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_DIR}"
mkdir -p data/raw data/parsed/pages data/parsed/markdown
mkdir -p data/indices/multivectors data/indices/chroma_index outputs

# GPU check
echo ""; echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# ──────────────────────────────────────────────────────────────
# Run Unified Offline Indexing Pipeline
# ──────────────────────────────────────────────────────────────
echo ""; echo "=== RUNNING MODULAR OFFLINE INDEXING PIPELINE ==="
python3 main.py --mode offline

echo ""; echo "=== JOB DONE: $(date) | Duration: ${SECONDS}s ==="
echo "Download: scp divyasaxena_rs@172.25.0.81:${WORK_DIR}/sci-rag-indices.zip ."
