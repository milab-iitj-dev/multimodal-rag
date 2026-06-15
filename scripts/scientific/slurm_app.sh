#!/bin/bash
#SBATCH --job-name=sci_rag_app
#SBATCH --output=/scratch/data/divyasaxena_rs/%x_%j.out
#SBATCH --error=/scratch/data/divyasaxena_rs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G

# ═══════════════════════════════════════════════════════════════
#  IITJ HPC — Run Streamlit App (port-forwarded to your laptop)
#
#  Submit  : sbatch scripts/slurm_app.sh
#  Then on YOUR LAPTOP open a NEW terminal and run:
#    ssh -L 8501:localhost:8501 divyasaxena_rs@172.25.0.81
#  Then open: http://localhost:8501
# ═══════════════════════════════════════════════════════════════

INTERN_NAME="Vineet"
HPC_USER="divyasaxena_rs"
WORK_DIR="/scratch/data/${HPC_USER}/${INTERN_NAME}_internship"
PROJECT_DIR="${WORK_DIR}"
VENV_DIR="${WORK_DIR}/.venv"
CACHE_DIR="${WORK_DIR}/.cache/huggingface"
PORT=8501

echo "════════════════════════════════════════════════════════════"
echo "  JOB  : ${SLURM_JOB_NAME} [ID: ${SLURM_JOB_ID}]"
echo "  NODE : ${SLURMD_NODENAME}"
echo "  START: $(date)"
echo "════════════════════════════════════════════════════════════"

# Load modules
module purge
module load python/3.10 2>/dev/null || module load python3
module load cuda/12.1   2>/dev/null || module load cuda

# Activate venv
source "${VENV_DIR}/bin/activate"

# Set env vars
export HF_HOME="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${CACHE_DIR}/hub"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU check
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

cd "${PROJECT_DIR}"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Starting Streamlit on port ${PORT}"
echo ""
echo "  On YOUR LAPTOP — open a new terminal and run:"
echo "  ssh -L ${PORT}:localhost:${PORT} divyasaxena_rs@172.25.0.81"
echo ""
echo "  Then open browser: http://localhost:${PORT}"
echo "════════════════════════════════════════════════════════════"
echo ""

streamlit run app/streamlit_app.py \
    --server.port ${PORT} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
