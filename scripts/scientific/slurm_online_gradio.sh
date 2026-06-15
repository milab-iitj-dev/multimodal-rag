#!/bin/bash
#SBATCH --job-name=rag_online
#SBATCH --output=/scratch/data/divyasaxena_rs/Vineet_internship/logs/rag_online_%j.out
#SBATCH --error=/scratch/data/divyasaxena_rs/Vineet_internship/logs/rag_online_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=24G

# ═══════════════════════════════════════════════════════════════
#  IITJ HPC — Online RAG Gradio Pipeline
#  Runs: online_rag_pipeline_with_modern_gradio.py on GPU node
#
#  Submit   : sbatch scripts/slurm_online_gradio.sh
#  Monitor  : squeue -u divyasaxena_rs
#             tail -f logs/rag_gradio_<JOB_ID>.out
#  Port-fwd : ssh -L 7860:localhost:7860 divyasaxena_rs@172.25.0.81
#  Browser  : http://localhost:7860
# ═══════════════════════════════════════════════════════════════

set -e

WORK_DIR="/scratch/data/divyasaxena_rs/Vineet_internship"
PROJECT_DIR="${WORK_DIR}/Scientific-Multimodal-RAG"
VENV_DIR="${WORK_DIR}/.venv"
CACHE_DIR="${WORK_DIR}/.cache/huggingface"
PORT=7860

echo "════════════════════════════════════════════════════════════"
echo "  JOB  : ${SLURM_JOB_NAME} [ID: ${SLURM_JOB_ID}]"
echo "  NODE : ${SLURMD_NODENAME}"
echo "  START: $(date)"
echo "════════════════════════════════════════════════════════════"

# ── Load modules ──
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module purge 2>/dev/null || true
module load python/3.10 2>/dev/null || module load python3 2>/dev/null || true
module load cuda/12.1   2>/dev/null || module load cuda   2>/dev/null || true

# ── GPU check ──
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null || true

# ── Activate venv ──
source "${VENV_DIR}/bin/activate"

# ── Env vars ──
export RAG_BASE_DIR="${WORK_DIR}"
export HF_HOME="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${CACHE_DIR}/hub"
export HF_DATASETS_CACHE="${CACHE_DIR}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STREAMLIT_PORT=${PORT}
export RAG_CONFIG_PATH="${WORK_DIR}/configs/config.yaml"

# ── Change to project directory ──
cd "${WORK_DIR}"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Starting Scientific RAG — Gradio Q&A App on port ${PORT}"
echo ""
echo "  On YOUR LAPTOP — open a new terminal and run:"
echo "  ssh -L ${PORT}:localhost:${PORT} divyasaxena_rs@172.25.0.81"
echo ""
echo "  Then open: http://localhost:${PORT}"
echo "════════════════════════════════════════════════════════════"
echo ""

export GRADIO_PORT=${PORT}
python3 -u app/gradio_app.py

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  JOB DONE: $(date)"
echo "════════════════════════════════════════════════════════════"

