#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  IITJ HPC — Scientific Multimodal RAG — Environment Setup
#  Run this ONCE after logging into the HPC node
#  Usage: bash scripts/hpc_setup.sh
# ═══════════════════════════════════════════════════════════════

set -e  # Exit on any error

# ── Config ────────────────────────────────────────────────────
INTERN_NAME="${INTERN_NAME:-Vineet}"   # Change or pass as env var
HPC_USER="divyasaxena_rs"
SCRATCH_ROOT="/scratch/data/${HPC_USER}"
WORK_DIR="${SCRATCH_ROOT}/${INTERN_NAME}_internship"
PROJECT_DIR="${WORK_DIR}/Scientific-Multimodal-RAG"
CACHE_DIR="${WORK_DIR}/.cache/huggingface"
VENV_DIR="${WORK_DIR}/rag_venv"
LOG_DIR="${WORK_DIR}/logs"

echo "════════════════════════════════════════════════════════════"
echo "  IITJ HPC — Scientific Multimodal RAG — Setup"
echo "  Intern    : ${INTERN_NAME}"
echo "  Work dir  : ${WORK_DIR}"
echo "════════════════════════════════════════════════════════════"

# ── Step 1: Create directory structure ──────────────────────────
echo ""
echo "[1/7] Creating directory structure..."
mkdir -p "${WORK_DIR}"
mkdir -p "${PROJECT_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${WORK_DIR}/outputs"
echo "  ✅ Directories created"

# ── Step 2: Load modules (IITJ HPC module system) ───────────────
echo ""
echo "[2/7] Loading HPC modules..."
module purge 2>/dev/null || true
module load python/3.10 2>/dev/null || module load python3 2>/dev/null || true
module load cuda/12.1   2>/dev/null || module load cuda 2>/dev/null || true
module load cudnn       2>/dev/null || true
echo "  Python: $(python3 --version)"
echo "  CUDA  : $(nvcc --version 2>/dev/null | head -1 || echo 'CUDA not in PATH')"
echo "  ✅ Modules loaded"

# ── Step 3: Create virtual environment ──────────────────────────
echo ""
echo "[3/7] Creating Python virtual environment at ${VENV_DIR}..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  ✅ Virtual environment created"
else
    echo "  ⏭  Already exists, skipping"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q

# ── Step 4: Install PyTorch with CUDA 12.1 ──────────────────────
echo ""
echo "[4/7] Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 -q
python3 -c "import torch; print(f'  torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')"
echo "  ✅ PyTorch installed"

# ── Step 5: Install project dependencies ────────────────────────
echo ""
echo "[5/7] Installing project dependencies..."
cd "${PROJECT_DIR}"
pip install colpali-engine==0.3.0 -q
pip install sentence-transformers==2.7.0 -q
pip install chromadb==1.5.9 -q
pip install transformers==4.41.2 accelerate==0.29.3 -q
pip install bitsandbytes -q
pip install pymupdf pdf2image arxiv -q
pip install numpy==1.26.4 pandas tqdm colorlog pyyaml python-dotenv Pillow -q
pip install nltk rouge-score sacrebleu -q
pip install gradio streamlit -q
# Install poppler for pdf2image
conda install -y poppler 2>/dev/null || \
    apt-get install -y poppler-utils 2>/dev/null || \
    echo "  ⚠️  Install poppler manually if pdf2image fails: conda install poppler"
echo "  ✅ All dependencies installed"

# ── Step 6: Install project as editable package ─────────────────
echo ""
echo "[6/7] Installing project as editable package..."
if [ -f "${PROJECT_DIR}/setup.py" ]; then
    pip install -e "${PROJECT_DIR}" -q
    echo "  ✅ Project installed (editable)"
else
    echo "  ⚠️  No setup.py found at ${PROJECT_DIR} — sync your repo first"
fi

# ── Step 7: Set environment variables ───────────────────────────
echo ""
echo "[7/7] Writing HPC .env file..."
cat > "${PROJECT_DIR}/.env.hpc" << EOF
# Auto-generated HPC environment config
# Source this before running: source .env.hpc

export HF_HOME=${CACHE_DIR}
export TRANSFORMERS_CACHE=${CACHE_DIR}/hub
export HF_DATASETS_CACHE=${CACHE_DIR}/datasets
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Project paths
export WORK_DIR=${WORK_DIR}
export PROJECT_DIR=${PROJECT_DIR}
export LOG_DIR=${LOG_DIR}

# Activate venv (add to your ~/.bashrc for convenience)
# source ${VENV_DIR}/bin/activate
EOF
echo "  ✅ .env.hpc written → ${PROJECT_DIR}/.env.hpc"

# ── Final summary ───────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  SETUP COMPLETE!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "  1.  cd ${WORK_DIR}"
echo "  2.  git clone <your-repo-url> Scientific-Multimodal-RAG"
echo "        OR scp/rsync from your laptop"
echo "  3.  source rag_venv/bin/activate"
echo "  4.  source Scientific-Multimodal-RAG/.env.hpc"
echo "  5.  sbatch Scientific-Multimodal-RAG/scripts/slurm_offline.sh"
echo ""
