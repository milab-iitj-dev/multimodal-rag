#!/usr/bin/env bash
# ==============================================================================
# MMRAG Unified — HPC Preflight Verification Script
# ==============================================================================
# Verifies environment, Python packages, GPU allocation, indices, and dataset
# paths BEFORE launching the FastAPI backend and React frontend.
#
# Usage:
#   chmod +x scripts/hpc_preflight.sh
#   ./scripts/hpc_preflight.sh
# ==============================================================================

set -euo pipefail

echo "=================================================================="
echo "               MMRAG UNIFIED — HPC PRE-FLIGHT CHECK              "
echo "=================================================================="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host:      $(hostname)"
echo "User:      $(whoami)"
echo "CWD:       $(pwd)"
echo "=================================================================="
echo ""

ERRORS=0
WARNINGS=0

# ------------------------------------------------------------------------------
# 1. Git Repository Check
# ------------------------------------------------------------------------------
echo "[1/10] Checking Git Repository..."
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    COMMIT_HASH=$(git rev-parse HEAD)
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    echo "  ✓ Git Branch: ${BRANCH}"
    echo "  ✓ Git Commit: ${COMMIT_HASH}"
else
    echo "  ⚠ WARNING: Not a git work tree or git binary missing"
    WARNINGS=$((WARNINGS + 1))
fi

# ------------------------------------------------------------------------------
# 2. Conda Environment & Python Check
# ------------------------------------------------------------------------------
echo ""
echo "[2/10] Checking Python & Environment..."
PY_PATH=$(which python 2>/dev/null || echo "NOT_FOUND")
echo "  Executable: ${PY_PATH}"

if [ "${PY_PATH}" = "NOT_FOUND" ]; then
    echo "  ✗ ERROR: Python binary not found in PATH"
    ERRORS=$((ERRORS + 1))
else
    PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    CONDA_ENV="${CONDA_DEFAULT_ENV:-none}"
    echo "  ✓ Python Version: ${PY_VER}"
    echo "  ✓ Conda Active Env: ${CONDA_ENV}"

    if [ "${CONDA_ENV}" != "mmrag" ]; then
        echo "  ⚠ WARNING: Expected conda environment 'mmrag', but found '${CONDA_ENV}'"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

# ------------------------------------------------------------------------------
# 3. PyTorch & CUDA Check
# ------------------------------------------------------------------------------
echo ""
echo "[3/10] Checking PyTorch & CUDA..."
python -c "
import sys
try:
    import torch
    print(f'  ✓ PyTorch Version: {torch.__version__}')
    print(f'  ✓ CUDA Available:  {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  ✓ CUDA Version:    {torch.version.cuda}')
        print(f'  ✓ Device Count:    {torch.cuda.device_count()}')
        print(f'  ✓ Current Device:  {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})')
    else:
        print('  ✗ ERROR: CUDA is NOT available to PyTorch')
        sys.exit(1)
except Exception as e:
    print(f'  ✗ ERROR checking PyTorch/CUDA: {e}')
    sys.exit(1)
" || ERRORS=$((ERRORS + 1))

# ------------------------------------------------------------------------------
# 4. Physical GPU Allocation Verification (Target: GPU 1)
# ------------------------------------------------------------------------------
echo ""
echo "[4/10] Checking Slurm & Physical GPU Allocation..."
SLURM_GPUS="${SLURM_JOB_GPUS:-NOT_SET}"
CUDA_VISIBLE="${CUDA_VISIBLE_DEVICES:-NOT_SET}"
echo "  SLURM_JOB_GPUS:       ${SLURM_GPUS}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  NVIDIA-SMI Output:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | sed 's/^/    /'
else
    echo "  ⚠ WARNING: nvidia-smi command not found"
    WARNINGS=$((WARNINGS + 1))
fi

if [ "${SLURM_GPUS}" != "NOT_SET" ]; then
    if [ "${SLURM_GPUS}" = "1" ]; then
        echo "  ✓ Slurm allocated physical GPU 1"
    else
        echo "  ⚠ WARNING: Slurm allocated physical GPU '${SLURM_GPUS}', but physical GPU 1 was required."
        echo "             Note: Under standard Slurm gres:gpu:1, Slurm assigns GPU 0 first."
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ℹ Note: Running outside Slurm environment (SLURM_JOB_GPUS not set)"
fi

# ------------------------------------------------------------------------------
# 5. System RAM Check
# ------------------------------------------------------------------------------
echo ""
echo "[5/10] Checking Memory & Disk..."
if command -v free >/dev/null 2>&1; then
    RAM_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')
    RAM_AVAIL=$(free -h | awk '/^Mem:/ {print $7}')
    echo "  ✓ Total RAM: ${RAM_TOTAL}"
    echo "  ✓ Available RAM: ${RAM_AVAIL}"
fi
DISK_AVAIL=$(df -h . | awk 'NR==2 {print $4}')
echo "  ✓ Working Dir Available Disk: ${DISK_AVAIL}"

# ------------------------------------------------------------------------------
# 6. Data & Index Path Validation
# ------------------------------------------------------------------------------
echo ""
echo "[6/10] Validating OpenI & ColQwen2 Data Assets..."

OPENI_DIR="data/openi"
OPENI_IMAGES="data/openi/images"
OPENI_REPORTS="data/openi/reports"
COLQWEN2_DIR="data/indexes/colqwen2_index"

# Check OpenI path
if [ -d "${OPENI_DIR}" ]; then
    echo "  ✓ OpenI base path exists: ${OPENI_DIR}"
    if [ -L "${OPENI_DIR}" ]; then
        TARGET=$(readlink -f "${OPENI_DIR}")
        echo "    (Symlink -> ${TARGET})"
    fi
else
    echo "  ✗ ERROR: OpenI data directory missing at ${OPENI_DIR}"
    ERRORS=$((ERRORS + 1))
fi

# Check OpenI images count
if [ -d "${OPENI_IMAGES}" ]; then
    IMG_COUNT=$(find "${OPENI_IMAGES}" -maxdepth 2 -type f \( -name "*.png" -o -name "*.dcm.png" -o -name "*.jpg" -o -name "*.png.png" \) 2>/dev/null | wc -l)
    echo "  ✓ Found ${IMG_COUNT} images in ${OPENI_IMAGES}"
    if [ "${IMG_COUNT}" -eq 100 ]; then
        echo "    ℹ Data State: SUBSET-ONLY (100 images present)"
    elif [ "${IMG_COUNT}" -ge 7000 ]; then
        echo "    ✓ Data State: FULL CANONICAL OPENI DATASET (${IMG_COUNT} images)"
    else
        echo "    ℹ Data State: PARTIAL DATASET (${IMG_COUNT} images)"
    fi
else
    echo "  ✗ ERROR: Image directory missing at ${OPENI_IMAGES}"
    ERRORS=$((ERRORS + 1))
fi

# Check ColQwen2 Index
if [ -d "${COLQWEN2_DIR}" ]; then
    echo "  ✓ ColQwen2 index directory exists: ${COLQWEN2_DIR}"
    if [ -L "${COLQWEN2_DIR}" ]; then
        TARGET=$(readlink -f "${COLQWEN2_DIR}")
        echo "    (Symlink -> ${TARGET})"
    fi

    # Check required index components
    for f in "document_store.json" "doc_ids.json"; do
        if [ -f "${COLQWEN2_DIR}/${f}" ]; then
            echo "    ✓ ${f} present"
        else
            echo "    ✗ ERROR: Index component missing: ${COLQWEN2_DIR}/${f}"
            ERRORS=$((ERRORS + 1))
        fi
    done

    # Embeddings file check (embeddings.pt or image_embeddings.pt)
    if [ -f "${COLQWEN2_DIR}/embeddings.pt" ] || [ -f "${COLQWEN2_DIR}/image_embeddings.pt" ]; then
        EMB_FILE=$(ls "${COLQWEN2_DIR}"/*embeddings.pt | head -n 1)
        SIZE=$(du -h "${EMB_FILE}" | cut -f1)
        echo "    ✓ Embeddings file present (${EMB_FILE}, ${SIZE})"
    else
        echo "    ✗ ERROR: No embeddings.pt found in ${COLQWEN2_DIR}"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  ✗ ERROR: ColQwen2 index directory missing at ${COLQWEN2_DIR}"
    ERRORS=$((ERRORS + 1))
fi

# ------------------------------------------------------------------------------
# 7. Model Config Check
# ------------------------------------------------------------------------------
echo ""
echo "[7/10] Checking Model Configurations..."
MODEL_CFG="configs/healthcare/model_config.yaml"
RETRIEVAL_CFG="configs/healthcare/retrieval_config.yaml"

if [ -f "${MODEL_CFG}" ]; then
    echo "  ✓ Healthcare model config present: ${MODEL_CFG}"
else
    echo "  ✗ ERROR: Missing ${MODEL_CFG}"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "${RETRIEVAL_CFG}" ]; then
    echo "  ✓ Healthcare retrieval config present: ${RETRIEVAL_CFG}"
else
    echo "  ✗ ERROR: Missing ${RETRIEVAL_CFG}"
    ERRORS=$((ERRORS + 1))
fi

# ------------------------------------------------------------------------------
# 8. Python Dependencies Import Test
# ------------------------------------------------------------------------------
echo ""
echo "[8/10] Testing Critical Python Imports..."
python -c "
import sys

modules = [
    ('fastapi', 'FastAPI'),
    ('pydantic', 'Pydantic'),
    ('PIL', 'Pillow'),
    ('torch', 'PyTorch'),
    ('transformers', 'HuggingFace Transformers'),
    ('accelerate', 'Accelerate'),
    ('yaml', 'PyYAML'),
]

failed = False
for mod, name in modules:
    try:
        __import__(mod)
        print(f'  ✓ {name} ({mod})')
    except ImportError as e:
        print(f'  ✗ ERROR importing {name} ({mod}): {e}')
        failed = True

# Domain specific optional checks
try:
    import colpali_engine
    print('  ✓ colpali_engine')
except ImportError:
    print('  ℹ colpali_engine: Not installed in current env')

try:
    import chromadb
    print('  ✓ chromadb')
except ImportError:
    print('  ℹ chromadb: Not installed in current env')

if failed:
    sys.exit(1)
" || ERRORS=$((ERRORS + 1))

# ------------------------------------------------------------------------------
# 9. Frontend & Node Check
# ------------------------------------------------------------------------------
echo ""
echo "[9/10] Checking Frontend Build Assets..."
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    echo "  ✓ Frontend directory & package.json present"
    if command -v node >/dev/null 2>&1; then
        NODE_VER=$(node -v)
        NPM_VER=$(npm -v)
        echo "  ✓ Node.js: ${NODE_VER}, npm: ${NPM_VER}"
    else
        echo "  ⚠ WARNING: Node.js/npm not found in PATH"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ✗ ERROR: frontend/package.json missing"
    ERRORS=$((ERRORS + 1))
fi

# ------------------------------------------------------------------------------
# 10. Port Availability Check
# ------------------------------------------------------------------------------
echo ""
echo "[10/10] Checking Port Availability (8000, 5173)..."
for port in 8000 5173; do
    if command -v netstat >/dev/null 2>&1; then
        if netstat -tuln | grep -q ":${port} "; then
            echo "  ⚠ WARNING: Port ${port} appears to be in use"
            WARNINGS=$((WARNINGS + 1))
        else
            echo "  ✓ Port ${port} is available"
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ss -tuln | grep -q ":${port} "; then
            echo "  ⚠ WARNING: Port ${port} appears to be in use"
            WARNINGS=$((WARNINGS + 1))
        else
            echo "  ✓ Port ${port} is available"
        fi
    else
        echo "  ℹ Port check tool (netstat/ss) not installed"
    fi
done

# ------------------------------------------------------------------------------
# Summary Report
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "                  PRE-FLIGHT SUMMARY REPORT                       "
echo "=================================================================="
echo "  Errors:   ${ERRORS}"
echo "  Warnings: ${WARNINGS}"

if [ "${ERRORS}" -eq 0 ]; then
    echo "  STATUS: PASSED (System ready for deployment/test)"
    echo "=================================================================="
    exit 0
else
    echo "  STATUS: FAILED (${ERRORS} critical errors detected)"
    echo "=================================================================="
    exit 1
fi
