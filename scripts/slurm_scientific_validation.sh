#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  MMRAG Unified — Scientific Domain Validation (SLURM)
# ════════════════════════════════════════════════════════════════════════
#
#  Validates that the Scientific pipeline is LIVE and returns real answers.
#
#  Phases:
#    1. Environment setup (GPU, venv, env vars, RAG_BASE_DIR)
#    2. Scientific asset verification (metadata, indices, models)
#    3. Start FastAPI server
#    4. Wait for /ready — verify BOTH domains are LIVE
#    5. Run scientific queries via generate_api_examples.py
#    6. Generate scientific validation report
#    7. Summary and shutdown
#
#  Submit:   sbatch scripts/slurm_scientific_validation.sh
#  Monitor:  tail -f outputs/logs/sci_validation_<JOBID>.log
#
# ════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=mmrag-sci-val
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/sci_validation_%j.log
#SBATCH --error=outputs/logs/sci_validation_%j.err

# ════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

PROJECT_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified"
VENV_DIR="${PROJECT_DIR}/.venv"
HC_DATA_ROOT="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag-healthcare"
SCI_DATA_ROOT="/scratch/data/divyasaxena_rs/Vineet_internship"
PORT=8847
HF_CACHE="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.cache/huggingface"
STARTUP_TIMEOUT=900

# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════

ts() { date '+%H:%M:%S'; }
ok()   { echo "  ✓ $1"; }
fail() { echo "  ✗ $1"; }
warn() { echo "  ⚠ $1"; }
SERVER_PID=""
UVICORN_LOG="${PROJECT_DIR}/outputs/logs/uvicorn_sci_val_${SLURM_JOB_ID:-$$}.log"

mkdir -p "${PROJECT_DIR}/outputs/logs"
mkdir -p "${PROJECT_DIR}/outputs/api_examples"
mkdir -p "${PROJECT_DIR}/outputs/reports"

cleanup() {
    echo ""
    echo "[$(ts)] Shutting down server..."
    [ -n "${SERVER_PID}" ] && kill ${SERVER_PID} 2>/dev/null || true
    [ -n "${SERVER_PID}" ] && wait ${SERVER_PID} 2>/dev/null || true
    echo "Done. $(date)"
}
trap cleanup EXIT INT TERM

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  MMRAG — Scientific Domain Validation                          ║"
echo "║  Job:  ${SLURM_JOB_ID:-interactive}                                              ║"
echo "║  Node: $(hostname)                                                    ║"
echo "║  Date: $(date '+%Y-%m-%d %H:%M:%S')                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 1 — ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

echo "[$(ts)] Phase 1: Environment setup"

# GPU check
if ! command -v nvidia-smi &>/dev/null; then
    fail "No GPU — must run via sbatch"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
ok "GPU: ${GPU_NAME}"

# Project directory
if [ ! -d "${PROJECT_DIR}" ]; then fail "Project dir not found"; exit 1; fi

# Venv activation (Anaconda-proof)
if [ ! -f "${VENV_DIR}/bin/activate" ]; then fail "No venv"; exit 1; fi
if command -v conda &>/dev/null; then conda deactivate 2>/dev/null || true; fi
CLEAN_PATH=""
IFS=':' read -ra PATH_PARTS <<< "$PATH"
for p in "${PATH_PARTS[@]}"; do
    case "$p" in *conda*|*anaconda*|*Anaconda*) ;; *) CLEAN_PATH="${CLEAN_PATH:+${CLEAN_PATH}:}${p}" ;; esac
done
export PATH="${CLEAN_PATH}"
source "${VENV_DIR}/bin/activate"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
hash -r
ok "Python: $(which python) ($(python --version 2>&1))"

# Environment variables
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAG_BASE_DIR="${SCI_DATA_ROOT}"
ok "HF_HOME=${HF_HOME}"
ok "RAG_BASE_DIR=${RAG_BASE_DIR}"

cd "${PROJECT_DIR}"
ok "CWD: $(pwd)"

# Healthcare symlinks (idempotent)
mkdir -p "${PROJECT_DIR}/data/indexes"
LINK_INDEX="${PROJECT_DIR}/data/indexes/colqwen2_index"
TARGET_INDEX="${HC_DATA_ROOT}/data/indexes/colqwen2_index"
if [ ! -L "${LINK_INDEX}" ] && [ ! -d "${LINK_INDEX}" ] && [ -d "${TARGET_INDEX}" ]; then
    ln -s "${TARGET_INDEX}" "${LINK_INDEX}"
fi

LINK_OPENI="${PROJECT_DIR}/data/openi"
TARGET_OPENI="${HC_DATA_ROOT}/data/openi"
if [ ! -L "${LINK_OPENI}" ] && [ ! -d "${LINK_OPENI}" ] && [ -d "${TARGET_OPENI}" ]; then
    ln -s "${TARGET_OPENI}" "${LINK_OPENI}"
fi

ok "Healthcare symlinks ready"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 2 — SCIENTIFIC ASSET VERIFICATION
# ════════════════════════════════════════════════════════════════════════

echo "[$(ts)] Phase 2: Scientific asset verification"
SCI_ERRORS=0

# Metadata files
SCI_METADATA="${SCI_DATA_ROOT}/data/indices/page_metadata.json"
SCI_DOCMAP="${SCI_DATA_ROOT}/data/indices/doc_mapping.json"
SCI_CHROMA="${SCI_DATA_ROOT}/data/indices/chroma_index"
SCI_MULTIVEC="${SCI_DATA_ROOT}/data/indices/multivectors"

if [ -f "${SCI_METADATA}" ]; then
    SCI_META_SIZE=$(du -h "${SCI_METADATA}" 2>/dev/null | cut -f1)
    ok "page_metadata.json (${SCI_META_SIZE})"
else
    fail "page_metadata.json NOT FOUND — scientific pipeline WILL fail"
    SCI_ERRORS=$((SCI_ERRORS + 1))
fi

if [ -f "${SCI_DOCMAP}" ]; then
    ok "doc_mapping.json"
else
    fail "doc_mapping.json NOT FOUND"
    SCI_ERRORS=$((SCI_ERRORS + 1))
fi

if [ -d "${SCI_CHROMA}" ]; then
    CHROMA_FILES=$(find "${SCI_CHROMA}" -type f 2>/dev/null | wc -l)
    ok "chroma_index (${CHROMA_FILES} files)"
else
    fail "chroma_index NOT FOUND"
    SCI_ERRORS=$((SCI_ERRORS + 1))
fi

if [ -d "${SCI_MULTIVEC}" ]; then
    NPY_COUNT=$(find "${SCI_MULTIVEC}" -name '*.npy' ! -name '*.meta.npy' 2>/dev/null | wc -l)
    ok "multivectors (${NPY_COUNT} .npy files)"
else
    fail "multivectors NOT FOUND"
    SCI_ERRORS=$((SCI_ERRORS + 1))
fi

# Page images
SCI_PAGES="${SCI_DATA_ROOT}/data/parsed/pages"
if [ -d "${SCI_PAGES}" ]; then
    PNG_COUNT=$(find "${SCI_PAGES}" -name "*.png" 2>/dev/null | wc -l)
    ok "Page images (${PNG_COUNT} PNGs)"
else
    warn "Page images directory not found (VLM context images may be missing)"
fi

# Scientific config
if [ -f "${PROJECT_DIR}/configs/scientific/config.yaml" ]; then
    ok "Scientific config.yaml"
else
    fail "configs/scientific/config.yaml NOT FOUND"
    SCI_ERRORS=$((SCI_ERRORS + 1))
fi

# Model cache check
echo ""
echo "  Model cache check:"
HF_HUB="${HF_CACHE}/hub"
for model_dir in "models--vidore--colpali-v1.2" "models--malteos--scincl" "models--Qwen--Qwen2-VL-2B-Instruct"; do
    if [ -d "${HF_HUB}/${model_dir}" ] || [ -L "${HF_HUB}/${model_dir}" ]; then
        ok "  ${model_dir}"
    else
        warn "  ${model_dir} NOT cached (will attempt download)"
    fi
done

if [ ${SCI_ERRORS} -gt 0 ]; then
    fail "Scientific asset verification failed (${SCI_ERRORS} errors)"
    echo "  Scientific pipeline will likely run in placeholder mode."
    echo "  Continuing anyway to verify..."
fi

echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 3 — START SERVER
# ════════════════════════════════════════════════════════════════════════

echo "[$(ts)] Phase 3: Starting FastAPI server"

fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 1

python -u -m uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --log-level info \
    2>&1 | tee "${UVICORN_LOG}" &

SERVER_PID=$!
ok "Server PID: ${SERVER_PID}"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 4 — WAIT FOR READY + VERIFY BOTH DOMAINS
# ════════════════════════════════════════════════════════════════════════

echo "[$(ts)] Phase 4: Waiting for server startup (up to ${STARTUP_TIMEOUT}s)"
echo "  Model loading takes 1–10 minutes..."

WAIT=0
while [ $WAIT -lt $STARTUP_TIMEOUT ]; do
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        fail "Server died during startup"
        tail -20 "${UVICORN_LOG}" 2>/dev/null
        exit 1
    fi
    if grep -q "Application startup complete" "${UVICORN_LOG}" 2>/dev/null; then
        break
    fi
    if grep -q "Uvicorn running on" "${UVICORN_LOG}" 2>/dev/null; then
        break
    fi
    sleep 5
    WAIT=$((WAIT + 5))
    if [ $((WAIT % 30)) -eq 0 ]; then
        echo "  [$(ts)] ... ${WAIT}s elapsed"
    fi
done

if [ $WAIT -ge $STARTUP_TIMEOUT ]; then
    fail "Startup timeout (${STARTUP_TIMEOUT}s)"
    exit 1
fi

ok "Application startup complete"

# Wait for /ready
R_WAIT=0
while [ $R_WAIT -lt 120 ]; do
    READY=$(curl -s --max-time 5 "http://localhost:${PORT}/ready" 2>/dev/null | \
        python -c "import sys,json; d=json.load(sys.stdin); print('YES' if d.get('ready') else 'NO')" 2>/dev/null || echo "NO")
    if [ "${READY}" = "YES" ]; then break; fi
    sleep 3
    R_WAIT=$((R_WAIT + 3))
done

if [ "${READY}" != "YES" ]; then
    fail "Server not ready after additional 120s"
    exit 1
fi

ok "Server ready"

# Check BOTH domains
echo ""
echo "  Domain registration check:"
READY_JSON=$(curl -s --max-time 10 "http://localhost:${PORT}/ready" 2>/dev/null)
echo "  /ready response: ${READY_JSON}"

HC_LIVE=$(echo "${READY_JSON}" | python -c "import sys,json; d=json.load(sys.stdin); print('YES' if 'healthcare' in d.get('domains',[]) else 'NO')" 2>/dev/null || echo "NO")
SCI_LIVE=$(echo "${READY_JSON}" | python -c "import sys,json; d=json.load(sys.stdin); print('YES' if 'scientific' in d.get('domains',[]) else 'NO')" 2>/dev/null || echo "NO")

if [ "${HC_LIVE}" = "YES" ]; then
    ok "Healthcare domain: LIVE"
else
    warn "Healthcare domain: NOT LIVE"
fi

if [ "${SCI_LIVE}" = "YES" ]; then
    ok "Scientific domain: LIVE"
else
    fail "Scientific domain: NOT LIVE — check RAG_BASE_DIR and data paths"
    echo ""
    echo "  Diagnostic info:"
    echo "    RAG_BASE_DIR=${RAG_BASE_DIR}"
    echo "    page_metadata.json exists: $([ -f '${SCI_METADATA}' ] && echo 'yes' || echo 'no')"
    grep -i "scientific" "${UVICORN_LOG}" 2>/dev/null | tail -5
    echo ""
    echo "  Continuing to run queries anyway (they will use placeholder responses)..."
fi

echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 5 — RUN SCIENTIFIC QUERIES
# ════════════════════════════════════════════════════════════════════════

echo "[$(ts)] Phase 5: Running API example generator (scientific + healthcare)"
echo ""

python tools/generate_api_examples.py \
    --config configs/api_examples.yaml \
    --server "http://localhost:${PORT}" \
    --no-wait

GEN_RC=$?

echo ""
echo "[$(ts)] Generator exit code: ${GEN_RC}"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 6 — GENERATE VALIDATION REPORT
# ════════════════════════════════════════════════════════════════════════

echo ""
echo "[$(ts)] Phase 6: Generating validation report"

REPORT_FILE="${PROJECT_DIR}/outputs/reports/scientific_validation.md"

cat > "${REPORT_FILE}" <<REPORT_EOF
# Scientific Domain — Production Validation Report

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Job ID:** ${SLURM_JOB_ID:-interactive}
**Node:** $(hostname)
**GPU:** ${GPU_NAME}

---

## Domain Status

| Domain | Status |
|--------|--------|
| Healthcare | ${HC_LIVE} |
| Scientific | ${SCI_LIVE} |

## Environment

| Variable | Value |
|----------|-------|
| RAG_BASE_DIR | \`${RAG_BASE_DIR}\` |
| HF_HOME | \`${HF_HOME}\` |
| CWD | \`$(pwd)\` |

## Scientific Assets

| Asset | Path | Status |
|-------|------|--------|
| page_metadata.json | \`${SCI_METADATA}\` | $([ -f "${SCI_METADATA}" ] && echo "✅" || echo "❌") |
| doc_mapping.json | \`${SCI_DOCMAP}\` | $([ -f "${SCI_DOCMAP}" ] && echo "✅" || echo "❌") |
| chroma_index/ | \`${SCI_CHROMA}\` | $([ -d "${SCI_CHROMA}" ] && echo "✅ (${CHROMA_FILES:-?} files)" || echo "❌") |
| multivectors/ | \`${SCI_MULTIVEC}\` | $([ -d "${SCI_MULTIVEC}" ] && echo "✅ (${NPY_COUNT:-?} .npy)" || echo "❌") |
| Page images | \`${SCI_PAGES}\` | $([ -d "${SCI_PAGES}" ] && echo "✅ (${PNG_COUNT:-?} PNGs)" || echo "⚠ not found") |

## Query Results

Exit code: \`${GEN_RC}\`

Full results saved to: \`outputs/api_examples/\`

## /ready Response

\`\`\`json
${READY_JSON}
\`\`\`

---

*Report generated by \`scripts/slurm_scientific_validation.sh\`*
REPORT_EOF

ok "Report saved: ${REPORT_FILE}"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 7 — FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
if [ ${GEN_RC} -eq 0 ] && [ "${SCI_LIVE}" = "YES" ]; then
    echo "║  ✅ SCIENTIFIC VALIDATION PASSED                               ║"
elif [ ${GEN_RC} -eq 0 ]; then
    echo "║  ⚠  QUERIES PASSED but Scientific domain NOT LIVE             ║"
else
    echo "║  ✗  VALIDATION FAILED                                         ║"
fi
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Healthcare: $(printf '%-52s' "${HC_LIVE}")║"
echo "║  Scientific: $(printf '%-52s' "${SCI_LIVE}")║"
echo "║  Queries:    $(printf '%-52s' "exit code ${GEN_RC}")║"
echo "║  Report:     $(printf '%-52s' "outputs/reports/scientific_validation.md")║"
echo "║  Examples:   $(printf '%-52s' "outputs/api_examples/")║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

exit ${GEN_RC}
