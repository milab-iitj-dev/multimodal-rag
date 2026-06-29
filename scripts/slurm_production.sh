#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  MMRAG Unified — Production Deployment with Public HTTPS Endpoint
# ════════════════════════════════════════════════════════════════════════
#
#  This is the FINAL production SLURM script. It performs:
#
#    Phase 1 — Environment Verification (Steps 1-11)
#      1.  Allocate 1 GPU
#      2.  Verify project directory
#      3.  Verify virtual environment
#      4.  Verify source files + configs
#      5.  Verify Healthcare indices exist
#      6.  Create symbolic links (indexes + OpenI data)
#      7.  Verify images and reports
#      8.  Load HPC modules + activate venv
#      9.  Verify Python + CUDA
#     10.  Verify import chain
#     11.  Log Scientific pipeline disabled
#
#    Phase 2 — Server Launch & Local Verification (Steps 12-16)
#     12.  Launch FastAPI (uvicorn)
#     13.  Wait until server is live
#     14.  Verify GET /health
#     15.  Verify GET /ready (Healthcare LIVE)
#     16.  Verify POST /query (real answer, not placeholder)
#
#    Phase 3 — Public HTTPS Tunnel (Steps 17-19)
#     17.  Download cloudflared (if not cached)
#     18.  Start Cloudflare Quick Tunnel
#     19.  Print public HTTPS URL
#
#    Phase 4 — Public Verification & Stability (Steps 20-22)
#     20.  Verify public /health, /ready, /query
#     21.  Repeat queries for stability (3 rounds)
#     22.  Keep server + tunnel alive for professor access
#
#  Submit:
#    sbatch scripts/slurm_production.sh
#
#  Check logs:
#    cat outputs/logs/production_<JOBID>.log
#    cat outputs/logs/production_<JOBID>.err
#
#  Find the public URL:
#    grep "PUBLIC_URL" outputs/logs/production_<JOBID>.log
#
#  Cancel when done:
#    scancel <JOBID>
#
#  Expected runtime: ~10 min setup + model load, then stays alive
# ════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=mmrag-production
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/production_%j.log
#SBATCH --error=outputs/logs/production_%j.err

set -euo pipefail

# ════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

PROJECT_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified"
VENV_DIR="${PROJECT_DIR}/.venv"
HC_DATA_ROOT="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag-healthcare"
PORT=8847
HF_CACHE="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.cache/huggingface"

# Cloudflared binary location (cached in project to avoid re-download)
CLOUDFLARED_BIN="${PROJECT_DIR}/.local/bin/cloudflared"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 1 — ENVIRONMENT VERIFICATION
# ════════════════════════════════════════════════════════════════════════

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  MMRAG Unified — Production Deployment with Public Endpoint    ║"
echo "║  Job ID : ${SLURM_JOB_ID:-interactive}                                       ║"
echo "║  Node   : ${SLURMD_NODENAME:-$(hostname)}                                            ║"
echo "║  Date   : $(date)                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

PREFLIGHT_FAIL=0

# ── Step 1: GPU allocation ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 1 — Environment Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "[STEP  1/22] GPU allocation..."
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi query failed")
    echo "  ✓ GPU: ${GPU_INFO}"
else
    echo "  ⚠ nvidia-smi not found on PATH (may be available after module load)"
fi

# ── Step 2: Project directory ──
echo "[STEP  2/22] Project directory..."
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "  ✗ FAIL: ${PROJECT_DIR} not found"
    PREFLIGHT_FAIL=1
else
    echo "  ✓ ${PROJECT_DIR}"
fi

# ── Step 3: Virtual environment ──
echo "[STEP  3/22] Virtual environment..."
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "  ✗ FAIL: No venv at ${VENV_DIR}"
    echo "  Fix: cd ${PROJECT_DIR} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    PREFLIGHT_FAIL=1
else
    echo "  ✓ ${VENV_DIR}"
fi

# ── Step 4: Source files + configs ──
echo "[STEP  4/22] Source files and configs..."
MISSING_FILES=0
for f in \
    "src/api/app.py" \
    "src/api/models.py" \
    "src/api/pipeline_factory.py" \
    "src/router/domain_router.py" \
    "pipelines/healthcare/adapter.py" \
    "pipelines/healthcare/rag_vqa.py" \
    "pipelines/scientific/adapter.py" \
    "configs/healthcare/model_config.yaml" \
    "configs/healthcare/retrieval_config.yaml" \
    "configs/healthcare/data_config.yaml" \
    "configs/unified_config.yaml" \
    "setup.py"
do
    if [ ! -f "${PROJECT_DIR}/${f}" ]; then
        echo "  ✗ Missing: ${f}"
        MISSING_FILES=1
    fi
done
if [ $MISSING_FILES -eq 0 ]; then
    echo "  ✓ All source files and configs present"
else
    echo "  ⚠ Some files missing (non-critical if scientific only)"
fi

# ── Step 5: Healthcare indices ──
echo "[STEP  5/22] Healthcare indices..."
if [ ! -d "${HC_DATA_ROOT}" ]; then
    echo "  ✗ FAIL: Healthcare data root not found: ${HC_DATA_ROOT}"
    PREFLIGHT_FAIL=1
elif [ ! -f "${HC_DATA_ROOT}/data/indexes/colqwen2_index/document_store.json" ]; then
    echo "  ✗ FAIL: document_store.json not found at ${HC_DATA_ROOT}/data/indexes/colqwen2_index/"
    PREFLIGHT_FAIL=1
else
    DOC_COUNT=$(python3 -c "import json; d=json.load(open('${HC_DATA_ROOT}/data/indexes/colqwen2_index/document_store.json')); print(len(d.get('documents',d)) if isinstance(d,dict) else len(d))" 2>/dev/null || echo "?")
    echo "  ✓ Healthcare index: ${HC_DATA_ROOT}/data/indexes/colqwen2_index/ (${DOC_COUNT} docs)"
fi

# Abort on critical failure
if [ $PREFLIGHT_FAIL -ne 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  PREFLIGHT FAILED — fix the issues above and resubmit          ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    exit 1
fi

# ── Step 6: Create symbolic links ──
echo "[STEP  6/22] Creating symbolic links..."
mkdir -p "${PROJECT_DIR}/data/indexes"

# Symlink: ColQwen2 index
LINK_INDEX="${PROJECT_DIR}/data/indexes/colqwen2_index"
TARGET_INDEX="${HC_DATA_ROOT}/data/indexes/colqwen2_index"
if [ -L "${LINK_INDEX}" ]; then
    echo "  ✓ Index symlink exists → $(readlink -f ${LINK_INDEX})"
elif [ -d "${LINK_INDEX}" ]; then
    echo "  ✓ Index directory exists (real, not symlink)"
else
    ln -s "${TARGET_INDEX}" "${LINK_INDEX}"
    echo "  ✓ Created: ${LINK_INDEX} → ${TARGET_INDEX}"
fi

# Symlink: OpenI dataset
LINK_OPENI="${PROJECT_DIR}/data/openi"
TARGET_OPENI="${HC_DATA_ROOT}/data/openi"
if [ -L "${LINK_OPENI}" ]; then
    echo "  ✓ OpenI symlink exists → $(readlink -f ${LINK_OPENI})"
elif [ -d "${LINK_OPENI}" ]; then
    echo "  ✓ OpenI directory exists (real, not symlink)"
elif [ -d "${TARGET_OPENI}" ]; then
    ln -s "${TARGET_OPENI}" "${LINK_OPENI}"
    echo "  ✓ Created: ${LINK_OPENI} → ${TARGET_OPENI}"
else
    echo "  ⚠ OpenI target not found: ${TARGET_OPENI} (image remapping may fail)"
fi

# ── Step 7: Verify images and reports ──
echo "[STEP  7/22] Verifying images and reports..."
if [ -f "${LINK_INDEX}/document_store.json" ]; then
    echo "  ✓ Index: document_store.json accessible via symlink"
else
    echo "  ✗ FAIL: document_store.json not accessible"
    exit 1
fi
if [ -d "${LINK_OPENI}/images" ] 2>/dev/null; then
    IMG_COUNT=$(find "${LINK_OPENI}/images/" -maxdepth 1 -type f -name "*.png" 2>/dev/null | wc -l)
    echo "  ✓ Images: ${IMG_COUNT} .png files in ${LINK_OPENI}/images/"
else
    echo "  ⚠ Images directory not accessible (retrieval still works, image display may not)"
fi
if [ -d "${LINK_OPENI}/reports" ] 2>/dev/null; then
    RPT_COUNT=$(find "${LINK_OPENI}/reports/" -maxdepth 1 -type f 2>/dev/null | wc -l)
    echo "  ✓ Reports: ${RPT_COUNT} files in ${LINK_OPENI}/reports/"
else
    echo "  ⚠ Reports directory not accessible"
fi

# ── Step 8: Load modules + activate venv ──
echo "[STEP  8/22] Loading HPC modules..."
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module purge 2>/dev/null || true
module load python/3.10 2>/dev/null || module load python3 2>/dev/null || true
module load cuda/12.1   2>/dev/null || module load cuda   2>/dev/null || true

echo "  Activating venv..."
source "${VENV_DIR}/bin/activate"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
hash -r

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_DIR}"
echo "  ✓ Python: $(which python) ($(python --version 2>&1))"
echo "  ✓ CWD: $(pwd)"

# ── Step 9: Verify CUDA from Python ──
echo "[STEP  9/22] Verifying CUDA..."
python -c "
import torch
print(f'  torch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print('  ✓ CUDA verified')
else:
    print('  ✗ FAIL: No CUDA GPU — Healthcare pipeline will NOT load')
    exit(1)
" || { echo "  CUDA verification failed"; exit 1; }

# ── Step 10: Verify import chain ──
echo "[STEP 10/22] Verifying import chain..."
python -c "
from src.api.models import QueryRequest, QueryResponse, HealthResponse, ReadyResponse
from src.api.models import RetrievalMetadata, RetrievalScores, VerificationResult
from src.router.domain_router import DomainRouter
from pipelines.healthcare.adapter import HealthcarePipeline
from pipelines.scientific.adapter import ScientificPipeline
from src.api.pipeline_factory import create_healthcare_pipeline, create_scientific_pipeline
from src.shared.schemas.response import UnifiedResponse, SourceItem
from src.shared.base_pipeline import BasePipeline
print('  ✓ All imports OK')
" || { echo "  ✗ Import chain broken"; exit 1; }

# ── Step 11: Scientific pipeline status ──
echo "[STEP 11/22] Scientific pipeline..."
echo "  ℹ Scientific pipeline disabled — no indices available"
echo "  ℹ Healthcare success determines job result"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 2 — SERVER LAUNCH & LOCAL VERIFICATION
# ════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 2 — Server Launch & Local Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 12: Launch FastAPI ──
echo "[STEP 12/22] Starting FastAPI server on port ${PORT}..."
mkdir -p outputs/logs

python -u -m uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --log-level info \
    2>&1 &

SERVER_PID=$!
echo "  Server PID: ${SERVER_PID}"

# ── Step 13: Wait until server is live ──
echo "[STEP 13/22] Waiting for server to be ready (up to 600s)..."
MAX_WAIT=600
WAITED=0
SERVER_UP=0

while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null | grep -q "200"; then
        SERVER_UP=1
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  ✗ FAIL: Server process died (PID ${SERVER_PID})"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "  ... waiting (${WAITED}s / ${MAX_WAIT}s)"
    fi
done

if [ $SERVER_UP -eq 0 ]; then
    echo "  ✗ FAIL: Server did not start within ${MAX_WAIT}s"
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
fi

echo "  ✓ Server is UP after ${WAITED}s"
echo ""

# ── Helper: run & validate curl tests ──
TMPFILE="/tmp/mmrag_test_${SLURM_JOB_ID:-$$}.json"
TEST_PASS=0
TEST_FAIL=0

curl_test() {
    local STEP="$1" NAME="$2" METHOD="$3" URL="$4" DATA="$5"
    shift 5
    # remaining args are python validation lines

    echo "[STEP ${STEP}] ${NAME}"

    local HTTP_CODE
    if [ "$METHOD" = "GET" ]; then
        HTTP_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" "${URL}" 2>/dev/null)
    else
        HTTP_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" \
            -X POST "${URL}" \
            -H "Content-Type: application/json" \
            -d "${DATA}" 2>/dev/null)
    fi

    local BODY
    BODY=$(cat "${TMPFILE}" 2>/dev/null || echo "NO RESPONSE")

    if [ "${HTTP_CODE}" != "200" ]; then
        echo "  ✗ FAIL: HTTP ${HTTP_CODE}"
        echo "  Body: $(echo ${BODY} | head -c 500)"
        TEST_FAIL=$((TEST_FAIL + 1))
        return 1
    fi

    # Run inline Python validation
    local RESULT
    RESULT=$(echo "${BODY}" | python -c "$@" 2>&1)
    local RC=$?

    if [ $RC -eq 0 ]; then
        echo "  ✓ PASS — ${RESULT}"
        TEST_PASS=$((TEST_PASS + 1))
        return 0
    else
        echo "  ✗ FAIL — ${RESULT}"
        echo "  Body: $(echo ${BODY} | head -c 500)"
        TEST_FAIL=$((TEST_FAIL + 1))
        return 1
    fi
}

BASE="http://localhost:${PORT}"

# ── Step 14: GET /health ──
curl_test "14/22" "GET /health (local)" "GET" "${BASE}/health" "" "
import sys, json
d = json.load(sys.stdin)
assert d.get('status') == 'healthy', f'status={d.get(\"status\")}'
print(f'status=healthy, service={d.get(\"service\")}, version={d.get(\"version\")}')
"

# ── Step 15: GET /ready ──
curl_test "15/22" "GET /ready (local)" "GET" "${BASE}/ready" "" "
import sys, json
d = json.load(sys.stdin)
assert d.get('ready') == True, f'ready={d.get(\"ready\")}'
assert 'healthcare' in d.get('domains', []), f'domains={d.get(\"domains\")}'
assert 'LIVE' in d.get('detail', ''), f'detail missing LIVE: {d.get(\"detail\")}'
print(f'ready=true, domains={d[\"domains\"]}, detail=\"{d[\"detail\"]}\"')
"

# ── Step 16: POST /query ──
curl_test "16/22" "POST /query — healthcare (local)" "POST" "${BASE}/query" \
    '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3,"include_images":true}' "
import sys, json
d = json.load(sys.stdin)
a = d.get('answer','')
assert a, 'empty answer'
assert 'Pipeline not loaded' not in a, f'PLACEHOLDER: {a[:80]}'
s = d.get('sources', [])
assert len(s) > 0, 'no sources'
rm = d.get('retrieval_metadata',{}).get('scores',{})
v = d.get('verification',{})
lat = d.get('latency_ms', 0)
print(f'answer={len(a)}ch conf={d.get(\"confidence\",0):.3f} sources={len(s)} '
      f'colpali={rm.get(\"colpali\",0):.4f} scincl={rm.get(\"scincl\",0):.4f} '
      f'fused={rm.get(\"fused\",0):.4f} attr={v.get(\"attribution\")} '
      f'faith={v.get(\"faithfulness\")} latency={lat}ms')
"

echo ""

# Check if local tests passed
if [ $TEST_FAIL -gt 0 ]; then
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  LOCAL TESTS FAILED — ${TEST_FAIL} failure(s). Aborting.                   ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    kill ${SERVER_PID} 2>/dev/null || true
    rm -f "${TMPFILE}"
    exit 1
fi

echo "  ✓ All local tests passed (${TEST_PASS}/${TEST_PASS})"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 3 — PUBLIC HTTPS TUNNEL (Cloudflare Quick Tunnel)
# ════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 3 — Public HTTPS Tunnel"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 17: Download cloudflared ──
echo "[STEP 17/22] Setting up cloudflared..."
TUNNEL_LOG="${PROJECT_DIR}/outputs/logs/tunnel_${SLURM_JOB_ID:-$$}.log"

if [ -x "${CLOUDFLARED_BIN}" ]; then
    echo "  ✓ cloudflared already cached at ${CLOUDFLARED_BIN}"
else
    echo "  Downloading cloudflared..."
    mkdir -p "$(dirname ${CLOUDFLARED_BIN})"
    curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" \
        -o "${CLOUDFLARED_BIN}" 2>/dev/null
    chmod +x "${CLOUDFLARED_BIN}"
    echo "  ✓ Downloaded cloudflared to ${CLOUDFLARED_BIN}"
fi

# Verify binary works
if "${CLOUDFLARED_BIN}" version 2>/dev/null | head -1; then
    echo "  ✓ cloudflared binary verified"
else
    echo "  ⚠ cloudflared version check failed — attempting tunnel anyway"
fi

# ── Step 18: Start tunnel ──
echo "[STEP 18/22] Starting Cloudflare Quick Tunnel..."
"${CLOUDFLARED_BIN}" tunnel --url "http://localhost:${PORT}" \
    --no-autoupdate \
    > "${TUNNEL_LOG}" 2>&1 &

TUNNEL_PID=$!
echo "  Tunnel PID: ${TUNNEL_PID}"
echo "  Waiting for tunnel URL..."

# Wait for the tunnel URL to appear in logs (up to 30s)
PUBLIC_URL=""
TUNNEL_WAIT=0
while [ $TUNNEL_WAIT -lt 30 ]; do
    sleep 2
    TUNNEL_WAIT=$((TUNNEL_WAIT + 2))

    # Extract the trycloudflare.com URL from tunnel logs
    PUBLIC_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "${TUNNEL_LOG}" 2>/dev/null | head -1)
    if [ -n "${PUBLIC_URL}" ]; then
        break
    fi

    if ! kill -0 ${TUNNEL_PID} 2>/dev/null; then
        echo "  ⚠ Tunnel process died. Check ${TUNNEL_LOG}"
        echo "  Continuing without public endpoint..."
        PUBLIC_URL=""
        break
    fi
done

echo ""
if [ -n "${PUBLIC_URL}" ]; then
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║  🌐 PUBLIC HTTPS ENDPOINT ACTIVE                               ║"
    echo "║                                                                ║"
    echo "║  PUBLIC_URL=${PUBLIC_URL}                                       "
    echo "║                                                                ║"
    echo "║  Endpoints:                                                    ║"
    echo "║    GET  ${PUBLIC_URL}/health                                    "
    echo "║    GET  ${PUBLIC_URL}/ready                                     "
    echo "║    POST ${PUBLIC_URL}/query                                     "
    echo "║    GET  ${PUBLIC_URL}/docs  (OpenAPI)                           "
    echo "║                                                                ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
else
    echo "  ⚠ Could not obtain public URL within 30s"
    echo "  Server is still running locally on port ${PORT}"
    echo "  Manual tunnel: cloudflared tunnel --url http://localhost:${PORT}"
fi
echo ""

# ── Step 19: Print public URL (repeated for easy grep) ──
echo "[STEP 19/22] Public URL summary"
echo "  PUBLIC_URL=${PUBLIC_URL:-NOT_AVAILABLE}"
echo "  LOCAL_URL=http://localhost:${PORT}"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 4 — PUBLIC VERIFICATION & STABILITY TESTING
# ════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 4 — Public Verification & Stability"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 20: Public endpoint verification ──
if [ -n "${PUBLIC_URL}" ]; then
    echo "[STEP 20/22] Verifying public endpoints..."

    # Public /health
    curl_test "20a" "GET /health (public)" "GET" "${PUBLIC_URL}/health" "" "
import sys, json
d = json.load(sys.stdin)
assert d.get('status') == 'healthy'
print('public /health OK')
"

    # Public /ready
    curl_test "20b" "GET /ready (public)" "GET" "${PUBLIC_URL}/ready" "" "
import sys, json
d = json.load(sys.stdin)
assert d.get('ready') == True
assert 'healthcare' in d.get('domains', [])
print(f'public /ready OK — {d.get(\"detail\")}')
"

    # Public /query
    curl_test "20c" "POST /query (public)" "POST" "${PUBLIC_URL}/query" \
        '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3}' "
import sys, json
d = json.load(sys.stdin)
a = d.get('answer','')
assert a and 'Pipeline not loaded' not in a
print(f'public /query OK — answer={len(a)}ch latency={d.get(\"latency_ms\",0)}ms')
"
else
    echo "[STEP 20/22] Skipped — no public URL available"
fi
echo ""

# ── Step 21: Stability testing (3 rounds of diverse queries) ──
echo "[STEP 21/22] Stability testing (3 rounds)..."
echo ""

STABILITY_QUERIES=(
    '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3}'
    '{"query":"Is there pleural effusion in this chest x-ray?","domain":"auto","top_k":3}'
    '{"query":"Describe the cardiac silhouette","domain":"healthcare","top_k":3}'
    '{"query":"Are there signs of pneumonia?","domain":"auto","top_k":3}'
    '{"query":"What does the lung field show?","domain":"healthcare","top_k":3}'
    '{"query":"Is there evidence of atelectasis?","domain":"auto","top_k":3}'
)

STABILITY_PASS=0
STABILITY_FAIL=0

for round in 1 2 3; do
    echo "  Round ${round}/3:"
    for i in "${!STABILITY_QUERIES[@]}"; do
        QUERY="${STABILITY_QUERIES[$i]}"
        HTTP_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" \
            -X POST "${BASE}/query" \
            -H "Content-Type: application/json" \
            -d "${QUERY}" 2>/dev/null)

        BODY=$(cat "${TMPFILE}" 2>/dev/null)
        ANSWER=$(echo "${BODY}" | python -c "import sys,json; d=json.load(sys.stdin); a=d.get('answer',''); print(f'{len(a)}ch' if a and 'Pipeline not loaded' not in a else 'FAIL')" 2>/dev/null || echo "FAIL")
        LATENCY=$(echo "${BODY}" | python -c "import sys,json; print(json.load(sys.stdin).get('latency_ms',0))" 2>/dev/null || echo "0")

        if [ "${HTTP_CODE}" = "200" ] && [ "${ANSWER}" != "FAIL" ]; then
            echo "    ✓ Query $((i+1)): HTTP=${HTTP_CODE} answer=${ANSWER} latency=${LATENCY}ms"
            STABILITY_PASS=$((STABILITY_PASS + 1))
        else
            echo "    ✗ Query $((i+1)): HTTP=${HTTP_CODE} answer=${ANSWER}"
            STABILITY_FAIL=$((STABILITY_FAIL + 1))
        fi
    done
    echo ""
done

STABILITY_TOTAL=$((STABILITY_PASS + STABILITY_FAIL))
echo "  Stability: ${STABILITY_PASS}/${STABILITY_TOTAL} queries successful"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  DEPLOYMENT SUMMARY                                            ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                ║"
echo "║  Local Tests    : ${TEST_PASS} passed, ${TEST_FAIL} failed                            ║"
echo "║  Stability      : ${STABILITY_PASS}/${STABILITY_TOTAL} queries passed                           ║"
echo "║  Server PID     : ${SERVER_PID}                                         ║"
echo "║  Local URL      : http://localhost:${PORT}                          ║"
if [ -n "${PUBLIC_URL}" ]; then
echo "║  Public URL     : ${PUBLIC_URL}  "
echo "║  Tunnel PID     : ${TUNNEL_PID}                                         ║"
fi
echo "║                                                                ║"
if [ $TEST_FAIL -eq 0 ] && [ $STABILITY_FAIL -eq 0 ]; then
echo "║  ✅ ALL CHECKS PASSED — Healthcare pipeline LIVE               ║"
else
echo "║  ⚠  Some checks failed — review logs above                    ║"
fi
echo "║                                                                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Print curl commands for the professor
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Curl commands for verification:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -n "${PUBLIC_URL}" ]; then
    echo ""
    echo "  # Health check"
    echo "  curl -s ${PUBLIC_URL}/health | python -m json.tool"
    echo ""
    echo "  # Readiness check"
    echo "  curl -s ${PUBLIC_URL}/ready | python -m json.tool"
    echo ""
    echo "  # Healthcare query"
    echo "  curl -s -X POST ${PUBLIC_URL}/query \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"query\":\"What is cardiomegaly?\",\"domain\":\"healthcare\",\"top_k\":3}' \\"
    echo "    | python -m json.tool"
    echo ""
    echo "  # Auto-routing query"
    echo "  curl -s -X POST ${PUBLIC_URL}/query \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"query\":\"Is there pleural effusion in this chest x-ray?\",\"domain\":\"auto\",\"top_k\":3}' \\"
    echo "    | python -m json.tool"
    echo ""
    echo "  # OpenAPI docs (browser)"
    echo "  ${PUBLIC_URL}/docs"
fi
echo ""

# ── Step 22: Keep server + tunnel alive ──
echo "[STEP 22/22] Server and tunnel are now running."
echo "  The server will remain active until the SLURM walltime (4h) or scancel."
echo "  To stop: scancel ${SLURM_JOB_ID:-<JOBID>}"
echo ""
echo "  Entering keep-alive loop..."
echo "  $(date)"
echo ""

# Cleanup handler
cleanup() {
    echo ""
    echo "  Shutting down..."
    kill ${SERVER_PID} 2>/dev/null || true
    [ -n "${TUNNEL_PID:-}" ] && kill ${TUNNEL_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
    [ -n "${TUNNEL_PID:-}" ] && wait ${TUNNEL_PID} 2>/dev/null || true
    rm -f "${TMPFILE}"
    echo "  Server and tunnel stopped."
    echo "  Finished: $(date)"
}
trap cleanup EXIT INT TERM

# Keep alive — periodically verify server is still running
while true; do
    sleep 60
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  ⚠ Server died unexpectedly at $(date)"
        exit 1
    fi
    # Silent health check every minute
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
    if [ "${HTTP_CODE}" != "200" ]; then
        echo "  ⚠ Health check failed (HTTP ${HTTP_CODE}) at $(date)"
    fi
done
