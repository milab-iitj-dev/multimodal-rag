#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  MMRAG Unified — Scientific Domain E2E Production Validation
# ════════════════════════════════════════════════════════════════════════
#
#  What this does:
#    Phase 1 — Environment & asset verification (GPU, venv, paths, models)
#    Phase 2 — Start FastAPI, wait for /ready, confirm BOTH domains LIVE
#    Phase 3 — Execute 5 scientific queries through the live API
#    Phase 4 — Validate every response field against production contract
#    Phase 5 — Generate markdown report with timings, scores, GPU stats
#    Phase 6 — Shutdown & exit (non-zero on any failure)
#
#  Submit:
#    sbatch scripts/slurm_scientific_e2e.sh
#
#  Monitor:
#    tail -f outputs/logs/sci_e2e_<JOBID>.log
#
#  Report:
#    cat outputs/reports/scientific_e2e_report.md
#
#  Expected runtime: 10-20 minutes (model loading + 5 queries)
# ════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=mmrag-sci-e2e
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=01:30:00
#SBATCH --output=outputs/logs/sci_e2e_%j.log
#SBATCH --error=outputs/logs/sci_e2e_%j.err

set -uo pipefail

# ════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

PROJECT_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified"
VENV_DIR="${PROJECT_DIR}/.venv"
HC_DATA_ROOT="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag-healthcare"
SCI_DATA_ROOT="/scratch/data/divyasaxena_rs/Vineet_internship"
PORT=8847
HF_CACHE="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.cache/huggingface"

# Output paths
REPORT_DIR="${PROJECT_DIR}/outputs/reports"
RESPONSE_DIR="${PROJECT_DIR}/outputs/api_examples/scientific_e2e"
REPORT_FILE="${REPORT_DIR}/scientific_e2e_report.md"
TMPFILE="/tmp/mmrag_sci_e2e_${SLURM_JOB_ID:-$$}.json"

# Timeouts
STARTUP_TIMEOUT=600    # 10 min for model loading
READY_TIMEOUT=120      # 2 min after health OK → ready

# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════

ts()   { date '+%H:%M:%S'; }
step() { echo "[$(ts)] [$1] $2"; }
ok()   { echo "  ✓ $1"; }
fail() { echo "  ✗ FAIL: $1"; }

SERVER_PID=""
TEST_PASS=0
TEST_FAIL=0
TEST_TOTAL=0
ASSET_ERRORS=0
DEPLOY_START=$(date +%s)

# Create output directories
mkdir -p "${REPORT_DIR}" "${RESPONSE_DIR}" "${PROJECT_DIR}/outputs/logs"

# Cleanup handler
cleanup() {
    echo ""
    echo "[$(ts)] Shutting down..."
    if [ -n "${SERVER_PID}" ]; then
        kill ${SERVER_PID} 2>/dev/null || true
        wait ${SERVER_PID} 2>/dev/null || true
        echo "  Server stopped (PID ${SERVER_PID})"
    fi
    rm -f "${TMPFILE}"
    echo "[$(ts)] Cleanup complete"
}
trap cleanup EXIT INT TERM

# ════════════════════════════════════════════════════════════════════════
#  BANNER
# ════════════════════════════════════════════════════════════════════════

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  MMRAG — Scientific Domain E2E Production Validation            ║"
echo "║  Job:  ${SLURM_JOB_ID:-interactive}                                              ║"
echo "║  Node: $(hostname)                                                    ║"
echo "║  Date: $(date '+%Y-%m-%d %H:%M:%S')                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 1 — ENVIRONMENT & ASSET VERIFICATION
# ════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 1 — Environment & Asset Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 1.1 GPU ──
step "1.1" "GPU allocation"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: ${GPU_NAME} (${GPU_MEM_TOTAL} MiB total, ${GPU_MEM_FREE} MiB free)"
    ok "Driver: ${DRIVER}"
else
    fail "nvidia-smi not found"
    exit 1
fi

# ── 1.2 Virtual environment ──
step "1.2" "Virtual environment"
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    fail "No venv at ${VENV_DIR}"
    exit 1
fi

# Anaconda-proof activation
if command -v conda &>/dev/null; then
    conda deactivate 2>/dev/null || true
fi
CLEAN_PATH=""
IFS=':' read -ra PATH_PARTS <<< "$PATH"
for p in "${PATH_PARTS[@]}"; do
    case "$p" in
        *conda*|*anaconda*|*Anaconda*) ;;
        *) CLEAN_PATH="${CLEAN_PATH:+${CLEAN_PATH}:}${p}" ;;
    esac
done
export PATH="${CLEAN_PATH}"

source "${VENV_DIR}/bin/activate"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
hash -r

PYTHON_PATH=$(which python 2>/dev/null)
if echo "${PYTHON_PATH}" | grep -q "${VENV_DIR}"; then
    ok "Python: ${PYTHON_PATH} ($(python --version 2>&1))"
else
    fail "Python NOT from venv: ${PYTHON_PATH}"
    exit 1
fi

# ── 1.3 Environment variables ──
step "1.3" "Environment variables"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAG_BASE_DIR="${SCI_DATA_ROOT}"
ok "HF_HOME=${HF_HOME}"
ok "RAG_BASE_DIR=${RAG_BASE_DIR}"
ok "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

cd "${PROJECT_DIR}"
ok "CWD: $(pwd)"

# ── 1.4 CUDA verification ──
step "1.4" "CUDA + torch"
python -c "
import torch
v = torch.__version__
cv = torch.version.cuda or 'NONE'
a = torch.cuda.is_available()
if not a:
    print('FATAL: CUDA not available')
    exit(1)
gn = torch.cuda.get_device_name(0)
gm = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'torch={v} cuda={cv} gpu={gn} vram={gm:.1f}GB')
" 2>&1 | while IFS= read -r line; do echo "  ${line}"; done
if [ ${PIPESTATUS[0]} -ne 0 ]; then fail "CUDA check failed"; exit 1; fi
ok "CUDA verified"

# ── 1.5 Scientific data assets ──
step "1.5" "Scientific data assets"

# page_metadata.json
SCI_METADATA="${SCI_DATA_ROOT}/data/indices/page_metadata.json"
if [ -f "${SCI_METADATA}" ]; then
    META_SIZE=$(stat --printf="%s" "${SCI_METADATA}" 2>/dev/null || echo "?")
    META_PAGES=$(python -c "import json; d=json.load(open('${SCI_METADATA}')); print(len(d))" 2>/dev/null || echo "?")
    ok "page_metadata.json: ${META_PAGES} pages (${META_SIZE} bytes)"
else
    fail "page_metadata.json NOT FOUND: ${SCI_METADATA}"
    ASSET_ERRORS=$((ASSET_ERRORS + 1))
fi

# doc_mapping.json
SCI_DOCMAP="${SCI_DATA_ROOT}/data/indices/doc_mapping.json"
if [ -f "${SCI_DOCMAP}" ]; then
    DOC_COUNT=$(python -c "import json; d=json.load(open('${SCI_DOCMAP}')); print(len(d))" 2>/dev/null || echo "?")
    ok "doc_mapping.json: ${DOC_COUNT} documents"
else
    fail "doc_mapping.json NOT FOUND: ${SCI_DOCMAP}"
    ASSET_ERRORS=$((ASSET_ERRORS + 1))
fi

# ChromaDB index
SCI_CHROMA="${SCI_DATA_ROOT}/data/indices/chroma_index"
if [ -d "${SCI_CHROMA}" ]; then
    CHROMA_FILES=$(find "${SCI_CHROMA}" -type f 2>/dev/null | wc -l)
    ok "chroma_index: ${CHROMA_FILES} files"
    # Verify ChromaDB can open it
    python -c "
import chromadb
c = chromadb.PersistentClient(path='${SCI_CHROMA}')
cols = c.list_collections()
names = [x.name for x in cols]
print(f'  Collections: {names}')
for name in names:
    col = c.get_collection(name)
    print(f'  {name}: {col.count()} records')
" 2>&1 | while IFS= read -r line; do echo "  ${line}"; done
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        fail "ChromaDB failed to open"
        ASSET_ERRORS=$((ASSET_ERRORS + 1))
    else
        ok "ChromaDB opened successfully"
    fi
else
    fail "chroma_index NOT FOUND: ${SCI_CHROMA}"
    ASSET_ERRORS=$((ASSET_ERRORS + 1))
fi

# Multivectors (.npy files)
SCI_MULTIVEC="${SCI_DATA_ROOT}/data/indices/multivectors"
if [ -d "${SCI_MULTIVEC}" ]; then
    NPY_COUNT=$(find "${SCI_MULTIVEC}" -name '*.npy' ! -name '*.meta.npy' 2>/dev/null | wc -l)
    NPY_SIZE=$(du -sh "${SCI_MULTIVEC}" 2>/dev/null | cut -f1)
    ok "multivectors: ${NPY_COUNT} .npy files (${NPY_SIZE})"
    # Verify we can load one
    python -c "
import numpy as np, os, glob
files = sorted(glob.glob('${SCI_MULTIVEC}/*.npy'))
files = [f for f in files if not f.endswith('.meta.npy')]
if files:
    arr = np.load(files[0])
    print(f'  Sample: {os.path.basename(files[0])} → shape={arr.shape} dtype={arr.dtype}')
" 2>&1 | while IFS= read -r line; do echo "  ${line}"; done
    ok "multivectors loadable"
else
    fail "multivectors NOT FOUND: ${SCI_MULTIVEC}"
    ASSET_ERRORS=$((ASSET_ERRORS + 1))
fi

# Page images
SCI_PAGES="${SCI_DATA_ROOT}/data/parsed/pages"
if [ -d "${SCI_PAGES}" ]; then
    IMG_COUNT=$(find "${SCI_PAGES}" -name '*.png' 2>/dev/null | wc -l)
    ok "page images: ${IMG_COUNT} .png files"
else
    echo "  ⚠ page images not found (retrieval works, generation images may miss)"
fi

# Scientific config
SCI_CONFIG="${PROJECT_DIR}/configs/scientific/config.yaml"
if [ -f "${SCI_CONFIG}" ]; then
    ok "configs/scientific/config.yaml exists"
else
    fail "configs/scientific/config.yaml NOT FOUND"
    ASSET_ERRORS=$((ASSET_ERRORS + 1))
fi

# ── 1.6 HuggingFace model cache ──
step "1.6" "HF model cache"

HF_HUB="${HF_CACHE}/hub"
SCI_MODELS=(
    "models--vidore--colpali-v1.2"
    "models--malteos--scincl"
    "models--Qwen--Qwen2-VL-2B-Instruct"
)

for model_dir in "${SCI_MODELS[@]}"; do
    MODEL_PATH="${HF_HUB}/${model_dir}"
    if [ -d "${MODEL_PATH}" ] || [ -L "${MODEL_PATH}" ]; then
        REAL_PATH=$(readlink -f "${MODEL_PATH}" 2>/dev/null || echo "${MODEL_PATH}")
        ok "${model_dir} → ${REAL_PATH}"
    else
        fail "${model_dir} NOT FOUND in ${HF_HUB}"
        ASSET_ERRORS=$((ASSET_ERRORS + 1))
    fi
done

# ── 1.7 Healthcare data (must still work) ──
step "1.7" "Healthcare data (regression check)"
HC_DOCSTORE="${HC_DATA_ROOT}/data/indexes/colqwen2_index/document_store.json"
if [ -f "${HC_DOCSTORE}" ]; then
    ok "Healthcare index exists"
else
    fail "Healthcare index NOT FOUND (regression!)"
    ASSET_ERRORS=$((ASSET_ERRORS + 1))
fi

# ── 1.8 Import chain ──
step "1.8" "Import chain"
python -c "
from src.api.models import QueryRequest, QueryResponse
from src.router.domain_router import DomainRouter
from pipelines.healthcare.adapter import HealthcarePipeline
from pipelines.scientific.adapter import ScientificPipeline
from src.api.pipeline_factory import create_healthcare_pipeline, create_scientific_pipeline
from pipelines.scientific.online_pipeline import OnlinePipeline
from src.domains.scientific.retrieval.colpali_retriever import ColPaliRetriever
from src.domains.scientific.retrieval.text_retriever import TextRetriever
from src.domains.scientific.retrieval.fusion_retriever import FusionRetriever
from src.domains.scientific.generation.self_check import DomainGuard
from src.domains.scientific.generation.rag_generator import RAGGenerator
from src.domains.scientific.models.loader import load_colpali, load_scincl, load_qwen2vl
print('All 12 imports OK')
" 2>&1
if [ $? -ne 0 ]; then
    fail "Import chain BROKEN"
    ASSET_ERRORS=$((ASSET_ERRORS + 1))
else
    ok "Import chain verified (12/12)"
fi

echo ""
if [ ${ASSET_ERRORS} -gt 0 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PHASE 1 FAILED — ${ASSET_ERRORS} critical issue(s)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 1
fi
echo "[$(ts)] Phase 1 PASSED — all assets verified"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 2 — START SERVER & VERIFY BOTH DOMAINS LIVE
# ════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 2 — Server Startup & Domain Readiness"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Healthcare symlinks (idempotent)
mkdir -p "${PROJECT_DIR}/data/indexes"
LINK_INDEX="${PROJECT_DIR}/data/indexes/colqwen2_index"
TARGET_INDEX="${HC_DATA_ROOT}/data/indexes/colqwen2_index"
if [ ! -L "${LINK_INDEX}" ] && [ ! -d "${LINK_INDEX}" ]; then
    ln -s "${TARGET_INDEX}" "${LINK_INDEX}"
fi

LINK_OPENI="${PROJECT_DIR}/data/openi"
TARGET_OPENI="${HC_DATA_ROOT}/data/openi"
if [ ! -L "${LINK_OPENI}" ] && [ ! -d "${LINK_OPENI}" ] && [ -d "${TARGET_OPENI}" ]; then
    ln -s "${TARGET_OPENI}" "${LINK_OPENI}"
fi

# Start server
step "2.1" "Starting FastAPI server on port ${PORT}"
python -u -m uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --log-level info \
    2>&1 &

SERVER_PID=$!
ok "Server PID: ${SERVER_PID}"

# Wait for /health (server process alive + accepting connections)
step "2.2" "Waiting for /health (up to ${STARTUP_TIMEOUT}s)"
WAITED=0
HEALTH_UP=0
while [ ${WAITED} -lt ${STARTUP_TIMEOUT} ]; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null | grep -q "200"; then
        HEALTH_UP=1
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        fail "Server process died"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "  ... waiting (${WAITED}s / ${STARTUP_TIMEOUT}s)"
    fi
done

if [ ${HEALTH_UP} -eq 0 ]; then
    fail "Server did not start within ${STARTUP_TIMEOUT}s"
    exit 1
fi
ok "Server is UP after ${WAITED}s"

# Wait for /ready with BOTH domains
step "2.3" "Waiting for /ready (up to ${READY_TIMEOUT}s more)"
WAITED=0
BOTH_LIVE=0
while [ ${WAITED} -lt ${READY_TIMEOUT} ]; do
    READY_BODY=$(curl -s "http://localhost:${PORT}/ready" 2>/dev/null)
    READY_RESULT=$(echo "${READY_BODY}" | python -c "
import sys, json
try:
    d = json.load(sys.stdin)
    domains = d.get('domains', [])
    if 'healthcare' in domains and 'scientific' in domains:
        print('BOTH_LIVE')
    elif domains:
        print('PARTIAL:' + ','.join(domains))
    else:
        print('NONE')
except:
    print('ERROR')
" 2>/dev/null)

    if [ "${READY_RESULT}" = "BOTH_LIVE" ]; then
        BOTH_LIVE=1
        break
    fi

    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 15)) -eq 0 ]; then
        echo "  ... status=${READY_RESULT} (${WAITED}s / ${READY_TIMEOUT}s)"
    fi
done

echo ""
echo "  /ready response:"
echo "  ${READY_BODY}" | python -m json.tool 2>/dev/null || echo "  ${READY_BODY}"
echo ""

if [ ${BOTH_LIVE} -eq 0 ]; then
    fail "BOTH domains not LIVE after $((STARTUP_TIMEOUT + WAITED))s"
    fail "Got: ${READY_RESULT}"
    echo ""
    echo "  This means create_scientific_pipeline() returned None."
    echo "  Check the server logs above for ERROR lines."
    exit 1
fi

STARTUP_ELAPSED=$(($(date +%s) - DEPLOY_START))
ok "BOTH domains LIVE (startup: ${STARTUP_ELAPSED}s)"
echo ""

# Record GPU state after loading
GPU_AFTER_LOAD=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null)
ok "GPU after model load: ${GPU_AFTER_LOAD}"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 3 — EXECUTE SCIENTIFIC QUERIES
# ════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 3 — Scientific Query Execution (5 queries)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

BASE="http://localhost:${PORT}"

# Query definitions: ID|QUERY
QUERIES=(
    "sci_01|What is the main architecture of Vision Transformer (ViT) and how does it process image patches?"
    "sci_02|How does DeiT achieve competitive accuracy without large-scale pretraining datasets?"
    "sci_03|What are the key differences between Swin Transformer and standard ViT?"
    "sci_04|How does EfficientFormer achieve MobileNet-level inference speed with transformer accuracy?"
    "sci_05|What scaling strategies are used in ViT-22B to train a 22 billion parameter vision model?"
)

run_scientific_query() {
    local QUERY_ID="$1"
    local QUERY_TEXT="$2"
    local RESPONSE_FILE="${RESPONSE_DIR}/${QUERY_ID}.json"

    echo "────────────────────────────────────────────────────────"
    echo "  ${QUERY_ID}: ${QUERY_TEXT:0:70}..."
    echo "────────────────────────────────────────────────────────"

    TEST_TOTAL=$((TEST_TOTAL + 1))

    # Execute query
    HTTP_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" \
        -X POST "${BASE}/query" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"${QUERY_TEXT}\",\"domain\":\"scientific\",\"top_k\":3,\"include_images\":true}" \
        2>/dev/null)

    if [ "${HTTP_CODE}" != "200" ]; then
        fail "HTTP ${HTTP_CODE}"
        cat "${TMPFILE}" 2>/dev/null | head -5
        TEST_FAIL=$((TEST_FAIL + 1))
        echo ""
        return
    fi

    # Save response
    cp "${TMPFILE}" "${RESPONSE_FILE}"

    # Validate all required fields
    VALIDATION=$(cat "${TMPFILE}" | python -c "
import sys, json

d = json.load(sys.stdin)
errors = []

# 1. answer — must exist, not placeholder
answer = d.get('answer', '')
if not answer:
    errors.append('answer is empty')
if 'Pipeline not loaded' in answer:
    errors.append('PLACEHOLDER response detected')

# 2. confidence — must be numeric > 0
conf = d.get('confidence', -1)
if not isinstance(conf, (int, float)) or conf < 0:
    errors.append(f'confidence invalid: {conf}')

# 3. sources — must have at least 1
sources = d.get('sources', [])
if len(sources) == 0:
    errors.append('no sources returned')

# 4. retrieval_metadata — must have scores with real values
rm = d.get('retrieval_metadata', {})
scores = rm.get('scores', {})
if 'fused' not in scores:
    errors.append('missing fused score')
elif scores.get('fused', 0) <= 0:
    errors.append(f'fused score is zero/negative: {scores.get(\"fused\")}')
if scores.get('colpali', 0) <= 0:
    errors.append(f'colpali score not propagated: {scores.get(\"colpali\", 0)}')
if scores.get('scincl', 0) <= 0:
    errors.append(f'scincl score not propagated: {scores.get(\"scincl\", 0)}')

# 5. verification — must have all 3 fields
v = d.get('verification', {})
for field in ['attribution', 'faithfulness', 'confidence_pass']:
    if field not in v:
        errors.append(f'missing verification.{field}')

# 6. latency — must be > 0
latency = d.get('latency_ms', 0)
if latency <= 0:
    errors.append(f'latency_ms is {latency}')

# 7. citations — sources must have titles (papers)
for i, s in enumerate(sources):
    if not s.get('title'):
        errors.append(f'source[{i}] has no title')
    if s.get('relevance_score', 0) <= 0:
        errors.append(f'source[{i}] has zero relevance_score')

if errors:
    print('FAIL|' + '; '.join(errors))
else:
    # Build summary
    n_src = len(sources)
    fused = scores.get('fused', 0)
    colpali = scores.get('colpali', 0)
    scincl = scores.get('scincl', 0)
    titles = [s['title'][:40] for s in sources[:2]]
    attr = v.get('attribution', '?')
    faith = v.get('faithfulness', '?')
    print(f'PASS|conf={conf:.4f} fused={fused:.4f} colpali={colpali:.4f} scincl={scincl:.4f} sources={n_src} latency={latency}ms attr={attr} faith={faith} papers={titles}')
" 2>&1)

    RESULT_STATUS=$(echo "${VALIDATION}" | cut -d'|' -f1)
    RESULT_DETAIL=$(echo "${VALIDATION}" | cut -d'|' -f2-)

    if [ "${RESULT_STATUS}" = "PASS" ]; then
        echo "  ✅ PASS — ${RESULT_DETAIL}"
        TEST_PASS=$((TEST_PASS + 1))
    else
        echo "  ❌ FAIL — ${RESULT_DETAIL}"
        TEST_FAIL=$((TEST_FAIL + 1))
    fi

    # Print answer snippet
    ANSWER_SNIPPET=$(cat "${TMPFILE}" | python -c "import sys,json; print(json.load(sys.stdin).get('answer','')[:150])" 2>/dev/null)
    echo "  Answer: ${ANSWER_SNIPPET}..."
    echo ""
}

# Execute all 5 queries
for entry in "${QUERIES[@]}"; do
    IFS='|' read -r QID QTEXT <<< "${entry}"
    run_scientific_query "${QID}" "${QTEXT}"
done

# ════════════════════════════════════════════════════════════════════════
#  PHASE 4 — HEALTHCARE REGRESSION CHECK
# ════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 4 — Healthcare Regression Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TEST_TOTAL=$((TEST_TOTAL + 1))
HC_RESPONSE="${RESPONSE_DIR}/healthcare_regression.json"

HTTP_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" \
    -X POST "${BASE}/query" \
    -H "Content-Type: application/json" \
    -d '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3,"include_images":true}' \
    2>/dev/null)

if [ "${HTTP_CODE}" = "200" ]; then
    cp "${TMPFILE}" "${HC_RESPONSE}"
    HC_ANSWER=$(cat "${TMPFILE}" | python -c "import sys,json; d=json.load(sys.stdin); print('OK' if d.get('answer') and 'Pipeline not loaded' not in d.get('answer','') else 'PLACEHOLDER')" 2>/dev/null)
    if [ "${HC_ANSWER}" = "OK" ]; then
        ok "Healthcare query returned real answer"
        TEST_PASS=$((TEST_PASS + 1))
    else
        fail "Healthcare query returned placeholder"
        TEST_FAIL=$((TEST_FAIL + 1))
    fi
else
    fail "Healthcare query HTTP ${HTTP_CODE}"
    TEST_FAIL=$((TEST_FAIL + 1))
fi
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 5 — GPU MEMORY & PERFORMANCE REPORT
# ════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 5 — Report Generation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Final GPU stats
GPU_FINAL_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
GPU_FINAL_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1)

TOTAL_ELAPSED=$(($(date +%s) - DEPLOY_START))

# Generate markdown report
python -c "
import os, json, glob, sys
from datetime import datetime

report_lines = []
r = report_lines.append

r('# Scientific Domain — E2E Production Validation Report')
r('')
r(f'**Date:** {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
r(f'**Job:** ${SLURM_JOB_ID:-interactive}')
r(f'**Node:** $(hostname)')
r(f'**GPU:** ${GPU_NAME}')
r(f'**Total Runtime:** ${TOTAL_ELAPSED}s')
r(f'**Startup Time:** ${STARTUP_ELAPSED}s')
r('')
r('## Results Summary')
r('')
r(f'| Metric | Value |')
r(f'|--------|-------|')
r(f'| Tests Passed | ${TEST_PASS} |')
r(f'| Tests Failed | ${TEST_FAIL} |')
r(f'| Total Tests | ${TEST_TOTAL} |')
r(f'| GPU After Load | ${GPU_AFTER_LOAD} |')
r(f'| GPU Final | ${GPU_FINAL_USED} used / ${GPU_FINAL_FREE} free |')
r(f'| Domains LIVE | healthcare, scientific |')
r('')

# Result badge
if ${TEST_FAIL} == 0:
    r('> **✅ ALL TESTS PASSED — Scientific domain PRODUCTION READY**')
else:
    r('> **❌ ${TEST_FAIL} TEST(S) FAILED — See details below**')
r('')

# Per-query details
r('## Scientific Query Results')
r('')
r('| Query | Confidence | Fused Score | Sources | Latency | Papers |')
r('|-------|-----------|-------------|---------|---------|--------|')

response_dir = '${RESPONSE_DIR}'
for i in range(1, 6):
    qid = f'sci_{i:02d}'
    fpath = os.path.join(response_dir, f'{qid}.json')
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        conf = d.get('confidence', 0)
        scores = d.get('retrieval_metadata', {}).get('scores', {})
        fused = scores.get('fused', 0)
        n_src = len(d.get('sources', []))
        latency = d.get('latency_ms', 0)
        titles = ', '.join(s.get('title', '?')[:30] for s in d.get('sources', [])[:2])
        r(f'| {qid} | {conf:.4f} | {fused:.4f} | {n_src} | {latency}ms | {titles} |')
    else:
        r(f'| {qid} | — | — | — | — | *response not saved* |')

r('')

# Healthcare regression
r('## Healthcare Regression')
r('')
hc_path = os.path.join(response_dir, 'healthcare_regression.json')
if os.path.exists(hc_path):
    with open(hc_path) as f:
        d = json.load(f)
    r(f'| Metric | Value |')
    r(f'|--------|-------|')
    r(f'| Answer Length | {len(d.get(\"answer\", \"\"))} chars |')
    r(f'| Confidence | {d.get(\"confidence\", 0):.4f} |')
    r(f'| Sources | {len(d.get(\"sources\", []))} |')
    r(f'| Latency | {d.get(\"latency_ms\", 0)}ms |')
else:
    r('*Healthcare regression response not saved.*')

r('')
r('---')
r(f'*Generated by slurm_scientific_e2e.sh — Job ${SLURM_JOB_ID:-interactive}*')

with open('${REPORT_FILE}', 'w') as f:
    f.write('\n'.join(report_lines))

print(f'Report saved to: ${REPORT_FILE}')
" 2>&1

ok "Report generated"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 6 — SUMMARY & EXIT
# ════════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  VALIDATION RESULTS                                             ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                 ║"
echo "║  Tests passed : ${TEST_PASS} / ${TEST_TOTAL}                                           ║"
echo "║  Tests failed : ${TEST_FAIL} / ${TEST_TOTAL}                                           ║"
echo "║  Startup time : ${STARTUP_ELAPSED}s                                            ║"
echo "║  Total time   : ${TOTAL_ELAPSED}s                                            ║"
echo "║  GPU used     : ${GPU_FINAL_USED}                                    ║"
echo "║                                                                 ║"

if [ ${TEST_FAIL} -eq 0 ]; then
    echo "║  ✅ ALL ${TEST_TOTAL} TESTS PASSED                                        ║"
    echo "║  Scientific domain: PRODUCTION READY                           ║"
else
    echo "║  ❌ ${TEST_FAIL} TEST(S) FAILED                                          ║"
    echo "║  Scientific domain: NOT READY                                  ║"
fi

echo "║                                                                 ║"
echo "║  Report: outputs/reports/scientific_e2e_report.md               ║"
echo "║  Responses: outputs/api_examples/scientific_e2e/                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

exit ${TEST_FAIL}
