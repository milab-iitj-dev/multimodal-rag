#!/usr/bin/env bash
# ==============================================================================
# MMRAG Unified — End-to-End API Smoke Test
# ==============================================================================

set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"

echo "=================================================================="
echo "Running E2E API Smoke Test against ${API_URL}..."
echo "=================================================================="

# 1. Health check
echo -n "[1/3] Testing GET /health ... "
HEALTH_RES=$(curl --silent --fail "${API_URL}/health")
echo "✓ PASSED: ${HEALTH_RES}"

# 2. Ready check
echo -n "[2/3] Testing GET /ready ... "
READY_RES=$(curl --silent --fail "${API_URL}/ready")
echo "✓ PASSED: ${READY_RES}"

# 3. Query check
echo "[3/3] Testing POST /query ..."
QUERY_PAYLOAD='{
  "query": "Is there evidence of pneumothorax or pleural effusion?",
  "domain": "healthcare",
  "top_k": 3
}'

QUERY_RES=$(curl --silent --fail -X POST "${API_URL}/query" \
  -H "Content-Type: application/json" \
  -d "${QUERY_PAYLOAD}")

echo "Response payload:"
echo "${QUERY_RES}" | python -m json.tool

ANSWER=$(echo "${QUERY_RES}" | python -c "import sys, json; print(json.load(sys.stdin).get('answer', ''))")
if [ -z "${ANSWER}" ]; then
    echo "✗ FAILED: Empty answer returned"
    exit 1
fi

echo ""
echo "=================================================================="
echo "SUCCESS: All E2E smoke tests passed!"
echo "=================================================================="
