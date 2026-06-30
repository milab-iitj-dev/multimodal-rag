#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  Cloudflare Quick Tunnel Launcher
# ════════════════════════════════════════════════════════════════
#
#  Creates a public HTTPS endpoint for the FastAPI server using
#  Cloudflare's free Quick Tunnel (no account required).
#
#  Usage:
#    bash scripts/start_cloudflare.sh              # default port 8847
#    bash scripts/start_cloudflare.sh 8000          # custom port
#
#  Why Cloudflare over ngrok:
#    ✓ No account signup required
#    ✓ No personal information collected
#    ✓ Automatic HTTPS with valid certificate
#    ✓ Single static binary — no installation
#    ✓ Outbound-only connection (HPC firewall friendly)
#    ✗ ngrok requires signup + auth token
#    ✗ ngrok logs request data to their servers
#    ✗ SSH tunnels require a public relay server + no HTTPS
#
# ════════════════════════════════════════════════════════════════

set -uo pipefail

PORT="${1:-8847}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CLOUDFLARED_BIN="${PROJECT_DIR}/.local/bin/cloudflared"
TUNNEL_LOG="${PROJECT_DIR}/outputs/logs/tunnel_$$.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Cloudflare Quick Tunnel"
echo "  Target: http://localhost:${PORT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 1: Verify server is running ──
echo "[1/4] Checking if FastAPI is running on port ${PORT}..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
if [ "${HTTP_CODE}" != "200" ]; then
    echo "  ✗ Server not responding on port ${PORT} (HTTP ${HTTP_CODE})"
    echo "  Start the server first:"
    echo "    python -u -m uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT} &"
    exit 1
fi
echo "  ✓ Server is running"

# ── Step 2: Download cloudflared if missing ──
echo "[2/4] Setting up cloudflared..."
if [ -x "${CLOUDFLARED_BIN}" ]; then
    echo "  ✓ Already cached at ${CLOUDFLARED_BIN}"
else
    echo "  Downloading..."
    mkdir -p "$(dirname "${CLOUDFLARED_BIN}")"
    curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" \
        -o "${CLOUDFLARED_BIN}"
    chmod +x "${CLOUDFLARED_BIN}"
    echo "  ✓ Downloaded to ${CLOUDFLARED_BIN}"
fi

"${CLOUDFLARED_BIN}" version 2>/dev/null | head -1 || true

# ── Step 3: Start tunnel ──
echo "[3/4] Starting tunnel..."
mkdir -p "$(dirname "${TUNNEL_LOG}")"
"${CLOUDFLARED_BIN}" tunnel --url "http://localhost:${PORT}" \
    --no-autoupdate \
    > "${TUNNEL_LOG}" 2>&1 &

TUNNEL_PID=$!
echo "  PID: ${TUNNEL_PID}"

# Wait for URL
PUBLIC_URL=""
WAITED=0
while [ $WAITED -lt 30 ]; do
    sleep 2
    WAITED=$((WAITED + 2))
    PUBLIC_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "${TUNNEL_LOG}" 2>/dev/null | head -1)
    if [ -n "${PUBLIC_URL}" ]; then
        break
    fi
    if ! kill -0 ${TUNNEL_PID} 2>/dev/null; then
        echo "  ✗ Tunnel process died. Check: ${TUNNEL_LOG}"
        exit 1
    fi
done

if [ -z "${PUBLIC_URL}" ]; then
    echo "  ✗ Could not obtain URL within 30s"
    echo "  Check: ${TUNNEL_LOG}"
    kill ${TUNNEL_PID} 2>/dev/null || true
    exit 1
fi

# ── Step 4: Print results ──
echo "[4/4] Tunnel active!"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║  🌐 PUBLIC HTTPS ENDPOINT                                  ║"
echo "║                                                            ║"
echo "║  PUBLIC_URL=${PUBLIC_URL}"
echo "║                                                            ║"
echo "║  Endpoints:                                                ║"
echo "║    GET  ${PUBLIC_URL}/health"
echo "║    GET  ${PUBLIC_URL}/ready"
echo "║    POST ${PUBLIC_URL}/query"
echo "║    GET  ${PUBLIC_URL}/docs"
echo "║                                                            ║"
echo "║  Tunnel PID: ${TUNNEL_PID}"
echo "║  Tunnel log: ${TUNNEL_LOG}"
echo "║                                                            ║"
echo "║  Stop: kill ${TUNNEL_PID}"
echo "║                                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Quick test:"
echo "  curl -s ${PUBLIC_URL}/health | python -m json.tool"
echo ""
echo "Run full test suite:"
echo "  bash scripts/test_api.sh ${PUBLIC_URL}"
echo ""

# Verify public endpoint
echo "Verifying public endpoint..."
PUB_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${PUBLIC_URL}/health" 2>/dev/null || echo "000")
if [ "${PUB_CODE}" = "200" ]; then
    echo "  ✓ Public /health returns 200"
else
    echo "  ⚠ Public /health returned ${PUB_CODE} (may need a few seconds)"
fi

echo ""
echo "Tunnel is running in the background. To stop:"
echo "  kill ${TUNNEL_PID}"
