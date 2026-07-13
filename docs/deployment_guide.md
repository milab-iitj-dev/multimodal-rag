# MMRAG Unified — Production Deployment Guide

## Prerequisites

| Requirement | Value |
|-------------|-------|
| HPC partition | `dgx` |
| GPU | NVIDIA A100-SXM4-40GB |
| Driver | 555.42.02 (CUDA ≤ 12.5) |
| Python | 3.11.7 |
| PyTorch | `2.5.1+cu124` (**not** cu130) |
| Virtual env | `mmrag_unified/.venv` |

---

## Quick Start (Interactive Session)

```bash
# 1. Get a GPU
salloc -p dgx --gres=gpu:1 --cpus-per-task=4 --mem-per-cpu=8192 --time=04:00:00
ssh gpu1

# 2. Setup
cd /scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified
source .venv/bin/activate
export PATH="$VIRTUAL_ENV/bin:$PATH"
hash -r

# 3. Pull latest code
git pull origin main

# 4. Verify CUDA
python -c "import torch; print(f'CUDA={torch.cuda.is_available()} GPU={torch.cuda.get_device_name(0)}')"
# Expected: CUDA=True GPU=NVIDIA A100-SXM4-40GB

# 5. Start the API
python -u -m uvicorn src.api.app:app --host 0.0.0.0 --port 8847 --log-level info &

# 6. Wait ~5-10 minutes for model loading, then test
curl -s http://localhost:8847/ready | python -m json.tool

# 7. Run full test suite
bash scripts/test_api.sh

# 8. Generate API examples
python tools/generate_api_examples.py --server http://localhost:8847 --no-wait
```

---

## SLURM Batch Deployment

```bash
# Submit production deployment (runs everything automatically)
sbatch scripts/slurm_production.sh

# Submit API example generation
sbatch scripts/slurm_generate_examples.sh

# Monitor
tail -f outputs/logs/production_<JOBID>.log
tail -f outputs/logs/gen_examples_<JOBID>.log

# Cancel when done
scancel <JOBID>
```

---

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/slurm_production.sh` | Full SLURM deployment (env + server + tests + keep-alive) | `sbatch scripts/slurm_production.sh` |
| `scripts/slurm_generate_examples.sh` | Generate verified API examples | `sbatch scripts/slurm_generate_examples.sh` |
| `scripts/test_api.sh` | Standalone API test (11 tests, 3 retrieval modes) | `bash scripts/test_api.sh [URL]` |
| `scripts/start_cloudflare.sh` | Standalone tunnel launcher (currently blocked) | `bash scripts/start_cloudflare.sh [PORT]` |
| `tools/generate_api_examples.py` | Permanent API example generator (Python) | `python tools/generate_api_examples.py` |

---

## Public Deployment Status

| Component | Status |
|-----------|--------|
| Local FastAPI deployment | ✅ Working |
| Healthcare pipeline | ✅ Working |
| Hybrid retrieval (ColQwen2 dual-index + RRF) | ✅ Working |
| Qwen2-VL generation | ✅ Working |
| Image input (multimodal queries) | ✅ Working |
| API validation (text / image / hybrid) | ✅ Working |
| SLURM deployment | ✅ Working |
| Cloudflare Tunnel | ❌ Blocked by HPC firewall |

### HPC Network Limitation

The HPC cluster blocks outbound QUIC/TCP traffic on Cloudflare Tunnel port (7844), preventing `cloudflared` from maintaining a persistent connection to Cloudflare Edge.

**Diagnostics:**

| Test | Result |
|------|--------|
| DNS Resolution | ✅ PASS |
| Cloudflare API (HTTPS) | ✅ PASS |
| UDP Connectivity (port 7844) | ❌ FAIL |
| TCP Connectivity (port 7844) | ❌ FAIL |

This is an **infrastructure restriction**, not an application issue. The Cloudflare Tunnel implementation is preserved in `scripts/slurm_production.sh` (commented out) and `scripts/start_cloudflare.sh` for future use.

### Planned Fallback Options

1. **SSH Port Forwarding** (preferred) — Forward HPC port to local machine via SSH tunnel. Zero infrastructure changes required.
2. **VPN-only demonstration** — Access the API directly from within the HPC network via VPN.
3. **Administrator-approved reverse proxy** — Request HPC admins to set up an authorized reverse proxy.
4. **Named Cloudflare Tunnel** — If HPC networking permits outbound port 7844 in the future, re-enable the existing Cloudflare implementation.

> **Note:** None of the fallback options are implemented yet. The API is fully functional on `localhost:8847` within the HPC node.

---

## API Image Support (Multimodal Queries)

The `/query` endpoint supports true multimodal retrieval via the `image_path` field:

```json
{
    "query": "Does this chest X-ray show cardiomegaly?",
    "domain": "healthcare",
    "image_path": "data/openi/images/1_IM-0001-4001.dcm.png",
    "top_k": 3
}
```

When `image_path` is provided:
1. The image is loaded and encoded by **ColQwen2** for visual similarity retrieval
2. Both image index and text index are queried (dual-index retrieval)
3. Results are fused using **Reciprocal Rank Fusion (RRF)**
4. Question-aware reranking is applied
5. **Qwen2-VL** generates the answer using both the query image and retrieved context

Without `image_path`, the query uses text-only retrieval.

---

## Professor-Ready Curl Commands

### Localhost (from GPU node)

```bash
# Health check
curl -s http://localhost:8847/health | python -m json.tool

# Readiness check
curl -s http://localhost:8847/ready | python -m json.tool

# Text-only healthcare query
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3}' \
  | python -m json.tool

# Multimodal query (image + text)
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Does this chest X-ray show cardiomegaly?","domain":"healthcare","image_path":"data/openi/images/1_IM-0001-4001.dcm.png","top_k":3}' \
  | python -m json.tool

# Auto-routing query
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Is there pleural effusion in this chest X-ray?","domain":"auto","top_k":3}' \
  | python -m json.tool
```

### OpenAPI Documentation

Open in browser (from HPC node): `http://localhost:8847/docs`

---

## Expected Responses

### GET /health
```json
{"status":"healthy","service":"mmrag-unified","version":"2.0.0"}
```

### GET /ready
```json
{"ready":true,"domains":["healthcare"],"detail":"Healthcare pipeline LIVE"}
```

### POST /query (text-only)
```json
{
  "answer": "Cardiomegaly refers to an enlarged heart...",
  "confidence": 0.82,
  "sources": [
    {"doc_id": "...", "page": 0, "title": "...", "relevance_score": 0.87, "snippet": "..."}
  ],
  "retrieval_metadata": {
    "method": "fused",
    "scores": {"colpali": 0.87, "scincl": 0.71, "fused": 0.82}
  },
  "verification": {
    "attribution": true,
    "faithfulness": true,
    "confidence_pass": true
  },
  "latency_ms": 3200
}
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `torch.cuda.is_available()` = False | Wrong torch build (cu130) | `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124` |
| `Pipeline not loaded` in answer | Index not found via symlink | Check `ls -la data/indexes/colqwen2_index/document_store.json` |
| Pydantic `ValidationError` on snippet | Source has `snippet=None` | Fixed in commit `893d2aa` — `git pull` |
| Server dies during startup | OOM | Check `--mem-per-cpu=8192` and GPU memory |
| Cloudflare Tunnel fails | HPC blocks outbound port 7844 | Use SSH port forwarding instead (see Planned Fallback Options) |
| `cloudflared` download fails | No internet on compute node | Download on login node: `curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o .local/bin/cloudflared && chmod +x .local/bin/cloudflared` |
| 500 error on `/query` | Check server logs | `cat outputs/logs/production_*.err` |
| `Image not found` on `/query` | Invalid `image_path` | Ensure the path is relative to project root or absolute. Check `ls data/openi/images/` |

---

## Deployment Checklist

- [ ] GPU allocated (`salloc` or `sbatch`)
- [ ] `torch==2.5.1+cu124` installed
- [ ] `torch.cuda.is_available()` = True
- [ ] Healthcare index symlinked (`data/indexes/colqwen2_index`)
- [ ] OpenI data symlinked (`data/openi`)
- [ ] FastAPI server started on port 8847
- [ ] `GET /health` → `{"status":"healthy"}`
- [ ] `GET /ready` → `{"ready":true,...,"detail":"Healthcare pipeline LIVE"}`
- [ ] `POST /query` (text) returns real answer (not placeholder)
- [ ] `POST /query` (image) with `image_path` returns real answer
- [ ] `POST /query` (hybrid) image + text returns real answer
- [ ] Sources non-empty
- [ ] Retrieval scores populated
- [ ] Verification fields populated
- [ ] Latency reported
- [ ] ~~Cloudflare tunnel active~~ (BLOCKED — HPC firewall)
- [ ] ~~Public URL obtained~~ (BLOCKED — HPC firewall)
- [ ] API examples generated (`python tools/generate_api_examples.py`)
- [ ] Server stable (no OOM, no crashes)
