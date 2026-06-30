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

# 7. Start public tunnel
bash scripts/start_cloudflare.sh

# 8. Run full test suite
bash scripts/test_api.sh
bash scripts/test_api.sh https://XXXXX.trycloudflare.com
```

---

## SLURM Batch Deployment

```bash
# Submit (runs everything automatically)
sbatch scripts/slurm_production.sh

# Monitor
tail -f outputs/logs/production_<JOBID>.log

# Find public URL
grep "PUBLIC_URL" outputs/logs/production_<JOBID>.log

# Cancel when done
scancel <JOBID>
```

---

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/slurm_production.sh` | Full SLURM deployment (22 steps + tunnel + stability) | `sbatch scripts/slurm_production.sh` |
| `scripts/test_api.sh` | Standalone API test (11 tests, 3 retrieval modes) | `bash scripts/test_api.sh [URL]` |
| `scripts/start_cloudflare.sh` | Standalone tunnel launcher | `bash scripts/start_cloudflare.sh [PORT]` |

---

## Cloudflare vs ngrok

| Feature | Cloudflare Quick Tunnel | ngrok |
|---------|------------------------|-------|
| Account required | **No** | Yes (signup + auth token) |
| Personal info | **None** | Email logged |
| HTTPS | **Automatic** | Automatic |
| Setup | Single binary, zero config | Requires auth token setup |
| HPC friendly | **Yes** — outbound only | May need config file |
| Privacy | **No tracking** | Requests logged |

**Decision: Cloudflare** — zero account, zero personal info, automatic HTTPS.

---

## Professor-Ready Curl Commands

### Localhost (from GPU node)

```bash
# Health check
curl -s http://localhost:8847/health | python -m json.tool

# Readiness check
curl -s http://localhost:8847/ready | python -m json.tool

# Healthcare query
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3}' \
  | python -m json.tool

# Auto-routing query
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Is there pleural effusion in this chest X-ray?","domain":"auto","top_k":3}' \
  | python -m json.tool
```

### Public URL (from anywhere)

```bash
# Replace XXXXX with actual subdomain from tunnel output
PUBLIC_URL=https://XXXXX.trycloudflare.com

curl -s $PUBLIC_URL/health | python -m json.tool
curl -s $PUBLIC_URL/ready | python -m json.tool

curl -s -X POST $PUBLIC_URL/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3}' \
  | python -m json.tool
```

### OpenAPI Documentation

Open in browser: `https://XXXXX.trycloudflare.com/docs`

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

### POST /query
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
| Tunnel URL not appearing | Firewall blocks outbound | Try `curl -s https://cloudflare.com` |
| `cloudflared` download fails | No internet on compute node | Download on login node: `curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o .local/bin/cloudflared && chmod +x .local/bin/cloudflared` |
| 500 error on `/query` | Check server logs | `cat outputs/logs/production_*.err` |

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
- [ ] `POST /query` returns real answer (not placeholder)
- [ ] Sources non-empty
- [ ] Retrieval scores populated
- [ ] Verification fields populated
- [ ] Latency reported
- [ ] Cloudflare tunnel active
- [ ] Public URL obtained
- [ ] Public `/health` → 200
- [ ] Public `/query` → real answer
- [ ] Stability test: 9 queries × 3 rounds pass
- [ ] Server stable (no OOM, no crashes)
