# MMRAG Unified API — Curl Command Reference

> **Server**: `http://localhost:8847` (default HPC port)
>
> Replace `localhost:8847` with your actual server address.

---

## Table of Contents

1. [System Endpoints](#1-system-endpoints)
2. [Healthcare — Text-only Queries](#2-healthcare--text-only-queries)
3. [Healthcare — Image-only Queries](#3-healthcare--image-only-queries)
4. [Healthcare — Hybrid Queries](#4-healthcare--hybrid-queries)
5. [Scientific Queries](#5-scientific-queries)
6. [Auto-routing Queries](#6-auto-routing-queries)
7. [Response Formatting](#7-response-formatting)

---

## 1. System Endpoints

### Health Check

**Linux / macOS:**
```bash
curl -s http://localhost:8847/health | python3 -m json.tool
```

**Windows CMD:**
```cmd
curl -s http://localhost:8847/health
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:8847/health | ConvertTo-Json
```

### Readiness Check

**Linux / macOS:**
```bash
curl -s http://localhost:8847/ready | python3 -m json.tool
```

**Windows CMD:**
```cmd
curl -s http://localhost:8847/ready
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:8847/ready | ConvertTo-Json -Depth 5
```

### OpenAPI Schema

**Linux / macOS:**
```bash
curl -s http://localhost:8847/openapi.json | python3 -m json.tool > openapi.json
```

**Windows CMD:**
```cmd
curl -s http://localhost:8847/openapi.json -o openapi.json
```

### Interactive API Docs

Open in browser:
```
http://localhost:8847/docs     # Swagger UI
http://localhost:8847/redoc    # ReDoc
```

---

## 2. Healthcare — Text-only Queries

Text-only queries send only a text question with no image.

**Expected response**: `retrieval_metadata.method = "scincl_only"`, only `scores.scincl > 0`.

### Query: "What is cardiomegaly?"

**Linux / macOS:**
```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is cardiomegaly?",
    "domain": "healthcare",
    "top_k": 3,
    "include_images": true
  }' | python3 -m json.tool
```

**Windows CMD:**
```cmd
curl -s -X POST http://localhost:8847/query -H "Content-Type: application/json" -d "{\"query\":\"What is cardiomegaly?\",\"domain\":\"healthcare\",\"top_k\":3,\"include_images\":true}"
```

**PowerShell:**
```powershell
$body = @{
    query = "What is cardiomegaly?"
    domain = "healthcare"
    top_k = 3
    include_images = $true
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri http://localhost:8847/query `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 10
```

### Query: "What is pneumothorax?"

**Linux / macOS:**
```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is pneumothorax and how is it identified on chest X-ray?",
    "domain": "healthcare",
    "top_k": 3,
    "include_images": true
  }' | python3 -m json.tool
```

### Query: "Explain pleural effusion."

**Linux / macOS:**
```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain pleural effusion and its radiographic findings.",
    "domain": "healthcare",
    "top_k": 3,
    "include_images": true
  }' | python3 -m json.tool
```

---

## 3. Healthcare — Image-only Queries

Image queries include an `image_path` pointing to a chest X-ray on disk.

**Expected response**: `retrieval_metadata.method = "colpali_only"`, only `scores.colpali > 0`.

### Query with image

**Linux / macOS:**
```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Retrieve visually similar chest X-rays.",
    "domain": "healthcare",
    "top_k": 3,
    "include_images": true,
    "image_path": "data/openi/images/1_IM-0001-4001.dcm.png"
  }' | python3 -m json.tool
```

**Windows CMD:**
```cmd
curl -s -X POST http://localhost:8847/query -H "Content-Type: application/json" -d "{\"query\":\"Retrieve visually similar chest X-rays.\",\"domain\":\"healthcare\",\"top_k\":3,\"include_images\":true,\"image_path\":\"data/openi/images/1_IM-0001-4001.dcm.png\"}"
```

**PowerShell:**
```powershell
$body = @{
    query = "Retrieve visually similar chest X-rays."
    domain = "healthcare"
    top_k = 3
    include_images = $true
    image_path = "data/openi/images/1_IM-0001-4001.dcm.png"
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri http://localhost:8847/query `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 10
```

> **Note**: `image_path` can be relative (to CWD) or absolute.
> On HPC, images are at `data/openi/images/*.dcm.png`.

---

## 4. Healthcare — Hybrid Queries

Hybrid queries combine a clinical question with a chest X-ray image.

**Expected response**: `retrieval_metadata.method = "fused"`, all three scores > 0.

### Query: Cardiomegaly + Image

**Linux / macOS:**
```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Does this chest X-ray show cardiomegaly?",
    "domain": "healthcare",
    "top_k": 3,
    "include_images": true,
    "image_path": "data/openi/images/1_IM-0001-4001.dcm.png"
  }' | python3 -m json.tool
```

### Query: Pleural Effusion + Image

**Linux / macOS:**
```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Is pleural effusion visible in this chest X-ray?",
    "domain": "healthcare",
    "top_k": 3,
    "include_images": true,
    "image_path": "data/openi/images/2_IM-0002-1001.dcm.png"
  }' | python3 -m json.tool
```

---

## 5. Scientific Queries

Scientific queries are text-only, targeting the 10-paper Vision Transformer corpus.

**Expected response**: `retrieval_metadata.method = "fused"`, ColPali, SciNCL, and fused scores all > 0.

### Query: Vision Transformer

**Linux / macOS:**
```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Vision Transformer (ViT) architecture and how does it process images?",
    "domain": "scientific",
    "top_k": 3,
    "include_images": true
  }' | python3 -m json.tool
```

**Windows CMD:**
```cmd
curl -s -X POST http://localhost:8847/query -H "Content-Type: application/json" -d "{\"query\":\"What is the Vision Transformer (ViT) architecture and how does it process images?\",\"domain\":\"scientific\",\"top_k\":3,\"include_images\":true}"
```

### Query: DeiT

```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does DeiT achieve competitive accuracy without large-scale pretraining datasets?",
    "domain": "scientific",
    "top_k": 3
  }' | python3 -m json.tool
```

### Query: Swin Transformer

```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key differences between Swin Transformer and standard ViT?",
    "domain": "scientific",
    "top_k": 3
  }' | python3 -m json.tool
```

### Query: EfficientFormer

```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does EfficientFormer achieve MobileNet-level inference speed with transformer accuracy?",
    "domain": "scientific",
    "top_k": 3
  }' | python3 -m json.tool
```

### Query: ConvNeXt

```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does ConvNeXt modernize convolutional networks to compete with vision transformers?",
    "domain": "scientific",
    "top_k": 3
  }' | python3 -m json.tool
```

---

## 6. Auto-routing Queries

Auto-routing uses `"domain": "auto"` (or omits the domain field).
The DomainRouter detects the target domain from query keywords.

### Healthcare auto-detection

```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is cardiomegaly and how is it diagnosed?",
    "domain": "auto",
    "top_k": 3
  }' | python3 -m json.tool
```

### Scientific auto-detection

```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Vision Transformer architecture?",
    "domain": "auto",
    "top_k": 3
  }' | python3 -m json.tool
```

### Ambiguous query (tests tie-breaking)

```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What medical imaging techniques use transformer architectures?",
    "domain": "auto",
    "top_k": 3
  }' | python3 -m json.tool
```

**PowerShell (auto-routing):**
```powershell
$body = @{
    query = "What is the Vision Transformer architecture?"
    domain = "auto"
    top_k = 3
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri http://localhost:8847/query `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 10
```

---

## 7. Response Formatting

### Pretty-print with jq (Linux)

```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3}' \
  | jq .
```

### Extract specific fields

```bash
# Answer only
curl -s ... | jq -r '.answer'

# Retrieval scores only
curl -s ... | jq '.retrieval_metadata.scores'

# Source titles
curl -s ... | jq '[.sources[].title]'

# Verification results
curl -s ... | jq '.verification'
```

### Save response to file

```bash
curl -s -X POST http://localhost:8847/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is ViT?","domain":"scientific","top_k":3}' \
  -o response.json
```

---

## Response Schema

Every `/query` response follows this frozen contract:

```json
{
  "answer": "string",
  "confidence": 0.0,
  "sources": [
    {
      "doc_id": "string",
      "page": 0,
      "title": "string",
      "relevance_score": 0.0,
      "snippet": "string"
    }
  ],
  "retrieval_metadata": {
    "method": "fused | colpali_only | scincl_only",
    "scores": {
      "colpali": 0.0,
      "scincl": 0.0,
      "fused": 0.0
    }
  },
  "verification": {
    "attribution": true,
    "faithfulness": true,
    "confidence_pass": true
  },
  "latency_ms": 0
}
```

---

*Generated for MMRAG Unified v2.0.0*
