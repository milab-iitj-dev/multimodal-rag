"""
STEP 1 (Extended) — Download up to 100 arXiv PDFs for Scientific RAG
Searches arXiv for Vision Transformer + related papers.
Already-downloaded papers (the original 10) are automatically skipped.

Usage:
    python scripts/hpc_step1_download_100.py
    python scripts/hpc_step1_download_100.py --max 50
    python scripts/hpc_step1_download_100.py --query "diffusion model image generation" --max 30
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Download arXiv PDFs for RAG ingestion")
parser.add_argument("--max",   type=int, default=100, help="Max papers to download (default: 100)")
parser.add_argument("--query", type=str, default=None, help="Custom arXiv search query")
parser.add_argument("--outdir", type=str, default="data/raw", help="Output directory for PDFs")
args = parser.parse_args()

RAW_DIR = args.outdir
Path(RAW_DIR).mkdir(parents=True, exist_ok=True)

# ── The original 10 fixed papers (always included) ───────────────────────────
FIXED_PAPERS = {
    "2010.11929": "An Image is Worth 16x16 Words",
    "2106.10270": "Training data-efficient image transformers",
    "2205.01580": "A Battle of Architectures ViT ResNet 3D Medical",
    "2108.00102": "DeepViT: Towards Deeper Vision Transformer",
    "2203.14465": "EfficientFormer: Vision Transformers at MobileNet Speed",
    "2211.11505": "Scaling Vision Transformers to 22 Billion Parameters",
    "2303.15326": "A Survey on Vision Transformer",
    "2106.14881": "Swin Transformer V2",
    "2111.09883": "Aggregating Nested Transformers",
    "2204.00636": "Egocentric Video Using Vision Transformers",
}

# ── arXiv search queries to gather 100 papers ────────────────────────────────
# These span Vision Transformers broadly for rich multimodal RAG coverage.
SEARCH_QUERIES = [
    "vision transformer image classification",
    "vision transformer object detection",
    "vision transformer semantic segmentation",
    "vision transformer medical image",
    "vision transformer self-supervised",
    "vision transformer contrastive learning",
    "vision transformer efficient lightweight",
    "swin transformer hierarchical",
    "deformable attention transformer",
    "multimodal transformer vision language",
]

# Use custom query if provided
if args.query:
    SEARCH_QUERIES = [args.query]

# ── Helper: query arXiv API ───────────────────────────────────────────────────
def search_arxiv(query: str, max_results: int = 20) -> list[dict]:
    """Search arXiv API and return list of {arxiv_id, title} dicts."""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"ti:{query} OR abs:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        r = requests.get(base_url, params=params, timeout=30)
        if r.status_code != 200:
            print(f"  ⚠️  arXiv API returned {r.status_code} for query: {query}")
            return []
        # Parse Atom XML (simple regex-free approach)
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        results = []
        for entry in root.findall("atom:entry", ns):
            id_tag = entry.find("atom:id", ns)
            title_tag = entry.find("atom:title", ns)
            if id_tag is None or title_tag is None:
                continue
            full_id = id_tag.text.strip()
            # Extract arxiv_id like "2010.11929" from the URL
            arxiv_id = full_id.split("/abs/")[-1].split("v")[0].strip()
            title    = " ".join(title_tag.text.strip().split())
            if arxiv_id:
                results.append({"arxiv_id": arxiv_id, "title": title})
        return results
    except Exception as e:
        print(f"  ❌ arXiv API error: {e}")
        return []


# ── Helper: download one PDF ──────────────────────────────────────────────────
def download_pdf(arxiv_id: str, title: str, raw_dir: str) -> dict:
    pdf_path = f"{raw_dir}/{arxiv_id}.pdf"
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 10000:
        return {"arxiv_id": arxiv_id, "title": title, "status": "exists", "pdf_path": pdf_path}
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.content) > 10000:
            with open(pdf_path, "wb") as f:
                f.write(r.content)
            return {"arxiv_id": arxiv_id, "title": title, "status": "success",
                    "pdf_path": pdf_path, "size_mb": round(len(r.content)/1e6, 2)}
        else:
            return {"arxiv_id": arxiv_id, "title": title, "status": "failed",
                    "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"arxiv_id": arxiv_id, "title": title, "status": "failed", "error": str(e)}


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print(f"  Scientific RAG — Extended Download ({args.max} papers)")
print(f"  Output: {RAW_DIR}")
print("=" * 65)

# Load existing download results if any
results_path = "data/download_results.json"
if os.path.exists(results_path):
    with open(results_path) as f:
        existing_results = {r["arxiv_id"]: r for r in json.load(f)}
    print(f"\n  Existing downloads: {len(existing_results)}")
else:
    existing_results = {}

# Collect paper candidates
all_candidates: dict[str, str] = dict(FIXED_PAPERS)  # arxiv_id -> title

print(f"\n  Searching arXiv for more papers ({len(SEARCH_QUERIES)} queries)...")
for q_idx, query in enumerate(SEARCH_QUERIES):
    if len(all_candidates) >= args.max + 20:  # buffer to account for failures
        break
    print(f"  [{q_idx+1}/{len(SEARCH_QUERIES)}] Query: \"{query}\"")
    found = search_arxiv(query, max_results=20)
    new_count = 0
    for item in found:
        aid = item["arxiv_id"]
        if aid not in all_candidates:
            all_candidates[aid] = item["title"]
            new_count += 1
    print(f"         → Found {len(found)}, added {new_count} new (total: {len(all_candidates)})")
    time.sleep(2)  # Be polite to arXiv API

print(f"\n  Total candidates: {len(all_candidates)}")
print(f"  Target downloads: {args.max}")
print("\n" + "=" * 65)

# Download
results = list(existing_results.values())
downloaded_ids = set(existing_results.keys())
success_count  = sum(1 for r in results if r["status"] in ("success", "exists"))
fail_count     = 0

papers_list = list(all_candidates.items())
i = 0
for arxiv_id, title in papers_list:
    if success_count >= args.max:
        print(f"\n  ✅ Reached target of {args.max} papers!")
        break

    if arxiv_id in downloaded_ids:
        continue  # already have it

    i += 1
    print(f"\n  [{i}] Downloading {arxiv_id}: {title[:55]}")
    result = download_pdf(arxiv_id, title, RAW_DIR)

    if result["status"] == "success":
        print(f"         ✅ {result.get('size_mb', '?')} MB")
        success_count += 1
    elif result["status"] == "exists":
        print(f"         ⏭  Already exists")
        success_count += 1
    else:
        print(f"         ❌ {result.get('error', 'unknown error')}")
        fail_count += 1

    results.append(result)
    downloaded_ids.add(arxiv_id)

    # Rate limit — arXiv requires ~3s between requests
    time.sleep(3)

# Save merged results
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 65)
print(f"  DOWNLOAD COMPLETE")
print(f"  Success : {success_count}")
print(f"  Failed  : {fail_count}")
print(f"  Results : {results_path}")
print("=" * 65)
