"""STEP 1 — Download arXiv PDFs"""
import requests, time, os, json

PAPERS = {
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

RAW_DIR = "data/raw"
results = []

for i, (arxiv_id, title) in enumerate(PAPERS.items()):
    pdf_path = f"{RAW_DIR}/{arxiv_id}.pdf"
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 10000:
        print(f"  [{i+1}/10] SKIP (exists): {arxiv_id}")
        results.append({"arxiv_id": arxiv_id, "title": title, "status": "exists", "pdf_path": pdf_path})
        continue
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"  [{i+1}/10] Downloading {arxiv_id}: {title[:55]}")
    try:
        r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.content) > 10000:
            with open(pdf_path, "wb") as f:
                f.write(r.content)
            print(f"         ✅ {len(r.content)/1e6:.1f} MB")
            results.append({"arxiv_id": arxiv_id, "title": title, "status": "success", "pdf_path": pdf_path})
        else:
            print(f"         ❌ HTTP {r.status_code}")
            results.append({"arxiv_id": arxiv_id, "title": title, "status": "failed"})
    except Exception as e:
        print(f"         ❌ {e}")
        results.append({"arxiv_id": arxiv_id, "title": title, "status": "failed"})
    time.sleep(3)

with open("data/download_results.json", "w") as f:
    json.dump(results, f, indent=2)

ok = sum(1 for r in results if r["status"] in ("success", "exists"))
print(f"\n  Download complete: {ok}/{len(PAPERS)} papers")
