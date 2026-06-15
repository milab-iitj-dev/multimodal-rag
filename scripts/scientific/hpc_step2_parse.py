"""STEP 2 — Parse PDFs into text + page images"""
import os, json, fitz
from pdf2image import convert_from_path

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

with open("data/download_results.json") as f:
    dl = [d for d in json.load(f) if d["status"] in ("success", "exists")]

doc_mapping, page_metadata, total_pages = {}, {}, 0

for idx, item in enumerate(dl):
    arxiv_id = item["arxiv_id"]
    title    = PAPERS.get(arxiv_id, arxiv_id)
    pdf_path = item["pdf_path"]
    print(f"\n  [{idx+1}/{len(dl)}] {arxiv_id}: {title[:50]}")

    # Text extraction
    try:
        doc = fitz.open(pdf_path)
        page_texts = [p.get_text("text") for p in doc]
        doc.close()
        print(f"         Text: {len(page_texts)} pages")
    except Exception as e:
        print(f"         ❌ Text failed: {e}")
        page_texts = []

    # Save markdown
    md_path = f"data/parsed/markdown/{arxiv_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\narXiv: {arxiv_id}\n\n")
        for i, t in enumerate(page_texts):
            f.write(f"## Page {i+1}\n\n{t}\n\n---\n\n")

    # Image extraction (150 DPI to save space on HPC scratch)
    page_images = []
    try:
        images = convert_from_path(pdf_path, dpi=150)
        for i, img in enumerate(images):
            img_path = f"data/parsed/pages/{arxiv_id}_page_{i+1}.png"
            img.save(img_path, "PNG")
            page_images.append(img_path)
        print(f"         Images: {len(page_images)} pages saved @ 150 DPI")
    except Exception as e:
        print(f"         ⚠️  Image extraction failed: {e} — text-only mode")

    doc_mapping[arxiv_id] = {
        "arxiv_id": arxiv_id, "title": title,
        "num_pages": len(page_texts), "page_images": page_images,
        "markdown_path": md_path, "status": "success",
    }
    for i in range(len(page_texts)):
        pk = f"{arxiv_id}_page_{i+1}"
        page_metadata[pk] = {
            "doc_id": arxiv_id, "page_num": i + 1,
            "image_path": page_images[i] if i < len(page_images) else "",
            "text": page_texts[i] if i < len(page_texts) else "",
            "paper_title": title,
        }
    total_pages += len(page_texts)
    print(f"         ✅ Done")

with open("data/indices/doc_mapping.json", "w", encoding="utf-8") as f:
    json.dump(doc_mapping, f, indent=2, ensure_ascii=False)
with open("data/indices/page_metadata.json", "w", encoding="utf-8") as f:
    json.dump(page_metadata, f, indent=2, ensure_ascii=False)

print(f"\n  Parse complete: {len(doc_mapping)} papers, {total_pages} total pages")
