#!/bin/bash
#SBATCH --job-name=sci_rag_100
#SBATCH --output=/scratch/data/divyasaxena_rs/%x_%j.out
#SBATCH --error=/scratch/data/divyasaxena_rs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL

# ═══════════════════════════════════════════════════════════════
#  IITJ HPC — Ingest 100 Papers (builds on top of existing 10)
#
#  Submit:  sbatch scripts/slurm_ingest_100.sh
#  Monitor: squeue -u divyasaxena_rs
#           tail -f /scratch/data/divyasaxena_rs/sci_rag_100_<ID>.out
#
#  Run this AFTER slurm_offline.sh has already completed
#  (existing 10 papers + their embeddings will be preserved)
# ═══════════════════════════════════════════════════════════════

set -e

INTERN_NAME="Vineet"
HPC_USER="divyasaxena_rs"
WORK_DIR="/scratch/data/${HPC_USER}/${INTERN_NAME}_internship"
PROJECT_DIR="${WORK_DIR}/Scientific-Multimodal-RAG"
VENV_DIR="${WORK_DIR}/rag_venv"
CACHE_DIR="${WORK_DIR}/.cache/huggingface"

echo "════════════════════════════════════════════════════════════"
echo "  JOB  : ${SLURM_JOB_NAME} [ID: ${SLURM_JOB_ID}]"
echo "  NODE : ${SLURMD_NODENAME}"
echo "  START: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load python/3.10 2>/dev/null || module load python3
module load cuda/12.1   2>/dev/null || module load cuda

source "${VENV_DIR}/bin/activate"
export HF_HOME="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${CACHE_DIR}/hub"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_DIR}"

mkdir -p data/raw data/parsed/pages data/parsed/markdown
mkdir -p data/indices/multivectors data/indices/chroma_index outputs

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

# ─────────────────────────────────────────────────────────────
# STEP 1: Download 100 papers (skips existing 10 automatically)
# ─────────────────────────────────────────────────────────────
echo ""; echo "=== STEP 1/4: Downloading 100 papers from arXiv ==="
python3 scripts/hpc_step1_download_100.py --max 100

# ─────────────────────────────────────────────────────────────
# STEP 2: Parse ONLY new PDFs (existing parsed files are skipped)
# ─────────────────────────────────────────────────────────────
echo ""; echo "=== STEP 2/4: Parsing new PDFs ==="
python3 - << 'PYEOF'
import os, json, fitz
from pathlib import Path
from pdf2image import convert_from_path

PAGES_DIR    = "data/parsed/pages"
MARKDOWN_DIR = "data/parsed/markdown"

# Load existing metadata (10 papers already parsed)
meta_path = "data/indices/page_metadata.json"
doc_map_path = "data/indices/doc_mapping.json"

page_metadata = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
doc_mapping   = json.load(open(doc_map_path)) if os.path.exists(doc_map_path) else {}

# Load download results
with open("data/download_results.json") as f:
    all_dl = [d for d in json.load(f) if d["status"] in ("success", "exists")]

# Only parse papers not already in doc_mapping
new_papers = [d for d in all_dl if d["arxiv_id"] not in doc_mapping]
print(f"  Already parsed : {len(doc_mapping)} papers")
print(f"  New to parse   : {len(new_papers)} papers")

total_new_pages = 0
for idx, item in enumerate(new_papers):
    arxiv_id = item["arxiv_id"]
    title    = item.get("title", arxiv_id)
    pdf_path = item["pdf_path"]
    print(f"\n  [{idx+1}/{len(new_papers)}] {arxiv_id}: {title[:55]}")

    try:
        doc = fitz.open(pdf_path)
        page_texts = [p.get_text("text") for p in doc]
        doc.close()
        print(f"         Text: {len(page_texts)} pages")
    except Exception as e:
        print(f"         ❌ Text failed: {e}")
        page_texts = []

    md_path = f"{MARKDOWN_DIR}/{arxiv_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\narXiv: {arxiv_id}\n\n")
        for i, t in enumerate(page_texts):
            f.write(f"## Page {i+1}\n\n{t}\n\n---\n\n")

    page_images = []
    try:
        images = convert_from_path(pdf_path, dpi=150)
        for i, img in enumerate(images):
            img_path = f"{PAGES_DIR}/{arxiv_id}_page_{i+1}.png"
            img.save(img_path, "PNG")
            page_images.append(img_path)
        print(f"         Images: {len(page_images)} pages @ 150 DPI")
    except Exception as e:
        print(f"         ⚠️  Images failed: {e}")

    doc_mapping[arxiv_id] = {
        "arxiv_id": arxiv_id, "title": title,
        "num_pages": len(page_texts), "page_images": page_images,
        "markdown_path": md_path, "status": "success",
    }
    for i in range(len(page_texts)):
        pk = f"{arxiv_id}_page_{i+1}"
        page_metadata[pk] = {
            "doc_id": arxiv_id, "page_num": i+1,
            "image_path": page_images[i] if i < len(page_images) else "",
            "text": page_texts[i] if i < len(page_texts) else "",
            "paper_title": title,
        }
    total_new_pages += len(page_texts)
    print(f"         ✅ Done")

# Save merged metadata
with open(doc_map_path, "w", encoding="utf-8") as f:
    json.dump(doc_mapping, f, indent=2, ensure_ascii=False)
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(page_metadata, f, indent=2, ensure_ascii=False)

print(f"\n  Parse done: {len(new_papers)} new papers, {total_new_pages} new pages")
print(f"  Total in index: {len(doc_mapping)} papers, {len(page_metadata)} pages")
PYEOF

# ─────────────────────────────────────────────────────────────
# STEP 3: ColPali embed ONLY new pages (existing .npy skipped)
# ─────────────────────────────────────────────────────────────
echo ""; echo "=== STEP 3/4: ColPali GPU Embedding (new pages only) ==="
python3 scripts/hpc_step3_colpali.py

# ─────────────────────────────────────────────────────────────
# STEP 4: SciNCL — upsert new pages into ChromaDB
# ─────────────────────────────────────────────────────────────
echo ""; echo "=== STEP 4/4: SciNCL upsert into ChromaDB ==="
python3 scripts/hpc_step4_scincl.py

# ─────────────────────────────────────────────────────────────
# Zip results
# ─────────────────────────────────────────────────────────────
echo ""; echo "=== Zipping results ==="
zip -r "${WORK_DIR}/sci-rag-indices-100.zip"  data/indices/ -q
zip -r "${WORK_DIR}/sci-rag-pages-100.zip"    data/parsed/  -q
echo "  ✅ sci-rag-indices-100.zip"
echo "  ✅ sci-rag-pages-100.zip"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  JOB COMPLETE: $(date) | ${SECONDS}s"
echo ""
echo "  Download results to your laptop:"
echo "  scp divyasaxena_rs@172.25.0.81:${WORK_DIR}/sci-rag-indices-100.zip ."
echo "  scp divyasaxena_rs@172.25.0.81:${WORK_DIR}/sci-rag-pages-100.zip ."
echo "════════════════════════════════════════════════════════════"
