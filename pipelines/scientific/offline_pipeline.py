"""
Offline Indexing Pipeline Orchestrator
======================================
Coordinates downloading, parsing, embedding, and indexing of research papers.
"""

from __future__ import annotations

import os
import json
import time
import requests
from src.domains.scientific.utils.helpers import clean_vram, ensure_directories, create_zip_archive
from src.domains.scientific.models.loader import load_colpali, load_scincl
from src.data.pdf_parser import PDFParser
from src.domains.scientific.embeddings.colpali_embedder import ColPaliEmbedder
from src.domains.scientific.embeddings.scincl_embedder import SciNCLEmbedder

class OfflinePipeline:
    """Orchestrates end-to-end paper ingestion and indexing."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.papers = cfg.get("papers", {})
        self.parsing = cfg.get("parsing", {})
        self.retrieval = cfg.get("retrieval", {})

        # Resolve paths with RAG_BASE_DIR override on HPC
        self.base = os.getenv("RAG_BASE_DIR", "")
        
        # Build path dictionary
        self.resolved_paths = {}
        for name, path in self.paths.items():
            self.resolved_paths[name] = os.path.join(self.base, path) if self.base else path

    def run(self) -> bool:
        """Executes the complete offline ingestion and indexing pipeline."""
        print("\n" + "=" * 60)
        print("  STARTING OFFLINE PIPELINE")
        print("=" * 60)

        # 1. Setup Directories
        ensure_directories(self.resolved_paths)

        # 2. Download PDFs
        downloaded_papers = self._download_papers()
        if not downloaded_papers:
            print("  ❌ No papers downloaded or verified. Exiting.")
            return False

        # 3. Parse PDFs
        doc_mapping, page_metadata = self._parse_papers(downloaded_papers)
        if not page_metadata:
            print("  ❌ No pages parsed. Exiting.")
            return False

        # 4. Generate ColPali Visual Embeddings
        self._generate_colpali_embeddings(page_metadata)

        # 5. Generate SciNCL Text Embeddings & Store in ChromaDB
        self._generate_scincl_embeddings(page_metadata)

        # 6. Save Summary Statistics
        self._save_summary(doc_mapping, page_metadata)

        # 7. Create Zip Archives
        self._create_zip_archives()

        print("\n" + "=" * 60)
        print("  ✅ OFFLINE PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        return True

    def _download_papers(self) -> list[dict]:
        """Downloads PDFs from arXiv to raw folder."""
        raw_dir = self.resolved_paths["raw"]
        download_results = []

        print(f"\n[Step 1/6] Downloading {len(self.papers)} arXiv PDFs...")
        for i, (arxiv_id, title) in enumerate(self.papers.items()):
            pdf_path = os.path.join(raw_dir, f"{arxiv_id}.pdf")
            
            # Skip if already exists
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 10000:
                print(f"  [{i+1:2d}/{len(self.papers)}] Skip (exists): {arxiv_id}")
                download_results.append({
                    "arxiv_id": arxiv_id, "title": title,
                    "status": "exists", "pdf_path": pdf_path
                })
                continue

            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            print(f"  [{i+1:2d}/{len(self.papers)}] Downloading: {arxiv_id}...")
            try:
                r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200 and len(r.content) > 10000:
                    with open(pdf_path, "wb") as f:
                        f.write(r.content)
                    print(f"           ✅ Saved ({len(r.content)/1e6:.1f} MB)")
                    download_results.append({
                        "arxiv_id": arxiv_id, "title": title,
                        "status": "success", "pdf_path": pdf_path
                    })
                else:
                    print(f"           ❌ HTTP {r.status_code} or small size")
            except Exception as e:
                print(f"           ❌ Download failed: {e}")
            
            # Rate limit politeness
            time.sleep(3)

        # Save logs
        with open(os.path.join(self.resolved_paths["indices"], "download_results.json"), "w") as f:
            json.dump(download_results, f, indent=2)

        return [d for d in download_results if d["status"] in ["success", "exists"]]

    def _parse_papers(self, downloaded_list: list[dict]) -> tuple[dict, dict]:
        """Parses PDFs into texts and high-resolution page images."""
        print(f"\n[Step 2/6] Parsing {len(downloaded_list)} PDFs (PyMuPDF + pdf2image)...")
        doc_mapping = {}
        page_metadata = {}
        dpi = self.parsing.get("dpi", 200)

        for i, dl in enumerate(downloaded_list):
            arxiv_id = dl["arxiv_id"]
            title = dl["title"]
            pdf_path = dl["pdf_path"]

            print(f"  [{i+1:2d}/{len(downloaded_list)}] Parsing: {arxiv_id}...")
            doc_info, page_chunk = PDFParser.build_metadata(pdf_path, arxiv_id, title, dpi, self.resolved_paths)
            
            doc_mapping[arxiv_id] = doc_info
            page_metadata.update(page_chunk)
            print(f"           ✅ Extracted {doc_info['num_pages']} pages")

        # Save metadata
        with open(os.path.join(self.resolved_paths["indices"], "doc_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(doc_mapping, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.resolved_paths["indices"], "page_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(page_metadata, f, indent=2, ensure_ascii=False)

        return doc_mapping, page_metadata

    def _generate_colpali_embeddings(self, page_metadata: dict):
        """Loads ColPali and embeds page images to .npy files."""
        print("\n[Step 3/6] Building ColPali Visual Embeddings...")
        npy_dir = self.resolved_paths["multivectors"]

        # Check existing embeddings
        existing = set(
            f.replace(".npy", "")
            for f in os.listdir(npy_dir)
            if f.endswith(".npy") and not f.endswith(".meta.npy")
        )

        to_embed = {}
        for pk, meta in page_metadata.items():
            if pk in existing:
                continue
            img_path = meta.get("image_path", "")
            if img_path and os.path.exists(img_path):
                to_embed[pk] = img_path

        print(f"  Already embedded: {len(existing)} pages")
        print(f"  To embed:         {len(to_embed)} pages")

        if to_embed:
            model_cfg = self.cfg.get("models", {}).get("colpali", "vidore/colpali-v1.2")
            device = "cuda"
            
            # Load
            print(f"  Loading ColPali...")
            model, processor = load_colpali(model_cfg, device=device)
            
            # Embed
            def progress(c, t, msg):
                print(f"  [{c}/{t}] {msg}")

            ColPaliEmbedder.batch_embed(to_embed, npy_dir, model, processor, status_callback=progress)
            
            # Unload
            print("  Unloading ColPali model...")
            del model, processor
            clean_vram()

        print("  ✅ ColPali visual embeddings complete!")

    def _generate_scincl_embeddings(self, page_metadata: dict):
        """Loads SciNCL and embeds texts into ChromaDB."""
        print("\n[Step 4/6] Building SciNCL Text Embeddings & ChromaDB Index...")
        chroma_dir = self.resolved_paths["chroma_index"]
        collection_name = self.retrieval.get("chroma_collection", "sci_rag_pages")

        model_cfg = self.cfg.get("models", {}).get("scincl", "malteos/scincl")
        device = "cuda"

        # Load
        print(f"  Loading SciNCL model...")
        model = load_scincl(model_cfg, device=device)

        # Index
        def progress(c, t, msg):
            print(f"  [{c}/{t}] {msg}")

        SciNCLEmbedder.embed_text_to_chroma(
            page_metadata,
            chroma_dir,
            collection_name,
            model,
            batch_size=32,
            status_callback=progress
        )

        # Unload
        print("  Unloading SciNCL model...")
        del model
        clean_vram()

        print("  ✅ SciNCL text embeddings complete!")

    def _save_summary(self, doc_mapping: dict, page_metadata: dict):
        """Saves overall indexing metadata summary.json."""
        print("\n[Step 5/6] Creating Index Summary Statistics...")
        npy_dir = self.resolved_paths["multivectors"]
        npy_files = [f for f in os.listdir(npy_dir) if f.endswith(".npy") and not f.endswith(".meta.npy")]

        total_pages = sum(d["num_pages"] for d in doc_mapping.values())
        npy_size_mb = sum(
            os.path.getsize(os.path.join(npy_dir, f)) / 1024**2
            for f in npy_files
        )

        summary = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_papers": len(doc_mapping),
            "total_pages": total_pages,
            "colpali_files": len(npy_files),
            "colpali_size_mb": round(npy_size_mb, 2),
            "chroma_entries": len(page_metadata),
            "models_used": {
                "colpali": self.cfg.get("models", {}).get("colpali", {}).get("model_name", "vidore/colpali-v1.2"),
                "scincl": self.cfg.get("models", {}).get("scincl", {}).get("model_name", "malteos/scincl")
            },
            "papers": {
                arxiv_id: {
                    "title": info["title"],
                    "num_pages": info["num_pages"]
                }
                for arxiv_id, info in doc_mapping.items()
            }
        }

        with open(os.path.join(self.resolved_paths["indices"], "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Summary saved -> {os.path.join(self.resolved_paths['indices'], 'summary.json')}")
        print(f"  Ingested {summary['num_papers']} papers ({summary['total_pages']} pages total)")

    def _create_zip_archives(self):
        """Packages indices and parsed directories as downloadable zip artifacts."""
        print("\n[Step 6/6] Packaging Zip Archives...")
        
        # Zip indices
        indices_dir = self.resolved_paths["indices"]
        indices_zip = os.path.join(self.base if self.base else ".", "data", "sci-rag-indices.zip")
        print(f"  Creating {indices_zip}...")
        create_zip_archive(indices_dir, indices_zip)

        # Zip pages
        parsed_dir = os.path.join(self.base if self.base else ".", "data", "parsed")
        pages_zip = os.path.join(self.base if self.base else ".", "data", "sci-rag-pages.zip")
        print(f"  Creating {pages_zip}...")
        create_zip_archive(parsed_dir, pages_zip)

        print("  ✅ Zip archives created!")
