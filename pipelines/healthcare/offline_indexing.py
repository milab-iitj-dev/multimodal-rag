"""
Offline Indexing Pipeline — Phase 3

Runs the complete offline knowledge-base preparation:
    1. Load OpenI dataset (image-report pairs)
    2. Initialize ColQwen2 embedder
    3. Build document store from dataset samples
    4. Encode all images with ColQwen2 → image_embeddings.pt
    5. Encode all report text with ColQwen2 → text_embeddings.pt
    6. Save dual index to disk

Supports three modes:
    --full         Build everything from scratch (default for new indexes)
    --text-only    Build ONLY text_embeddings.pt from existing image index
    --images-only  Build ONLY image_embeddings.pt (Phase 2 behavior)

Run this ONCE before using the RAG pipeline.

Usage:
    python -m pipelines.offline_indexing
    python -m pipelines.offline_indexing --max-samples 50
    python -m pipelines.offline_indexing --text-only
    python -m pipelines.offline_indexing --data-config configs/data_config.yaml
"""

import json
import time
from pathlib import Path
from typing import Optional

from src.domains.healthcare.ingestion.dicom_loader import OpenIDataset
from src.domains.healthcare.embeddings.colqwen2_embedder import ColQwen2Embedder
from src.domains.healthcare.indexing.index_builder import ColQwen2IndexBuilder
from src.shared.logging_utils import setup_logger

logger = setup_logger("pipeline.offline_indexing")


class OfflineIndexingPipeline:
    """
    Phase 3 offline pipeline: OpenI → ColQwen2 dual index.

    Processes all OpenI image-report pairs, encodes each image and
    report text with ColQwen2, and saves a persistent dual retrieval
    index to disk.

    The saved index is later loaded by the RAG VQA pipeline for
    online retrieval across both image and text modalities.

    Supports incremental indexing: if an image index already exists,
    --text-only mode will only build the text embeddings without
    re-encoding any images.
    """

    def __init__(
        self,
        data_config: dict,
        retrieval_config: dict,
        index_dir: str = "data/indexes/colqwen2_index",
    ):
        """
        Args:
            data_config:      Dataset configuration dict.
            retrieval_config: Retrieval configuration dict.
            index_dir:        Directory to save the built index.
        """
        self.data_config = data_config
        self.retrieval_config = retrieval_config
        self.index_dir = index_dir

    # ------------------------------------------------------------------ #
    #  Full build (images + text)                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        max_samples: Optional[int] = None,
        text_only: bool = False,
        images_only: bool = False,
    ) -> dict:
        """
        Execute the offline indexing pipeline.

        Args:
            max_samples: Cap on number of samples to index (None = all).
            text_only:   If True, only build text embeddings from an
                         existing image index. Skips image encoding.
            images_only: If True, only build image embeddings (Phase 2).

        Returns:
            Summary dict with build statistics.
        """
        total_start = time.time()

        logger.info("=" * 60)
        if text_only:
            logger.info("Phase 3: Text-Only Indexing Pipeline")
            logger.info("(Building text_embeddings.pt from existing index)")
        elif images_only:
            logger.info("Phase 2: Image-Only Indexing Pipeline")
        else:
            logger.info("Phase 3: Full Dual Indexing Pipeline")
            logger.info("(Building image_embeddings.pt + text_embeddings.pt)")
        logger.info("=" * 60)

        # Step 1: Initialize ColQwen2 embedder
        logger.info("\n--- Step 1: Loading ColQwen2 embedder ---")
        embedder = ColQwen2Embedder()
        embedder.load(self.retrieval_config)
        logger.info("ColQwen2 embedder ready")

        # Route to appropriate build mode
        if text_only:
            summary = self._run_text_only(embedder)
        else:
            summary = self._run_full(
                embedder,
                max_samples=max_samples,
                build_text_index=not images_only,
            )

        # Cleanup — free VRAM
        logger.info("\n--- Cleanup: Unloading embedder ---")
        embedder.unload()

        total_time = time.time() - total_start
        summary["total_time_seconds"] = round(total_time, 2)
        summary["index_dir"] = self.index_dir

        logger.info("\n" + "=" * 60)
        logger.info("Offline Indexing Complete")
        logger.info(f"  Documents indexed: {summary.get('num_indexed', 0)}")
        logger.info(
            f"  Image embeddings:  "
            f"{summary.get('num_image_embeddings', 0)}"
        )
        logger.info(
            f"  Text embeddings:   "
            f"{summary.get('num_text_embeddings', 0)}"
        )
        logger.info(f"  Index saved to:    {self.index_dir}")
        logger.info(f"  Total time:        {total_time:.1f}s")
        logger.info("=" * 60)

        return summary

    def _run_full(
        self,
        embedder: ColQwen2Embedder,
        max_samples: Optional[int] = None,
        build_text_index: bool = True,
    ) -> dict:
        """Full build: dataset → images + text → save."""
        # Load dataset
        logger.info("\n--- Step 2: Loading OpenI dataset ---")
        dataset = self._load_dataset(max_samples)
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        logger.info(f"Dataset summary: {dataset.summary()}")

        # Build index
        logger.info("\n--- Step 3: Building ColQwen2 dual index ---")
        builder = ColQwen2IndexBuilder(
            embedder=embedder,
            config=self.retrieval_config,
        )

        batch_size = (
            self.retrieval_config
            .get("retrieval", {})
            .get("colqwen2", {})
            .get("batch_size", 4)
        )

        builder.build_from_dataset(
            dataset=dataset,
            max_samples=max_samples,
            batch_size=batch_size,
            build_text_index=build_text_index,
        )

        # Save
        logger.info("\n--- Step 4: Saving index ---")
        builder.save(self.index_dir)

        return builder.summary()

    def _run_text_only(self, embedder: ColQwen2Embedder) -> dict:
        """
        Text-only build: load existing index → encode text → save.

        This mode reuses the existing image_embeddings.pt and
        document_store.json, only adding text_embeddings.pt.
        """
        index_path = Path(self.index_dir)

        # Verify existing index
        logger.info("\n--- Step 2: Verifying existing index ---")
        self._verify_existing_index(index_path)

        # Build text index
        logger.info("\n--- Step 3: Building text embeddings ---")
        builder = ColQwen2IndexBuilder(
            embedder=embedder,
            config=self.retrieval_config,
        )
        builder.build_text_index_from_existing(self.index_dir)

        # Save (only writes text_embeddings.pt + updates metadata)
        logger.info("\n--- Step 4: Saving text index ---")
        builder.save(self.index_dir)

        return builder.summary()

    # ------------------------------------------------------------------ #
    #  Index verification                                                  #
    # ------------------------------------------------------------------ #

    def _verify_existing_index(self, index_path: Path) -> None:
        """
        Verify that an existing image index is valid and complete.

        Checks:
            1. Index directory exists
            2. Image embeddings file exists and loads successfully
            3. Document store exists and loads
            4. Doc IDs file exists and aligns with embeddings

        Raises:
            FileNotFoundError: If required files are missing.
            RuntimeError: If the index is invalid or inconsistent.
        """
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index directory not found: {index_path}\n"
                f"Run full indexing first: "
                f"python -m pipelines.offline_indexing"
            )

        # Check image embeddings
        img_emb_path = index_path / "image_embeddings.pt"
        old_emb_path = index_path / "embeddings.pt"

        if img_emb_path.exists():
            emb_file = img_emb_path
        elif old_emb_path.exists():
            emb_file = old_emb_path
        else:
            raise FileNotFoundError(
                f"No image embeddings found in {index_path}. "
                f"Run full indexing first."
            )

        # Verify embeddings load successfully
        import torch
        try:
            embs = torch.load(str(emb_file), map_location="cpu")
            n_embs = len(embs)
            logger.info(
                f"  ✓ Image embeddings: {n_embs} tensors "
                f"(file: {emb_file.name})"
            )
            if n_embs > 0:
                logger.info(
                    f"    Shape of first: {list(embs[0].shape)}"
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image embeddings from {emb_file}: {e}"
            )

        # Check document store
        docstore_path = index_path / "document_store.json"
        if not docstore_path.exists():
            raise FileNotFoundError(
                f"Document store not found: {docstore_path}"
            )
        with open(docstore_path, "r") as f:
            docstore_data = json.load(f)
        n_docs = len(docstore_data.get("documents", {}))
        logger.info(f"  ✓ Document store: {n_docs} documents")

        # Check doc IDs
        doc_ids_path = index_path / "doc_ids.json"
        if not doc_ids_path.exists():
            raise FileNotFoundError(
                f"Doc IDs not found: {doc_ids_path}"
            )
        with open(doc_ids_path, "r") as f:
            doc_ids = json.load(f)
        n_ids = len(doc_ids)
        logger.info(f"  ✓ Doc IDs: {n_ids} entries")

        # Verify alignment
        if n_embs != n_ids:
            raise RuntimeError(
                f"Index inconsistent: {n_embs} image embeddings "
                f"vs {n_ids} doc IDs. Rebuild with full indexing."
            )

        # Check for existing text embeddings
        txt_emb_path = index_path / "text_embeddings.pt"
        if txt_emb_path.exists():
            txt_embs = torch.load(str(txt_emb_path), map_location="cpu")
            logger.info(
                f"  ⚠ Text embeddings already exist: "
                f"{len(txt_embs)} tensors (will be overwritten)"
            )

        logger.info(
            f"  ✓ Existing index is valid: {n_embs} images, "
            f"{n_docs} documents, {n_ids} IDs"
        )

    # ------------------------------------------------------------------ #
    #  Dataset loading                                                     #
    # ------------------------------------------------------------------ #

    def _load_dataset(
        self, max_samples: Optional[int] = None
    ) -> OpenIDataset:
        """Load and return the OpenI dataset."""
        ds_cfg = self.data_config.get("dataset", {})

        dataset = OpenIDataset(
            images_dir=ds_cfg.get("images_dir", "data/openi/images"),
            reports_dir=ds_cfg.get("reports_dir", "data/openi/reports"),
            max_samples=max_samples or ds_cfg.get("max_samples"),
            load_images=False,   # Images loaded on-demand during indexing
        )
        dataset.load()
        return dataset


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #

def main():
    """Run the offline indexing pipeline from command line."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Phase 3: Offline ColQwen2 Dual Indexing Pipeline"
    )
    parser.add_argument(
        "--data-config",
        default="configs/data_config.yaml",
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--retrieval-config",
        default="configs/retrieval_config.yaml",
        help="Path to retrieval config YAML",
    )
    parser.add_argument(
        "--index-dir",
        default="data/indexes/colqwen2_index",
        help="Directory to save the index",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to index (None = all)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help=(
            "Build ONLY text embeddings from an existing image index. "
            "Does NOT re-encode images. Requires a valid existing index."
        ),
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help=(
            "Build ONLY image embeddings (Phase 2 behavior). "
            "Skips text encoding."
        ),
    )
    args = parser.parse_args()

    if args.text_only and args.images_only:
        parser.error("Cannot use --text-only and --images-only together")

    # Load configs
    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)
    with open(args.retrieval_config) as f:
        retrieval_config = yaml.safe_load(f)

    # Resolve relative data paths to project root
    from src.shared.config_loader import resolve_data_paths
    data_config = resolve_data_paths(data_config)

    # Run pipeline
    pipeline = OfflineIndexingPipeline(
        data_config=data_config,
        retrieval_config=retrieval_config,
        index_dir=args.index_dir,
    )

    summary = pipeline.run(
        max_samples=args.max_samples,
        text_only=args.text_only,
        images_only=args.images_only,
    )

    # Save summary
    summary_path = Path(args.index_dir) / "build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Build summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
