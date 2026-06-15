"""
Offline index builder for the medical knowledge base.

Orchestrates the full offline indexing pipeline:
    1. Load OpenI dataset (image-report pairs)
    2. Build document store from dataset samples
    3. Encode all document images with ColQwen2
    4. Encode all document text with ColQwen2 (Phase 3)
    5. Save multi-vector embeddings + document store to disk

This runs ONCE (or on dataset updates), not at query time.
The saved index is loaded by the ColQwen2Retriever at query time.

Index format on disk:
    data/indexes/colqwen2_index/
    ├── document_store.json        # all document metadata
    ├── image_embeddings.pt        # image multi-vector tensors
    ├── text_embeddings.pt         # text multi-vector tensors (Phase 3)
    ├── doc_ids.json               # ordered list of doc IDs
    └── index_metadata.json        # build info (timestamp, model, count)

Backward compatibility:
    - Old indexes with 'embeddings.pt' are auto-detected and
      treated as image-only indexes during load().
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from tqdm import tqdm

from src.domains.healthcare.indexing.document_store import DocumentStore, Document
from src.domains.healthcare.embeddings.colqwen2_embedder import ColQwen2Embedder
from src.domains.healthcare.ingestion.base_loader import BaseDataset
from src.shared.logging_utils import setup_logger
from src.shared.image_utils import load_image

logger = setup_logger("indexing.builder")


class ColQwen2IndexBuilder:
    """
    Offline index builder for ColQwen2-based retrieval.

    Takes an OpenI dataset, encodes all images AND report text with
    ColQwen2, and saves a persistent dual index that can be loaded
    at query time.

    Supports three indexing modes:
      - Full build: images + text (default for new indexes)
      - Text-only build: add text embeddings to existing image index
      - Image-only build: backward compatible with Phase 2

    Usage:
        builder = ColQwen2IndexBuilder(embedder, config)
        builder.build_from_dataset(dataset)
        builder.save("data/indexes/colqwen2_index/")

        # Or: add text index to existing image index
        builder.build_text_index_from_existing("data/indexes/colqwen2_index/")
    """

    def __init__(
        self,
        embedder: ColQwen2Embedder,
        config: Optional[dict] = None,
    ):
        """
        Args:
            embedder: Loaded ColQwen2Embedder instance.
            config:   Optional config dict for index builder settings.
        """
        self.embedder = embedder
        self.config = config or {}
        self.document_store = DocumentStore()
        self._image_embeddings: List[torch.Tensor] = []
        self._text_embeddings: List[torch.Tensor] = []
        self._doc_ids: List[str] = []
        self._build_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #  Full build from dataset (images + text)                             #
    # ------------------------------------------------------------------ #

    def build_from_dataset(
        self,
        dataset: BaseDataset,
        max_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        build_text_index: bool = True,
    ) -> None:
        """
        Build the ColQwen2 index from an OpenI dataset.

        Steps:
            1. Iterate dataset samples → build document store
            2. Load images for all documents
            3. Encode images with ColQwen2 in batches
            4. Encode report text with ColQwen2 (if build_text_index=True)
            5. Store embeddings + metadata

        Args:
            dataset:          Loaded OpenI dataset.
            max_samples:      Cap on number of samples to index (None = all).
            batch_size:       Override embedder batch size.
            build_text_index: Whether to also build text embeddings.
        """
        start_time = time.time()
        n_samples = len(dataset)
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        logger.info(f"Building ColQwen2 index from {n_samples} samples")

        # Step 1: Build document store and collect images + text
        images = []
        doc_texts = []
        doc_ids = []
        skipped = 0

        for idx in tqdm(range(n_samples), desc="Loading documents"):
            sample = dataset[idx]

            # Skip samples without usable content
            if not sample.image_path:
                skipped += 1
                continue

            # Try loading the image
            try:
                image = load_image(sample.image_path)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Skipping {sample.sample_id}: {e}")
                skipped += 1
                continue

            # Create document entry
            doc = Document(
                doc_id=sample.sample_id,
                text=sample.report or "",
                image_path=sample.image_path,
                findings=sample.findings,
                impression=sample.impression,
                metadata=sample.metadata,
            )
            self.document_store.add_document(doc)

            images.append(image)
            doc_ids.append(sample.sample_id)

            # Collect text for text index
            doc_text = self._build_document_text(
                sample.findings, sample.impression
            )
            doc_texts.append(doc_text)

        logger.info(
            f"Document store built: {len(doc_ids)} documents "
            f"({skipped} skipped)"
        )

        # Step 2: Encode all images with ColQwen2
        logger.info("Encoding images with ColQwen2...")
        self._image_embeddings = self.embedder.encode_images(
            images,
            batch_size=batch_size,
        )
        self._doc_ids = doc_ids

        # Step 3: Encode all document text with ColQwen2
        if build_text_index:
            text_cfg = (
                self.config
                .get("retrieval", {})
                .get("colqwen2", {})
                .get("dual_index", {})
            )
            text_max_length = text_cfg.get("text_max_length", 256)
            text_batch_size = text_cfg.get(
                "text_batch_size", batch_size or 8
            )

            logger.info(
                f"Encoding document text with ColQwen2 "
                f"(max_length={text_max_length})..."
            )
            self._text_embeddings = self.embedder.encode_document_text(
                doc_texts,
                batch_size=text_batch_size,
                max_length=text_max_length,
            )
            logger.info(
                f"Text index built: {len(self._text_embeddings)} embeddings"
            )
        else:
            logger.info("Text index building skipped (build_text_index=False)")

        # Step 4: Store build metadata
        elapsed = time.time() - start_time
        self._build_metadata = {
            "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.embedder.model_name,
            "num_documents": len(doc_ids),
            "num_skipped": skipped,
            "build_time_seconds": round(elapsed, 2),
            "has_image_index": True,
            "has_text_index": build_text_index,
            "image_embedding_shapes": [
                list(emb.shape) for emb in self._image_embeddings[:3]
            ],
            "text_embedding_shapes": [
                list(emb.shape) for emb in self._text_embeddings[:3]
            ] if self._text_embeddings else [],
        }

        logger.info(
            f"Index built: {len(self._image_embeddings)} image embeddings"
            + (f", {len(self._text_embeddings)} text embeddings"
               if self._text_embeddings else "")
            + f" in {elapsed:.1f}s"
        )

    # ------------------------------------------------------------------ #
    #  Text-only build from existing index                                 #
    # ------------------------------------------------------------------ #

    def build_text_index_from_existing(
        self,
        index_dir: str,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Build ONLY the text index from an existing image index.

        This loads the existing document store and doc IDs, encodes
        the report text for each document, and saves text_embeddings.pt
        alongside the existing image index files.

        The existing image_embeddings.pt (or embeddings.pt) is NOT
        modified or re-encoded.

        Args:
            index_dir:  Path to existing ColQwen2 index directory.
            batch_size: Override embedder batch size for text encoding.
        """
        index_path = Path(index_dir)
        start_time = time.time()

        logger.info(f"Building text index from existing index: {index_path}")

        # Load existing document store
        docstore_path = index_path / "document_store.json"
        if not docstore_path.exists():
            raise FileNotFoundError(
                f"Document store not found: {docstore_path}"
            )
        self.document_store.load(str(docstore_path))
        logger.info(
            f"Loaded document store: {len(self.document_store)} documents"
        )

        # Load existing doc IDs
        doc_ids_path = index_path / "doc_ids.json"
        if not doc_ids_path.exists():
            raise FileNotFoundError(f"Doc IDs not found: {doc_ids_path}")
        with open(doc_ids_path, "r", encoding="utf-8") as f:
            self._doc_ids = json.load(f)
        logger.info(f"Loaded doc IDs: {len(self._doc_ids)}")

        # Load existing image embeddings (don't re-encode)
        img_emb_path = index_path / "image_embeddings.pt"
        old_emb_path = index_path / "embeddings.pt"

        if img_emb_path.exists():
            self._image_embeddings = torch.load(
                str(img_emb_path), map_location="cpu"
            )
            logger.info(
                f"Existing image embeddings loaded: "
                f"{len(self._image_embeddings)} tensors (NOT re-encoded)"
            )
        elif old_emb_path.exists():
            self._image_embeddings = torch.load(
                str(old_emb_path), map_location="cpu"
            )
            logger.info(
                f"Existing embeddings.pt loaded (old format): "
                f"{len(self._image_embeddings)} tensors (NOT re-encoded)"
            )
        else:
            logger.warning(
                "No existing image embeddings found. "
                "Only text index will be built."
            )

        # Verify alignment
        if self._image_embeddings and \
                len(self._image_embeddings) != len(self._doc_ids):
            logger.warning(
                f"Mismatch: {len(self._image_embeddings)} image embeddings "
                f"vs {len(self._doc_ids)} doc IDs"
            )

        # Build text for each document (in doc_id order)
        doc_texts = []
        for doc_id in self._doc_ids:
            doc = self.document_store.get_document(doc_id)
            if doc is not None:
                doc_text = self._build_document_text(
                    doc.findings, doc.impression
                )
            else:
                doc_text = "no report available"
                logger.warning(f"Doc {doc_id} not in document store")
            doc_texts.append(doc_text)

        # Encode text
        text_cfg = (
            self.config
            .get("retrieval", {})
            .get("colqwen2", {})
            .get("dual_index", {})
        )
        text_max_length = text_cfg.get("text_max_length", 256)
        text_batch_size = batch_size or text_cfg.get("text_batch_size", 8)

        logger.info(
            f"Encoding {len(doc_texts)} document texts "
            f"(max_length={text_max_length}, "
            f"batch_size={text_batch_size})..."
        )
        self._text_embeddings = self.embedder.encode_document_text(
            doc_texts,
            batch_size=text_batch_size,
            max_length=text_max_length,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"Text index built: {len(self._text_embeddings)} embeddings "
            f"in {elapsed:.1f}s"
        )

        # Update metadata
        self._build_metadata = {
            "text_build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.embedder.model_name,
            "num_documents": len(self._doc_ids),
            "text_build_time_seconds": round(elapsed, 2),
            "has_image_index": bool(self._image_embeddings),
            "has_text_index": True,
            "text_max_length": text_max_length,
            "text_embedding_shapes": [
                list(emb.shape) for emb in self._text_embeddings[:3]
            ],
        }

    # ------------------------------------------------------------------ #
    #  Helper: build document text from findings + impression              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_document_text(
        findings: Optional[str],
        impression: Optional[str],
    ) -> str:
        """
        Combine findings and impression into a single document text
        string for text encoding.

        Args:
            findings:   Findings section of the radiology report.
            impression: Impression/conclusion section.

        Returns:
            Combined text string. Falls back to placeholder if both
            are empty.
        """
        parts = []
        if findings and findings.strip():
            parts.append(findings.strip())
        if impression and impression.strip():
            parts.append(impression.strip())
        return " ".join(parts) if parts else "no report available"

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, index_dir: str) -> None:
        """
        Save the complete index to disk.

        Creates:
            index_dir/
            ├── document_store.json
            ├── image_embeddings.pt     (if image index exists)
            ├── text_embeddings.pt      (if text index exists)
            ├── doc_ids.json
            └── index_metadata.json

        Args:
            index_dir: Directory to save the index in.
        """
        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)

        # Save document store
        docstore_path = index_path / "document_store.json"
        self.document_store.save(str(docstore_path))

        # Save image embeddings
        if self._image_embeddings:
            img_path = index_path / "image_embeddings.pt"
            torch.save(self._image_embeddings, str(img_path))
            logger.info(
                f"Image embeddings saved: "
                f"{len(self._image_embeddings)} tensors → {img_path}"
            )

        # Save text embeddings
        if self._text_embeddings:
            txt_path = index_path / "text_embeddings.pt"
            torch.save(self._text_embeddings, str(txt_path))
            logger.info(
                f"Text embeddings saved: "
                f"{len(self._text_embeddings)} tensors → {txt_path}"
            )

        # Save ordered doc IDs
        doc_ids_path = index_path / "doc_ids.json"
        with open(doc_ids_path, "w", encoding="utf-8") as f:
            json.dump(self._doc_ids, f, indent=2)
        logger.info(
            f"Doc IDs saved: {len(self._doc_ids)} → {doc_ids_path}"
        )

        # Save build metadata
        metadata_path = index_path / "index_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._build_metadata, f, indent=2)
        logger.info(f"Index metadata saved → {metadata_path}")

        logger.info(f"Complete index saved to: {index_path}")

    def load(self, index_dir: str) -> None:
        """
        Load a previously saved index from disk.

        Handles backward compatibility:
            - New format: image_embeddings.pt + text_embeddings.pt
            - Old format: embeddings.pt (treated as image-only)

        Args:
            index_dir: Directory containing the saved index.

        Raises:
            FileNotFoundError: If the directory or required files
                               are missing.
        """
        index_path = Path(index_dir)
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index directory not found: {index_path}"
            )

        # Load document store
        docstore_path = index_path / "document_store.json"
        self.document_store.load(str(docstore_path))

        # Load image embeddings (handle both naming conventions)
        img_emb_path = index_path / "image_embeddings.pt"
        old_emb_path = index_path / "embeddings.pt"

        if img_emb_path.exists():
            self._image_embeddings = torch.load(
                str(img_emb_path), map_location="cpu"
            )
            logger.info(
                f"Image embeddings loaded: "
                f"{len(self._image_embeddings)} tensors"
            )
        elif old_emb_path.exists():
            # Backward compatibility: old format uses 'embeddings.pt'
            self._image_embeddings = torch.load(
                str(old_emb_path), map_location="cpu"
            )
            logger.info(
                f"Image embeddings loaded (old format 'embeddings.pt'): "
                f"{len(self._image_embeddings)} tensors"
            )
        else:
            logger.warning("No image embeddings found in index")

        # Load text embeddings (optional — may not exist in old indexes)
        txt_emb_path = index_path / "text_embeddings.pt"
        if txt_emb_path.exists():
            self._text_embeddings = torch.load(
                str(txt_emb_path), map_location="cpu"
            )
            logger.info(
                f"Text embeddings loaded: "
                f"{len(self._text_embeddings)} tensors"
            )
        else:
            self._text_embeddings = []
            logger.info(
                "No text embeddings found — text retrieval disabled"
            )

        # Load doc IDs
        doc_ids_path = index_path / "doc_ids.json"
        if not doc_ids_path.exists():
            raise FileNotFoundError(
                f"Doc IDs file not found: {doc_ids_path}"
            )
        with open(doc_ids_path, "r", encoding="utf-8") as f:
            self._doc_ids = json.load(f)
        logger.info(f"Doc IDs loaded: {len(self._doc_ids)}")

        # Load build metadata
        metadata_path = index_path / "index_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self._build_metadata = json.load(f)

        logger.info(
            f"Index loaded from {index_path}: "
            f"{len(self._image_embeddings)} image embeddings, "
            f"{len(self._text_embeddings)} text embeddings, "
            f"{len(self.document_store)} documents"
        )

    # ------------------------------------------------------------------ #
    #  Accessors                                                           #
    # ------------------------------------------------------------------ #

    @property
    def image_embeddings(self) -> List[torch.Tensor]:
        """All stored image embeddings."""
        return self._image_embeddings

    @property
    def text_embeddings(self) -> List[torch.Tensor]:
        """All stored text embeddings."""
        return self._text_embeddings

    @property
    def embeddings(self) -> List[torch.Tensor]:
        """Image embeddings (backward compatibility alias)."""
        return self._image_embeddings

    @property
    def doc_ids(self) -> List[str]:
        """Ordered list of doc IDs (maps index → doc_id)."""
        return self._doc_ids

    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        return len(self._doc_ids)

    @property
    def has_text_index(self) -> bool:
        """Whether text embeddings are available."""
        return bool(self._text_embeddings)

    @property
    def build_metadata(self) -> Dict[str, Any]:
        """Metadata from the last build."""
        return self._build_metadata

    def summary(self) -> Dict[str, Any]:
        """Summary of the current index state."""
        return {
            "num_indexed": len(self._doc_ids),
            "num_image_embeddings": len(self._image_embeddings),
            "num_text_embeddings": len(self._text_embeddings),
            "has_text_index": self.has_text_index,
            "num_documents": len(self.document_store),
            "doc_ids_match": (
                len(self._image_embeddings) == len(self._doc_ids)
            ),
            "build_metadata": self._build_metadata,
            "document_store_summary": self.document_store.summary(),
        }
