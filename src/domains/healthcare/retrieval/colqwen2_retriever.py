"""
ColQwen2 late-interaction retrieval for multimodal documents.

Uses ColQwen2 (ColPali architecture with Qwen2-VL backbone) for
late-interaction retrieval. Unlike CLIP's single-vector matching,
ColQwen2 produces per-token embeddings and uses MaxSim scoring
for fine-grained matching between query tokens and document patches.

Supports three query modes:
  - Text-only query:   encode query text → search text index
  - Image-only query:  encode query image → search image index
  - Image + text query: search both indexes, delegate fusion to caller

The retriever loads a pre-built dual index (from ColQwen2IndexBuilder)
and performs online similarity search at query time.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from PIL import Image

from src.domains.healthcare.retrieval.base_retriever import BaseRetriever, RetrievedDocument
from src.domains.healthcare.embeddings.colqwen2_embedder import ColQwen2Embedder
from src.domains.healthcare.indexing.document_store import DocumentStore
from src.shared.logging_utils import setup_logger
from src.shared.image_utils import load_image

logger = setup_logger("retrieval.colqwen2")


class ColQwen2Retriever(BaseRetriever):
    """
    ColQwen2 late-interaction retriever with dual-index support.

    Loads a pre-built ColQwen2 index containing both image and text
    embeddings, and performs MaxSim-based retrieval for user queries.

    Supports three retrieval paths:
      - Image → image_embeddings (visual similarity)
      - Text → text_embeddings (text-to-text matching)
      - Image + Text → both paths (results fused by HybridRetriever)

    Usage:
        retriever = ColQwen2Retriever(embedder)
        retriever.load_index("data/indexes/colqwen2_index/")

        # Image-only retrieval
        results = retriever.retrieve("describe this", query_image=img)

        # Text-only retrieval
        results = retriever.retrieve("signs of pneumonia")

        # Dual retrieval for fusion
        img_results = retriever.retrieve_by_image(img, top_k=15)
        txt_results = retriever.retrieve_by_text("pneumonia?", top_k=15)
    """

    # Known image path prefixes from different environments
    _KAGGLE_PREFIX = (
        "/kaggle/input/datasets/raddar/"
        "chest-xrays-indiana-university/images/images_normalized/"
    )

    def __init__(
        self,
        embedder: ColQwen2Embedder,
        config: Optional[dict] = None,
    ):
        """
        Args:
            embedder: Loaded ColQwen2Embedder instance (shared with indexing).
            config:   Optional retrieval config dict. Used to read
                      colqwen2.image_base_dir for path remapping.
        """
        self.embedder = embedder
        self.document_store = DocumentStore()
        self._image_embeddings: List[torch.Tensor] = []
        self._text_embeddings: List[torch.Tensor] = []
        self._doc_ids: List[str] = []
        self._index_loaded = False

        # Image path remapping: handles Kaggle→HPC path translation.
        # If image_base_dir is set, Kaggle paths in the document store
        # are remapped to this directory at retrieval time.
        config = config or {}
        self._image_base_dir = (
            config
            .get("retrieval", {})
            .get("colqwen2", {})
            .get("image_base_dir", "")
        )
        if self._image_base_dir:
            logger.info(
                f"Image path remapping enabled: → {self._image_base_dir}"
            )

    # ------------------------------------------------------------------ #
    #  BaseRetriever: index()                                              #
    # ------------------------------------------------------------------ #

    def index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the retrieval index from a list of document dicts.

        This is an alternative to loading a pre-built index. It encodes
        all document images with ColQwen2 and stores the embeddings.

        Args:
            documents: List of dicts, each with at least:
                - 'doc_id': str
                - 'image_path': str (path to the image file)
                - 'text': str (report text, optional)
                - 'findings': str (optional)
                - 'impression': str (optional)
                - 'metadata': dict (optional)
        """
        from src.domains.healthcare.indexing.document_store import Document

        logger.info(f"Indexing {len(documents)} documents with ColQwen2")

        images = []
        doc_ids = []

        for doc_data in documents:
            doc_id = doc_data["doc_id"]
            image_path = doc_data.get("image_path", "")

            # Add to document store
            doc = Document(
                doc_id=doc_id,
                text=doc_data.get("text", ""),
                image_path=image_path,
                findings=doc_data.get("findings"),
                impression=doc_data.get("impression"),
                metadata=doc_data.get("metadata", {}),
            )
            self.document_store.add_document(doc)

            # Load image for encoding
            if image_path:
                try:
                    image = load_image(image_path)
                    images.append(image)
                    doc_ids.append(doc_id)
                except (FileNotFoundError, ValueError) as e:
                    logger.warning(f"Skipping {doc_id}: {e}")

        # Encode all images
        if images:
            self._image_embeddings = self.embedder.encode_images(images)
            self._doc_ids = doc_ids
            self._index_loaded = True
            logger.info(f"Indexed {len(self._image_embeddings)} documents")
        else:
            logger.warning("No images found to index")

    # ------------------------------------------------------------------ #
    #  BaseRetriever: retrieve() — unified entry point                     #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        query_image: Optional[Image.Image] = None,
        top_k: int = 3,
    ) -> List[RetrievedDocument]:
        """
        Retrieve the top-k most relevant documents for a query.

        Routes to the appropriate retrieval path based on inputs:
          - Image only:  query_image provided → image index
          - Text only:   query string only → text index (if available),
                         falls back to cross-modal image index
          - Image + text: query_image + query → image index only
                         (for full dual retrieval, use retrieve_by_image()
                          and retrieve_by_text() separately via
                          HybridRetriever)

        Args:
            query:       Text query string.
            query_image: Optional query image (for multimodal retrieval).
            top_k:       Number of documents to return.

        Returns:
            List of RetrievedDocument sorted by relevance (highest first).
        """
        if not self._index_loaded:
            raise RuntimeError(
                "No index loaded. Call load_index() or index() first."
            )

        if not self._image_embeddings and not self._text_embeddings:
            logger.warning("Index is empty, no documents to retrieve from")
            return []

        # Route based on available inputs
        if query_image is not None:
            # Image provided → use image retrieval (primary path)
            return self.retrieve_by_image(
                query_image, query=query, top_k=top_k
            )
        elif self._text_embeddings and query and query.strip():
            # Text-only with text index available → text retrieval
            return self.retrieve_by_text(query, top_k=top_k)
        elif query and query.strip():
            # Text-only but no text index → cross-modal fallback
            logger.info(
                "Text-only query but no text index — "
                "using cross-modal image index"
            )
            return self._retrieve_cross_modal_text(query, top_k=top_k)
        else:
            logger.warning("No query text or image provided")
            return []

    # ------------------------------------------------------------------ #
    #  Image retrieval path                                                #
    # ------------------------------------------------------------------ #

    def retrieve_by_image(
        self,
        query_image: Image.Image,
        query: str = "",
        top_k: int = 3,
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents using image query against the image index.

        Args:
            query_image: Query image (chest X-ray).
            query:       Optional query text (passed to encoder but
                         currently not used by encode_image_queries).
            top_k:       Number of documents to return.

        Returns:
            List of RetrievedDocument with source="colqwen2_image".
        """
        if not self._image_embeddings:
            logger.warning("No image embeddings — cannot do image retrieval")
            return []

        logger.info(f"Retrieval mode: image → image index ({top_k})")

        query_embeddings = self.embedder.encode_image_queries(
            images=[query_image],
            queries=[query or "describe this medical image"],
        )

        scores = self.embedder.score(
            query_embeddings=query_embeddings,
            doc_embeddings=self._image_embeddings,
        )

        return self._build_results(
            scores.squeeze(0), top_k, source="colqwen2_image"
        )

    # ------------------------------------------------------------------ #
    #  Text retrieval path (text → text index)                             #
    # ------------------------------------------------------------------ #

    def retrieve_by_text(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents using text query against the text index.

        This is text-to-text retrieval: the query text is encoded by
        ColQwen2's text encoder (process_queries), and matched against
        document text embeddings (also produced by process_queries
        during offline indexing) via MaxSim scoring.

        Args:
            query:  Text query string.
            top_k:  Number of documents to return.

        Returns:
            List of RetrievedDocument with source="colqwen2_text".
        """
        if not self._text_embeddings:
            logger.warning(
                "No text embeddings — falling back to cross-modal"
            )
            return self._retrieve_cross_modal_text(query, top_k=top_k)

        logger.info(f"Retrieval mode: text → text index ({top_k})")

        query_embeddings = self.embedder.encode_queries([query])

        scores = self.embedder.score(
            query_embeddings=query_embeddings,
            doc_embeddings=self._text_embeddings,
        )

        return self._build_results(
            scores.squeeze(0), top_k, source="colqwen2_text"
        )

    # ------------------------------------------------------------------ #
    #  Cross-modal fallback (text → image index)                           #
    # ------------------------------------------------------------------ #

    def _retrieve_cross_modal_text(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[RetrievedDocument]:
        """
        Fallback: text query against image index (cross-modal matching).

        This is the Phase 2 behavior — text tokens matched against
        X-ray image patches. Works but is weaker than text-to-text.

        Args:
            query:  Text query string.
            top_k:  Number of documents to return.

        Returns:
            List of RetrievedDocument with source="colqwen2_crossmodal".
        """
        logger.info(
            f"Retrieval mode: text → image index (cross-modal) ({top_k})"
        )

        query_embeddings = self.embedder.encode_queries([query])

        scores = self.embedder.score(
            query_embeddings=query_embeddings,
            doc_embeddings=self._image_embeddings,
        )

        return self._build_results(
            scores.squeeze(0), top_k, source="colqwen2_crossmodal"
        )

    # ------------------------------------------------------------------ #
    #  Build result objects from scores                                     #
    # ------------------------------------------------------------------ #

    def _build_results(
        self,
        scores: torch.Tensor,
        top_k: int,
        source: str = "colqwen2",
    ) -> List[RetrievedDocument]:
        """
        Convert a score tensor into a ranked list of RetrievedDocument.

        Args:
            scores:  1D tensor of shape [n_docs].
            top_k:   Number of results to return.
            source:  Source label for the results.

        Returns:
            List of RetrievedDocument sorted by score (highest first).
        """
        k = min(top_k, len(self._doc_ids))
        top_scores, top_indices = torch.topk(scores, k=k)

        results = []
        for rank, (score_val, idx) in enumerate(
            zip(top_scores.tolist(), top_indices.tolist())
        ):
            doc_id = self._doc_ids[idx]
            doc = self.document_store.get_document(doc_id)

            if doc is None:
                logger.warning(f"Document {doc_id} not found in store")
                continue

            # Try loading the image for the retrieved document
            resolved_path = self._resolve_image_path(doc.image_path)
            retrieved_image = None
            if resolved_path:
                try:
                    retrieved_image = load_image(resolved_path)
                except (FileNotFoundError, ValueError):
                    pass

            result = RetrievedDocument(
                doc_id=doc_id,
                score=score_val,
                text=doc.text,
                image=retrieved_image,
                image_path=doc.image_path,
                source=source,
                metadata={
                    "rank": rank + 1,
                    "findings": doc.findings,
                    "impression": doc.impression,
                    **doc.metadata,
                },
            )
            results.append(result)

        logger.info(
            f"Retrieved {len(results)} documents via {source} "
            f"(scores: {[f'{r.score:.4f}' for r in results]})"
        )
        return results

    # ------------------------------------------------------------------ #
    #  Image path resolution                                               #
    # ------------------------------------------------------------------ #

    def _resolve_image_path(self, path: str) -> str:
        """
        Resolve a document image path to the current environment.

        The existing document store may contain paths from a different
        environment (e.g., Kaggle paths like /kaggle/input/...). This
        method remaps them to the configured image_base_dir.

        Remapping logic:
            1. If path starts with the known Kaggle prefix and
               image_base_dir is configured → remap to local path
            2. If path exists as-is → use it directly
            3. Otherwise → return path as-is (let load_image fail)

        Args:
            path: Original image path from the document store.

        Returns:
            Resolved path string (may or may not exist).
        """
        if not path:
            return ""

        # Remap known prefixes
        if self._image_base_dir and path.startswith(self._KAGGLE_PREFIX):
            filename = path[len(self._KAGGLE_PREFIX):]
            resolved = os.path.join(self._image_base_dir, filename)
            return resolved

        # Also handle paths that start with /kaggle/ but may have
        # slightly different directory structures
        if self._image_base_dir and "/kaggle/" in path:
            filename = os.path.basename(path)
            resolved = os.path.join(self._image_base_dir, filename)
            return resolved

        return path

    # ------------------------------------------------------------------ #
    #  BaseRetriever: save/load index                                      #
    # ------------------------------------------------------------------ #

    def save_index(self, path: str) -> None:
        """
        Save the retrieval index to disk.

        Args:
            path: Directory path to save the index.
        """
        index_path = Path(path)
        index_path.mkdir(parents=True, exist_ok=True)

        # Save document store
        self.document_store.save(str(index_path / "document_store.json"))

        # Save image embeddings
        if self._image_embeddings:
            torch.save(
                self._image_embeddings,
                str(index_path / "image_embeddings.pt"),
            )

        # Save text embeddings
        if self._text_embeddings:
            torch.save(
                self._text_embeddings,
                str(index_path / "text_embeddings.pt"),
            )

        # Save doc IDs
        with open(
            index_path / "doc_ids.json", "w", encoding="utf-8"
        ) as f:
            json.dump(self._doc_ids, f, indent=2)

        logger.info(f"Index saved to {index_path}")

    def load_index(self, path: str) -> None:
        """
        Load a previously saved index from disk.

        Handles backward compatibility:
          - New format: image_embeddings.pt + text_embeddings.pt
          - Old format: embeddings.pt (treated as image-only)

        Args:
            path: Directory path containing the saved index.

        Raises:
            FileNotFoundError: If required index files are missing.
        """
        index_path = Path(path)
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index directory not found: {index_path}"
            )

        # Load document store
        docstore_path = index_path / "document_store.json"
        if not docstore_path.exists():
            raise FileNotFoundError(
                f"Document store not found: {docstore_path}"
            )
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
            self._image_embeddings = torch.load(
                str(old_emb_path), map_location="cpu"
            )
            logger.info(
                f"Image embeddings loaded (old format): "
                f"{len(self._image_embeddings)} tensors"
            )
        else:
            logger.warning("No image embeddings found in index directory")

        # Load text embeddings (optional — Phase 3+)
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
                "No text embeddings found — text retrieval disabled. "
                "Run offline indexing with --text-only to build."
            )

        # Load doc IDs
        doc_ids_path = index_path / "doc_ids.json"
        if not doc_ids_path.exists():
            raise FileNotFoundError(
                f"Doc IDs not found: {doc_ids_path}"
            )
        with open(doc_ids_path, "r", encoding="utf-8") as f:
            self._doc_ids = json.load(f)

        self._index_loaded = True

        # Validate consistency
        if self._image_embeddings and \
                len(self._image_embeddings) != len(self._doc_ids):
            logger.warning(
                f"Mismatch: {len(self._image_embeddings)} image embeddings "
                f"vs {len(self._doc_ids)} doc IDs"
            )
        if self._text_embeddings and \
                len(self._text_embeddings) != len(self._doc_ids):
            logger.warning(
                f"Mismatch: {len(self._text_embeddings)} text embeddings "
                f"vs {len(self._doc_ids)} doc IDs"
            )

        logger.info(
            f"Index loaded: {len(self._image_embeddings)} image embeddings, "
            f"{len(self._text_embeddings)} text embeddings, "
            f"{len(self.document_store)} documents"
        )

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    @property
    def is_index_loaded(self) -> bool:
        """Whether an index has been loaded or built."""
        return self._index_loaded

    @property
    def has_text_index(self) -> bool:
        """Whether text embeddings are available."""
        return bool(self._text_embeddings)

    @property
    def num_indexed(self) -> int:
        """Number of indexed documents."""
        return len(self._doc_ids)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the retriever state."""
        return {
            "retriever": "ColQwen2Retriever",
            "index_loaded": self._index_loaded,
            "num_indexed": len(self._doc_ids),
            "num_image_embeddings": len(self._image_embeddings),
            "num_text_embeddings": len(self._text_embeddings),
            "has_text_index": self.has_text_index,
            "num_documents": len(self.document_store),
            "embedder_loaded": self.embedder.is_loaded,
        }
