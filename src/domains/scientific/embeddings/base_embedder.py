"""
Base Embedder Interface for the Scientific Multimodal RAG Project.

Defines the abstract contract that every embedding backend must
implement.  Two concrete implementations exist:

* :class:`~src.embeddings.colpali_embedder.ColPaliEmbedder` —
  multi-vector vision embeddings via ColPali-v1.2 (output shape
  ``[N, 128]``).
* :class:`~src.embeddings.scincl_embedder.SciNCLEmbedder` —
  dense text embeddings via SciNCL (output shape ``[768]``).

Both embedders follow the same load → embed → unload lifecycle as
the VLM models, enabling the staggered-loading strategy on
GPU-constrained environments.

Example:
    >>> from src.domains.scientific.embeddings.colpali_embedder import ColPaliEmbedder
    >>> embedder = ColPaliEmbedder()
    >>> embedder.load()
    >>> output = embedder.embed_image(pil_image)
    >>> print(output.vectors.shape)  # [num_tokens, 128]
    >>> embedder.unload()
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from src.shared.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# EmbeddingOutput dataclass
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingOutput:
    """Structured output returned by every embedder call.

    Attributes:
        vectors: The embedding tensor.  Shape depends on the backend:

            * ColPali: ``(num_tokens, 128)`` — one 128-dim vector per
              visual patch token (multi-vector for MaxSim scoring).
            * SciNCL: ``(768,)`` — a single dense 768-dim vector.

        doc_id: Identifier for the source document, e.g.
            ``"2305.12345"`` (arXiv ID) or a filesystem path.
        page_num: 1-indexed page number within the document, or
            ``None`` for text-only embeddings that are not page-bound.
        metadata: Arbitrary key-value metadata (model name, timestamp,
            etc.) useful for provenance tracking.
        embedding_time: Wall-clock seconds for the embedding call,
            measured from input pre-processing to tensor output.
    """

    vectors: torch.Tensor
    doc_id: str
    page_num: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_time: float = 0.0

    def __post_init__(self) -> None:
        """Validate fields after initialisation.

        Raises:
            ValueError: If *vectors* is not a torch.Tensor, *doc_id*
                is empty, or *embedding_time* is negative.
        """
        if not isinstance(self.vectors, torch.Tensor):
            raise ValueError(
                f"EmbeddingOutput.vectors must be a torch.Tensor, "
                f"got {type(self.vectors).__name__}"
            )
        if not self.doc_id:
            raise ValueError("EmbeddingOutput.doc_id must not be empty.")
        if self.embedding_time < 0:
            raise ValueError(
                f"EmbeddingOutput.embedding_time must be >= 0, "
                f"got {self.embedding_time}"
            )


# ---------------------------------------------------------------------------
# BaseEmbedder abstract base class
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Abstract base class for embedding models.

    Every concrete embedder in the pipeline must subclass this and
    implement :meth:`load`, :meth:`embed_image`, :meth:`embed_text`,
    :meth:`embed_batch`, :meth:`save_vectors`, and :meth:`unload`.

    The lifecycle mirrors :class:`~src.models.base_vlm.BaseVLM`:

    1. **load()** — Download (if needed) and initialise the model.
    2. **embed_image / embed_text / embed_batch** — Produce
       :class:`EmbeddingOutput` instances.
    3. **save_vectors** — Persist embeddings to disk (``.npy`` or
       ChromaDB depending on the backend).
    4. **unload()** — Release GPU memory.

    Note:
        Not every backend supports both image and text embedding.
        Callers should handle :class:`NotImplementedError` gracefully.
    """

    # -----------------------------------------------------------------
    # Abstract methods
    # -----------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load the embedding model and its processor / tokenizer.

        This method must:
            * Download model weights from Hugging Face Hub if not cached.
            * Move the model to the configured device.
            * Switch the model to evaluation mode.
            * Log VRAM usage after loading.

        Raises:
            RuntimeError: If the model cannot be loaded (e.g. OOM,
                download failure, GPU unavailable).
        """

    @abstractmethod
    def embed_image(self, image: Any) -> EmbeddingOutput:
        """Embed a single PIL image.

        Args:
            image: A ``PIL.Image.Image`` instance.

        Returns:
            An :class:`EmbeddingOutput` whose ``vectors`` field contains
            the image embedding.

        Raises:
            NotImplementedError: If the backend does not support image
                embedding (e.g. SciNCL is text-only).
            RuntimeError: If the model has not been loaded.
        """

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingOutput:
        """Embed a single text string.

        Args:
            text: The input text to embed.

        Returns:
            An :class:`EmbeddingOutput` whose ``vectors`` field contains
            the text embedding.

        Raises:
            NotImplementedError: If the backend does not support text
                embedding (e.g. ColPali is image-only).
            RuntimeError: If the model has not been loaded.
            ValueError: If *text* is empty.
        """

    @abstractmethod
    def embed_batch(
        self,
        items: List[Any],
        item_type: str = "image",
    ) -> List[EmbeddingOutput]:
        """Embed a batch of items for efficiency.

        Args:
            items: A list of items to embed.  When *item_type* is
                ``"image"``, each element should be a ``PIL.Image.Image``.
                When ``"text"``, each element should be a string.
            item_type: Either ``"image"`` or ``"text"`` to indicate the
                kind of input.

        Returns:
            A list of :class:`EmbeddingOutput` objects, one per item,
            in the same order as the input list.

        Raises:
            NotImplementedError: If the backend does not support the
                requested *item_type*.
            RuntimeError: If the model has not been loaded or an OOM
                error occurs during batch processing.
            ValueError: If *items* is empty or *item_type* is invalid.
        """

    @abstractmethod
    def save_vectors(self, output: EmbeddingOutput, filepath: Union[str, Path]) -> None:
        """Persist an :class:`EmbeddingOutput` to disk.

        The storage format depends on the backend:

            * ColPali → ``.npy`` (NumPy) because ChromaDB does not
              natively support multi-vector storage.
            * SciNCL → ChromaDB collection for ANN search.

        Args:
            output: The embedding output to save.
            filepath: Destination path (including filename and
                extension).

        Raises:
            IOError: If the file cannot be written.
            ValueError: If *output* or *filepath* is invalid.
        """

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and release GPU memory.

        This method must:
            * Delete model and processor / tokenizer references.
            * Call ``gc.collect()`` and ``torch.cuda.empty_cache()``.
            * Log the amount of VRAM freed.

        Raises:
            RuntimeError: If unloading fails unexpectedly.
        """

    # -----------------------------------------------------------------
    # Concrete helper methods
    # -----------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Check whether the model is currently loaded and ready.

        Returns:
            ``True`` if the model is loaded, ``False`` otherwise.
        """
        has_model = hasattr(self, "_model") and self._model is not None
        logger.debug("Embedder loaded check: %s", has_model)
        return has_model

    def _validate_item_type(self, item_type: str) -> None:
        """Validate the item_type parameter.

        Args:
            item_type: Must be ``"image"`` or ``"text"``.

        Raises:
            ValueError: If *item_type* is not one of the allowed values.
        """
        if item_type not in ("image", "text"):
            raise ValueError(
                f"item_type must be 'image' or 'text', got {item_type!r}"
            )
