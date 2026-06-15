"""
Abstract base class for all embedding models.

Every embedding model (CLIP, BiomedCLIP, ColQwen2) implements this interface.
Provides a unified API for encoding images and/or text into dense vectors.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Union

import numpy as np
from PIL import Image


class BaseEmbedder(ABC):
    """
    Abstract interface for embedding models.

    Subclasses:
        - CLIPEmbedder        (CLIP / BiomedCLIP image-text embeddings)
        - ColQwen2Embedder    (ColQwen2 multi-vector embeddings)
    """

    @abstractmethod
    def load(self, config: dict) -> None:
        """Load model weights and processor."""
        ...

    @abstractmethod
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of text strings into embedding vectors.

        Args:
            texts: List of text strings.

        Returns:
            np.ndarray of shape (N, embedding_dim).
        """
        ...

    @abstractmethod
    def encode_image(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode a batch of images into embedding vectors.

        Args:
            images: List of PIL images.

        Returns:
            np.ndarray of shape (N, embedding_dim).
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of output embeddings."""
        ...
