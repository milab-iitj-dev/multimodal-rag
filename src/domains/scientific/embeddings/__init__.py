"""
Embeddings Package — Vision & Text Embedding Backends.

Re-exports the core classes so that downstream modules can import
from the package root::

    from src.embeddings import (
        BaseEmbedder, EmbeddingOutput,
        ColPaliEmbedder, SciNCLEmbedder,
    )
"""

from src.domains.scientific.embeddings.base_embedder import BaseEmbedder, EmbeddingOutput
from src.domains.scientific.embeddings.colpali_embedder import ColPaliEmbedder
from src.domains.scientific.embeddings.scincl_embedder import SciNCLEmbedder

__all__ = [
    "BaseEmbedder",
    "EmbeddingOutput",
    "ColPaliEmbedder",
    "SciNCLEmbedder",
]
