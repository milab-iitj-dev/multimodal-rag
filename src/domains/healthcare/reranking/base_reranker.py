"""
Abstract base class for reranking models.

Rerankers score (query, document) pairs and re-order retrieval results.
They run after the initial retrieval stage to improve precision.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseReranker(ABC):
    """
    Abstract interface for reranking models.

    Subclasses:
        - CrossEncoderReranker  (transformer cross-encoder)
        - MedicalReranker       (domain-specific medical reranker)
    """

    @abstractmethod
    def load(self, config: dict) -> None:
        """Load the reranking model."""
        ...

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of retrieved documents for a given query.

        Args:
            query:     The original query text.
            documents: List of dicts with 'doc_id', 'text', 'score', etc.
            top_k:     Number of documents to return after reranking.

        Returns:
            Reranked list of documents (highest relevance first).
        """
        ...
