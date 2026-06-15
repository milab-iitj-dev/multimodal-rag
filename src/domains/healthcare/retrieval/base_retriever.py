"""
Abstract base class for all retriever implementations.

Every retriever (BM25, ColQwen2, CLIP, hybrid) implements this interface.
Pipelines call retrieve() and get back a list of RetrievedDocument objects —
they never know which retriever is running behind the scenes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from PIL import Image


@dataclass
class RetrievedDocument:
    """
    A single retrieved document / evidence from the knowledge base.

    This is the universal retrieval result format. All retrievers
    must produce RetrievedDocument instances.
    """
    doc_id: str                                     # unique document identifier
    score: float = 0.0                              # relevance score
    text: Optional[str] = None                      # report text / findings
    image: Optional[Image.Image] = None             # associated image (if any)
    image_path: str = ""                            # path to the image file
    source: str = ""                                # which retriever produced this
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """
    Abstract interface for all retrieval methods.

    Subclasses:
        - BM25Retriever      (sparse text retrieval)
        - CLIPRetriever       (dense image-text retrieval)
        - ColQwen2Retriever   (late-interaction vision retrieval)
        - HybridRetriever     (RRF fusion of multiple retrievers)
    """

    @abstractmethod
    def index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build the retrieval index from a list of documents.

        Args:
            documents: List of dicts with at least 'doc_id' and 'text' or 'image_path'.
        """
        ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        query_image: Optional[Image.Image] = None,
        top_k: int = 5,
    ) -> List[RetrievedDocument]:
        """
        Retrieve the top-k most relevant documents for a query.

        Args:
            query:       Text query string.
            query_image: Optional query image (for multimodal retrieval).
            top_k:       Number of documents to return.

        Returns:
            List of RetrievedDocument sorted by relevance (highest first).
        """
        ...

    @abstractmethod
    def save_index(self, path: str) -> None:
        """Save the index to disk for reuse."""
        ...

    @abstractmethod
    def load_index(self, path: str) -> None:
        """Load a previously saved index from disk."""
        ...
