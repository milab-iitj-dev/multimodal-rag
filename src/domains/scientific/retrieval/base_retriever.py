"""
Base Retriever Interface for the Scientific Multimodal RAG Project.

Defines the core data structures and abstract contract that every
retrieval backend must implement.  Two concrete implementations exist:

* :class:`~src.retrieval.colpali_retriever.ColPaliRetriever` —
  multi-vector page retrieval via MaxSim scoring over ``.npy`` files.
* :class:`~src.retrieval.text_retriever.TextRetriever` —
  dense text retrieval via ChromaDB ANN search.

A third class, :class:`~src.retrieval.fusion_retriever.FusionRetriever`,
combines the two with weighted score fusion.

Example:
    >>> from src.retrieval import ColPaliRetriever
    >>> retriever = ColPaliRetriever(npy_dir="data/indices/multivectors/")
    >>> retriever.load_index("data/indices/multivectors/")
    >>> results = retriever.retrieve(query_embedding, top_k=5)
    >>> for doc in results:
    ...     print(doc.doc_id, doc.page_num, doc.score)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import PIL.Image

from src.shared.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SourceCitation dataclass
# ---------------------------------------------------------------------------

@dataclass
class SourceCitation:
    """Provenance metadata for a retrieved document page.

    Captures enough information to trace a retrieval result back to the
    original scientific paper, including an arXiv URL for direct access.

    Attributes:
        paper_title: Title of the source paper.
        paper_id: Unique identifier for the paper (typically the arXiv
            ID, e.g. ``"2305.12345"``).
        arxiv_url: Full URL to the paper on arXiv, e.g.
            ``"https://arxiv.org/abs/2305.12345"``.
        page_numbers: List of page numbers within the paper that are
            relevant to the query (1-indexed).
        relevance_score: Raw similarity score assigned by the retrieval
            backend before any normalisation.
        page_images: Rendered page images for the cited pages.  These
            can be passed directly to the VLM for answer generation.
        text_snippet: Short text excerpt from the relevant section,
            useful for citation display in the UI.
    """

    paper_title: str
    paper_id: str
    arxiv_url: str
    page_numbers: List[int]
    relevance_score: float
    page_images: List[PIL.Image.Image] = field(default_factory=list)
    text_snippet: str = ""

    def __post_init__(self) -> None:
        """Validate fields after initialisation.

        Raises:
            ValueError: If *paper_id* is empty, *page_numbers* is
                empty, or *relevance_score* is not a finite number.
        """
        if not self.paper_id:
            raise ValueError("SourceCitation.paper_id must not be empty.")
        if not self.page_numbers:
            raise ValueError(
                "SourceCitation.page_numbers must contain at least one page."
            )
        if not isinstance(self.relevance_score, (int, float)):
            raise ValueError(
                f"SourceCitation.relevance_score must be numeric, "
                f"got {type(self.relevance_score).__name__}"
            )


# ---------------------------------------------------------------------------
# RetrievedDocument dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievedDocument:
    """A single retrieval result representing one page of a document.

    Each ``RetrievedDocument`` corresponds to a specific page within a
    paper, carrying both the content (image and/or text) and the
    provenance information needed for citation.

    Attributes:
        doc_id: Unique document identifier, typically an arXiv ID or
            filesystem path (e.g. ``"2305.12345"``).
        page_num: 1-indexed page number within the document.
        score: Similarity score from the retrieval backend.  Higher
            values indicate greater relevance.  Scores may be raw or
            normalised depending on the retriever.
        image: Rendered page image as a ``PIL.Image.Image``, or
            ``None`` if not available.
        text: Extracted text content of the page, or ``None`` if not
            available.
        source_citation: Full provenance metadata linking back to the
            original paper.
        retrieval_method: Name of the retrieval backend that produced
            this result (e.g. ``"colpali"``, ``"scincl"``, ``"fusion"``).
    """

    doc_id: str
    page_num: int
    score: float
    image: Optional[PIL.Image.Image]
    text: Optional[str]
    source_citation: SourceCitation
    retrieval_method: str

    def __post_init__(self) -> None:
        """Validate fields after initialisation.

        Raises:
            ValueError: If *doc_id* is empty, *page_num* is not
                positive, or *retrieval_method* is empty.
        """
        if not self.doc_id:
            raise ValueError("RetrievedDocument.doc_id must not be empty.")
        if self.page_num < 1:
            raise ValueError(
                f"RetrievedDocument.page_num must be >= 1, "
                f"got {self.page_num}"
            )
        if not self.retrieval_method:
            raise ValueError(
                "RetrievedDocument.retrieval_method must not be empty."
            )


# ---------------------------------------------------------------------------
# BaseRetriever abstract base class
# ---------------------------------------------------------------------------

class BaseRetriever(ABC):
    """Abstract base class for retrieval backends.

    Every concrete retriever in the pipeline must subclass this and
    implement :meth:`load_index` and :meth:`retrieve`.

    The lifecycle is:

    1. **load_index(index_path)** — Load pre-computed embeddings from
       disk (``.npy`` files, ChromaDB, etc.).
    2. **retrieve(query_embedding, top_k)** — Score the query against
       all indexed pages and return the top-k results as
       :class:`RetrievedDocument` instances.

    Subclasses may add additional parameters (e.g. collection name,
    persistence directory) via ``__init__``, but must always support
    the two core methods above.

    Note:
        ``load_index`` is intentionally separate from ``__init__`` so
        that a retriever can be instantiated without immediately
        loading large indices into memory.  This is useful when
        embedding and retrieval happen in separate pipeline stages.
    """

    @abstractmethod
    def load_index(self, index_path: str) -> None:
        """Load a pre-computed index from disk.

        Reads embedding data from the specified path and prepares the
        retriever for query operations.  The exact format depends on
        the concrete implementation:

        * ColPaliRetriever → directory of ``.npy`` files.
        * TextRetriever → ChromaDB persistent client directory.

        Args:
            index_path: Filesystem path to the index data.  This may
                be a directory or file depending on the backend.

        Raises:
            FileNotFoundError: If *index_path* does not exist.
            RuntimeError: If the index cannot be loaded (corrupt data,
                missing dependencies, etc.).
        """

    @abstractmethod
    def retrieve(
        self,
        query_embedding: object,
        top_k: int = 5,
    ) -> List[RetrievedDocument]:
        """Retrieve the top-k most relevant documents for a query.

        Scores the query embedding against all indexed pages and
        returns the *top_k* results sorted by descending relevance.

        Args:
            query_embedding: The query representation.  The expected
                type depends on the concrete retriever:

                * ColPaliRetriever — ``torch.Tensor`` of shape
                  ``(num_query_tokens, 128)``.
                * TextRetriever — ``list[float]`` of length 768.
            top_k: Number of results to return.  Defaults to 5.

        Returns:
            A list of :class:`RetrievedDocument` objects sorted by
            descending score.  The list length is at most *top_k*.

        Raises:
            RuntimeError: If the index has not been loaded.
            ValueError: If *top_k* is not a positive integer.
        """
