"""
Document store for managing the medical knowledge base.

Provides a clean interface to store and retrieve documents (reports,
images, metadata) by ID. Acts as the central repository that the
index builder populates and the retrievers query against.

Persistence: JSON on disk. Each document is an image-report pair
from the OpenI dataset, with full metadata for traceability.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, field, asdict

from src.shared.logging_utils import setup_logger

logger = setup_logger("indexing.docstore")


@dataclass
class Document:
    """
    A single indexable document — one image-report pair.

    This is the retrieval unit for the ColQwen2 index. Each document
    corresponds to one OpenI sample: a chest X-ray image paired with
    its radiology report text and clinical metadata.
    """
    doc_id: str                                     # unique identifier (OpenI uid)
    text: str = ""                                  # full report text (findings + impression)
    image_path: str = ""                            # absolute path to X-ray image
    findings: Optional[str] = None                  # findings section
    impression: Optional[str] = None                # impression / conclusion section
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata may contain: mesh_terms, problems, indication, projection, etc.


class DocumentStore:
    """
    In-memory document store with JSON persistence.

    Stores image-report pair documents that form the retrieval
    knowledge base. The index builder populates this store during
    offline indexing, and retrievers look up documents by ID
    after retrieval.

    Usage:
        store = DocumentStore()
        store.add_document(Document(doc_id="123", text="...", image_path="..."))
        store.save("data/indexes/document_store.json")

        # Later...
        store = DocumentStore()
        store.load("data/indexes/document_store.json")
        doc = store.get_document("123")
    """

    def __init__(self):
        self._documents: Dict[str, Document] = {}

    # ------------------------------------------------------------------ #
    #  CRUD operations                                                     #
    # ------------------------------------------------------------------ #

    def add_document(self, doc: Document) -> None:
        """
        Add a document to the store.

        Args:
            doc: Document to add. Overwrites if doc_id already exists.
        """
        self._documents[doc.doc_id] = doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID.

        Args:
            doc_id: The document identifier.

        Returns:
            Document if found, None otherwise.
        """
        return self._documents.get(doc_id)

    def has_document(self, doc_id: str) -> bool:
        """Check if a document exists in the store."""
        return doc_id in self._documents

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document by ID.

        Returns:
            True if document was found and removed, False otherwise.
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    def list_documents(self) -> Iterator[Document]:
        """Iterate over all documents in the store."""
        yield from self._documents.values()

    def get_all_doc_ids(self) -> List[str]:
        """Return all document IDs in insertion order."""
        return list(self._documents.keys())

    def __len__(self) -> int:
        """Return the number of documents in the store."""
        return len(self._documents)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._documents

    # ------------------------------------------------------------------ #
    #  Persistence (JSON)                                                  #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """
        Save the document store to a JSON file.

        Args:
            path: Output file path (e.g., 'data/indexes/document_store.json').
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "2.0",
            "num_documents": len(self._documents),
            "documents": {
                doc_id: asdict(doc)
                for doc_id, doc in self._documents.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Document store saved: {len(self._documents)} documents -> {path}")

    def load(self, path: str) -> None:
        """
        Load the document store from a JSON file.

        Args:
            path: Input file path.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document store not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._documents.clear()

        for doc_id, doc_data in data.get("documents", {}).items():
            # Handle both field naming conventions:
            #   - DocumentStore.save() writes: doc_id, text, findings, impression
            #   - Kaggle index builder writes: case_id, report, findings, impression
            resolved_id = doc_data.get("doc_id", doc_data.get("case_id", doc_id))
            resolved_text = doc_data.get("text", doc_data.get("report", ""))

            self._documents[doc_id] = Document(
                doc_id=resolved_id,
                text=resolved_text,
                image_path=doc_data.get("image_path", ""),
                findings=doc_data.get("findings"),
                impression=doc_data.get("impression"),
                metadata=doc_data.get("metadata", {}),
            )

        logger.info(f"Document store loaded: {len(self._documents)} documents <- {path}")

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics about the document store."""
        has_text = sum(1 for d in self._documents.values() if d.text)
        has_image = sum(1 for d in self._documents.values() if d.image_path)
        has_findings = sum(1 for d in self._documents.values() if d.findings)
        has_impression = sum(1 for d in self._documents.values() if d.impression)

        return {
            "total_documents": len(self._documents),
            "with_text": has_text,
            "with_image": has_image,
            "with_findings": has_findings,
            "with_impression": has_impression,
        }
