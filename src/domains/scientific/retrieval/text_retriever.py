"""
Text Retriever
==============
Performs textual document similarity queries using ChromaDB and SciNCL.
"""

import chromadb
from src.shared.logging_utils import get_logger

logger = get_logger(__name__)

class TextRetriever:
    """Retrieves document pages using ChromaDB text searches."""

    @staticmethod
    def retrieve(
        query_embedding: list[float],
        chroma_dir: str,
        collection_name: str,
        page_metadata: dict,
        top_k: int = 10
    ) -> list[dict]:
        """Queries ChromaDB and returns top k results."""
        logger.info("Executing ChromaDB text search in database: %s", chroma_dir)
        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        collections = chroma_client.list_collections()
        
        # Verify collection exists
        col_names = [c.name for c in collections]
        if collection_name not in col_names:
            if collections:
                logger.warning("Requested collection %s not found. Defaulting to %s", collection_name, collections[0].name)
                collection_name = collections[0].name
            else:
                logger.error("No collections found in ChromaDB.")
                return []

        logger.debug("Querying collection '%s' for top %d matches...", collection_name, top_k)
        collection = chroma_client.get_collection(collection_name)
        results_db = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        if not results_db or not results_db.get("ids") or not results_db["ids"][0]:
            logger.warning("No matches returned from ChromaDB.")
            return []

        # Build results
        results = []
        for i, doc_id in enumerate(results_db["ids"][0]):
            meta = results_db["metadatas"][0][i]
            dist = results_db["distances"][0][i]
            sim = 1.0 - dist
            page_meta = page_metadata.get(doc_id, {})
            results.append({
                "page_key": doc_id,
                "score": sim,
                "doc_id": meta.get("doc_id", ""),
                "page_num": int(meta.get("page_num", 0)),
                "paper_title": meta.get("paper_title", ""),
                "image_path": page_meta.get("image_path", ""),
                "text": page_meta.get("text", "")
            })
            logger.debug("ChromaDB Hit [%d]: Page %s (Similarity: %.4f)", i+1, doc_id, sim)

        logger.info("ChromaDB query completed. Retrieved %d candidate pages.", len(results))
        return results
