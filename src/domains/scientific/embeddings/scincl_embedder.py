"""
SciNCL Text Embedder
====================
Encodes text paragraphs and stores them in ChromaDB using SciNCL.
"""

import os
import chromadb
from src.shared.logging_utils import get_logger

logger = get_logger(__name__)

class SciNCLEmbedder:
    """Encodes paragraph text and registers with ChromaDB."""

    @staticmethod
    def embed_text_to_chroma(
        page_metadata: dict,
        chroma_dir: str,
        collection_name: str,
        model,
        batch_size: int = 32,
        status_callback: callable = None
    ) -> int:
        """Encodes page texts in batches and stores in a local ChromaDB collection."""
        # 1. Initialize ChromaDB
        logger.info("Initializing ChromaDB client at: %s", chroma_dir)
        chroma_client = chromadb.PersistentClient(path=chroma_dir)

        # Reset collection if exists (fresh start)
        try:
            logger.debug("Deleting existing ChromaDB collection: %s", collection_name)
            chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        logger.info("Creating fresh ChromaDB collection: %s", collection_name)
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # 2. Filter & prepare text data
        texts_to_embed = []
        metas_to_embed = []
        ids_to_embed = []

        for page_key, meta in page_metadata.items():
            text = meta.get("text", "")
            if not text or len(text.strip()) < 20:
                continue
            # Cap text length to SciNCL max input limit
            texts_to_embed.append(text[:512])
            metas_to_embed.append({
                "doc_id": meta["doc_id"],
                "page_num": str(meta["page_num"]),
                "paper_title": meta.get("paper_title", meta["doc_id"])
            })
            ids_to_embed.append(page_key)

        total = len(texts_to_embed)
        logger.info("Found %d paragraphs to encode with SciNCL...", total)
        embedded_count = 0

        # 3. Batch encode & upload
        for i in range(0, total, batch_size):
            batch_texts = texts_to_embed[i : i + batch_size]
            batch_ids = ids_to_embed[i : i + batch_size]
            batch_metas = metas_to_embed[i : i + batch_size]

            embeddings = model.encode(
                batch_texts,
                show_progress_bar=False,
                batch_size=batch_size,
                convert_to_numpy=True
            )

            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings.tolist(),
                documents=[t[:500] for t in batch_texts],
                metadatas=batch_metas
            )

            embedded_count += len(batch_ids)
            if status_callback:
                status_callback(embedded_count, total, f"Embedded {embedded_count}/{total} pages")
            logger.info("Progress: SciNCL indexed %d/%d pages in ChromaDB", embedded_count, total)

        return embedded_count
