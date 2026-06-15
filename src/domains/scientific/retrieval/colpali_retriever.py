"""
ColPali Retriever
=================
Performs visual document page similarity searches using ColPali MaxSim.
"""

import os
import numpy as np
import torch
from src.shared.logging_utils import get_logger

logger = get_logger(__name__)

class ColPaliRetriever:
    """Retrieves document pages using multi-vector similarity matches."""

    @staticmethod
    def retrieve(
        query: str,
        npy_index: dict[str, str],
        page_metadata: dict,
        model,
        processor,
        top_k: int = 10
    ) -> list[dict]:
        """Calculates visual similarity scores and returns top k results."""
        logger.info("Executing ColPali visual similarity search...")
        # 1. Encode query
        batch = processor.process_queries(queries=[query])
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            query_emb = model(**batch)

        query_vec = query_emb[0].cpu().float().numpy()

        # Clean query vectors from VRAM
        del batch, query_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Score pages
        scores = []
        logger.debug("Calculating MaxSim scores over %d index .npy files...", len(npy_index))
        for page_key, npy_path in npy_index.items():
            if not os.path.exists(npy_path):
                continue
            try:
                page_vec = np.load(npy_path)
                sim_matrix = np.matmul(query_vec, page_vec.T)
                score = float(sim_matrix.max(axis=1).mean())
                scores.append((page_key, score))
            except Exception as e:
                logger.error("Error loading/scoring page vector %s: %s", page_key, e)
                continue

        # 3. Sort & build output
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]

        results = []
        for i, (page_key, score) in enumerate(top_scores):
            meta = page_metadata.get(page_key, {})
            results.append({
                "page_key": page_key,
                "score": score,
                "doc_id": meta.get("doc_id", ""),
                "page_num": meta.get("page_num", 0),
                "paper_title": meta.get("paper_title", ""),
                "image_path": meta.get("image_path", ""),
                "text": meta.get("text", "")
            })
            logger.debug("ColPali Hit [%d]: Page %s (MaxSim Score: %.4f)", i+1, page_key, score)

        logger.info("ColPali search completed. Retrieved %d visual matches.", len(results))
        return results
