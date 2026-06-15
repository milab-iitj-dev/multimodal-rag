"""
ColPali Visual Embedder
=======================
Encodes PDF page images to multi-vector representations using ColPali.
"""

import os
import gc
import numpy as np
import torch
from PIL import Image
from src.shared.logging_utils import get_logger

logger = get_logger(__name__)

class ColPaliEmbedder:
    """Encodes page images into multi-vector npy arrays."""

    @staticmethod
    def embed_page(image_path: str, model, processor) -> np.ndarray:
        """Generates visual embeddings for a single page image."""
        img = Image.open(image_path).convert("RGB")
        batch = processor.process_images(images=[img])
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            embeddings = model(**batch)

        vectors = embeddings[0].cpu().float().numpy()
        
        # Explicit VRAM cleaning
        del batch, embeddings, img
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return vectors

    @classmethod
    def batch_embed(cls, image_paths_dict: dict[str, str], output_dir: str, model, processor, status_callback: callable = None):
        """Batch processes page images and saves them as .npy files."""
        total = len(image_paths_dict)
        count = 0
        logger.info("Starting batch ColPali visual embedding for %d pages...", total)
        
        for i, (page_key, img_path) in enumerate(image_paths_dict.items()):
            try:
                vectors = cls.embed_page(img_path, model, processor)
                npy_path = os.path.join(output_dir, f"{page_key}.npy")
                np.save(npy_path, vectors)
                count += 1
                logger.debug("Successfully embedded page: %s", page_key)
                
                if status_callback and ((i + 1) % 10 == 0 or (i + 1) == total):
                    status_callback(i + 1, total, f"Embedded {count}/{total} pages")
                    logger.info("Progress: Embedded %d/%d pages", count, total)
            except torch.cuda.OutOfMemoryError:
                logger.warning("OutOfMemoryError on page %s — clearing cache and skipping...", page_key)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error("Error embedding page %s: %s", page_key, e)
