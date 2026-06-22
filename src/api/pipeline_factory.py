"""
Pipeline Factory — lazy-loads real Healthcare and Scientific pipelines.

On HPC (GPU available):
    Loads the actual RAGVQAPipeline and OnlinePipeline with real models.

Locally (no GPU):
    Returns None for inner_pipeline, adapters fall back to placeholder mode.

This module is the ONLY place where heavy model loading happens.
The factory is called once at first /query request (lazy init).

Usage:
    from src.api.pipeline_factory import create_healthcare_pipeline
    from src.api.pipeline_factory import create_scientific_pipeline

    health_inner = create_healthcare_pipeline()  # RAGVQAPipeline or None
    sci_inner = create_scientific_pipeline()      # OnlinePipeline or None
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import yaml

logger = logging.getLogger("mmrag.factory")


def _has_gpu() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _resolve_config_path(relative: str) -> str:
    """Resolve a config path relative to mmrag_unified root."""
    # Try from current directory first
    if os.path.exists(relative):
        return relative

    # Try from the directory where this file lives (src/api/)
    base = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
    resolved = os.path.join(base, relative)
    if os.path.exists(resolved):
        return resolved

    return relative


# ── Healthcare Pipeline ─────────────────────────────────────


def create_healthcare_pipeline() -> Optional[object]:
    """Create a real RAGVQAPipeline if GPU and index are available.

    Returns:
        RAGVQAPipeline instance, or None if resources are unavailable.
    """
    if not _has_gpu():
        logger.warning(
            "Healthcare pipeline: No GPU available — using placeholder mode"
        )
        return None

    try:
        model_config_path = _resolve_config_path(
            "configs/healthcare/model_config.yaml"
        )
        retrieval_config_path = _resolve_config_path(
            "configs/healthcare/retrieval_config.yaml"
        )

        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        with open(retrieval_config_path, "r") as f:
            retrieval_config = yaml.safe_load(f)

        # Determine index directory
        index_dir = (
            retrieval_config
            .get("retrieval", {})
            .get("colqwen2", {})
            .get("index_path", "data/indexes/colqwen2_index")
        )
        index_dir = _resolve_config_path(index_dir)

        if not os.path.exists(os.path.join(index_dir, "document_store.json")):
            logger.warning(
                f"Healthcare index not found at {index_dir} — "
                "using placeholder mode"
            )
            return None

        # Load VLM
        from src.domains.healthcare.generation.model_factory import create_model
        model_name = model_config["model"]["name"]
        logger.info(f"Loading Healthcare VLM: {model_name}...")
        vlm = create_model(model_config)
        vlm.load(model_config)

        # Create pipeline
        from pipelines.healthcare.rag_vqa import RAGVQAPipeline
        pipeline = RAGVQAPipeline(
            vlm=vlm,
            retrieval_config=retrieval_config,
            index_dir=index_dir,
            top_k=3,
        )

        logger.info("Healthcare pipeline loaded successfully")
        return pipeline

    except Exception as e:
        logger.error(f"Failed to load Healthcare pipeline: {e}")
        return None


# ── Scientific Pipeline ─────────────────────────────────────


def create_scientific_pipeline() -> Optional[object]:
    """Create a real OnlinePipeline if GPU and indices are available.

    Returns:
        OnlinePipeline instance, or None if resources are unavailable.
    """
    if not _has_gpu():
        logger.warning(
            "Scientific pipeline: No GPU available — using placeholder mode"
        )
        return None

    try:
        config_path = _resolve_config_path(
            "configs/scientific/config.yaml"
        )

        with open(config_path, "r") as f:
            sci_config = yaml.safe_load(f)

        # Verify indices exist
        base = os.getenv("RAG_BASE_DIR", "")
        indices_dir = sci_config.get("paths", {}).get("indices", "data/indices")
        if base:
            indices_dir = os.path.join(base, indices_dir)
        else:
            indices_dir = _resolve_config_path(indices_dir)

        metadata_path = os.path.join(indices_dir, "page_metadata.json")
        if not os.path.exists(metadata_path):
            logger.warning(
                f"Scientific index not found at {metadata_path} — "
                "using placeholder mode"
            )
            return None

        # Create pipeline
        from pipelines.scientific.online_pipeline import OnlinePipeline
        pipeline = OnlinePipeline(sci_config)

        logger.info("Scientific pipeline loaded successfully")
        return pipeline

    except Exception as e:
        logger.error(f"Failed to load Scientific pipeline: {e}")
        return None
