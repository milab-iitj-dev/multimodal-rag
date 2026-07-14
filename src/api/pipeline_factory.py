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
import time
import logging
import traceback
from typing import Optional

import yaml

logger = logging.getLogger("mmrag.factory")


def _has_gpu() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            logger.info(
                f"GPU check: available=True, "
                f"device_count={torch.cuda.device_count()}, "
                f"device={torch.cuda.get_device_name(0)}, "
                f"torch={torch.__version__}, "
                f"cuda={torch.version.cuda}"
            )
        else:
            logger.warning(
                f"GPU check: available=False, "
                f"torch={torch.__version__}, "
                f"cuda_build={torch.version.cuda or 'NONE'}"
            )
        return available
    except ImportError:
        logger.warning("GPU check: torch not installed")
        return False


def _resolve_config_path(relative: str) -> str:
    """Resolve a config path relative to mmrag_unified root."""
    # Try from current directory first
    if os.path.exists(relative):
        return os.path.abspath(relative)

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
    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info("Healthcare Pipeline Factory — BEGIN")
    logger.info(f"  CWD: {os.getcwd()}")
    logger.info(f"  __file__: {os.path.abspath(__file__)}")

    if not _has_gpu():
        logger.warning(
            "Healthcare pipeline: No GPU available → placeholder mode"
        )
        return None

    try:
        # ── Config resolution ──
        model_config_path = _resolve_config_path(
            "configs/healthcare/model_config.yaml"
        )
        retrieval_config_path = _resolve_config_path(
            "configs/healthcare/retrieval_config.yaml"
        )
        logger.info(f"  model_config_path:     {model_config_path}")
        logger.info(f"    exists: {os.path.exists(model_config_path)}")
        logger.info(f"  retrieval_config_path: {retrieval_config_path}")
        logger.info(f"    exists: {os.path.exists(retrieval_config_path)}")

        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        with open(retrieval_config_path, "r") as f:
            retrieval_config = yaml.safe_load(f)

        # ── Index resolution ──
        index_dir = (
            retrieval_config
            .get("retrieval", {})
            .get("colqwen2", {})
            .get("index_path", "data/indexes/colqwen2_index")
        )
        logger.info(f"  index_path (from config): {index_dir}")
        index_dir = _resolve_config_path(index_dir)
        logger.info(f"  index_path (resolved):    {index_dir}")

        doc_store_path = os.path.join(index_dir, "document_store.json")
        logger.info(f"  document_store.json:      {doc_store_path}")
        logger.info(f"    exists: {os.path.exists(doc_store_path)}")

        if not os.path.exists(doc_store_path):
            logger.warning(
                f"Healthcare index not found at {index_dir} → "
                "placeholder mode"
            )
            return None

        # ── Load VLM ──
        from src.domains.healthcare.generation.model_factory import create_model
        model_name = model_config["model"]["name"]
        logger.info(f"  Loading VLM: {model_name} ...")
        t_vlm = time.monotonic()
        vlm = create_model(model_config)
        vlm.load(model_config)
        logger.info(f"  VLM loaded in {time.monotonic() - t_vlm:.1f}s")

        # ── Create pipeline ──
        from pipelines.healthcare.rag_vqa import RAGVQAPipeline
        logger.info("  Creating RAGVQAPipeline ...")
        t_pipe = time.monotonic()
        pipeline = RAGVQAPipeline(
            vlm=vlm,
            retrieval_config=retrieval_config,
            index_dir=index_dir,
            top_k=3,
        )
        logger.info(
            f"  RAGVQAPipeline created in {time.monotonic() - t_pipe:.1f}s"
        )
        logger.info(
            f"  Pipeline object: {type(pipeline).__name__} id={id(pipeline)}"
        )

        elapsed = time.monotonic() - t0
        logger.info(f"Healthcare Pipeline Factory — SUCCESS ({elapsed:.1f}s)")
        logger.info("=" * 60)
        return pipeline

    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error(
            f"Healthcare Pipeline Factory — FAILED ({elapsed:.1f}s): {e}"
        )
        logger.error(traceback.format_exc())
        return None


# ── Scientific Pipeline ─────────────────────────────────────


def create_scientific_pipeline() -> Optional[object]:
    """Create a real OnlinePipeline if GPU and indices are available.

    Returns:
        OnlinePipeline instance, or None if resources are unavailable.
    """
    logger.info("Scientific Pipeline Factory — BEGIN")

    if not _has_gpu():
        logger.info("Scientific pipeline: No GPU → disabled")
        return None

    try:
        config_path = _resolve_config_path(
            "configs/scientific/config.yaml"
        )

        if not os.path.exists(config_path):
            logger.info(
                f"Scientific pipeline: config not found at {config_path} → "
                "disabled"
            )
            return None

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
        logger.info(f"  Scientific index: {metadata_path}")
        logger.info(f"    exists: {os.path.exists(metadata_path)}")

        if not os.path.exists(metadata_path):
            logger.info(
                f"Scientific pipeline: index not found at "
                f"{metadata_path} → disabled"
            )
            return None

        # Create pipeline
        from pipelines.scientific.online_pipeline import OnlinePipeline
        pipeline = OnlinePipeline(sci_config)

        logger.info(
            f"Scientific Pipeline Factory — SUCCESS "
            f"(id={id(pipeline)})"
        )
        return pipeline

    except Exception as e:
        logger.error(f"Scientific pipeline disabled — {e}")
        logger.error(traceback.format_exc())
        return None
