"""
Shared VLM Registry — lightweight model discovery and caching.

This is intentionally minimal. It does NOT replace domain-specific
model loading logic. It only provides:
  1. A registry of VLM names → loader callables
  2. A cache to avoid loading the same model twice
  3. A single get_model() function for callers

Domain-specific model wrappers (Qwen2VLModel, etc.) stay in their
domain packages. This registry just knows how to find and cache them.
"""

from __future__ import annotations

from typing import Dict, Callable, Any, Optional
from src.shared.logging_utils import setup_logger

logger = setup_logger("shared.model_registry")

# Global model cache: model_key → loaded model instance
_MODEL_CACHE: Dict[str, Any] = {}

# Global registry: model_key → loader_callable
_MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(key: str, loader: Callable) -> None:
    """
    Register a model loader.

    Args:
        key:    Unique identifier (e.g. "qwen2vl-7b-healthcare").
        loader: Callable that returns a loaded model instance.
                Will be called lazily on first get_model() call.
    """
    _MODEL_REGISTRY[key] = loader
    logger.info(f"Registered model loader: {key}")


def get_model(key: str, force_reload: bool = False) -> Any:
    """
    Get a model by key, loading it on first call and caching.

    Args:
        key:          Model identifier.
        force_reload: If True, reload even if cached.

    Returns:
        The loaded model instance.

    Raises:
        KeyError: If no loader is registered for the key.
    """
    if not force_reload and key in _MODEL_CACHE:
        logger.info(f"Model cache hit: {key}")
        return _MODEL_CACHE[key]

    if key not in _MODEL_REGISTRY:
        raise KeyError(
            f"No model loader registered for '{key}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    logger.info(f"Loading model: {key}...")
    model = _MODEL_REGISTRY[key]()
    _MODEL_CACHE[key] = model
    logger.info(f"Model loaded and cached: {key}")
    return model


def clear_cache(key: Optional[str] = None) -> None:
    """
    Clear model cache.

    Args:
        key: If provided, clear only that model. Otherwise clear all.
    """
    if key:
        _MODEL_CACHE.pop(key, None)
        logger.info(f"Cleared model cache: {key}")
    else:
        _MODEL_CACHE.clear()
        logger.info("Cleared all model cache")


def list_models() -> list:
    """List registered model keys."""
    return list(_MODEL_REGISTRY.keys())
