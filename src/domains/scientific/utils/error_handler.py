"""
Error Handler — Retry Logic, Fallback Chains, and OOM Recovery.

Provides robust error handling utilities for the Scientific Multimodal
RAG pipeline, including:

* Exponential backoff retry for transient failures (network, API).
* Fallback chains that try alternative functions when the primary
  function fails.
* OOM (Out-of-Memory) auto-recovery that reduces batch size.
* Query validation to catch malformed inputs early.
* Structured error logging for debugging and monitoring.

Fallback Constants
------------------
The following string constants define the fallback strategy names
used throughout the pipeline for logging and result tracking:

* ``FALLBACK_COLPALI_TO_SCINCL`` — ColPali failed, using SciNCL only.
* ``FALLBACK_SCINCL_TO_TFIDF`` — SciNCL failed, using TF-IDF.
* ``FALLBACK_VLM_TO_SOURCES`` — VLM failed, returning sources only.

Example:
    >>> from src.domains.scientific.utils.error_handler import retry_with_backoff, validate_query
    >>> result = retry_with_backoff(my_api_call, max_retries=3)
    >>> is_valid, msg = validate_query("What is attention?")
    >>> print(is_valid, msg)
    True ''
"""

from __future__ import annotations

import functools
import re
import time
import traceback
from typing import Any, Callable, List, Optional, Tuple, TypeVar

from src.shared.logging_utils import get_logger

logger = get_logger(__name__)

# Type variable for generic function signatures
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Fallback strategy constants
# ---------------------------------------------------------------------------

FALLBACK_COLPALI_TO_SCINCL: str = "colpali_to_scincl"
"""ColPali retrieval failed — falling back to SciNCL only."""

FALLBACK_SCINCL_TO_TFIDF: str = "scincl_to_tfidf"
"""SciNCL retrieval failed — falling back to keyword TF-IDF search."""

FALLBACK_VLM_TO_SOURCES: str = "vlm_to_sources"
"""VLM generation failed — returning sources only (no answer)."""


# ---------------------------------------------------------------------------
# retry_with_backoff
# ---------------------------------------------------------------------------

def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    delay: float = 30.0,
    backoff_factor: float = 2.0,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Execute a function with exponential backoff retry logic.

    Calls *func* with the provided arguments.  If the call raises an
    exception, waits *delay* seconds and retries, doubling the delay
    each time (exponential backoff).  After *max_retries* failures,
    the last exception is re-raised.

    This is useful for transient failures such as network timeouts,
    API rate limits, or temporary GPU memory pressure.

    Args:
        func: The callable to execute.
        max_retries: Maximum number of retry attempts.  Defaults to 3
            (total of 4 attempts including the initial call).
        delay: Initial delay in seconds before the first retry.
            Defaults to 30 seconds.
        backoff_factor: Multiplier applied to the delay after each
            retry.  Defaults to 2 (delay doubles each time).
        *args: Positional arguments to pass to *func*.
        **kwargs: Keyword arguments to pass to *func*.

    Returns:
        The return value of *func* on the first successful call.

    Raises:
        Exception: Re-raises the last exception if all retries fail.

    Example:
        >>> def flaky_api():
        ...     # Simulates an API that sometimes fails
        ...     import random
        ...     if random.random() < 0.5:
        ...         raise ConnectionError("API timeout")
        ...     return "success"
        >>> result = retry_with_backoff(flaky_api, max_retries=3, delay=1.0)
    """
    last_exception: Optional[Exception] = None
    current_delay = delay

    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            if attempt > 0:
                logger.info(
                    "retry_with_backoff succeeded on attempt %d/%d",
                    attempt + 1,
                    max_retries + 1,
                )
            return result

        except Exception as exc:
            last_exception = exc

            if attempt < max_retries:
                logger.warning(
                    "retry_with_backoff: attempt %d/%d failed — %s.  "
                    "Retrying in %.1f s (backoff_factor=%.1f).",
                    attempt + 1,
                    max_retries + 1,
                    str(exc)[:200],
                    current_delay,
                    backoff_factor,
                )
                time.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                logger.error(
                    "retry_with_backoff: all %d attempts failed — %s",
                    max_retries + 1,
                    str(exc)[:200],
                )

    # All retries exhausted — re-raise the last exception
    raise last_exception  # type: ignore[misc]


# ---------------------------------------------------------------------------
# fallback_chain
# ---------------------------------------------------------------------------

def fallback_chain(*funcs: Callable[..., T]) -> T:
    """Try each function in sequence until one succeeds.

    Calls each function in order.  If a function raises an exception,
    the next function is tried.  If all functions fail, the last
    exception is re-raised.

    This implements a simple fallback pattern where alternative
    strategies are attempted in priority order.

    Args:
        *funcs: Callables to try in order.  Each callable should
            accept no arguments (use ``functools.partial`` or
            lambda to bind arguments).

    Returns:
        The return value of the first function that succeeds.

    Raises:
        ValueError: If no functions are provided.
        Exception: Re-raises the last exception if all functions fail.

    Example:
        >>> def primary():
        ...     raise RuntimeError("Primary failed")
        >>> def secondary():
        ...     return "secondary result"
        >>> result = fallback_chain(primary, secondary)
        >>> print(result)
        'secondary result'
    """
    if not funcs:
        raise ValueError("fallback_chain requires at least one function.")

    last_exception: Optional[Exception] = None

    for i, func in enumerate(funcs):
        try:
            result = func()
            if i > 0:
                logger.info(
                    "fallback_chain: function %d succeeded after %d failure(s).",
                    i + 1,
                    i,
                )
            return result

        except Exception as exc:
            last_exception = exc
            logger.warning(
                "fallback_chain: function %d/%d failed — %s",
                i + 1,
                len(funcs),
                str(exc)[:200],
            )

    # All functions failed
    logger.error(
        "fallback_chain: all %d functions failed.",
        len(funcs),
    )
    raise last_exception  # type: ignore[misc]


# ---------------------------------------------------------------------------
# handle_oom
# ---------------------------------------------------------------------------

def handle_oom(
    embedder: Any,
    current_batch_size: int,
) -> int:
    """Auto-reduce batch size on GPU Out-of-Memory errors.

    Called after catching a ``torch.cuda.OutOfMemoryError`` during
    batch embedding.  Empties the CUDA cache and returns a reduced
    batch size for the next attempt.

    The reduction strategy is:
    - If current batch size > 4: reduce by half
    - If current batch size is 2-4: reduce to 1
    - If current batch size is 1: recommend CPU fallback

    Args:
        embedder: The embedder object (must have an ``unload()``
            method or be ``None``).  Currently unused but kept for
            future model-specific recovery logic.
        current_batch_size: The batch size that caused the OOM.

    Returns:
        A reduced batch size that should fit in available VRAM.
        Returns 0 if CPU fallback is recommended.

    Example:
        >>> try:
        ...     embeddings = model.embed_batch(images, batch_size=8)
        ... except torch.cuda.OutOfMemoryError:
        ...     new_batch_size = handle_oom(model, 8)
        ...     embeddings = model.embed_batch(images, batch_size=new_batch_size)
    """
    import gc

    import torch

    logger.warning(
        "OOM with batch_size=%d — clearing cache and reducing batch size.",
        current_batch_size,
    )

    # Clear CUDA cache to free fragmented memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reduce batch size
    if current_batch_size > 4:
        new_batch_size = current_batch_size // 2
    elif current_batch_size > 1:
        new_batch_size = 1
    else:
        # Already at batch_size=1 — recommend CPU fallback
        logger.error(
            "OOM at batch_size=1 — GPU memory exhausted.  "
            "Recommend CPU fallback."
        )
        return 0

    logger.info(
        "handle_oom: batch_size reduced from %d to %d.",
        current_batch_size,
        new_batch_size,
    )

    return new_batch_size


# ---------------------------------------------------------------------------
# validate_query
# ---------------------------------------------------------------------------

def validate_query(query: str) -> Tuple[bool, str]:
    """Validate that a query is well-formed and not gibberish.

    Checks for the following conditions:

    1. **Not empty**: Query must contain non-whitespace characters.
    2. **Not too short**: Query must have at least 3 words after
       normalisation.
    3. **Not gibberish**: Query must contain at least one alphabetic
       word of length >= 3 (filters out keyboard mashing like
       "asdf asdf").

    Args:
        query: The user's query string.

    Returns:
        A tuple ``(is_valid, message)`` where *is_valid* is ``True``
        if the query passes all checks, and *message* is an empty
        string on success or a descriptive error message on failure.

    Example:
        >>> validate_query("What is the attention mechanism?")
        (True, '')
        >>> validate_query("")
        (False, 'Query is empty.')
        >>> validate_query("hi")
        (False, 'Query is too short — must contain at least 3 words.')
        >>> validate_query("asdf qwer zxcv")
        (False, 'Query appears to be gibberish — no meaningful words detected.')
    """
    if not query or not query.strip():
        return False, "Query is empty."

    # Normalize: strip, collapse whitespace
    normalized = " ".join(query.strip().split())

    # Check word count
    words = normalized.split()
    if len(words) < 3:
        return False, (
            "Query is too short — must contain at least 3 words."
        )

    # Check for meaningful words (alphabetic, length >= 3)
    meaningful_words = [
        w for w in words
        if len(w) >= 3 and re.match(r"^[a-zA-Z]+$", w)
    ]

    if not meaningful_words:
        return False, (
            "Query appears to be gibberish — no meaningful words detected."
        )

    return True, ""


# ---------------------------------------------------------------------------
# log_error
# ---------------------------------------------------------------------------

def log_error(
    module: str,
    error: Exception,
    context: str = "",
) -> None:
    """Log a structured error message with module and context information.

    Provides a consistent error logging format across the entire
    pipeline, making it easy to search and filter logs for debugging.

    The log entry includes:
    * **Module**: The pipeline component where the error occurred.
    * **Error type**: The exception class name.
    * **Error message**: The exception's string representation.
    * **Context**: Additional context about what was happening when
      the error occurred.
    * **Traceback**: The full traceback at DEBUG level.

    Args:
        module: Name of the module or component where the error
            occurred, e.g. ``"colpali_embedder"``, ``"rag_generator"``.
        error: The exception instance.
        context: Additional context string describing what operation
            was being performed when the error occurred.

    Example:
        >>> try:
        ...     result = model.embed_image(img)
        ... except RuntimeError as e:
        ...     log_error("colpali_embedder", e, "embedding page 3")
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # Structured log entry at ERROR level
    log_message = (
        f"[{module}] {error_type}: {error_msg}"
    )
    if context:
        log_message += f" | Context: {context}"

    logger.error(log_message)

    # Full traceback at DEBUG level for detailed debugging
    logger.debug(
        "[%s] Full traceback for %s:\n%s",
        module,
        error_type,
        traceback.format_exc(),
    )
