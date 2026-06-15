"""
Base Vision-Language Model (VLM) Interface.

Defines the abstract contract that every VLM backend in the
Scientific Multimodal RAG project must implement.  Concrete
subclasses (e.g. :class:`Qwen2VLModel`) handle model loading,
generation, and unloading while this module provides the shared
:data:`VLMOutput` dataclass and :class:`BaseVLM` abstract base class.

The staggered-loading strategy required on Kaggle P100 (16 GB VRAM)
means models are loaded **one at a time**: ``load()`` → ``generate()`` →
``unload()``.  Every method therefore logs VRAM usage so operators can
verify that GPU memory is properly released.

Example:
    >>> from src.domains.scientific.models.qwen2vl_model import Qwen2VLModel
    >>> vlm = Qwen2VLModel()
    >>> vlm.load()
    >>> result = vlm.generate("What is the main contribution?", images=[pil_img])
    >>> print(result.answer)
    >>> vlm.unload()
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.shared.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# VLMOutput dataclass
# ---------------------------------------------------------------------------

@dataclass
class VLMOutput:
    """Structured output returned by every VLM ``generate`` call.

    Attributes:
        answer: The generated text answer.  This is the clean, human-readable
            response produced by the model after post-processing (strip
            whitespace, remove special tokens, etc.).
        confidence: A float in [0.0, 1.0] indicating the model's confidence
            in its answer, typically derived from the mean log-probability
            of generated tokens.  Values below ``confidence_threshold``
            (default 0.6) may trigger a retry in the caller.
        source_pages: A list of page-identifier strings cited by the model,
            e.g. ``["page_3", "page_7"]``.  These are extracted from
            bracket-style citations (``[page X]``) in the raw output.
        raw_output: The unprocessed text returned by the model, including
            any special formatting or citation markers.  Useful for
            debugging and for downstream parsers that need the original
            structure.
        logprobs: Optional dictionary mapping each generated token (str)
            to its log-probability (float).  May be ``None`` when the
            model backend does not expose log-probabilities or when
            generation is run without ``output_logprobs=True``.
        generation_time: Wall-clock seconds for the ``generate`` call,
            measured from just before the forward pass to just after
            decoding.  Does **not** include image pre-processing time.
    """

    answer: str
    confidence: float
    source_pages: List[str] = field(default_factory=list)
    raw_output: Optional[str] = None
    logprobs: Optional[Dict[str, float]] = None
    generation_time: float = 0.0

    def __post_init__(self) -> None:
        """Validate fields after initialisation.

        Raises:
            ValueError: If *confidence* is outside [0.0, 1.0] or
                *answer* is an empty string.
        """
        if not self.answer:
            raise ValueError("VLMOutput.answer must not be empty.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"VLMOutput.confidence must be in [0.0, 1.0], got {self.confidence}"
            )


# ---------------------------------------------------------------------------
# BaseVLM abstract base class
# ---------------------------------------------------------------------------

class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models.

    Every concrete VLM used in the pipeline must subclass this and
    implement :meth:`load`, :meth:`generate`, and :meth:`unload`.
    The class enforces a consistent lifecycle:

    1. **load()** — Download (if needed) and initialise the model on GPU.
    2. **generate()** — Run inference with a text prompt and optional
       images; return a :class:`VLMOutput`.
    3. **unload()** — Release GPU memory so the next model in the
       staggered-loading sequence can be loaded.

    Subclasses should call the logger at every major step so that VRAM
    usage can be monitored in production logs.

    Note:
        On Kaggle P100 (16 GB VRAM) only **one** model may be loaded at a
        time.  The caller is responsible for calling ``unload()`` before
        loading a different model.
    """

    # -----------------------------------------------------------------
    # Abstract methods
    # -----------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load the model and its processor / tokenizer onto the device.

        This method must:
            * Download model weights from Hugging Face Hub if not cached.
            * Move the model to the configured device (``"cuda"`` or
              ``"cpu"``).
            * Switch the model to evaluation mode (``model.eval()``).
            * Log peak VRAM usage after loading.

        Raises:
            RuntimeError: If the model cannot be loaded (e.g. OOM,
                download failure, GPU unavailable).
        """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ) -> VLMOutput:
        """Generate an answer given a text prompt and optional images.

        Args:
            prompt: The user question or instruction text.
            images: Optional list of PIL images to condition the model.
                Each element should be a ``PIL.Image.Image`` instance.
                When ``None``, the model operates in text-only mode.
            max_new_tokens: Maximum number of new tokens to generate.
                Must be a positive integer.
            temperature: Sampling temperature.  Lower values (e.g. 0.1)
                produce more deterministic / factual outputs; higher
                values (e.g. 0.8) increase diversity.

        Returns:
            A :class:`VLMOutput` containing the answer, confidence,
            source pages, raw output, log-probabilities, and generation
            time.

        Raises:
            RuntimeError: If the model has not been loaded, or if an
                OOM error occurs during generation.
            ValueError: If *prompt* is empty or *max_new_tokens* <= 0.
        """

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and release GPU memory.

        This method must:
            * Delete the model and processor references.
            * Call ``gc.collect()`` and ``torch.cuda.empty_cache()`` to
              reclaim VRAM.
            * Log the amount of VRAM freed.

        After calling this method, :meth:`load` must be called again
        before the next :meth:`generate` call.

        Raises:
            RuntimeError: If unloading fails unexpectedly.
        """

    # -----------------------------------------------------------------
    # Concrete helper methods
    # -----------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Check whether the model is currently loaded and ready.

        Returns:
            ``True`` if the model has been loaded and not yet unloaded;
            ``False`` otherwise.
        """
        has_model = hasattr(self, "_model") and self._model is not None
        logger.debug("Model loaded check: %s", has_model)
        return has_model

    def _validate_generate_inputs(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        """Validate common inputs to :meth:`generate`.

        Args:
            prompt: The user prompt string.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Raises:
            ValueError: If *prompt* is empty, *max_new_tokens* <= 0,
                or *temperature* < 0.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        if max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be > 0, got {max_new_tokens}"
            )
        if temperature < 0:
            raise ValueError(
                f"temperature must be >= 0, got {temperature}"
            )
        logger.debug(
            "Input validation passed: prompt_len=%d, max_new_tokens=%d, "
            "temperature=%.2f",
            len(prompt),
            max_new_tokens,
            temperature,
        )
