"""
Abstract base class for all Vision-Language Models.

Every VLM (LLaVA, Qwen2-VL, future models) implements this interface.
Pipelines, training scripts, and inference scripts program against
BaseVLM — never against a concrete model class. This is what makes
model swapping a config change instead of a code rewrite.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from PIL import Image


@dataclass
class VLMOutput:
    """
    Standard output from any VLM generate() call.

    Every model wrapper must return this exact structure so that
    downstream code (evaluation, logging, grounding) works unchanged.
    """
    answer: str                                     # generated text answer
    raw_output: str                                 # full raw model output
    generation_time_sec: float = 0.0                # wall-clock generation time
    input_token_count: int = 0                      # tokens in the prompt
    output_token_count: int = 0                     # tokens generated
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseVLM(ABC):
    """
    Abstract interface for Vision-Language Models.

    Lifecycle:
        1. __init__()  → store config (no heavy loading)
        2. load()      → load model weights onto GPU
        3. generate()  → run inference
        4. caption()   → generate image caption (optional)

    Subclasses must implement all abstract methods.
    """

    @abstractmethod
    def load(self, config: dict) -> None:
        """
        Load model weights, processor, and tokenizer from config.

        This is where VRAM gets allocated. Call this explicitly
        so that scripts control when the GPU is used.

        Args:
            config: Full model config dict (from model_config.yaml).
        """
        ...

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        question: str,
        context: Optional[str] = None,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> VLMOutput:
        """
        Generate an answer from image + question.

        Args:
            image:          Input image (PIL RGB).
            question:       The clinical question.
            context:        Optional retrieved evidence text (Phase 2+).
            max_new_tokens: Maximum tokens to generate.
            **kwargs:       Additional params (e.g. query_type for routing).

        Returns:
            VLMOutput with the generated answer and metadata.
        """
        ...

    @abstractmethod
    def caption(self, image: Image.Image) -> str:
        """
        Generate a clinical caption for a medical image.

        Used in the offline pipeline to create searchable descriptions.
        If a model doesn't support captioning, return empty string.

        Args:
            image: Input medical image.

        Returns:
            Clinical caption string.
        """
        ...

    @abstractmethod
    def get_memory_footprint(self) -> Dict[str, float]:
        """
        Report current VRAM usage of this model.

        Returns:
            Dict with keys like "model_size_gb", "allocated_gb", etc.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the human-readable model name."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model weights are loaded in memory."""
        ...
