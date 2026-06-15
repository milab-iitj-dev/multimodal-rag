"""
Medical image and text preprocessing.

Centralizes all transforms so that training and inference use identical
preprocessing. Never preprocess inside a model wrapper or pipeline script.
"""

from typing import Tuple, Optional
from PIL import Image


class MedicalImagePreprocessor:
    """
    Stateless image preprocessor for medical images.

    Handles resizing and basic normalization. The VLM processor
    (e.g., LlavaProcessor) does its own pixel normalization internally,
    so this class focuses on format standardization only.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (336, 336),
    ):
        self.target_size = target_size

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply full preprocessing pipeline."""
        image = self.ensure_rgb(image)
        image = self.resize(image)
        return image

    def ensure_rgb(self, image: Image.Image) -> Image.Image:
        """Convert to RGB if needed (grayscale X-rays → 3-channel)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def resize(self, image: Image.Image) -> Image.Image:
        """Resize to target dimensions."""
        if image.size != self.target_size:
            image = image.resize(self.target_size, Image.LANCZOS)
        return image


def clean_report_text(text: Optional[str]) -> str:
    """
    Clean and normalize radiology report text.

    Removes artifacts common in OpenI reports:
      - XXXX placeholders (anonymized data)
      - excessive whitespace
      - common filler phrases
    """
    if not text:
        return ""

    # Remove anonymization placeholders
    text = text.replace("XXXX", "").replace("xxxx", "")

    # Normalize whitespace
    text = " ".join(text.split())

    # Remove trailing punctuation artifacts
    text = text.strip(" .,;:")

    return text


def truncate_text(text: str, max_tokens: int = 512) -> str:
    """
    Truncate text to approximately max_tokens.

    Uses a simple word-based approximation (1 token ≈ 0.75 words).
    The actual tokenizer in the model wrapper handles precise truncation.
    """
    if not text:
        return ""
    max_words = int(max_tokens * 0.75)
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."
