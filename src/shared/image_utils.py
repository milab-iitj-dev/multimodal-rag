"""
Image loading and display utilities.

Handles image I/O for medical images (PNG, JPEG, DICOM future).
All image operations go through this module so format handling is centralized.
"""

from pathlib import Path
from typing import Tuple, Optional, Union

from PIL import Image


def load_image(
    path: Union[str, Path],
    convert_rgb: bool = True,
) -> Image.Image:
    """
    Load an image from disk.

    Args:
        path:        Path to the image file.
        convert_rgb: Convert to RGB (required for most VLMs).

    Returns:
        PIL Image.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the file cannot be opened as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path)
        if convert_rgb:
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Cannot open image {path}: {e}")


def resize_image(
    image: Image.Image,
    size: Tuple[int, int] = (336, 336),
) -> Image.Image:
    """
    Resize an image to the target dimensions.

    Args:
        image: PIL Image to resize.
        size:  Target (width, height).

    Returns:
        Resized PIL Image.
    """
    return image.resize(size, Image.LANCZOS)


def get_image_info(image: Image.Image) -> dict:
    """Return basic metadata about an image."""
    return {
        "size": image.size,
        "mode": image.mode,
        "format": image.format,
    }
