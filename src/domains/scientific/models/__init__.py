"""
Models Package — Vision-Language Model Backends.

Re-exports the core classes so that downstream modules can import
from the package root::

    from src.models import BaseVLM, VLMOutput, Qwen2VLModel, ModelFactory
"""

from src.domains.scientific.models.base_vlm import BaseVLM, VLMOutput
from src.domains.scientific.models.qwen2vl_model import Qwen2VLModel
from src.domains.scientific.models.model_factory import ModelFactory

__all__ = [
    "BaseVLM",
    "VLMOutput",
    "Qwen2VLModel",
    "ModelFactory",
]
