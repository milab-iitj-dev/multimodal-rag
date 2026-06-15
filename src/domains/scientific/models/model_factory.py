"""
Model Factory for the Scientific Multimodal RAG Project.

Implements the **Factory** pattern with an internal registry that maps
human-readable model type strings (e.g. ``"qwen2vl"``) to their
concrete :class:`BaseVLM` subclasses.  This decouples configuration-
driven model selection from instantiation logic, making it trivial to
add new VLM backends without modifying caller code.

Example:
    >>> from src.domains.scientific.models.model_factory import ModelFactory
    >>> config = {"model_name": "Qwen/Qwen2-VL-2B-Instruct", "quantization": "4bit"}
    >>> vlm = ModelFactory.get_model("qwen2vl", config)
    >>> vlm.load()
"""

from __future__ import annotations

from typing import Any, Dict, Type

from src.domains.scientific.models.base_vlm import BaseVLM
from src.shared.logging_utils import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Factory class that creates VLM instances from a type string and config.

    The factory maintains an internal **registry** (a class-level
    dictionary) that maps model type identifiers to their implementing
    classes.  New backends can be registered via
    :meth:`register_model` or by directly extending the
    ``_REGISTRY`` dictionary.

    Attributes:
        _REGISTRY: Class-level mapping of model type strings to
            :class:`BaseVLM` subclasses.
    """

    _REGISTRY: Dict[str, Type[BaseVLM]] = {}

    # -----------------------------------------------------------------
    # Registry management
    # -----------------------------------------------------------------

    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseVLM]) -> None:
        """Register a new VLM class under a given type key.

        Args:
            model_type: A short identifier, e.g. ``"qwen2vl"``.
            model_class: The :class:`BaseVLM` subclass to instantiate
                when this type is requested.

        Raises:
            TypeError: If *model_class* is not a subclass of
                :class:`BaseVLM`.
            ValueError: If *model_type* is already registered and the
                caller is overwriting a different class.
        """
        if not issubclass(model_class, BaseVLM):
            raise TypeError(
                f"model_class must be a subclass of BaseVLM, "
                f"got {model_class!r}"
            )

        existing = cls._REGISTRY.get(model_type)
        if existing is not None and existing is not model_class:
            logger.warning(
                "Overwriting existing registry entry for '%s': "
                "%s → %s",
                model_type,
                existing.__name__,
                model_class.__name__,
            )

        cls._REGISTRY[model_type] = model_class
        logger.info("Registered model type '%s' → %s", model_type, model_class.__name__)

    @classmethod
    def list_models(cls) -> list[str]:
        """Return a sorted list of all registered model type strings.

        Returns:
            A sorted list of keys in the registry, e.g.
            ``["qwen2vl"]``.
        """
        return sorted(cls._REGISTRY.keys())

    # -----------------------------------------------------------------
    # Factory method
    # -----------------------------------------------------------------

    @classmethod
    def get_model(cls, model_type: str, config: Dict[str, Any]) -> BaseVLM:
        """Create and return a VLM instance for the given type.

        The *config* dictionary is unpacked as keyword arguments into
        the model class constructor.  Only keys that the constructor
        accepts are forwarded; unknown keys are silently ignored so
        that a full YAML config section can be passed without trimming.

        Args:
            model_type: Registry key, e.g. ``"qwen2vl"``.
            config: Dictionary of keyword arguments forwarded to the
                model constructor.  Example::

                    {
                        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
                        "quantization": "4bit",
                        "device": "cuda",
                    }

        Returns:
            An instantiated (but **not yet loaded**) :class:`BaseVLM`
            subclass.  The caller must call ``.load()`` before
            ``.generate()``.

        Raises:
            ValueError: If *model_type* is not found in the registry.
            TypeError: If *config* is not a dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError(
                f"config must be a dict, got {type(config).__name__}"
            )

        if model_type not in cls._REGISTRY:
            available = cls.list_models()
            raise ValueError(
                f"Unknown model type '{model_type}'.  "
                f"Available types: {available}.  "
                f"Use ModelFactory.register_model() to add a new type."
            )

        model_class = cls._REGISTRY[model_type]
        logger.info(
            "Creating %s from config with keys: %s",
            model_class.__name__,
            list(config.keys()),
        )

        # Filter config to only include parameters accepted by the constructor.
        import inspect
        sig = inspect.signature(model_class.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        filtered_config = {
            k: v for k, v in config.items() if k in valid_params
        }

        if len(filtered_config) < len(config):
            skipped = set(config.keys()) - set(filtered_config.keys())
            logger.debug(
                "Skipped unrecognized config keys for %s: %s",
                model_class.__name__,
                skipped,
            )

        instance = model_class(**filtered_config)
        logger.info("Created %s instance.", model_class.__name__)
        return instance


# ---------------------------------------------------------------------------
# Register built-in models
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    """Import and register all built-in VLM implementations.

    This is called at module import time so that the factory is
    ready to use immediately.
    """
    from src.domains.scientific.models.qwen2vl_model import Qwen2VLModel

    ModelFactory.register_model("qwen2vl", Qwen2VLModel)

    logger.debug("Built-in models registered: %s", ModelFactory.list_models())


_register_builtins()
