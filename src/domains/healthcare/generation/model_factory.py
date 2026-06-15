"""
Model Factory — config-driven model instantiation.

Usage:
    config = yaml.safe_load(open("configs/model_config.yaml"))
    model = create_model(config)
    model.load(config)

To add a new model:
    1. Create src/models/newmodel.py implementing BaseVLM
    2. Add an entry to MODEL_REGISTRY below
    3. Update model_config.yaml with the new model's settings
"""

from src.domains.healthcare.generation.base_generator import BaseVLM
from src.shared.logging_utils import setup_logger

logger = setup_logger("models.factory")


# ------------------------------------------------------------------ #
#  Registry — maps config names to model classes                       #
# ------------------------------------------------------------------ #
# Lazy imports inside the factory function to avoid loading
# heavy dependencies (transformers, torch) at import time.

MODEL_REGISTRY = {
    "qwen2-vl-7b": "src.domains.healthcare.generation.qwen2_vl_generator.Qwen2VLModel",
}


def create_model(config: dict) -> BaseVLM:
    """
    Instantiate a VLM from config.

    Args:
        config: Full config dict (must have config["model"]["name"]).

    Returns:
        An unloaded BaseVLM instance. Call .load(config) to load weights.

    Raises:
        ValueError: If the model name is not in the registry.
    """
    model_name = config["model"]["name"]

    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. Available models: {available}"
        )

    class_path = MODEL_REGISTRY[model_name]
    module_path, class_name = class_path.rsplit(".", 1)

    # Dynamic import
    import importlib
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    logger.info(f"Creating model: {model_name} ({class_path})")
    return model_class()
