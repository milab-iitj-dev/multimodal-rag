"""
Unified Model Loader
====================
Provides memory-efficient loaders for ColPali, SciNCL, and Qwen2-VL.
"""

import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor

try:
    from transformers import Qwen2VLForConditionalGeneration
    QWEN_CLASS = Qwen2VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForVision2Seq
    QWEN_CLASS = AutoModelForVision2Seq

def get_model_name(model_cfg, default_name: str) -> str:
    """Helper to extract model_name string if config is a dict."""
    if isinstance(model_cfg, dict):
        return model_cfg.get("model_name", default_name)
    return model_cfg or default_name

def load_colpali(model_cfg, device: str = "cuda"):
    """Loads and returns ColPali model and processor."""
    model_name = get_model_name(model_cfg, "vidore/colpali-v1.2")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    device_map = "cuda" if device == "cuda" else None
    
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    if device == "cpu":
        model = model.to("cpu")
    processor = ColPaliProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor

def load_scincl(model_cfg, device: str = "cuda"):
    """Loads and returns SciNCL SentenceTransformer model."""
    model_name = get_model_name(model_cfg, "malteos/scincl")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model

def load_qwen2vl(model_cfg, device: str = "cuda"):
    """Loads and returns Qwen2-VL model and processor."""
    model_name = get_model_name(model_cfg, "Qwen/Qwen2-VL-2B-Instruct")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    device_map = "auto" if device == "cuda" else None
    
    model = QWEN_CLASS.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    if device == "cpu":
        model = model.to("cpu")
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor
