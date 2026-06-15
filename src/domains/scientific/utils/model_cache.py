# src/utils/model_cache.py
"""Utility to set a permanent HuggingFace cache directory.

The repository may be moved between machines, so we keep the cache
inside the project (but ignored via .gitignore).  Importing this module
sets the environment variable *before* any transformer model is
instantiated.
"""
import os
from pathlib import Path

# Directory: <repo_root>/.cache/huggingface
HF_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Export for all downstream libraries (transformers, sentence‑transformers, …)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
