"""
Shared Utilities and Helper Functions
====================================
"""

import os
import gc
import zipfile
import torch

def clean_vram():
    """Unloads GPU memory cache and triggers garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ensure_directories(dirs_dict: dict, base_dir: str = ""):
    """Creates directory paths listed in the directories dictionary."""
    for name, path in dirs_dict.items():
        full_path = os.path.join(base_dir, path) if base_dir else path
        os.makedirs(full_path, exist_ok=True)

def extract_zip_archive(zip_path: str, extract_to: str) -> bool:
    """Extracts a zip file to the specified target directory."""
    if not os.path.exists(zip_path):
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        return True
    except Exception:
        return False

def create_zip_archive(source_dir: str, zip_path: str) -> bool:
    """Zips the contents of source_dir and writes to zip_path."""
    if not os.path.exists(source_dir):
        return False
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zf.write(file_path, arcname)
        return True
    except Exception:
        return False
