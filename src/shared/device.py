"""
GPU and device utilities.

Provides device detection, VRAM reporting, and environment diagnostics.
Used by every module that touches the GPU to ensure consistent device placement.
"""

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

import platform
import sys
from typing import Dict, Any


def get_device_info() -> Dict[str, Any]:
    """
    Collect full device and environment information.

    Returns:
        Dictionary with GPU name, VRAM, CUDA version, Python version, etc.
    """
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "torch_version": torch.__version__ if _TORCH_AVAILABLE else "not installed",
        "cuda_available": torch.cuda.is_available() if _TORCH_AVAILABLE else False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpus": [],
    }

    if _TORCH_AVAILABLE and torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            gpu = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1e9, 2
                ),
                "allocated_memory_gb": round(
                    torch.cuda.memory_allocated(i) / 1e9, 2
                ),
                "reserved_memory_gb": round(
                    torch.cuda.memory_reserved(i) / 1e9, 2
                ),
            }
            info["gpus"].append(gpu)

    return info


def print_gpu_status(logger=None):
    """Print a formatted GPU status report."""
    info = get_device_info()
    lines = [
        "=" * 50,
        "DEVICE STATUS",
        "=" * 50,
        f"  Python:       {info['python_version']}",
        f"  Platform:     {info['platform']}",
        f"  PyTorch:      {info['torch_version']}",
        f"  CUDA:         {info['cuda_available']}",
    ]

    if info["cuda_available"]:
        lines.append(f"  CUDA Version: {info['cuda_version']}")
        for gpu in info["gpus"]:
            lines.append(f"  GPU {gpu['index']}:       {gpu['name']}")
            lines.append(f"    Total VRAM:   {gpu['total_memory_gb']} GB")
            lines.append(f"    Allocated:    {gpu['allocated_memory_gb']} GB")
            lines.append(f"    Reserved:     {gpu['reserved_memory_gb']} GB")
    else:
        lines.append("  [!] No GPU detected -- running on CPU")

    lines.append("=" * 50)
    report = "\n".join(lines)

    if logger:
        logger.info(report)
    else:
        print(report)

    return info


def get_device(preference: str = "auto") -> "torch.device":
    """
    Resolve a device preference string to a torch.device.

    Args:
        preference: "auto", "cuda:0", "cpu", etc.

    Returns:
        torch.device for model placement.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Install it: pip install torch")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def get_vram_usage_gb(device_index: int = 0) -> Dict[str, float]:
    """Get current VRAM usage in GB for a specific GPU."""
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}
    return {
        "allocated": round(torch.cuda.memory_allocated(device_index) / 1e9, 2),
        "reserved": round(torch.cuda.memory_reserved(device_index) / 1e9, 2),
        "total": round(
            torch.cuda.get_device_properties(device_index).total_memory / 1e9, 2
        ),
    }
