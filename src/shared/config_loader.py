"""
Unified configuration loader.

Loads YAML config files and resolves relative paths to be
relative to the project root (the directory containing setup.py).
"""

from pathlib import Path
from typing import Dict, Any

import yaml


def get_project_root() -> Path:
    """
    Find the project root directory.

    Walks up from this file until it finds setup.py.
    Falls back to cwd if not found.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "setup.py").exists():
            return current
        current = current.parent
    return Path.cwd()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML config file.

    If config_path is relative, it is resolved relative to the project root.
    """
    root = get_project_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve relative data paths in a config dict to absolute paths.

    Looks for 'images_dir' and 'reports_dir' keys inside config['dataset']
    and makes them absolute relative to the project root.
    """
    root = get_project_root()
    if "dataset" in config:
        ds = config["dataset"]
        for key in ("images_dir", "reports_dir"):
            if key in ds and not Path(ds[key]).is_absolute():
                ds[key] = str(root / ds[key])
    return config
