"""
Scientific Multimodal RAG — Main Entry Point
=============================================

Usage
-----
  # Run full offline ingestion (download → parse → embed → index)
  python main.py --mode offline

  # Launch Streamlit Q&A app (requires completed offline index)
  python main.py --mode online

  # Run everything end-to-end
  python main.py --mode full

  # Custom config
  python main.py --mode online --config configs/config.yaml

HPC Usage (SLURM)
-----------------
  sbatch scripts/slurm_offline.sh      # offline pipeline
  sbatch scripts/slurm_online_gradio.sh # online Streamlit app
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load and resolve paths in the YAML config.

    On HPC, ``RAG_BASE_DIR`` env var is prepended to all relative paths.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base = os.getenv("RAG_BASE_DIR", "")
    if base:
        # Resolve all paths relative to RAG_BASE_DIR
        for key, val in cfg.get("paths", {}).items():
            if not os.path.isabs(val):
                cfg["paths"][key] = os.path.join(base, val)
        print(f"  [config] RAG_BASE_DIR = {base}")

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Mode: offline — ingestion pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_offline(cfg: dict) -> None:
    """Run the full offline ingestion pipeline.

    Steps:
        1. Download arXiv PDFs
        2. Parse PDFs → page images + extracted text
        3. Build ColPali visual embeddings (.npy)
        4. Build SciNCL text embeddings → ChromaDB index
    """
    from pipelines.offline_pipeline import OfflinePipeline

    print("\n" + "═" * 60)
    print("  OFFLINE PIPELINE — Ingestion")
    print("  Steps: Download → Parse → ColPali Embed → SciNCL Index")
    print("═" * 60 + "\n")

    pipeline = OfflinePipeline(cfg)
    pipeline.run()

    print("\n" + "═" * 60)
    print("  ✅ OFFLINE PIPELINE COMPLETE")
    print("  Run: python main.py --mode online  to start the app")
    print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Mode: online — Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

def run_online(cfg: dict) -> None:
    """Launch the Streamlit Q&A application.

    Requires the offline index to be built first.
    Checks for required index files before launching.
    """
    app_cfg = cfg.get("app", {})
    port    = int(os.getenv("STREAMLIT_PORT", app_cfg.get("streamlit_port", 8501)))

    # On HPC bind to 0.0.0.0 (for port-forwarding); locally use localhost
    is_hpc  = bool(os.getenv("RAG_BASE_DIR"))
    host    = "0.0.0.0" if is_hpc else "localhost"

    # Verify indices exist before launching
    paths = cfg.get("paths", {})
    chroma_dir = paths.get("chroma_index", "data/indices/chroma_index")
    npy_dir    = paths.get("multivectors", "data/indices/multivectors")

    if not os.path.exists(chroma_dir):
        print(f"\n  ❌ ChromaDB index not found: {chroma_dir}")
        print("  Run:  python main.py --mode offline  first!")
        sys.exit(1)

    npy_files = [f for f in os.listdir(npy_dir) if f.endswith(".npy")] if os.path.exists(npy_dir) else []
    if not npy_files:
        print(f"\n  ❌ No ColPali .npy embeddings found in: {npy_dir}")
        print("  Run:  python main.py --mode offline  first!")
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  ONLINE PIPELINE — Streamlit Q&A App")
    print(f"  ChromaDB: {len(os.listdir(chroma_dir))} files")
    print(f"  ColPali : {len(npy_files)} page embeddings")
    print(f"  Port    : {port}")
    print("═" * 60)
    print(f"\n  ✅ Open in browser → http://localhost:{port}")
    if is_hpc:
        print(f"  (Port-forward: ssh -L {port}:localhost:{port} divyasaxena_rs@172.25.0.81)")
    print()

    # Pass config path to the Streamlit app via env
    env = os.environ.copy()
    env["RAG_CONFIG_PATH"] = "configs/config.yaml"
    if os.getenv("RAG_BASE_DIR"):
        env["RAG_BASE_DIR"] = os.getenv("RAG_BASE_DIR")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "app/streamlit_app.py",
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    subprocess.run(cmd, env=env)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scientific Multimodal RAG — Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "online", "full"],
        default="online",
        help="Pipeline mode: offline=ingest, online=serve, full=both",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file (default: configs/config.yaml)",
    )

    args = parser.parse_args()

    # ── Banner ──
    print("\n" + "═" * 60)
    print("  Scientific Multimodal RAG System")
    print("  ColPali + SciNCL + Qwen2-VL + Streamlit")
    print(f"  Mode   : {args.mode}")
    print(f"  Config : {args.config}")
    print("═" * 60)

    # ── Load config ──
    cfg = load_config(args.config)

    # ── Create output directories ──
    for key, path in cfg.get("paths", {}).items():
        os.makedirs(path, exist_ok=True)

    # ── Run requested mode ──
    if args.mode in ("offline", "full"):
        run_offline(cfg)

    if args.mode in ("online", "full"):
        run_online(cfg)


if __name__ == "__main__":
    main()
