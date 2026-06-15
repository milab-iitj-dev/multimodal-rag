"""
Healthcare MRAG — UI Launcher.

CLI entry point for launching the professional inference UI.
Loads models, initializes the RAG pipeline, and starts Gradio.

Usage:
    # Minimal (uses default configs)
    python scripts/launch_ui.py

    # Full options
    python scripts/launch_ui.py \\
        --model-config configs/model_config.yaml \\
        --retrieval-config configs/retrieval_config.yaml \\
        --index-dir data/indexes/colqwen2_index \\
        --port 7860 \\
        --share

    # HPC / DGX
    srun --gres=gpu:1 --mem=32G python scripts/launch_ui.py --share

    # UI preview only (no GPU needed)
    python scripts/launch_ui.py --ui-only
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Healthcare MRAG — Professional Inference UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/launch_ui.py                          # Default configs\n"
            "  python scripts/launch_ui.py --share                  # Public URL\n"
            "  python scripts/launch_ui.py --ui-only                # Layout preview\n"
            "  python scripts/launch_ui.py --index-dir /path/to/idx # Custom index\n"
        ),
    )
    parser.add_argument(
        "--model-config",
        default="configs/model_config.yaml",
        help="Path to model config YAML (default: configs/model_config.yaml)",
    )
    parser.add_argument(
        "--retrieval-config",
        default="configs/retrieval_config.yaml",
        help="Path to retrieval config YAML (default: configs/retrieval_config.yaml)",
    )
    parser.add_argument(
        "--index-dir",
        default="data/indexes/colqwen2_index",
        help="Path to pre-built ColQwen2 index directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve (default: 3)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1, use 0.0.0.0 for external access)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link (useful for HPC)",
    )
    parser.add_argument(
        "--ui-only",
        action="store_true",
        help="Launch UI without loading models (layout preview mode)",
    )
    args = parser.parse_args()

    # ── Banner ──
    print()
    print("=" * 60)
    print("  Healthcare MRAG — Professional Inference UI")
    print("=" * 60)

    pipeline = None
    gpu_name = None

    if args.ui_only:
        # UI preview mode — no model loading
        print("  Mode:  UI preview (no models loaded)")
        print("  Note:  Inference disabled. Use this to preview layout.")
    else:
        # Full pipeline mode — load everything
        import yaml
        import torch

        # GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU:   {gpu_name} ({vram:.1f} GB)")
        else:
            print("  GPU:   Not available (CPU mode)")

        # Load configs
        print(f"  Model config:     {args.model_config}")
        print(f"  Retrieval config: {args.retrieval_config}")
        print(f"  Index dir:        {args.index_dir}")
        print("-" * 60)

        with open(args.model_config, "r") as f:
            model_config = yaml.safe_load(f)
        with open(args.retrieval_config, "r") as f:
            retrieval_config = yaml.safe_load(f)

        # Load VLM via existing factory pattern
        from src.domains.healthcare.generation.model_factory import create_model

        print("  Loading VLM model...")
        t0 = time.time()
        vlm = create_model(model_config)
        vlm.load(model_config)
        print(f"  VLM loaded in {time.time() - t0:.1f}s")

        # Create RAG pipeline via existing class
        from pipelines.rag_vqa import RAGVQAPipeline

        print("  Initializing RAG pipeline...")
        t0 = time.time()
        pipeline = RAGVQAPipeline(
            vlm=vlm,
            retrieval_config=retrieval_config,
            index_dir=args.index_dir,
            top_k=args.top_k,
        )
        print(f"  Pipeline ready in {time.time() - t0:.1f}s")

    # ── Launch UI ──
    print("-" * 60)
    print(f"  Server:  http://{args.server_name}:{args.port}")
    if args.share:
        print("  Share:   Public URL will be generated")
    print("=" * 60)
    print()

    from ui.app import create_app

    app, launch_kwargs = create_app(pipeline=pipeline, gpu_name=gpu_name)
    app.queue(max_size=5)
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True,
        **launch_kwargs,
    )


if __name__ == "__main__":
    main()
