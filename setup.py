"""MMRAG Unified — Setup configuration."""

from setuptools import setup, find_packages

setup(
    name="mmrag-unified",
    version="1.0.0",
    description=(
        "Unified Multimodal Retrieval-Augmented Generation "
        "for Healthcare and Scientific domains"
    ),
    author="MILab IITJ",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "gradio>=4.0.0",
    ],
    extras_require={
        "healthcare": [
            "colpali-engine>=0.3.0",
            "qwen-vl-utils>=0.0.2",
            "peft>=0.7.0",
        ],
        "scientific": [
            "chromadb>=0.4.0",
            "sentence-transformers>=2.2.0",
            "pymupdf>=1.23.0",
            "arxiv>=2.0.0",
        ],
        "all": [
            "colpali-engine>=0.3.0",
            "qwen-vl-utils>=0.0.2",
            "peft>=0.7.0",
            "chromadb>=0.4.0",
            "sentence-transformers>=2.2.0",
            "pymupdf>=1.23.0",
            "arxiv>=2.0.0",
            "streamlit>=1.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mmrag-api=src.api.app:app",
        ],
    },
)
