"""
Scientific Multimodal RAG — Source Package
==========================================
A vision + text hybrid retrieval-augmented generation system
for scientific research papers.
"""

__version__ = "0.1.0"
__author__ = "Vineet"

# Expose key modular interfaces
from src.domains.scientific.utils.helpers import clean_vram, ensure_directories, extract_zip_archive, create_zip_archive
from src.domains.scientific.models.loader import load_colpali, load_scincl, load_qwen2vl
from src.data.pdf_parser import PDFParser
from src.domains.scientific.embeddings.colpali_embedder import ColPaliEmbedder
from src.domains.scientific.embeddings.scincl_embedder import SciNCLEmbedder
from src.domains.scientific.retrieval.colpali_retriever import ColPaliRetriever
from src.domains.scientific.retrieval.text_retriever import TextRetriever
from src.domains.scientific.retrieval.fusion_retriever import FusionRetriever
from src.domains.scientific.generation.self_check import DomainGuard
from src.domains.scientific.generation.rag_generator import RAGGenerator
