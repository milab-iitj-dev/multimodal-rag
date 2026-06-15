"""
Generation Package — Answer generation and verification.
"""

from src.domains.scientific.generation.self_check import DomainGuard
from src.domains.scientific.generation.rag_generator import RAGGenerator

__all__ = [
    "DomainGuard",
    "RAGGenerator",
]
