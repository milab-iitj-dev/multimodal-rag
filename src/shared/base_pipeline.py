"""
BasePipeline — The single interface all domain pipelines must expose.

The router calls pipeline.run(query, image=...) and gets a
UnifiedResponse. It never needs to know whether the pipeline
is healthcare, scientific, or anything else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any
from PIL import Image

from src.shared.schemas.response import UnifiedResponse


class BasePipeline(ABC):
    """
    Abstract base class for all domain pipelines.

    Every concrete pipeline (Healthcare, Scientific, or future)
    MUST implement .run() and return a UnifiedResponse.

    Usage:
        pipeline = HealthcarePipeline(config)
        result: UnifiedResponse = pipeline.run("Is there effusion?", image=img)
    """

    @abstractmethod
    def run(
        self,
        query: str,
        image: Optional[Image.Image] = None,
        top_k: int = 3,
        **kwargs: Any,
    ) -> UnifiedResponse:
        """
        Execute the full pipeline and return a unified response.

        Args:
            query:  The user's question.
            image:  Optional PIL image (used by healthcare, ignored by scientific).
            top_k:  Number of documents to retrieve.
            **kwargs: Domain-specific overrides.

        Returns:
            UnifiedResponse with answer, confidence, sources, metadata.
        """
        raise NotImplementedError
