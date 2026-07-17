"""
Scientific Pipeline Adapter — wraps OnlinePipeline behind BasePipeline.

Translates scientific-specific RAGResult into UnifiedResponse
without changing any retrieval, generation, or self-check logic.
"""

from __future__ import annotations

from typing import Optional, Any
from PIL import Image

from src.shared.base_pipeline import BasePipeline
from src.shared.schemas.response import UnifiedResponse, SourceItem
from src.shared.logging_utils import setup_logger

logger = setup_logger("pipeline.scientific_adapter")


class ScientificPipeline(BasePipeline):
    """
    Wraps the existing OnlinePipeline to expose the BasePipeline interface.

    All internal logic (ColPali, SciNCL, ChromaDB, weighted fusion,
    self-check) is delegated unchanged. Only the output is translated
    to UnifiedResponse.
    """

    def __init__(self, inner_pipeline=None, **kwargs):
        """
        Args:
            inner_pipeline: A loaded OnlinePipeline instance.
                            If None, the pipeline is in demo/placeholder mode.
        """
        self.inner = inner_pipeline

    def run(
        self,
        query: str,
        image: Optional[Image.Image] = None,
        top_k: int = 3,
        **kwargs: Any,
    ) -> UnifiedResponse:
        """
        Execute scientific RAG pipeline and return UnifiedResponse.

        Delegates to OnlinePipeline.query(), then converts
        the scientific-specific RAGResult into a UnifiedResponse.

        Note: image is ignored — scientific pipeline is text-only.
        """
        if self.inner is None:
            return UnifiedResponse(
                domain="scientific",
                answer=(
                    f"[Scientific] Pipeline not loaded. "
                    f"Query received: '{query}'"
                ),
                confidence=0.0,
                sources=[],
                metadata={"status": "placeholder"},
            )

        # Delegate to the real pipeline
        result = self.inner.query(query)

        # Convert sources: SourceCitation → SourceItem
        sources = []

        if hasattr(result, 'sources') and result.sources:
            for s in result.sources:
                rel_score = getattr(s, 'relevance_score', 0.0)
                sources.append(SourceItem(
                    title=getattr(s, 'paper_title', 'Unknown'),
                    score=rel_score,
                    snippet=getattr(s, 'text_snippet', ''),
                    url=getattr(s, 'arxiv_url', ''),
                    page_numbers=getattr(s, 'page_numbers', []),
                    metadata={
                        "paper_id": getattr(s, 'paper_id', ''),
                        "colpali_norm_score": getattr(s, 'colpali_norm_score', 0.0),
                        "scincl_norm_score": getattr(s, 'scincl_norm_score', 0.0),
                    },
                ))

        # Extract top-ranked component scores propagated by OnlinePipeline
        top_colpali_score = getattr(result, 'top_colpali_score', 0.0)
        top_scincl_score = getattr(result, 'top_scincl_score', 0.0)
        top_fused_score = getattr(result, 'top_fused_score', 0.0)

        # Build metadata from check_result
        check_passed = False
        attr_passed = False
        faith_passed = False
        if hasattr(result, 'check_result') and result.check_result:
            check_passed = result.check_result.passed
            attr_passed = result.check_result.attribution_passed
            faith_passed = result.check_result.faithfulness_passed

        return UnifiedResponse(
            domain="scientific",
            answer=result.answer,
            confidence=result.confidence,
            sources=sources,
            metadata={
                # Verification fields for API contract
                "self_check_passed": check_passed,
                "attribution_passed": attr_passed,
                "faithfulness_passed": faith_passed,
                # Retrieval metadata for API contract
                # Scientific always runs both ColPali + SciNCL → fused
                "retrieval_method": "fused",
                "visual_score": top_colpali_score,
                "text_score": top_scincl_score,
                "fusion_score": top_fused_score,
                # Pipeline internals
                "total_time_sec": result.total_time,
                "retries": getattr(result, 'retries', 0),
                "num_retrieved": len(sources),
            },
        )
