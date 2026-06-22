"""
Healthcare Pipeline Adapter — wraps RAGVQAPipeline behind BasePipeline.

Translates healthcare-specific RAGOutput into UnifiedResponse
without changing any retrieval, grounding, or generation logic.
"""

from __future__ import annotations

from typing import Optional, Any
from PIL import Image

from src.shared.base_pipeline import BasePipeline
from src.shared.schemas.response import UnifiedResponse, SourceItem
from src.shared.logging_utils import setup_logger

logger = setup_logger("pipeline.healthcare_adapter")


class HealthcarePipeline(BasePipeline):
    """
    Wraps the existing RAGVQAPipeline to expose the BasePipeline interface.

    All internal logic (ColQwen2 retrieval, RRF fusion, evidence
    aggregation, grounding, confidence) is delegated unchanged.
    Only the output is translated to UnifiedResponse.
    """

    def __init__(self, inner_pipeline=None, **kwargs):
        """
        Args:
            inner_pipeline: A loaded RAGVQAPipeline instance.
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
        Execute healthcare RAG pipeline and return UnifiedResponse.

        Delegates to RAGVQAPipeline.run_single(), then converts
        the healthcare-specific RAGOutput into a UnifiedResponse
        with full metadata for the frozen API contract.
        """
        if self.inner is None:
            return UnifiedResponse(
                domain="healthcare",
                answer=(
                    f"[Healthcare] Pipeline not loaded. "
                    f"Query received: '{query}'"
                ),
                confidence=0.0,
                sources=[],
                metadata={"status": "placeholder"},
            )

        # Delegate to the real pipeline
        output = self.inner.run_single(
            query=query,
            query_image=image,
            top_k=top_k,
        )

        # Convert sources: RetrievedDocument → SourceItem
        sources = []
        for doc in output.retrieved_docs:
            # Extract per-doc retrieval scores for the API contract
            doc_meta = doc.metadata or {}
            sources.append(SourceItem(
                title=f"Case {doc.doc_id}",
                score=doc.score,
                snippet=doc_meta.get(
                    "findings", doc.text[:200] if doc.text else ""
                ),
                url="",
                page_numbers=[],
                metadata={
                    "doc_id": doc.doc_id,
                    "rank": doc_meta.get("rank", 0),
                    "impression": doc_meta.get("impression", ""),
                    # Per-doc scores for retrieval_metadata mapping:
                    "image_score": doc_meta.get("image_score", 0.0),
                    "text_score": doc_meta.get("text_score", 0.0),
                    "rrf_score": doc_meta.get("rrf_score", doc.score),
                },
            ))

        # Convert confidence: ConfidenceResult → float
        conf_score = 0.0
        conf_level = "UNKNOWN"
        if output.confidence:
            conf_score = output.confidence.score
            conf_level = output.confidence.level

        # Extract grounding result
        grounding_passed = True
        was_corrected = False
        if output.grounding_result:
            grounding_passed = output.grounding_result.is_grounded
            was_corrected = output.grounding_result.was_corrected

        # Determine retrieval method for contract
        retrieval_method = "fused"
        query_type = output.metadata.get("query_type", "")
        if query_type == "text_only":
            retrieval_method = "scincl_only"

        return UnifiedResponse(
            domain="healthcare",
            answer=output.answer,
            confidence=conf_score,
            sources=sources,
            metadata={
                # Confidence & verification fields
                "confidence_level": conf_level,
                "grounding_passed": grounding_passed,
                # Retrieval metadata for API contract
                "retrieval_method": retrieval_method,
                # Pipeline internals
                "query_type": query_type,
                "consensus": output.metadata.get("consensus", ""),
                "was_corrected": was_corrected,
                "retrieval_time_sec": output.retrieval_time_sec,
                "generation_time_sec": output.generation_time_sec,
                "total_time_sec": output.total_time_sec,
                "num_retrieved": len(output.retrieved_docs),
            },
        )
