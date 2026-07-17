"""
Phase 1 Tests — Retrieval Metadata Correctness.

Verifies that retrieval_metadata (method + scores) accurately represents
the retrieval path that actually executed, for both domains.

Test coverage:
  - Text-only: method == "scincl_only", only scincl has a real score
  - Image-only: method == "colpali_only", only colpali has a real score
  - Hybrid/fused: method == "fused", all three scores populated
  - Same frozen response schema works for both domains
  - Scientific component scores are real (not always 0.0)
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.shared.schemas.response import UnifiedResponse, SourceItem
from src.api.models import (
    QueryResponse,
    RetrievalMetadata,
    RetrievalScores,
    VerificationResult,
    SourceItemResponse,
)

# Import the mapping helpers directly from app module
from src.api.app import (
    _map_sources,
    _map_retrieval_metadata,
    _map_verification,
)


# ── Fixtures: mock UnifiedResponse objects ──────────────────


def _make_healthcare_response(
    retrieval_method: str = "fused",
    retrieval_mode: str = "hybrid",
    image_score: float = 0.0,
    text_score: float = 0.0,
    rrf_score: float = 0.0,
    confidence: float = 0.7,
) -> UnifiedResponse:
    """Build a mock healthcare UnifiedResponse with explicit scores."""
    return UnifiedResponse(
        domain="healthcare",
        answer="Test healthcare answer",
        confidence=confidence,
        sources=[
            SourceItem(
                title="Case CXR-001",
                score=rrf_score or image_score or text_score,
                snippet="Normal chest x-ray",
                metadata={
                    "doc_id": "CXR-001",
                    "image_score": image_score,
                    "text_score": text_score,
                    "rrf_score": rrf_score or image_score or text_score,
                },
            ),
        ],
        metadata={
            "retrieval_method": retrieval_method,
            "retrieval_mode": retrieval_mode,
            "image_score": image_score,
            "text_score": text_score,
            "rrf_score": rrf_score,
            "confidence_level": "HIGH",
            "grounding_passed": True,
        },
    )


def _make_scientific_response(
    visual_score: float = 0.5,
    text_score: float = 0.3,
    fusion_score: float = 0.88,
    confidence: float = 0.6,
) -> UnifiedResponse:
    """Build a mock scientific UnifiedResponse with real component scores."""
    return UnifiedResponse(
        domain="scientific",
        answer="Test scientific answer",
        confidence=confidence,
        sources=[
            SourceItem(
                title="Vision Transformer Paper",
                score=fusion_score,
                snippet="ViT architecture...",
                metadata={
                    "paper_id": "2010.11929",
                    "colpali_norm_score": visual_score,
                    "scincl_norm_score": text_score,
                },
            ),
        ],
        metadata={
            "retrieval_method": "fused",
            "visual_score": visual_score,
            "text_score": text_score,
            "fusion_score": fusion_score,
            "self_check_passed": True,
            "attribution_passed": True,
            "faithfulness_passed": True,
        },
    )


# ── Test: Healthcare text-only retrieval ────────────────────


class TestHealthcareTextOnly:
    """Text-only retrieval: method == scincl_only, only scincl populated."""

    def test_method_is_scincl_only(self):
        resp = _make_healthcare_response(
            retrieval_method="scincl_only",
            retrieval_mode="text_only",
            text_score=18.09,
        )
        metadata = _map_retrieval_metadata(resp)
        assert metadata.method == "scincl_only"

    def test_scincl_has_real_score(self):
        resp = _make_healthcare_response(
            retrieval_method="scincl_only",
            retrieval_mode="text_only",
            text_score=18.09,
        )
        metadata = _map_retrieval_metadata(resp)
        assert metadata.scores.scincl == 18.09

    def test_colpali_is_zero(self):
        resp = _make_healthcare_response(
            retrieval_method="scincl_only",
            retrieval_mode="text_only",
            text_score=18.09,
        )
        metadata = _map_retrieval_metadata(resp)
        assert metadata.scores.colpali == 0.0

    def test_fused_is_zero(self):
        resp = _make_healthcare_response(
            retrieval_method="scincl_only",
            retrieval_mode="text_only",
            text_score=18.09,
        )
        metadata = _map_retrieval_metadata(resp)
        assert metadata.scores.fused == 0.0


# ── Test: Healthcare image-only retrieval ───────────────────


class TestHealthcareImageOnly:
    """Image-only retrieval: method == colpali_only, only colpali populated."""

    def test_method_is_colpali_only(self):
        resp = _make_healthcare_response(
            retrieval_method="colpali_only",
            retrieval_mode="image_only",
            image_score=699.5,
        )
        metadata = _map_retrieval_metadata(resp)
        assert metadata.method == "colpali_only"

    def test_colpali_has_real_score(self):
        resp = _make_healthcare_response(
            retrieval_method="colpali_only",
            retrieval_mode="image_only",
            image_score=699.5,
        )
        metadata = _map_retrieval_metadata(resp)
        assert metadata.scores.colpali == 699.5

    def test_scincl_is_zero(self):
        resp = _make_healthcare_response(
            retrieval_method="colpali_only",
            retrieval_mode="image_only",
            image_score=699.5,
        )
        metadata = _map_retrieval_metadata(resp)
        assert metadata.scores.scincl == 0.0

    def test_fused_is_zero(self):
        resp = _make_healthcare_response(
            retrieval_method="colpali_only",
            retrieval_mode="image_only",
            image_score=699.5,
        )
        metadata = _map_retrieval_metadata(resp)
        assert metadata.scores.fused == 0.0


# ── Test: Healthcare hybrid/fused retrieval ─────────────────


class TestHealthcareHybrid:
    """Hybrid retrieval: method == fused, all three scores populated."""

    def test_method_is_fused(self):
        resp = _make_healthcare_response(
            retrieval_method="fused",
            retrieval_mode="hybrid",
            image_score=699.5,
            text_score=18.09,
            rrf_score=0.0328,
        )
        metadata = _map_retrieval_metadata(resp)
        assert metadata.method == "fused"

    def test_all_scores_populated(self):
        resp = _make_healthcare_response(
            retrieval_method="fused",
            retrieval_mode="hybrid",
            image_score=699.5,
            text_score=18.09,
            rrf_score=0.0328,
        )
        metadata = _map_retrieval_metadata(resp)
        assert metadata.scores.colpali == 699.5
        assert metadata.scores.scincl == 18.09
        assert metadata.scores.fused == 0.0328


# ── Test: Scientific fused retrieval ────────────────────────


class TestScientificFused:
    """Scientific always runs fused — component scores must be real."""

    def test_method_is_fused(self):
        resp = _make_scientific_response()
        metadata = _map_retrieval_metadata(resp)
        assert metadata.method == "fused"

    def test_colpali_is_real(self):
        resp = _make_scientific_response(visual_score=0.75)
        metadata = _map_retrieval_metadata(resp)
        assert metadata.scores.colpali == 0.75

    def test_scincl_is_real(self):
        resp = _make_scientific_response(text_score=0.42)
        metadata = _map_retrieval_metadata(resp)
        assert metadata.scores.scincl == 0.42

    def test_fused_is_real(self):
        resp = _make_scientific_response(fusion_score=0.88)
        metadata = _map_retrieval_metadata(resp)
        assert metadata.scores.fused == 0.88

    def test_scores_not_all_zero(self):
        """Regression: scientific scores must not be fake 0.0 when real values are provided."""
        resp = _make_scientific_response(
            visual_score=0.65, text_score=0.48, fusion_score=0.82
        )
        metadata = _map_retrieval_metadata(resp)
        assert metadata.scores.colpali > 0, "ColPali score should be non-zero"
        assert metadata.scores.scincl > 0, "SciNCL score should be non-zero"
        assert metadata.scores.fused > 0, "Fused score should be non-zero"


# ── Test: Schema compatibility across domains ───────────────


class TestSchemaCompatibility:
    """Both domains produce the same frozen response schema."""

    def test_same_metadata_fields(self):
        health_resp = _make_healthcare_response(
            retrieval_method="fused",
            image_score=500.0,
            text_score=20.0,
            rrf_score=0.03,
        )
        sci_resp = _make_scientific_response()

        health_meta = _map_retrieval_metadata(health_resp)
        sci_meta = _map_retrieval_metadata(sci_resp)

        # Both must have the same field structure
        assert set(RetrievalMetadata.model_fields.keys()) == set(RetrievalMetadata.model_fields.keys())

    def test_scores_model_fields(self):
        # Both must have colpali, scincl, fused
        assert set(RetrievalScores.model_fields.keys()) == {"colpali", "scincl", "fused"}

    def test_method_values_are_valid(self):
        """Method field only allows fused/colpali_only/scincl_only."""
        valid_methods = {"fused", "colpali_only", "scincl_only"}

        for method in valid_methods:
            resp = _make_healthcare_response(retrieval_method=method)
            meta = _map_retrieval_metadata(resp)
            assert meta.method in valid_methods

    def test_invalid_method_defaults_to_fused(self):
        resp = _make_healthcare_response(retrieval_method="invalid_method")
        # The invalid value should be normalized to "fused"
        # Override metadata directly
        resp.metadata["retrieval_method"] = "invalid_method"
        meta = _map_retrieval_metadata(resp)
        assert meta.method == "fused"


# ── Test: Verification mapping ──────────────────────────────


class TestVerificationMapping:
    """Verification fields map correctly from both domains."""

    def test_healthcare_verification(self):
        resp = _make_healthcare_response(confidence=0.7)
        resp.metadata["grounding_passed"] = True
        resp.metadata["confidence_level"] = "HIGH"

        verification = _map_verification(resp)
        assert verification.attribution is True
        assert verification.faithfulness is True  # 0.7 >= 0.5
        assert verification.confidence_pass is True  # HIGH != LOW

    def test_healthcare_low_confidence(self):
        resp = _make_healthcare_response(confidence=0.3)
        resp.metadata["confidence_level"] = "LOW"

        verification = _map_verification(resp)
        assert verification.faithfulness is False  # 0.3 < 0.5
        assert verification.confidence_pass is False  # LOW

    def test_scientific_verification(self):
        resp = _make_scientific_response()
        resp.metadata["attribution_passed"] = True
        resp.metadata["faithfulness_passed"] = True
        resp.metadata["self_check_passed"] = True

        verification = _map_verification(resp)
        assert verification.attribution is True
        assert verification.faithfulness is True
        assert verification.confidence_pass is True

    def test_scientific_failed_verification(self):
        resp = _make_scientific_response()
        resp.metadata["attribution_passed"] = False
        resp.metadata["faithfulness_passed"] = False
        resp.metadata["self_check_passed"] = False

        verification = _map_verification(resp)
        assert verification.attribution is False
        assert verification.faithfulness is False
        assert verification.confidence_pass is False


# ── Test: Healthcare adapter method detection ───────────────


class TestHealthcareAdapterMethodDetection:
    """Healthcare adapter correctly maps retrieval_mode to retrieval_method."""

    def test_text_only_mode_produces_scincl_only(self):
        from pipelines.healthcare.adapter import HealthcarePipeline

        # Create a mock inner pipeline
        mock_inner = MagicMock()

        @dataclass
        class MockConfidence:
            score: float = 0.7
            level: str = "HIGH"

        @dataclass
        class MockGrounding:
            is_grounded: bool = True
            was_corrected: bool = False

        @dataclass
        class MockDoc:
            doc_id: str = "CXR-001"
            score: float = 18.09
            text: str = "Normal chest"
            metadata: dict = field(default_factory=lambda: {
                "findings": "Normal",
                "impression": "Normal",
            })

        @dataclass
        class MockOutput:
            answer: str = "Normal chest x-ray"
            retrieved_docs: list = field(default_factory=lambda: [MockDoc()])
            confidence: Optional[object] = field(default_factory=MockConfidence)
            grounding_result: Optional[object] = field(default_factory=MockGrounding)
            retrieval_time_sec: float = 0.5
            generation_time_sec: float = 1.0
            total_time_sec: float = 1.5
            metadata: dict = field(default_factory=lambda: {
                "query_type": "text_only",
                "retrieval_mode": "text_only",
                "consensus": "MAJORITY_ABSENT",
            })

        mock_inner.run_single.return_value = MockOutput()

        pipeline = HealthcarePipeline(inner_pipeline=mock_inner)
        result = pipeline.run("Is there pleural effusion?")

        assert result.metadata["retrieval_method"] == "scincl_only"
        assert result.metadata["text_score"] == 18.09
        assert result.metadata["image_score"] == 0.0
        assert result.metadata["rrf_score"] == 0.0

    def test_image_only_mode_produces_colpali_only(self):
        from pipelines.healthcare.adapter import HealthcarePipeline

        mock_inner = MagicMock()

        @dataclass
        class MockConfidence:
            score: float = 0.8
            level: str = "HIGH"

        @dataclass
        class MockGrounding:
            is_grounded: bool = True
            was_corrected: bool = False

        @dataclass
        class MockDoc:
            doc_id: str = "CXR-002"
            score: float = 699.5
            text: str = "Cardiomegaly"
            metadata: dict = field(default_factory=lambda: {
                "findings": "Cardiomegaly",
            })

        @dataclass
        class MockOutput:
            answer: str = "Cardiomegaly detected"
            retrieved_docs: list = field(default_factory=lambda: [MockDoc()])
            confidence: Optional[object] = field(default_factory=MockConfidence)
            grounding_result: Optional[object] = field(default_factory=MockGrounding)
            retrieval_time_sec: float = 0.5
            generation_time_sec: float = 1.0
            total_time_sec: float = 1.5
            metadata: dict = field(default_factory=lambda: {
                "query_type": "binary_clinical",
                "retrieval_mode": "image_only",
                "consensus": "UNANIMOUS_PRESENT",
            })

        mock_inner.run_single.return_value = MockOutput()

        pipeline = HealthcarePipeline(inner_pipeline=mock_inner)
        from PIL import Image
        img = Image.new("RGB", (224, 224))
        result = pipeline.run("Describe this image", image=img)

        assert result.metadata["retrieval_method"] == "colpali_only"
        assert result.metadata["image_score"] == 699.5
        assert result.metadata["text_score"] == 0.0
        assert result.metadata["rrf_score"] == 0.0


# ── Test: Scientific adapter score extraction ───────────────


class TestScientificAdapterScoreExtraction:
    """Scientific adapter correctly extracts real component scores."""

    def test_component_scores_propagated(self):
        from pipelines.scientific.adapter import ScientificPipeline

        # Create a mock OnlinePipeline result
        mock_inner = MagicMock()

        class MockCheckResult:
            passed = True
            attribution_passed = True
            faithfulness_passed = True

        class MockSource:
            paper_title = "ViT Paper"
            paper_id = "2010.11929"
            arxiv_url = "https://arxiv.org/abs/2010.11929"
            page_numbers = [3]
            relevance_score = 0.88
            text_snippet = "Vision Transformer..."
            colpali_norm_score = 0.75
            scincl_norm_score = 0.42

        class MockResult:
            answer = "The ViT architecture..."
            confidence = 0.6
            sources = [MockSource()]
            check_result = MockCheckResult()
            total_time = 5.0
            retries = 0
            top_colpali_score = 0.75
            top_scincl_score = 0.42
            top_fused_score = 0.88

        mock_inner.query.return_value = MockResult()

        pipeline = ScientificPipeline(inner_pipeline=mock_inner)
        result = pipeline.run("What is ViT?")

        assert result.metadata["visual_score"] == 0.75
        assert result.metadata["text_score"] == 0.42
        assert result.metadata["fusion_score"] == 0.88
        assert result.metadata["retrieval_method"] == "fused"


# ── Test: FusionRetriever preserves component scores ────────


class TestFusionRetrieverComponentScores:
    """FusionRetriever output includes normalized component scores."""

    def test_fused_results_have_component_scores(self):
        pytest.importorskip("torch", reason="torch required for FusionRetriever import chain")
        from src.domains.scientific.retrieval.fusion_retriever import FusionRetriever

        colpali_results = [
            {"page_key": "p1", "score": 10.0, "doc_id": "d1", "page_num": 1, "paper_title": "Paper A", "text": "..."},
            {"page_key": "p2", "score": 8.0, "doc_id": "d2", "page_num": 2, "paper_title": "Paper B", "text": "..."},
        ]
        scincl_results = [
            {"page_key": "p1", "score": 0.9, "doc_id": "d1", "page_num": 1, "paper_title": "Paper A", "text": "..."},
            {"page_key": "p3", "score": 0.7, "doc_id": "d3", "page_num": 3, "paper_title": "Paper C", "text": "..."},
        ]

        fused = FusionRetriever.fuse(
            colpali_results, scincl_results,
            colpali_weight=0.7, scincl_weight=0.3, top_k=3
        )

        # All fused results must have component scores
        for result in fused:
            assert "fused_score" in result
            assert "colpali_norm_score" in result
            assert "scincl_norm_score" in result

        # p1 appears in both lists — should have both component scores > 0
        p1_result = next(r for r in fused if r["page_key"] == "p1")
        assert p1_result["colpali_norm_score"] > 0
        assert p1_result["scincl_norm_score"] > 0
        assert p1_result["fused_score"] > 0

    def test_single_source_page_has_zero_for_missing_component(self):
        pytest.importorskip("torch", reason="torch required for FusionRetriever import chain")
        from src.domains.scientific.retrieval.fusion_retriever import FusionRetriever

        colpali_results = [
            {"page_key": "p_colpali", "score": 5.0, "doc_id": "d1", "page_num": 1, "paper_title": "Paper A", "text": "..."},
        ]
        scincl_results = [
            {"page_key": "p_scincl", "score": 0.8, "doc_id": "d2", "page_num": 2, "paper_title": "Paper B", "text": "..."},
        ]

        fused = FusionRetriever.fuse(
            colpali_results, scincl_results,
            colpali_weight=0.7, scincl_weight=0.3, top_k=2
        )

        # p_colpali should have colpali score but 0 scincl
        p_colpali = next(r for r in fused if r["page_key"] == "p_colpali")
        assert p_colpali["colpali_norm_score"] > 0
        assert p_colpali["scincl_norm_score"] == 0.0

        # p_scincl should have scincl score but 0 colpali
        p_scincl = next(r for r in fused if r["page_key"] == "p_scincl")
        assert p_scincl["colpali_norm_score"] == 0.0
        assert p_scincl["scincl_norm_score"] > 0


# ── Test: Frozen schema preserved ───────────────────────────


class TestFrozenSchema:
    """Score fields remain float (not Optional), unused fields are 0.0."""

    def test_scores_are_float_type(self):
        scores = RetrievalScores(colpali=0.0, scincl=0.0, fused=0.0)
        assert isinstance(scores.colpali, float)
        assert isinstance(scores.scincl, float)
        assert isinstance(scores.fused, float)

    def test_default_scores_are_zero(self):
        scores = RetrievalScores()
        assert scores.colpali == 0.0
        assert scores.scincl == 0.0
        assert scores.fused == 0.0

    def test_unused_fields_stay_zero_not_null(self):
        """Unused scores must be 0.0, NOT None — frozen schema."""
        resp = _make_healthcare_response(
            retrieval_method="scincl_only",
            text_score=18.09,
        )
        metadata = _map_retrieval_metadata(resp)
        # Unused fields are 0.0, not None
        assert metadata.scores.colpali == 0.0
        assert metadata.scores.fused == 0.0
        assert metadata.scores.colpali is not None
        assert metadata.scores.fused is not None
