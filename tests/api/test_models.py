"""
Tests for Pydantic API models — frozen contract validation.

Verifies that:
  - QueryRequest validates all fields correctly
  - QueryResponse matches the exact contract structure
  - Default values are correct
  - Invalid inputs are rejected
"""

import pytest
from pydantic import ValidationError

from src.api.models import (
    QueryRequest,
    QueryResponse,
    SourceItemResponse,
    RetrievalMetadata,
    RetrievalScores,
    VerificationResult,
    HealthResponse,
    ReadyResponse,
)


# ── QueryRequest ────────────────────────────────────────────


class TestQueryRequest:
    """Tests for the request model."""

    def test_minimal_request(self):
        """Only query is required."""
        req = QueryRequest(query="What is cardiomegaly?")
        assert req.query == "What is cardiomegaly?"
        assert req.domain == "auto"
        assert req.top_k == 3
        assert req.include_images is True

    def test_full_request(self):
        """All fields specified."""
        req = QueryRequest(
            query="Is there effusion?",
            domain="healthcare",
            top_k=5,
            include_images=False,
        )
        assert req.domain == "healthcare"
        assert req.top_k == 5
        assert req.include_images is False

    def test_auto_domain(self):
        req = QueryRequest(query="test", domain="auto")
        assert req.domain == "auto"

    def test_scientific_domain(self):
        req = QueryRequest(query="test", domain="scientific")
        assert req.domain == "scientific"

    def test_invalid_domain_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", domain="invalid_domain")

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_top_k_bounds(self):
        """top_k must be 1–20."""
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=21)

        # Valid bounds
        req_min = QueryRequest(query="test", top_k=1)
        assert req_min.top_k == 1
        req_max = QueryRequest(query="test", top_k=20)
        assert req_max.top_k == 20


# ── QueryResponse ───────────────────────────────────────────


class TestQueryResponse:
    """Tests for the response model."""

    def test_minimal_response(self):
        """Answer is required, everything else has defaults."""
        resp = QueryResponse(answer="Cardiomegaly is an enlarged heart.")
        assert resp.answer == "Cardiomegaly is an enlarged heart."
        assert resp.confidence == 0.0
        assert resp.sources == []
        assert resp.retrieval_metadata.method == "fused"
        assert resp.retrieval_metadata.scores.colpali == 0.0
        assert resp.verification.attribution is True
        assert resp.latency_ms == 0

    def test_full_response(self):
        """All fields populated."""
        resp = QueryResponse(
            answer="The chest X-ray shows cardiomegaly.",
            confidence=0.85,
            sources=[
                SourceItemResponse(
                    doc_id="CXR1234",
                    page=1,
                    title="Case CXR1234",
                    relevance_score=0.92,
                    snippet="Enlarged cardiac silhouette noted.",
                ),
            ],
            retrieval_metadata=RetrievalMetadata(
                method="fused",
                scores=RetrievalScores(
                    colpali=0.88,
                    scincl=0.76,
                    fused=0.82,
                ),
            ),
            verification=VerificationResult(
                attribution=True,
                faithfulness=True,
                confidence_pass=True,
            ),
            latency_ms=245,
        )
        assert resp.confidence == 0.85
        assert len(resp.sources) == 1
        assert resp.sources[0].doc_id == "CXR1234"
        assert resp.retrieval_metadata.scores.colpali == 0.88
        assert resp.verification.attribution is True
        assert resp.latency_ms == 245

    def test_response_json_contract(self):
        """Verify the JSON output matches the frozen contract exactly."""
        resp = QueryResponse(
            answer="Test answer",
            confidence=0.75,
            sources=[],
            retrieval_metadata=RetrievalMetadata(
                method="fused",
                scores=RetrievalScores(colpali=0.5, scincl=0.3, fused=0.4),
            ),
            verification=VerificationResult(
                attribution=True,
                faithfulness=False,
                confidence_pass=True,
            ),
            latency_ms=100,
        )
        data = resp.model_dump()

        # Top-level keys
        assert set(data.keys()) == {
            "answer", "confidence", "sources",
            "retrieval_metadata", "verification", "latency_ms",
        }

        # Nested keys
        assert set(data["retrieval_metadata"].keys()) == {"method", "scores"}
        assert set(data["retrieval_metadata"]["scores"].keys()) == {
            "colpali", "scincl", "fused",
        }
        assert set(data["verification"].keys()) == {
            "attribution", "faithfulness", "confidence_pass",
        }

    def test_confidence_bounds(self):
        """Confidence must be 0.0–1.0."""
        with pytest.raises(ValidationError):
            QueryResponse(answer="test", confidence=-0.1)
        with pytest.raises(ValidationError):
            QueryResponse(answer="test", confidence=1.1)


# ── Component models ────────────────────────────────────────


class TestSourceItemResponse:
    def test_defaults(self):
        s = SourceItemResponse()
        assert s.doc_id == ""
        assert s.page == 0
        assert s.title == ""
        assert s.relevance_score == 0.0
        assert s.snippet == ""


class TestRetrievalMetadata:
    def test_defaults(self):
        m = RetrievalMetadata()
        assert m.method == "fused"
        assert m.scores.colpali == 0.0

    def test_valid_methods(self):
        for method in ("fused", "colpali_only", "scincl_only"):
            m = RetrievalMetadata(method=method)
            assert m.method == method

    def test_invalid_method(self):
        with pytest.raises(ValidationError):
            RetrievalMetadata(method="invalid")


class TestVerificationResult:
    def test_defaults(self):
        v = VerificationResult()
        assert v.attribution is True
        assert v.faithfulness is True
        assert v.confidence_pass is True


class TestHealthResponse:
    def test_defaults(self):
        h = HealthResponse()
        assert h.status == "healthy"
        assert h.service == "mmrag-unified"
        assert h.version == "2.0.0"


class TestReadyResponse:
    def test_defaults(self):
        r = ReadyResponse()
        assert r.ready is True
        assert r.domains == []
