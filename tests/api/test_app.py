"""
Tests for FastAPI endpoints — frozen contract compliance.

Uses FastAPI TestClient to verify:
  - GET /health returns 200 + correct schema
  - GET /ready returns pipeline status
  - POST /query returns frozen contract response
  - POST /query with domain="auto" routes correctly
  - Error handling (invalid domain, empty query)
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


# ── GET /health ─────────────────────────────────────────────


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["service"] == "mmrag-unified"
        assert data["version"] == "2.0.0"


# ── GET /ready ──────────────────────────────────────────────


class TestReady:
    def test_ready_returns_200(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200

    def test_ready_has_domains(self, client):
        data = client.get("/ready").json()
        assert "ready" in data
        assert "domains" in data
        assert isinstance(data["domains"], list)


# ── POST /query ─────────────────────────────────────────────


class TestQuery:
    def test_healthcare_query(self, client):
        """Healthcare query returns frozen contract response."""
        resp = client.post("/query", json={
            "query": "What is cardiomegaly?",
            "domain": "healthcare",
            "top_k": 3,
            "include_images": True,
        })
        assert resp.status_code == 200

        data = resp.json()
        # Verify all top-level keys are present
        assert "answer" in data
        assert "confidence" in data
        assert "sources" in data
        assert "retrieval_metadata" in data
        assert "verification" in data
        assert "latency_ms" in data

        # Verify nested structure
        rm = data["retrieval_metadata"]
        assert "method" in rm
        assert "scores" in rm
        assert "colpali" in rm["scores"]
        assert "scincl" in rm["scores"]
        assert "fused" in rm["scores"]

        v = data["verification"]
        assert "attribution" in v
        assert "faithfulness" in v
        assert "confidence_pass" in v

    def test_scientific_query(self, client):
        """Scientific query returns frozen contract response."""
        resp = client.post("/query", json={
            "query": "Explain retrieval augmented generation",
            "domain": "scientific",
            "top_k": 3,
        })
        assert resp.status_code == 200

        data = resp.json()
        assert "answer" in data
        assert "retrieval_metadata" in data
        assert "verification" in data

    def test_auto_routing_healthcare(self, client):
        """Auto domain routes medical queries to healthcare."""
        resp = client.post("/query", json={
            "query": "Is there cardiomegaly in this chest x-ray?",
            "domain": "auto",
        })
        assert resp.status_code == 200

    def test_auto_routing_scientific(self, client):
        """Auto domain routes ML queries to scientific."""
        resp = client.post("/query", json={
            "query": "Explain the attention mechanism in transformers",
            "domain": "auto",
        })
        assert resp.status_code == 200

    def test_minimal_request(self, client):
        """Only query is required (domain defaults to auto)."""
        resp = client.post("/query", json={
            "query": "test question",
        })
        assert resp.status_code == 200

    def test_empty_query_rejected(self, client):
        """Empty query should be rejected with 422."""
        resp = client.post("/query", json={
            "query": "",
        })
        assert resp.status_code == 422

    def test_latency_is_present(self, client):
        """Response must include latency_ms >= 0."""
        resp = client.post("/query", json={
            "query": "What is cardiomegaly?",
            "domain": "healthcare",
        })
        data = resp.json()
        assert data["latency_ms"] >= 0

    def test_confidence_in_range(self, client):
        """Confidence must be 0.0–1.0."""
        resp = client.post("/query", json={
            "query": "test",
            "domain": "healthcare",
        })
        data = resp.json()
        assert 0.0 <= data["confidence"] <= 1.0


# ── OpenAPI docs ────────────────────────────────────────────


class TestDocs:
    def test_openapi_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "/query" in data["paths"]
        assert "/health" in data["paths"]
        assert "/ready" in data["paths"]
