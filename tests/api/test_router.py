"""
Tests for the enhanced DomainRouter — auto-routing logic.

Verifies:
  - Explicit domain selection works
  - "auto" triggers keyword detection
  - Healthcare queries route correctly
  - Scientific queries route correctly
  - Ambiguous queries fall to default
  - Bigram matching works
  - Routing confidence is returned
"""

import pytest
from unittest.mock import MagicMock

from src.router.domain_router import DomainRouter
from src.shared.schemas.response import UnifiedResponse


@pytest.fixture
def router():
    """Router with mock pipelines."""
    r = DomainRouter()

    # Mock healthcare pipeline
    health_pipe = MagicMock()
    health_pipe.run.return_value = UnifiedResponse(
        domain="healthcare",
        answer="Healthcare answer",
        confidence=0.9,
    )

    # Mock scientific pipeline
    sci_pipe = MagicMock()
    sci_pipe.run.return_value = UnifiedResponse(
        domain="scientific",
        answer="Scientific answer",
        confidence=0.8,
    )

    r.register("healthcare", health_pipe)
    r.register("scientific", sci_pipe)
    return r


# ── Explicit domain ────────────────────────────────────────


class TestExplicitDomain:
    def test_explicit_healthcare(self, router):
        domain, conf = router.detect_domain("anything", domain_hint="healthcare")
        assert domain == "healthcare"
        assert conf == 1.0

    def test_explicit_scientific(self, router):
        domain, conf = router.detect_domain("anything", domain_hint="scientific")
        assert domain == "scientific"
        assert conf == 1.0

    def test_explicit_case_insensitive(self, router):
        domain, _ = router.detect_domain("test", domain_hint="Healthcare")
        assert domain == "healthcare"


# ── Auto-routing: healthcare ───────────────────────────────


class TestAutoHealthcare:
    @pytest.mark.parametrize("query", [
        "What is cardiomegaly?",
        "Is there pleural effusion?",
        "Describe the chest x-ray findings",
        "Is there pneumonia in this lung image?",
        "What does the radiology report show?",
        "Does the patient have atelectasis?",
    ])
    def test_healthcare_queries(self, router, query):
        domain, conf = router.detect_domain(query, domain_hint="auto")
        assert domain == "healthcare", f"'{query}' routed to {domain}"
        assert conf > 0.0

    def test_healthcare_bigram(self, router):
        domain, conf = router.detect_domain(
            "Analyze this chest x-ray image", domain_hint="auto"
        )
        assert domain == "healthcare"
        assert conf > 0.3  # Bigram gives strong signal


# ── Auto-routing: scientific ───────────────────────────────


class TestAutoScientific:
    @pytest.mark.parametrize("query", [
        "Explain retrieval augmented generation",
        "How does the attention mechanism work in transformers?",
        "What is the vision transformer architecture?",
        "Summarize the deep learning methodology",
        "Compare BERT and GPT architectures",
        "What benchmark was used for evaluation?",
    ])
    def test_scientific_queries(self, router, query):
        domain, conf = router.detect_domain(query, domain_hint="auto")
        assert domain == "scientific", f"'{query}' routed to {domain}"
        assert conf > 0.0

    def test_scientific_bigram(self, router):
        domain, conf = router.detect_domain(
            "Explain the attention mechanism in neural networks",
            domain_hint="auto",
        )
        assert domain == "scientific"
        assert conf > 0.3


# ── Ambiguous / default ────────────────────────────────────


class TestDefaultRouting:
    def test_ambiguous_query_uses_default(self, router):
        domain, conf = router.detect_domain("hello world", domain_hint="auto")
        assert domain == "healthcare"  # default
        assert conf == 0.0

    def test_none_hint_triggers_auto(self, router):
        domain, _ = router.detect_domain(
            "Is there cardiomegaly?", domain_hint=None
        )
        assert domain == "healthcare"


# ── Route method ───────────────────────────────────────────


class TestRoute:
    def test_route_calls_pipeline(self, router):
        result = router.route("Is there effusion?", domain_hint="healthcare")
        assert result.domain == "healthcare"
        assert result.answer == "Healthcare answer"
        assert result.metadata["routed_domain"] == "healthcare"
        assert result.metadata["routing_method"] == "explicit"

    def test_route_auto_injects_metadata(self, router):
        result = router.route(
            "What is cardiomegaly?", domain_hint="auto"
        )
        assert result.metadata["routed_domain"] == "healthcare"
        assert result.metadata["routing_method"] == "auto_keyword"
        assert result.metadata["routing_confidence"] > 0.0


# ── Registration ───────────────────────────────────────────


class TestRegistration:
    def test_unregistered_domain_raises(self, router):
        with pytest.raises(KeyError):
            router.get_pipeline("unknown_domain")

    def test_registered_domains(self, router):
        assert "healthcare" in router._pipelines
        assert "scientific" in router._pipelines
