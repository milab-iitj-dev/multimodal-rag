"""
End-to-end smoke test — verifies the full stack:
  Router → Pipeline adapter → UnifiedResponse → UI formatter

Runs in placeholder mode (no GPU, no models, no data).
Tests that the architecture is wired correctly.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.router.domain_router import DomainRouter
from src.shared.schemas.response import UnifiedResponse, SourceItem
from src.shared.base_pipeline import BasePipeline
from pipelines.healthcare.adapter import HealthcarePipeline
from pipelines.scientific.adapter import ScientificPipeline


def test_unified_response_schema():
    """Verify UnifiedResponse can be constructed."""
    r = UnifiedResponse(
        domain="test",
        answer="Test answer",
        confidence=0.85,
        sources=[
            SourceItem(title="Source 1", score=0.9, snippet="test snippet"),
        ],
        metadata={"key": "value"},
    )
    assert r.domain == "test"
    assert r.confidence == 0.85
    assert len(r.sources) == 1
    assert r.sources[0].title == "Source 1"
    print("  [PASS] UnifiedResponse schema")


def test_base_pipeline_interface():
    """Verify BasePipeline can't be instantiated."""
    try:
        _ = BasePipeline()
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    print("  [PASS] BasePipeline is abstract")


def test_healthcare_adapter():
    """Verify HealthcarePipeline returns UnifiedResponse."""
    pipeline = HealthcarePipeline(inner_pipeline=None)
    result = pipeline.run(query="Is there pleural effusion?")
    assert isinstance(result, UnifiedResponse)
    assert result.domain == "healthcare"
    assert "Pipeline not loaded" in result.answer
    print("  [PASS] HealthcarePipeline → UnifiedResponse (placeholder)")


def test_scientific_adapter():
    """Verify ScientificPipeline returns UnifiedResponse."""
    pipeline = ScientificPipeline(inner_pipeline=None)
    result = pipeline.run(query="What is Vision Transformer?")
    assert isinstance(result, UnifiedResponse)
    assert result.domain == "scientific"
    assert "Pipeline not loaded" in result.answer
    print("  [PASS] ScientificPipeline → UnifiedResponse (placeholder)")


def test_router_detection():
    """Verify domain auto-detection."""
    router = DomainRouter()

    # Healthcare keywords
    assert router.detect_domain("Is there pleural effusion in the chest x-ray?") == "healthcare"
    # Scientific keywords
    assert router.detect_domain("What is the vision transformer architecture in the paper?") == "scientific"
    # Default (no strong keywords)
    assert router.detect_domain("Hello world") == "healthcare"
    # Explicit hint overrides
    assert router.detect_domain("Hello", domain_hint="scientific") == "scientific"
    print("  [PASS] Domain detection (healthcare/scientific/default/explicit)")


def test_router_e2e():
    """Verify router.route() returns UnifiedResponse."""
    router = DomainRouter()
    router.register("healthcare", HealthcarePipeline(inner_pipeline=None))
    router.register("scientific", ScientificPipeline(inner_pipeline=None))

    # Healthcare route
    r1 = router.route(query="Is there cardiomegaly?", domain_hint="healthcare")
    assert isinstance(r1, UnifiedResponse)
    assert r1.domain == "healthcare"

    # Scientific route
    r2 = router.route(query="What is ViT?", domain_hint="scientific")
    assert isinstance(r2, UnifiedResponse)
    assert r2.domain == "scientific"

    # Both return same schema
    assert type(r1) == type(r2) == UnifiedResponse
    print("  [PASS] Router E2E: route() → UnifiedResponse for both domains")


def test_identical_schema():
    """Verify healthcare and scientific return identical field structure."""
    router = DomainRouter()
    router.register("healthcare", HealthcarePipeline(inner_pipeline=None))
    router.register("scientific", ScientificPipeline(inner_pipeline=None))

    r1 = router.route("test", domain_hint="healthcare")
    r2 = router.route("test", domain_hint="scientific")

    # Both have exactly the same fields
    r1_fields = set(vars(r1).keys())
    r2_fields = set(vars(r2).keys())
    assert r1_fields == r2_fields, f"Mismatch: {r1_fields ^ r2_fields}"
    print("  [PASS] Identical field structure across domains")


if __name__ == "__main__":
    print("=" * 60)
    print("  MMRAG Unified — Smoke Tests")
    print("=" * 60)

    tests = [
        test_unified_response_schema,
        test_base_pipeline_interface,
        test_healthcare_adapter,
        test_scientific_adapter,
        test_router_detection,
        test_router_e2e,
        test_identical_schema,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
