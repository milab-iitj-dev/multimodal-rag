"""
Domain Router — domain-agnostic query dispatcher.

Detects the domain, selects the pipeline, and executes run().
The router NEVER needs to know whether the pipeline is healthcare
or scientific. It only knows: detect → select → execute.

Usage:
    router = DomainRouter(config)
    router.register("healthcare", healthcare_pipeline)
    router.register("scientific", scientific_pipeline)
    result = router.route(query="Is there effusion?", image=img)
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from PIL import Image

from src.shared.base_pipeline import BasePipeline
from src.shared.schemas.response import UnifiedResponse
from src.shared.logging_utils import setup_logger

logger = setup_logger("router.domain")


class DomainRouter:
    """
    Domain-agnostic query router.

    Responsibilities (ONLY these):
      1. Detect which domain a query belongs to
      2. Select the registered pipeline for that domain
      3. Execute pipeline.run()

    The router has ZERO knowledge of healthcare or scientific internals.
    """

    VALID_DOMAINS = {"healthcare", "scientific"}

    # ── Keyword sets for auto-detection ──
    _HEALTHCARE_KEYWORDS = {
        "x-ray", "xray", "chest", "lung", "pleural", "effusion",
        "cardiomegaly", "pneumonia", "pneumothorax", "atelectasis",
        "edema", "consolidation", "fracture", "nodule", "opacity",
        "radiology", "radiograph", "medical", "patient", "clinical",
        "dicom", "findings", "impression", "diagnosis",
    }

    _SCIENTIFIC_KEYWORDS = {
        "paper", "arxiv", "transformer", "attention", "bert",
        "neural", "network", "architecture", "benchmark", "dataset",
        "experiment", "ablation", "table", "figure", "equation",
        "algorithm", "training", "loss", "accuracy", "citation",
        "abstract", "methodology", "results", "conclusion",
        "vision transformer", "vit", "resnet", "convolution",
    }

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Unified config dict.
                    config["unified"]["default_domain"] = "healthcare"
        """
        self.config = config or {}
        unified = self.config.get("unified", {})
        self.default_domain = unified.get("default_domain", "healthcare")

        # Pipeline registry: domain_name → BasePipeline instance
        self._pipelines: Dict[str, BasePipeline] = {}

        logger.info(f"DomainRouter initialized (default: {self.default_domain})")

    # ── Registration ──

    def register(self, domain: str, pipeline: BasePipeline) -> None:
        """
        Register a pipeline for a domain.

        Args:
            domain:   Domain name (e.g. "healthcare", "scientific").
            pipeline: A BasePipeline instance.
        """
        self._pipelines[domain] = pipeline
        logger.info(f"Registered pipeline: {domain} -> {type(pipeline).__name__}")

    # ── Pipeline selection ──

    def get_pipeline(self, domain: str) -> BasePipeline:
        """
        Get the registered pipeline for a domain.

        Raises KeyError if no pipeline is registered for the domain.
        """
        if domain not in self._pipelines:
            raise KeyError(
                f"No pipeline registered for domain '{domain}'. "
                f"Available: {list(self._pipelines.keys())}"
            )
        return self._pipelines[domain]

    # ── Domain detection ──

    def detect_domain(
        self,
        query: str = "",
        domain_hint: Optional[str] = None,
    ) -> str:
        """
        Detect the appropriate domain for a query.

        Priority:
          1. Explicit domain_hint (if valid)
          2. Keyword-based auto-detection
          3. Config default
        """
        # Priority 1: Explicit hint
        if domain_hint and domain_hint.lower() in self.VALID_DOMAINS:
            logger.info(f"Domain: {domain_hint} (explicit)")
            return domain_hint.lower()

        # Priority 2: Keyword detection
        if query:
            detected = self._detect_by_keywords(query)
            if detected:
                logger.info(f"Domain: {detected} (auto-detected)")
                return detected

        # Priority 3: Config default
        logger.info(f"Domain: {self.default_domain} (default)")
        return self.default_domain

    # ── The main entry point ──

    def route(
        self,
        query: str,
        domain_hint: Optional[str] = None,
        image: Optional[Image.Image] = None,
        top_k: int = 3,
        **kwargs: Any,
    ) -> UnifiedResponse:
        """
        Detect domain → select pipeline → execute run().

        This is the ONLY method the API/UI layer needs to call.

        Args:
            query:       The user's query text.
            domain_hint: Explicit domain override.
            image:       Optional PIL image.
            top_k:       Number of documents to retrieve.

        Returns:
            UnifiedResponse from the appropriate pipeline.
        """
        domain = self.detect_domain(query, domain_hint)
        pipeline = self.get_pipeline(domain)
        result = pipeline.run(query=query, image=image, top_k=top_k, **kwargs)
        return result

    # ── Internal ──

    def _detect_by_keywords(self, query: str) -> Optional[str]:
        """Detect domain from query keywords."""
        q_lower = query.lower()

        health_score = sum(1 for kw in self._HEALTHCARE_KEYWORDS if kw in q_lower)
        sci_score = sum(1 for kw in self._SCIENTIFIC_KEYWORDS if kw in q_lower)

        if health_score > sci_score and health_score >= 2:
            return "healthcare"
        if sci_score > health_score and sci_score >= 2:
            return "scientific"

        return None
