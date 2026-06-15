"""
Domain Router — dispatches queries to the correct pipeline.

Detects whether a query should be handled by the healthcare or
scientific pipeline based on explicit domain selection or
config-driven defaults.

Usage:
    router = DomainRouter(config)
    domain = router.detect_domain(query, domain_hint="healthcare")
    pipeline = router.get_pipeline(domain)
    result = pipeline.run(query)
"""

from typing import Optional
from src.shared.logging_utils import setup_logger

logger = setup_logger("router.domain")


class DomainRouter:
    """
    Routes queries to the correct domain pipeline.

    Supports:
      - Explicit domain selection via parameter
      - Config-driven default domain
      - (Future) Auto-detection via keyword analysis

    Usage:
        router = DomainRouter(config)
        domain = router.detect_domain("Is there pleural effusion?")
        # → "healthcare"

        domain = router.detect_domain("What is the Vision Transformer?")
        # → "scientific"
    """

    VALID_DOMAINS = {"healthcare", "scientific"}

    # Keywords that strongly indicate healthcare domain
    _HEALTHCARE_KEYWORDS = {
        "x-ray", "xray", "chest", "lung", "pleural", "effusion",
        "cardiomegaly", "pneumonia", "pneumothorax", "atelectasis",
        "edema", "consolidation", "fracture", "nodule", "opacity",
        "radiology", "radiograph", "medical", "patient", "clinical",
        "dicom", "findings", "impression", "diagnosis",
    }

    # Keywords that strongly indicate scientific domain
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
            config: Unified config dict. Expected structure:
                config["unified"]["default_domain"] = "healthcare"
        """
        self.config = config or {}
        unified = self.config.get("unified", {})
        self.default_domain = unified.get("default_domain", "healthcare")

        logger.info(
            f"DomainRouter initialized (default: {self.default_domain})"
        )

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

        Args:
            query:       The user's query text.
            domain_hint: Explicit domain override.

        Returns:
            "healthcare" or "scientific"
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

    def _detect_by_keywords(self, query: str) -> Optional[str]:
        """
        Detect domain from query keywords.

        Counts keyword matches for each domain and returns the
        domain with more matches, or None if tied/zero.
        """
        q_lower = query.lower()
        words = set(q_lower.split())

        health_score = sum(
            1 for kw in self._HEALTHCARE_KEYWORDS
            if kw in q_lower
        )
        sci_score = sum(
            1 for kw in self._SCIENTIFIC_KEYWORDS
            if kw in q_lower
        )

        if health_score > sci_score and health_score >= 2:
            return "healthcare"
        if sci_score > health_score and sci_score >= 2:
            return "scientific"

        return None
