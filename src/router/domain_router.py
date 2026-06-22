"""
Domain Router — domain-agnostic query dispatcher with auto-routing.

Detects the domain, selects the pipeline, and executes run().
The router NEVER needs to know whether the pipeline is healthcare
or scientific. It only knows: detect → select → execute.

Auto-routing strategy:
    1. Explicit domain parameter → use directly
    2. "auto" or None → keyword + bigram detection
    3. No match → config default (healthcare)

The keyword router uses curated unigram and bigram term lists
for each domain. This is the recommended production approach
for 2 domains with clearly distinct vocabularies (zero latency,
deterministic, testable). See implementation_plan.md for the
full option analysis.

Usage:
    router = DomainRouter(config)
    router.register("healthcare", healthcare_pipeline)
    router.register("scientific", scientific_pipeline)
    result = router.route(query="Is there effusion?", image=img)

    # Auto-routing:
    result = router.route(query="Is there effusion?", domain_hint="auto")
"""

from __future__ import annotations

import re
from typing import Optional, Dict, Any, Tuple
from PIL import Image

from src.shared.base_pipeline import BasePipeline
from src.shared.schemas.response import UnifiedResponse
from src.shared.logging_utils import setup_logger

logger = setup_logger("router.domain")


class DomainRouter:
    """
    Domain-agnostic query router with enhanced auto-detection.

    Responsibilities (ONLY these):
      1. Detect which domain a query belongs to
      2. Select the registered pipeline for that domain
      3. Execute pipeline.run()

    The router has ZERO knowledge of healthcare or scientific internals.

    Auto-detection uses curated keyword lists and bigram matching:
      - Zero latency (no model inference)
      - Deterministic (same query → same routing)
      - Testable (keyword lists are explicit)
    """

    VALID_DOMAINS = {"healthcare", "scientific", "auto"}

    # ── Keyword sets for auto-detection ──
    # Unigrams: single terms strongly associated with each domain.

    _HEALTHCARE_KEYWORDS = {
        # Anatomy / imaging
        "x-ray", "xray", "chest", "lung", "lungs", "heart", "cardiac",
        "thorax", "thoracic", "rib", "ribs", "spine", "diaphragm",
        "mediastinum", "hilum", "hilar", "aorta", "aortic",
        # Conditions
        "pleural", "effusion", "cardiomegaly", "pneumonia", "pneumothorax",
        "atelectasis", "edema", "consolidation", "fracture", "nodule",
        "opacity", "fibrosis", "emphysema", "calcification", "scoliosis",
        "tuberculosis", "mass", "lesion", "infiltrate",
        # Clinical
        "radiology", "radiograph", "radiographic", "medical", "patient",
        "clinical", "diagnosis", "prognosis", "symptom", "treatment",
        "pathology", "disease", "condition",
        # Report terms
        "dicom", "findings", "impression", "report",
    }

    _SCIENTIFIC_KEYWORDS = {
        # ML / AI
        "transformer", "attention", "bert", "gpt", "llm", "llms",
        "neural", "network", "deep", "learning", "model", "pretrained",
        "fine-tuning", "finetuning", "embedding", "embeddings",
        "classification", "segmentation", "detection",
        # Research
        "paper", "arxiv", "research", "literature", "survey",
        "benchmark", "dataset", "experiment", "ablation",
        "methodology", "evaluation", "baseline",
        # Architecture
        "architecture", "encoder", "decoder", "convolution", "resnet",
        "vit", "cnn", "rnn", "lstm", "gan",
        # Metrics / training
        "accuracy", "precision", "recall", "f1", "loss", "gradient",
        "optimization", "backpropagation", "epoch", "batch",
        # Writing structure
        "abstract", "conclusion", "equation", "algorithm",
        "table", "figure", "citation", "reference", "section",
        # RAG-specific
        "retrieval", "augmented", "generation", "rag", "indexing",
        "vector", "similarity", "dense",
    }

    # Bigrams: two-word phrases that are strong domain signals.
    # Checked as lowercased substrings of the query.

    _HEALTHCARE_BIGRAMS = [
        "chest x-ray", "chest xray", "x-ray image", "xray image",
        "pleural effusion", "pulmonary edema", "cardiac silhouette",
        "lung field", "lung fields", "heart size",
        "medical image", "medical imaging", "clinical finding",
        "radiology report", "chest radiograph",
    ]

    _SCIENTIFIC_BIGRAMS = [
        "vision transformer", "language model", "large language",
        "retrieval augmented", "augmented generation",
        "attention mechanism", "self-attention", "cross-attention",
        "neural network", "deep learning", "machine learning",
        "transfer learning", "knowledge distillation",
        "contrastive learning", "self-supervised",
        "state of the art", "state-of-the-art",
        "training data", "test set", "validation set",
    ]

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
    ) -> Tuple[str, float]:
        """
        Detect the appropriate domain for a query.

        Priority:
          1. Explicit domain_hint (if valid and not "auto")
          2. Keyword + bigram auto-detection
          3. Config default

        Returns:
            Tuple of (domain_name, routing_confidence).
            Confidence is 1.0 for explicit hints, 0.0–1.0 for auto.
        """
        # Priority 1: Explicit hint (not "auto")
        if domain_hint and domain_hint.lower() in ("healthcare", "scientific"):
            logger.info(f"Domain: {domain_hint} (explicit)")
            return domain_hint.lower(), 1.0

        # Priority 2: Keyword + bigram detection
        if query:
            detected, confidence = self._detect_by_keywords(query)
            if detected:
                logger.info(
                    f"Domain: {detected} (auto-detected, "
                    f"confidence={confidence:.2f})"
                )
                return detected, confidence

        # Priority 3: Config default
        logger.info(f"Domain: {self.default_domain} (default)")
        return self.default_domain, 0.0

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
            domain_hint: Explicit domain override, or "auto".
            image:       Optional PIL image.
            top_k:       Number of documents to retrieve.

        Returns:
            UnifiedResponse from the appropriate pipeline.
        """
        domain, routing_confidence = self.detect_domain(query, domain_hint)
        pipeline = self.get_pipeline(domain)
        result = pipeline.run(query=query, image=image, top_k=top_k, **kwargs)

        # Inject routing metadata into the response
        result.metadata["routed_domain"] = domain
        result.metadata["routing_confidence"] = round(routing_confidence, 4)
        result.metadata["routing_method"] = (
            "explicit" if routing_confidence == 1.0
            else "auto_keyword" if routing_confidence > 0.0
            else "default"
        )

        return result

    # ── Internal: keyword + bigram detection ──

    def _detect_by_keywords(self, query: str) -> Tuple[Optional[str], float]:
        """Detect domain from query keywords and bigrams.

        Scoring:
            - Each unigram match = 1 point
            - Each bigram match = 2 points (stronger signal)
            - Domain with higher score wins
            - Minimum 1 point required (lowered from 2 for better recall)
            - Confidence = winner_score / (winner_score + loser_score + 1)

        Returns:
            Tuple of (domain_or_None, confidence_score).
        """
        q_lower = query.lower()

        # Tokenize for unigram matching (word boundaries)
        q_tokens = set(re.findall(r"[a-z][a-z0-9-]+", q_lower))

        # Unigram scores
        health_score = len(q_tokens & self._HEALTHCARE_KEYWORDS)
        sci_score = len(q_tokens & self._SCIENTIFIC_KEYWORDS)

        # Bigram scores (substring match, +2 each)
        for bigram in self._HEALTHCARE_BIGRAMS:
            if bigram in q_lower:
                health_score += 2

        for bigram in self._SCIENTIFIC_BIGRAMS:
            if bigram in q_lower:
                sci_score += 2

        logger.debug(
            f"Routing scores: healthcare={health_score}, "
            f"scientific={sci_score}"
        )

        # Decision: winner must have >= 1 point and be strictly greater
        if health_score > sci_score and health_score >= 1:
            confidence = health_score / (health_score + sci_score + 1)
            return "healthcare", round(confidence, 4)

        if sci_score > health_score and sci_score >= 1:
            confidence = sci_score / (health_score + sci_score + 1)
            return "scientific", round(confidence, 4)

        # Tie or no matches
        return None, 0.0
