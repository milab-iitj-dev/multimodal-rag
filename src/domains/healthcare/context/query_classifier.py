"""
Query Classifier — unified query-type routing for the MRAG pipeline.

Classifies clinical queries into one of four types:

    BINARY_CLINICAL    — yes/no disease-specific questions
    DESCRIPTIVE_IMAGE  — open-ended image description requests
    TEXT_ONLY          — text-based queries without an image
    MIXED_IMAGE_TEXT   — specific clinical questions with an image

The classifier uses a pattern-based approach that inspects:
  1. Question structure (starts with "is there", "describe", etc.)
  2. Medical keyword presence (disease names, clinical terms)
  3. Descriptive/open-ended wording
  4. Image availability

This replaces the scattered ad-hoc classification logic that was
previously embedded in _generate_text_only_answer(), _extract_topic(),
and _build_messages().

The QueryType flows through the entire pipeline:
  classify → aggregate → prompt → generate → verify → confidence

Usage:
    classifier = QueryClassifier()
    qtype = classifier.classify("Is there pleural effusion?", has_image=True)
    # → QueryType.BINARY_CLINICAL

    qtype = classifier.classify("Describe the findings", has_image=True)
    # → QueryType.DESCRIPTIVE_IMAGE
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from src.shared.logging_utils import setup_logger

logger = setup_logger("context.query_classifier")


class QueryType(Enum):
    """The four query-type categories."""
    BINARY_CLINICAL = "binary_clinical"
    DESCRIPTIVE_IMAGE = "descriptive_image"
    TEXT_ONLY = "text_only"
    MIXED_IMAGE_TEXT = "mixed_image_text"


@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: QueryType
    is_binary: bool               # yes/no question?
    is_descriptive: bool          # open-ended description?
    has_image: bool               # image available?
    detected_topic: Optional[str] = None  # medical topic if found
    confidence: float = 1.0       # classification confidence
    reason: str = ""              # human-readable explanation


class QueryClassifier:
    """
    Pattern-based query classifier for clinical VQA routing.

    Not a hardcoded list — uses linguistic patterns to classify
    ANY query into the correct type. Extensible via the pattern
    lists without modifying core logic.

    Usage:
        classifier = QueryClassifier()
        result = classifier.classify("Is there pleural effusion?", has_image=True)
        print(result.query_type)      # QueryType.BINARY_CLINICAL
        print(result.detected_topic)  # "pleural effusion"
    """

    # ------------------------------------------------------------------ #
    #  Binary question patterns (yes/no clinical questions)                #
    # ------------------------------------------------------------------ #

    # Prefixes that strongly indicate a yes/no question
    _BINARY_PREFIXES = [
        r"^is there\b",
        r"^are there\b",
        r"^does (?:the |this |it )?\w+\s+(?:show|have|demonstrate|indicate|suggest)\b",
        r"^do you (?:see|observe|note|detect|identify)\b",
        r"^is (?:the |this |it )?\w+\s+(?:enlarged|present|visible|seen|noted)\b",
        r"^has (?:the |this )?\w+\s+(?:changed|worsened|improved|increased|decreased)\b",
        r"^can you (?:see|confirm|identify|detect)\b",
        r"^any (?:signs?|evidence|indication) of\b",
    ]

    # ------------------------------------------------------------------ #
    #  Descriptive/open-ended patterns                                     #
    # ------------------------------------------------------------------ #

    # Prefixes that strongly indicate a descriptive request
    _DESCRIPTIVE_PREFIXES = [
        r"^describe\b",
        r"^summarize\b",
        r"^explain\b",
        r"^list (?:the |all )?\b",
        r"^report (?:the |on |all )?\b",
        r"^what (?:does|do|are|is)\b.*(?:show|visible|seen|findings?|abnormalit|present)\b",
        r"^what can (?:you |be )?\b",
        r"^tell me (?:about |what )\b",
        r"^provide (?:a |an )?\b.*(?:summary|description|analysis|overview)\b",
        r"^generate (?:a |an )?\b.*(?:report|summary|description)\b",
    ]

    # Keywords anywhere in the query that indicate descriptive intent
    _DESCRIPTIVE_KEYWORDS = [
        r"\bdescribe\b",
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\boverview\b",
        r"\ball\s+(?:findings?|abnormalit|observations?)\b",
        r"\bkey\s+findings?\b",
        r"\bclinical(?:ly)?\s+significant\b",
        r"\bvisible\s+(?:in|on)\b",
        r"\bwhat\s+(?:does|do)\s+(?:this|the)\s+(?:image|x-?ray|radiograph|scan)\s+show\b",
    ]

    # ------------------------------------------------------------------ #
    #  Medical topic patterns (for extracting the specific condition)       #
    # ------------------------------------------------------------------ #

    # Common clinical findings/conditions
    _MEDICAL_TOPICS = [
        "pleural effusion", "cardiomegaly", "pneumothorax",
        "consolidation", "atelectasis", "pulmonary edema", "edema",
        "pneumonia", "mass", "nodule", "nodules", "fracture",
        "emphysema", "fibrosis", "calcification", "opacity",
        "opacities", "infiltrate", "infiltrates",
        "hilar prominence", "mediastinal widening",
        "pericardial effusion", "interstitial disease",
        "hyperinflation", "scoliosis", "aortic calcification",
        "widened mediastinum", "enlarged heart",
        "lung mass", "lung nodule", "rib fracture",
        "pleural thickening", "bronchiectasis",
    ]

    # Question prefix stripping patterns (for topic extraction)
    _TOPIC_STRIP_PREFIXES = [
        r"^is there\s+(?:any\s+)?(?:evidence\s+of\s+|signs?\s+of\s+)?",
        r"^are there\s+(?:any\s+)?(?:signs?\s+of\s+)?",
        r"^does (?:the |this )?\w+ (?:show|have|demonstrate|indicate)\s+",
        r"^what (?:are |is )?(?:the )?(?:signs? of |evidence of )?\s*",
        r"^any\s+(?:signs?\s+of\s+)?",
        r"^do you see\s+",
        r"^can you (?:see|identify|detect)\s+",
    ]

    def __init__(self):
        self._binary_re = [
            re.compile(p, re.IGNORECASE) for p in self._BINARY_PREFIXES
        ]
        self._descriptive_prefix_re = [
            re.compile(p, re.IGNORECASE) for p in self._DESCRIPTIVE_PREFIXES
        ]
        self._descriptive_keyword_re = re.compile(
            "|".join(self._DESCRIPTIVE_KEYWORDS), re.IGNORECASE
        )
        self._topic_strip_re = [
            re.compile(p, re.IGNORECASE) for p in self._TOPIC_STRIP_PREFIXES
        ]
        # Pre-sort topics by length (longest first for greedy matching)
        self._sorted_topics = sorted(
            self._MEDICAL_TOPICS, key=len, reverse=True
        )

    def classify(
        self,
        query: str,
        has_image: bool = False,
    ) -> QueryClassification:
        """
        Classify a query into a QueryType.

        Decision tree:
          1. Check for descriptive patterns → DESCRIPTIVE_IMAGE (if image)
          2. Check for binary clinical patterns → BINARY_CLINICAL (if image)
          3. No image available → TEXT_ONLY
          4. Image + specific clinical topic → MIXED_IMAGE_TEXT
          5. Image + unclear intent → DESCRIPTIVE_IMAGE (safe default)

        Args:
            query:     The user's question text.
            has_image: Whether an image is available for this query.

        Returns:
            QueryClassification with type, topic, and reasoning.
        """
        q = query.strip()
        q_lower = q.lower()

        is_binary = self._is_binary_question(q_lower)
        is_descriptive = self._is_descriptive_question(q_lower)
        detected_topic = self._extract_medical_topic(q_lower)

        # ── Decision logic ──

        if not has_image:
            # No image → text-only path regardless of question form
            result = QueryClassification(
                query_type=QueryType.TEXT_ONLY,
                is_binary=is_binary,
                is_descriptive=is_descriptive,
                has_image=False,
                detected_topic=detected_topic,
                reason="No image available — text-only evidence path",
            )

        elif is_descriptive and not is_binary:
            # Clearly descriptive + has image → descriptive image
            result = QueryClassification(
                query_type=QueryType.DESCRIPTIVE_IMAGE,
                is_binary=False,
                is_descriptive=True,
                has_image=True,
                detected_topic=detected_topic,
                reason="Descriptive/open-ended query with image",
            )

        elif is_binary and detected_topic:
            # Clear yes/no question about a specific condition + image
            result = QueryClassification(
                query_type=QueryType.BINARY_CLINICAL,
                is_binary=True,
                is_descriptive=False,
                has_image=True,
                detected_topic=detected_topic,
                reason=f"Binary clinical question about '{detected_topic}'",
            )

        elif is_binary and not detected_topic:
            # Binary form but no specific topic → still binary clinical
            result = QueryClassification(
                query_type=QueryType.BINARY_CLINICAL,
                is_binary=True,
                is_descriptive=False,
                has_image=True,
                detected_topic=None,
                confidence=0.8,
                reason="Binary question form but no specific topic detected",
            )

        elif detected_topic and has_image:
            # Specific topic mentioned + image but not binary → mixed
            result = QueryClassification(
                query_type=QueryType.MIXED_IMAGE_TEXT,
                is_binary=False,
                is_descriptive=False,
                has_image=True,
                detected_topic=detected_topic,
                reason=f"Clinical query about '{detected_topic}' with image",
            )

        else:
            # Ambiguous with image → default to descriptive
            result = QueryClassification(
                query_type=QueryType.DESCRIPTIVE_IMAGE,
                is_binary=False,
                is_descriptive=True,
                has_image=True,
                detected_topic=detected_topic,
                confidence=0.6,
                reason="Ambiguous query with image — defaulting to descriptive",
            )

        logger.info(
            f"Query classified: '{q[:60]}' -> {result.query_type.value} "
            f"(topic={result.detected_topic}, "
            f"binary={result.is_binary}, "
            f"descriptive={result.is_descriptive})"
        )

        return result

    # ------------------------------------------------------------------ #
    #  Pattern detection methods                                           #
    # ------------------------------------------------------------------ #

    def _is_binary_question(self, q_lower: str) -> bool:
        """Check if the query is a yes/no binary question."""
        for pattern in self._binary_re:
            if pattern.search(q_lower):
                return True
        return False

    def _is_descriptive_question(self, q_lower: str) -> bool:
        """Check if the query is a descriptive/open-ended question."""
        # Check prefix patterns
        for pattern in self._descriptive_prefix_re:
            if pattern.search(q_lower):
                return True

        # Check keyword patterns
        if self._descriptive_keyword_re.search(q_lower):
            return True

        return False

    def _extract_medical_topic(self, q_lower: str) -> Optional[str]:
        """
        Extract a specific medical topic/condition from the query.

        Uses two strategies:
          1. Match known clinical findings (greedy, longest first)
          2. Strip question prefixes and extract remaining topic

        Returns None if no specific topic is found.
        """
        # Strategy 1: Match known clinical findings
        for topic in self._sorted_topics:
            if topic in q_lower:
                return topic

        # Strategy 2: Strip question prefixes and take what's left
        stripped = q_lower.rstrip("?.,! ")
        for pattern in self._topic_strip_re:
            stripped = pattern.sub("", stripped)
        stripped = stripped.strip().rstrip("?.,! ")

        # Remove trailing articles/prepositions
        stripped = re.sub(
            r"\b(?:a|an|the|in|on|of|for|this|that)\s*$", "", stripped
        ).strip()

        # If result is short and specific (1-4 words), it's likely a topic
        if stripped and 1 <= len(stripped.split()) <= 4:
            return stripped

        return None
