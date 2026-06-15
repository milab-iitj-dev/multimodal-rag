"""
Confidence Estimator — evidence-based answer confidence scoring.

Scores how confident we should be in the generated answer based on
multiple factors from the retrieval and grounding pipeline. No extra
model needed — purely evidence-based scoring.

Factors:
  1. Evidence agreement (do retrieved reports agree?)
  2. Answer-evidence consistency (does the answer match evidence?)
  3. Retrieval quality (how relevant are the retrieved documents?)
  4. Evidence specificity (does evidence directly address the question?)

Output: a confidence score (0.0-1.0) with a human-readable level
(HIGH, MEDIUM, LOW) and a breakdown of contributing factors.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.domains.healthcare.context.evidence_aggregator import EvidenceSummary
from src.domains.healthcare.generation.grounding import GroundingResult
from src.domains.healthcare.retrieval.base_retriever import RetrievedDocument
from src.shared.logging_utils import setup_logger

logger = setup_logger("generation.confidence")


@dataclass
class ConfidenceResult:
    """Confidence assessment for a generated answer."""
    score: float = 0.0                  # 0.0 - 1.0
    level: str = "LOW"                  # HIGH, MEDIUM, LOW
    factors: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""               # human-readable explanation
    formatted_text: str = ""            # text to append to answer


class ConfidenceEstimator:
    """
    Evidence-based confidence scoring.

    Combines four factors to produce a confidence score:
      - Agreement: Do the retrieved reports agree?
      - Consistency: Does the answer match the evidence?
      - Retrieval quality: How good are the retrieval scores?
      - Specificity: Does the evidence address the question?

    Usage:
        estimator = ConfidenceEstimator()
        confidence = estimator.estimate(
            evidence_summary=summary,
            grounding_result=grounding,
            retrieved_docs=docs,
        )
        print(confidence.score, confidence.level)
    """

    # Weight of each factor in the final score
    _WEIGHTS = {
        "agreement": 0.35,
        "consistency": 0.30,
        "retrieval_quality": 0.20,
        "specificity": 0.15,
    }

    # Thresholds for confidence levels
    _HIGH_THRESHOLD = 0.70
    _MEDIUM_THRESHOLD = 0.40

    def estimate(
        self,
        evidence_summary: Optional[EvidenceSummary] = None,
        grounding_result: Optional[GroundingResult] = None,
        retrieved_docs: Optional[List[RetrievedDocument]] = None,
    ) -> ConfidenceResult:
        """
        Estimate confidence in the generated answer.

        Args:
            evidence_summary: From the EvidenceAggregator.
            grounding_result: From the GroundingVerifier.
            retrieved_docs:   The raw retrieved documents.

        Returns:
            ConfidenceResult with score, level, and breakdown.
        """
        factors = {}

        # Factor 1: Evidence agreement (0.0 - 1.0)
        factors["agreement"] = self._score_agreement(evidence_summary)

        # Factor 2: Answer-evidence consistency (0.0 - 1.0)
        factors["consistency"] = self._score_consistency(
            grounding_result
        )

        # Factor 3: Retrieval quality (0.0 - 1.0)
        factors["retrieval_quality"] = self._score_retrieval_quality(
            retrieved_docs
        )

        # Factor 4: Evidence specificity (0.0 - 1.0)
        factors["specificity"] = self._score_specificity(
            evidence_summary
        )

        # Weighted final score
        score = sum(
            factors[k] * self._WEIGHTS[k] for k in self._WEIGHTS
        )

        # Determine level
        if score >= self._HIGH_THRESHOLD:
            level = "HIGH"
        elif score >= self._MEDIUM_THRESHOLD:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Build explanation
        explanation = self._build_explanation(factors, score, level)

        # Build formatted text for output
        formatted = self._format_confidence(
            score, level, factors, evidence_summary
        )

        result = ConfidenceResult(
            score=round(score, 2),
            level=level,
            factors={k: round(v, 2) for k, v in factors.items()},
            explanation=explanation,
            formatted_text=formatted,
        )

        logger.info(
            f"Confidence: {level} ({score:.2f}) — "
            f"agree={factors['agreement']:.2f}, "
            f"consist={factors['consistency']:.2f}, "
            f"quality={factors['retrieval_quality']:.2f}, "
            f"specific={factors['specificity']:.2f}"
        )

        return result

    # ------------------------------------------------------------------ #
    #  Factor scoring                                                      #
    # ------------------------------------------------------------------ #

    def _score_agreement(
        self, summary: Optional[EvidenceSummary]
    ) -> float:
        """
        Score how well the retrieved reports agree with each other.

        Unanimous = 1.0, Majority = 0.7, Mixed = 0.3, None = 0.0
        """
        if not summary:
            return 0.0

        consensus_scores = {
            "UNANIMOUS_ABSENT": 1.0,
            "UNANIMOUS_PRESENT": 1.0,
            "MAJORITY_ABSENT": 0.7,
            "MAJORITY_PRESENT": 0.7,
            "MIXED_LEAN_ABSENT": 0.3,
            "MIXED_LEAN_PRESENT": 0.3,
            "INSUFFICIENT": 0.1,
        }

        return consensus_scores.get(summary.consensus, 0.0)

    def _score_consistency(
        self, grounding: Optional[GroundingResult]
    ) -> float:
        """
        Score how consistent the answer is with the evidence.

        Grounded + no correction = 1.0
        Corrected = 0.8 (we fixed it, so the output is now consistent)
        Contradiction flagged but not corrected = 0.2
        No grounding check = 0.5
        """
        if not grounding:
            return 0.5

        if grounding.is_grounded and not grounding.contradiction_detected:
            return 1.0

        if grounding.was_corrected:
            # Answer was corrected to match evidence — now consistent
            return 0.8

        if grounding.contradiction_detected:
            # Contradiction flagged but not corrected
            return 0.2

        return 0.5

    def _score_retrieval_quality(
        self, docs: Optional[List[RetrievedDocument]]
    ) -> float:
        """
        Score the quality of retrieval results.

        Based on the average retrieval score of the top documents.
        Higher scores = more relevant evidence = more confidence.
        """
        if not docs:
            return 0.0

        scores = [d.score for d in docs if d.score > 0]
        if not scores:
            return 0.0

        avg_score = sum(scores) / len(scores)

        # Normalize: typical ColQwen2 scores range from 0.3 to 1.0
        # Map this to 0.0-1.0 confidence
        normalized = max(0.0, min(1.0, (avg_score - 0.3) / 0.7))
        return normalized

    def _score_specificity(
        self, summary: Optional[EvidenceSummary]
    ) -> float:
        """
        Score how specifically the evidence addresses the question.

        More relevant findings = higher specificity.
        """
        if not summary:
            return 0.0

        if not summary.relevant_findings:
            return 0.1  # evidence exists but doesn't address question

        # More findings = more specific
        n_findings = len(summary.relevant_findings)
        if n_findings >= 3:
            return 1.0
        if n_findings >= 2:
            return 0.8
        if n_findings >= 1:
            return 0.5

        return 0.1

    # ------------------------------------------------------------------ #
    #  Explanation and formatting                                          #
    # ------------------------------------------------------------------ #

    def _build_explanation(
        self,
        factors: Dict[str, float],
        score: float,
        level: str,
    ) -> str:
        """Build a human-readable explanation of the confidence score."""
        parts = [f"Confidence: {level} ({score:.2f})"]

        if factors["agreement"] >= 0.7:
            parts.append("  + Retrieved reports strongly agree")
        elif factors["agreement"] <= 0.3:
            parts.append("  - Retrieved reports are mixed or insufficient")

        if factors["consistency"] >= 0.8:
            parts.append("  + Answer is consistent with evidence")
        elif factors["consistency"] <= 0.3:
            parts.append("  - Answer may contradict evidence")

        if factors["retrieval_quality"] >= 0.7:
            parts.append("  + High-quality retrieval matches")
        elif factors["retrieval_quality"] <= 0.3:
            parts.append("  - Low retrieval relevance scores")

        if factors["specificity"] >= 0.7:
            parts.append("  + Evidence directly addresses the question")
        elif factors["specificity"] <= 0.3:
            parts.append("  - Evidence may not address the question directly")

        return "\n".join(parts)

    def _format_confidence(
        self,
        score: float,
        level: str,
        factors: Dict[str, float],
        summary: Optional[EvidenceSummary] = None,
    ) -> str:
        """Format confidence for appending to the answer output."""
        parts = [
            f"\n--- Confidence: {level} ({score:.2f}) ---",
        ]

        # Factor breakdown
        parts.append(
            f"  Evidence agreement:  {factors['agreement']:.2f} "
            f"(weight: {self._WEIGHTS['agreement']})"
        )
        parts.append(
            f"  Answer consistency:  {factors['consistency']:.2f} "
            f"(weight: {self._WEIGHTS['consistency']})"
        )
        parts.append(
            f"  Retrieval quality:   {factors['retrieval_quality']:.2f} "
            f"(weight: {self._WEIGHTS['retrieval_quality']})"
        )
        parts.append(
            f"  Evidence specificity: {factors['specificity']:.2f} "
            f"(weight: {self._WEIGHTS['specificity']})"
        )

        if summary and summary.relevant_findings:
            n = len(summary.relevant_findings)
            parts.append(
                f"  Supporting evidence:  "
                f"{n} finding(s) from {summary.total_reports} reports"
            )

        return "\n".join(parts)
