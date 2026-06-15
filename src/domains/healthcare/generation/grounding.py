"""
Grounding Verifier — post-generation answer verification.

After the VLM generates an answer, this module verifies that the
answer is consistent with the retrieved evidence. If the answer
contradicts unanimous evidence, it flags the contradiction and
can optionally correct the answer.

This is a lightweight, rule-based verifier (no extra model needed).
It uses the EvidenceSummary from the aggregator to check:
  1. Does the answer direction (yes/no) match the evidence consensus?
  2. Does the answer mention findings that contradict the evidence?
  3. Is the answer grounded in at least some evidence?

Design philosophy:
  - Conservative corrections only (unanimous evidence + clear contradiction)
  - Transparent: always explains what was changed and why
  - Does NOT rewrite the answer — only flags or overrides yes/no direction
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from src.domains.healthcare.context.evidence_aggregator import EvidenceSummary
from src.shared.logging_utils import setup_logger

logger = setup_logger("generation.grounding")


@dataclass
class GroundingResult:
    """Result of grounding verification."""
    original_answer: str              # the VLM's raw answer
    verified_answer: str              # the final answer (may be corrected)
    is_grounded: bool                 # whether the answer is grounded
    was_corrected: bool = False       # whether the answer was changed
    contradiction_detected: bool = False
    correction_reason: str = ""       # why it was corrected
    grounding_details: Dict = field(default_factory=dict)


class GroundingVerifier:
    """
    Post-generation answer verification against evidence.

    Checks if the VLM's answer is consistent with the evidence
    consensus from the EvidenceAggregator. Corrects clear
    contradictions when evidence is unanimous.

    Usage:
        verifier = GroundingVerifier()
        result = verifier.verify(answer, evidence_summary)
        print(result.verified_answer)
        print(result.was_corrected)
    """

    # Patterns indicating YES answer
    _YES_PATTERNS = [
        r"^\s*yes\b",
        r"\bthere (?:is|are)\b(?! no\b)",
        r"\bpresent\b",
        r"\bidentified\b",
        r"\bdemonstrates?\b",
        r"\bshows?\b(?! no\b)",
        r"\bevidence of\b",
        r"\bconsistent with\b",
        r"\bfindings? (?:of|suggest|indicate)\b",
    ]

    # Patterns indicating NO answer
    _NO_PATTERNS = [
        r"^\s*no\b",
        r"\bno evidence\b",
        r"\bno signs?\b",
        r"\bnot? (?:seen|identified|observed|present|detected)\b",
        r"\babsence\b",
        r"\babsent\b",
        r"\bwithout\b",
        r"\bnegative\b",
        r"\bnormal\b(?! size| limits)",
        r"\bunremarkable\b",
        r"\bdoes not (?:show|demonstrate|indicate)\b",
        r"\bno (?:definite|significant|focal)\b",
    ]

    def __init__(self):
        self._yes_re = re.compile(
            "|".join(self._YES_PATTERNS), re.IGNORECASE
        )
        self._no_re = re.compile(
            "|".join(self._NO_PATTERNS), re.IGNORECASE
        )

    def verify(
        self,
        answer: str,
        evidence_summary: Optional[EvidenceSummary] = None,
        question: str = "",
        query_type=None,
    ) -> GroundingResult:
        """
        Verify that the answer is grounded in evidence.

        For BINARY_CLINICAL queries: checks YES/NO direction against
        evidence consensus and corrects contradictions.

        For DESCRIPTIVE_IMAGE queries: skips direction checking since
        descriptive answers have no binary direction to verify.

        Args:
            answer:           The VLM's generated answer.
            evidence_summary: The structured evidence from aggregator.
            question:         The original question.
            query_type:       QueryType from the classifier.

        Returns:
            GroundingResult with verification details.
        """
        if not evidence_summary or not answer:
            return GroundingResult(
                original_answer=answer,
                verified_answer=answer,
                is_grounded=True,  # no evidence to contradict
            )

        # For descriptive queries, skip YES/NO direction checking.
        # Descriptive answers are open-ended — checking binary direction
        # would produce false contradiction signals.
        from src.domains.healthcare.context.query_classifier import QueryType

        if query_type in (QueryType.DESCRIPTIVE_IMAGE, QueryType.MIXED_IMAGE_TEXT):
            logger.info(
                f"Grounding check: SKIP direction check "
                f"(query_type={query_type.value})"
            )
            return GroundingResult(
                original_answer=answer,
                verified_answer=answer,
                is_grounded=True,
                grounding_details={
                    "query_type": query_type.value,
                    "skip_reason": "descriptive query — no binary direction",
                    "consensus": evidence_summary.consensus,
                },
            )

        # Step 1: Detect answer direction (yes/no/neutral)
        answer_direction = self._detect_direction(answer)

        # Step 2: Get evidence direction from consensus
        evidence_direction = self._get_evidence_direction(
            evidence_summary
        )

        logger.info(
            f"Grounding check: answer={answer_direction}, "
            f"evidence={evidence_direction}, "
            f"consensus={evidence_summary.consensus}"
        )

        # Step 3: Check for contradiction
        contradiction = self._check_contradiction(
            answer_direction, evidence_direction, evidence_summary
        )

        if contradiction:
            # Step 4: Correct if evidence is strong enough
            corrected = self._correct_answer(
                answer, evidence_summary, answer_direction,
                evidence_direction,
            )
            return corrected

        return GroundingResult(
            original_answer=answer,
            verified_answer=answer,
            is_grounded=True,
            grounding_details={
                "answer_direction": answer_direction,
                "evidence_direction": evidence_direction,
                "consensus": evidence_summary.consensus,
                "consensus_strength": evidence_summary.consensus_strength,
            },
        )

    # ------------------------------------------------------------------ #
    #  Direction detection                                                 #
    # ------------------------------------------------------------------ #

    def _detect_direction(self, answer: str) -> str:
        """
        Detect whether the answer says YES, NO, or is NEUTRAL.

        Checks the first ~100 chars of the answer for direction cues.
        """
        # Check the beginning of the answer first (most indicative)
        first_part = answer[:150].lower()

        # Strong YES at start
        if re.match(r'^\s*yes\b', first_part):
            return "YES"

        # Strong NO at start
        if re.match(r'^\s*no\b', first_part):
            return "NO"

        # Count pattern matches
        yes_matches = len(self._yes_re.findall(first_part))
        no_matches = len(self._no_re.findall(first_part))

        if yes_matches > no_matches:
            return "YES"
        if no_matches > yes_matches:
            return "NO"

        return "NEUTRAL"

    def _get_evidence_direction(
        self, summary: EvidenceSummary
    ) -> str:
        """Get the direction from evidence consensus."""
        if summary.consensus in (
            "UNANIMOUS_ABSENT", "MAJORITY_ABSENT", "MIXED_LEAN_ABSENT"
        ):
            return "ABSENT"
        if summary.consensus in (
            "UNANIMOUS_PRESENT", "MAJORITY_PRESENT", "MIXED_LEAN_PRESENT"
        ):
            return "PRESENT"
        return "INSUFFICIENT"

    # ------------------------------------------------------------------ #
    #  Contradiction detection                                             #
    # ------------------------------------------------------------------ #

    def _check_contradiction(
        self,
        answer_direction: str,
        evidence_direction: str,
        summary: EvidenceSummary,
    ) -> bool:
        """
        Check if the answer contradicts the evidence.

        A contradiction occurs when:
          - Answer says YES but evidence says ABSENT
          - Answer says NO but evidence says PRESENT

        Only flags contradictions when evidence consensus is strong.
        """
        if answer_direction == "NEUTRAL":
            return False

        if evidence_direction == "INSUFFICIENT":
            return False

        # YES answer + ABSENT evidence = contradiction
        if answer_direction == "YES" and evidence_direction == "ABSENT":
            # Only flag if consensus is strong enough
            if summary.consensus_strength >= 0.6:
                logger.warning(
                    f"CONTRADICTION: Answer says YES but "
                    f"{summary.num_absent}/{summary.total_reports} "
                    f"reports say ABSENT "
                    f"(consensus: {summary.consensus})"
                )
                return True

        # NO answer + PRESENT evidence = contradiction
        if answer_direction == "NO" and evidence_direction == "PRESENT":
            if summary.consensus_strength >= 0.6:
                logger.warning(
                    f"CONTRADICTION: Answer says NO but "
                    f"{summary.num_present}/{summary.total_reports} "
                    f"reports say PRESENT "
                    f"(consensus: {summary.consensus})"
                )
                return True

        return False

    # ------------------------------------------------------------------ #
    #  Answer correction                                                   #
    # ------------------------------------------------------------------ #

    def _correct_answer(
        self,
        answer: str,
        summary: EvidenceSummary,
        answer_direction: str,
        evidence_direction: str,
    ) -> GroundingResult:
        """
        Correct a contradicted answer.

        Only corrects when evidence is UNANIMOUS. For mixed evidence,
        flags the contradiction but does not change the answer.
        """
        topic = summary.question_topic

        # Only auto-correct for UNANIMOUS consensus
        if summary.consensus.startswith("UNANIMOUS"):
            if evidence_direction == "ABSENT":
                corrected = (
                    f"NO. Based on retrieved evidence, there is no "
                    f"{topic} identified. "
                    f"{summary.num_absent}/{summary.total_reports} "
                    f"similar cases show absence of {topic}."
                )
                reason = (
                    f"Original answer said YES but "
                    f"{summary.num_absent}/{summary.total_reports} "
                    f"reports unanimously indicate absence of {topic}."
                )
            else:
                corrected = (
                    f"YES. Based on retrieved evidence, {topic} is "
                    f"present. "
                    f"{summary.num_present}/{summary.total_reports} "
                    f"similar cases confirm presence of {topic}."
                )
                reason = (
                    f"Original answer said NO but "
                    f"{summary.num_present}/{summary.total_reports} "
                    f"reports unanimously indicate presence of {topic}."
                )

            logger.info(f"Answer CORRECTED: {reason}")

            return GroundingResult(
                original_answer=answer,
                verified_answer=corrected,
                is_grounded=False,
                was_corrected=True,
                contradiction_detected=True,
                correction_reason=reason,
                grounding_details={
                    "answer_direction": answer_direction,
                    "evidence_direction": evidence_direction,
                    "consensus": summary.consensus,
                    "consensus_strength": summary.consensus_strength,
                    "correction_type": "unanimous_override",
                },
            )

        # For non-unanimous: flag but don't correct
        return GroundingResult(
            original_answer=answer,
            verified_answer=answer,  # keep original
            is_grounded=False,
            was_corrected=False,
            contradiction_detected=True,
            correction_reason=(
                f"Answer direction ({answer_direction}) conflicts with "
                f"evidence direction ({evidence_direction}), but evidence "
                f"is not unanimous ({summary.consensus}). "
                f"Keeping original answer."
            ),
            grounding_details={
                "answer_direction": answer_direction,
                "evidence_direction": evidence_direction,
                "consensus": summary.consensus,
                "consensus_strength": summary.consensus_strength,
                "correction_type": "flagged_only",
            },
        )
