"""
Evidence Aggregator — condense retrieved reports into structured summaries.

Transforms raw retrieved documents (noisy radiology reports with metadata)
into clean, question-focused evidence summaries that are optimized for
VLM comprehension and grounded answer generation.

Instead of dumping raw reports to the VLM, this module:
  1. Extracts sentences relevant to the question
  2. Detects negation and assertion patterns
  3. Counts consensus across reports
  4. Produces a structured summary with agreement statistics

Example:
  Input:  3 reports, each containing "No pleural effusion"
  Output: "3/3 reports indicate ABSENCE of pleural effusion. Consensus: UNANIMOUS"

This dramatically improves grounding because the VLM receives a clear
signal rather than noisy raw text.
"""

import re
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from src.domains.healthcare.retrieval.base_retriever import RetrievedDocument
from src.shared.logging_utils import setup_logger

logger = setup_logger("context.aggregator")


# ------------------------------------------------------------------ #
#  Negation detection patterns                                         #
# ------------------------------------------------------------------ #

# Patterns that indicate a finding is ABSENT
_NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnone\b",
    r"\bnor\b",
    r"\bwithout\b",
    r"\babsence\b",
    r"\babsent\b",
    r"\bnegative\b",
    r"\bnormal\b",
    r"\bno evidence\b",
    r"\bno signs?\b",
    r"\bnot? (?:seen|identified|observed|noted|detected|present)\b",
    r"\bfree of\b",
    r"\bclear of\b",
    r"\bunremarkable\b",
    r"\bwithin normal limits\b",
    r"\bno acute\b",
    r"\bdenies\b",
    r"\bruled out\b",
]

_NEGATION_RE = re.compile(
    "|".join(_NEGATION_PATTERNS), re.IGNORECASE
)

# Patterns that indicate a finding is PRESENT
_ASSERTION_PATTERNS = [
    r"\bpresent\b",
    r"\bidentified\b",
    r"\bobserved\b",
    r"\bnoted\b",
    r"\bdetected\b",
    r"\bconfirmed\b",
    r"\bconsistent with\b",
    r"\bcompatible with\b",
    r"\bsuggestive of\b",
    r"\bevidence of\b",
    r"\bfindings? of\b",
    r"\bdemonstrat\w+\b",
    r"\benlarg\w+\b",
    r"\bincreas\w+\b",
    r"\bopacit\w+\b",
    r"\beffusion\b",
    r"\bconsolidation\b",
    r"\binfiltra\w+\b",
]

_ASSERTION_RE = re.compile(
    "|".join(_ASSERTION_PATTERNS), re.IGNORECASE
)

# Markers that indicate a general/descriptive query (not disease-specific)
_GENERAL_QUERY_MARKERS = [
    r"\bfindings?\b", r"\babnormalit", r"\bvisible\b",
    r"\bshow\b", r"\bsee\b", r"\bidentify\b",
    r"\bdescribe\b", r"\bsummar", r"\boverall\b",
    r"\bchest x-?ray\b", r"\bradiograph\b",
]

_GENERAL_RE = re.compile(
    "|".join(_GENERAL_QUERY_MARKERS), re.IGNORECASE
)


# ------------------------------------------------------------------ #
#  Data classes                                                        #
# ------------------------------------------------------------------ #

@dataclass
class ExtractedFinding:
    """A single finding extracted from a report."""
    text: str                    # the relevant sentence/phrase
    doc_id: str                  # source document ID
    score: float                 # retrieval score
    is_negated: bool = False     # whether this finding is negated
    source_field: str = ""       # "findings" or "impression"


@dataclass
class EvidenceSummary:
    """Structured summary of aggregated evidence."""
    question: str                           # the original question
    question_topic: str                     # extracted topic keyword
    total_reports: int                      # number of reports analyzed
    relevant_findings: List[ExtractedFinding] = field(
        default_factory=list
    )
    num_absent: int = 0                     # reports saying finding is absent
    num_present: int = 0                    # reports saying finding is present
    num_ambiguous: int = 0                  # reports with unclear stance
    consensus: str = "INSUFFICIENT"         # UNANIMOUS_ABSENT, UNANIMOUS_PRESENT, MIXED, INSUFFICIENT
    consensus_strength: float = 0.0         # 0.0 to 1.0
    additional_findings: List[str] = field( # other relevant findings
        default_factory=list
    )
    formatted_text: str = ""                # final formatted summary for VLM

    @property
    def has_consensus(self) -> bool:
        return self.consensus in (
            "UNANIMOUS_ABSENT", "UNANIMOUS_PRESENT"
        )


# ------------------------------------------------------------------ #
#  Evidence Aggregator                                                 #
# ------------------------------------------------------------------ #

class EvidenceAggregator:
    """
    Condense retrieved medical reports into structured evidence.

    Workflow:
      1. Extract the question topic (e.g., "pleural effusion")
      2. Search each report for sentences mentioning that topic
      3. Classify each mention as negated or asserted
      4. Count consensus: how many reports agree?
      5. Format a clean summary for the VLM

    Usage:
        aggregator = EvidenceAggregator()
        summary = aggregator.aggregate(retrieved_docs, "Is there pleural effusion?")
        print(summary.formatted_text)  # send this to VLM instead of raw reports
    """

    # Common medical question prefixes to strip
    _QUESTION_PREFIXES = [
        r"^is there\s+",
        r"^are there\s+",
        r"^does (?:the |this )?\w+ (?:show|have|demonstrate|indicate)\s+",
        r"^what (?:are |is )?\s*(?:the )?(?:signs? of |evidence of )?\s*",
        r"^describe\s+(?:the\s+)?",
        r"^any\s+(?:signs?\s+of\s+)?",
        r"^do you see\s+",
        r"^can you identify\s+",
    ]

    def __init__(self):
        self._prefix_patterns = [
            re.compile(p, re.IGNORECASE) for p in self._QUESTION_PREFIXES
        ]

    def aggregate(
        self,
        retrieved_docs: List[RetrievedDocument],
        question: str,
        query_type=None,
    ) -> EvidenceSummary:
        """
        Aggregate retrieved documents into a structured evidence summary.

        Args:
            retrieved_docs: List of RetrievedDocument from the retriever.
            question:       The clinical question being asked.
            query_type:     Optional QueryType from the classifier.
                            If DESCRIPTIVE_IMAGE, uses full-scan mode.

        Returns:
            EvidenceSummary with consensus analysis and formatted text.
        """
        if not retrieved_docs:
            return EvidenceSummary(
                question=question,
                question_topic="",
                total_reports=0,
                formatted_text="No evidence retrieved.",
            )

        # Step 1: Determine aggregation mode.
        # Primary signal: query_type from the classifier.
        # Fallback: _extract_topic's __GENERAL__ detection.
        from src.domains.healthcare.context.query_classifier import QueryType

        topic = self._extract_topic(question)
        logger.info(f"Evidence aggregation: topic='{topic}' from '{question}'")

        use_full_scan = (
            topic == "__GENERAL__"
            or query_type == QueryType.DESCRIPTIVE_IMAGE
        )

        # Step 2: Extract findings — different strategy per mode
        if use_full_scan:
            # DESCRIPTIVE / GENERAL: scan all reports for ALL findings
            all_findings, additional = self._extract_all_findings_scan(
                retrieved_docs
            )
            topic_display = "general findings"
        else:
            # SPECIFIC QUERY: look for the specific topic
            all_findings = []
            for doc in retrieved_docs:
                findings = self._extract_relevant_findings(doc, topic)
                all_findings.extend(findings)

            # Fallback: check full reports if no findings in structured fields
            if not all_findings:
                for doc in retrieved_docs:
                    report_text = self._get_report_text(doc)
                    if topic.lower() in report_text.lower():
                        is_neg = bool(_NEGATION_RE.search(
                            self._get_sentence_with_topic(
                                report_text, topic
                            )
                        ))
                        all_findings.append(ExtractedFinding(
                            text=self._get_sentence_with_topic(
                                report_text, topic
                            ),
                            doc_id=doc.doc_id,
                            score=doc.score,
                            is_negated=is_neg,
                            source_field="report",
                        ))

            additional = self._extract_additional_findings(
                retrieved_docs, topic
            )
            topic_display = topic

        # Step 3: Classify and count consensus
        num_absent = sum(1 for f in all_findings if f.is_negated)
        num_present = sum(1 for f in all_findings if not f.is_negated)

        num_ambiguous = len(retrieved_docs) - (
            len(set(f.doc_id for f in all_findings))
        )

        # Step 4: Determine consensus
        consensus, strength = self._compute_consensus(
            num_absent, num_present, num_ambiguous, len(retrieved_docs)
        )

        # Step 6: Format for VLM
        summary = EvidenceSummary(
            question=question,
            question_topic=topic_display,
            total_reports=len(retrieved_docs),
            relevant_findings=all_findings,
            num_absent=num_absent,
            num_present=num_present,
            num_ambiguous=num_ambiguous,
            consensus=consensus,
            consensus_strength=strength,
            additional_findings=additional,
        )
        summary.formatted_text = self._format_summary(summary)

        logger.info(
            f"  Aggregated: {len(all_findings)} findings, "
            f"consensus={consensus} ({strength:.2f})"
        )

        return summary

    # ------------------------------------------------------------------ #
    #  Topic extraction                                                    #
    # ------------------------------------------------------------------ #

    def _extract_topic(self, question: str) -> str:
        """
        Extract the medical topic from a clinical question.

        Examples:
            "Is there pleural effusion?"      → "pleural effusion"
            "What are signs of cardiomegaly?" → "cardiomegaly"
            "Describe the findings"           → "__GENERAL__"
            "What does this X-ray show?"      → "__GENERAL__"

        Returns "__GENERAL__" for broad descriptive queries that
        don't target a specific finding.
        """
        topic = question.strip().rstrip("?.,!")

        for pattern in self._prefix_patterns:
            topic = pattern.sub("", topic)

        topic = topic.strip().rstrip("?.,!")

        # Remove trailing articles and prepositions
        topic = re.sub(r"\b(?:a|an|the|in|on|of|for|this|that)\s*$", "", topic)
        topic = topic.strip()

        if not topic:
            topic = question.strip().rstrip("?.,!")

        # Detect general/descriptive queries:
        # - Topic is too long (>4 words) — probably not a specific finding
        # - Topic contains general markers ("findings", "visible", "show")
        word_count = len(topic.split())
        if word_count > 4 or _GENERAL_RE.search(topic):
            logger.info(
                f"  General query detected (topic='{topic}', "
                f"words={word_count}) — using full-scan mode"
            )
            return "__GENERAL__"

        return topic

    # ------------------------------------------------------------------ #
    #  Finding extraction                                                  #
    # ------------------------------------------------------------------ #

    def _extract_relevant_findings(
        self,
        doc: RetrievedDocument,
        topic: str,
    ) -> List[ExtractedFinding]:
        """Extract findings from a document that mention the topic."""
        findings = []
        topic_lower = topic.lower()

        # Check findings field
        doc_findings = doc.metadata.get("findings", "") or ""
        if doc_findings and topic_lower in doc_findings.lower():
            sentence = self._get_sentence_with_topic(
                doc_findings, topic
            )
            if sentence:
                is_neg = self._is_negated(sentence, topic)
                findings.append(ExtractedFinding(
                    text=sentence.strip(),
                    doc_id=doc.doc_id,
                    score=doc.score,
                    is_negated=is_neg,
                    source_field="findings",
                ))

        # Check impression field
        doc_impression = doc.metadata.get("impression", "") or ""
        if doc_impression and topic_lower in doc_impression.lower():
            sentence = self._get_sentence_with_topic(
                doc_impression, topic
            )
            if sentence:
                is_neg = self._is_negated(sentence, topic)
                # Avoid duplicate if same sentence found in both
                if not any(
                    f.text == sentence.strip() for f in findings
                ):
                    findings.append(ExtractedFinding(
                        text=sentence.strip(),
                        doc_id=doc.doc_id,
                        score=doc.score,
                        is_negated=is_neg,
                        source_field="impression",
                    ))

        return findings

    def _get_sentence_with_topic(
        self, text: str, topic: str
    ) -> str:
        """Extract the sentence containing the topic keyword."""
        # Split into sentences
        sentences = re.split(r'[.!]\s+', text)
        topic_lower = topic.lower()

        for sentence in sentences:
            if topic_lower in sentence.lower():
                return sentence.strip()

        # If no sentence-level match, return the portion around the topic
        idx = text.lower().find(topic_lower)
        if idx >= 0:
            start = max(0, idx - 80)
            end = min(len(text), idx + len(topic) + 80)
            return text[start:end].strip()

        return text[:200]

    def _is_negated(self, sentence: str, topic: str) -> bool:
        """
        Determine if the topic is negated in the sentence.

        Strategy: check if negation words appear BEFORE the topic
        in the sentence. "No pleural effusion" → negated.
        "Pleural effusion present" → not negated.
        """
        topic_idx = sentence.lower().find(topic.lower())
        if topic_idx < 0:
            # Topic not in sentence — check overall negation
            return bool(_NEGATION_RE.search(sentence))

        # Check text BEFORE the topic for negation
        prefix = sentence[:topic_idx]
        if _NEGATION_RE.search(prefix):
            return True

        # Check if the full sentence is a negation pattern
        # e.g., "There is no evidence of pleural effusion"
        if _NEGATION_RE.search(sentence[:topic_idx + len(topic) + 20]):
            # But only if the negation is before the topic
            neg_match = _NEGATION_RE.search(sentence)
            if neg_match and neg_match.start() < topic_idx:
                return True

        return False

    def _get_report_text(self, doc: RetrievedDocument) -> str:
        """Get the full text content of a document."""
        parts = []
        findings = doc.metadata.get("findings", "")
        impression = doc.metadata.get("impression", "")

        if findings:
            parts.append(findings)
        if impression:
            parts.append(impression)
        if not parts and doc.text:
            parts.append(doc.text)

        return " ".join(parts)

    # ------------------------------------------------------------------ #
    #  Consensus computation                                               #
    # ------------------------------------------------------------------ #

    def _compute_consensus(
        self,
        num_absent: int,
        num_present: int,
        num_ambiguous: int,
        total: int,
    ) -> Tuple[str, float]:
        """
        Compute consensus classification and strength.

        Returns:
            (consensus_label, strength_score)
        """
        if total == 0:
            return "INSUFFICIENT", 0.0

        evidence_count = num_absent + num_present

        if evidence_count == 0:
            return "INSUFFICIENT", 0.0

        if num_absent > 0 and num_present == 0:
            strength = num_absent / total
            if strength >= 0.8:
                return "UNANIMOUS_ABSENT", strength
            return "MAJORITY_ABSENT", strength

        if num_present > 0 and num_absent == 0:
            strength = num_present / total
            if strength >= 0.8:
                return "UNANIMOUS_PRESENT", strength
            return "MAJORITY_PRESENT", strength

        # Mixed evidence
        total_ev = num_absent + num_present
        majority = max(num_absent, num_present)
        strength = majority / total_ev
        if num_absent > num_present:
            return "MIXED_LEAN_ABSENT", strength
        return "MIXED_LEAN_PRESENT", strength

    # ------------------------------------------------------------------ #
    #  Full-scan extraction (for general queries)                           #
    # ------------------------------------------------------------------ #

    # Clinical findings to scan for in general-query mode
    _SCAN_FINDINGS = [
        "cardiomegaly", "pleural effusion", "pneumothorax",
        "consolidation", "atelectasis", "edema", "pulmonary edema",
        "opacity", "opacities", "pneumonia", "mass", "nodule",
        "fracture", "emphysema", "fibrosis", "calcification",
        "hilar prominence", "mediastinal widening", "aortic",
        "pericardial effusion", "interstitial", "infiltrate",
        "hyperinflation", "scoliosis", "degenerative",
    ]

    def _extract_all_findings_scan(
        self,
        docs: List[RetrievedDocument],
    ) -> Tuple[List[ExtractedFinding], List[str]]:
        """
        Scan all reports for ALL clinical findings (general query mode).

        Instead of searching for one specific topic, this method scans
        every report for every known clinical finding, classifies each
        as present/absent, and returns all findings.

        Used when _extract_topic returns "__GENERAL__".

        Returns:
            (all_findings, additional_summary_lines)
        """
        all_findings = []
        finding_counts: Dict[str, Dict[str, int]] = {}
        # finding_counts[finding] = {"present": N, "absent": N}

        for doc in docs:
            report_text = self._get_report_text(doc)
            report_lower = report_text.lower()

            for finding in self._SCAN_FINDINGS:
                if finding in report_lower:
                    sentence = self._get_sentence_with_topic(
                        report_text, finding
                    )
                    is_neg = self._is_negated(sentence, finding)

                    all_findings.append(ExtractedFinding(
                        text=sentence.strip(),
                        doc_id=doc.doc_id,
                        score=doc.score,
                        is_negated=is_neg,
                        source_field="report",
                    ))

                    if finding not in finding_counts:
                        finding_counts[finding] = {
                            "present": 0, "absent": 0
                        }
                    if is_neg:
                        finding_counts[finding]["absent"] += 1
                    else:
                        finding_counts[finding]["present"] += 1

        # Build additional summary lines
        additional = []
        for finding, counts in sorted(
            finding_counts.items(),
            key=lambda x: x[1]["present"],
            reverse=True,
        ):
            total_mentions = counts["present"] + counts["absent"]
            if counts["present"] > 0:
                additional.append(
                    f"{finding.title()} PRESENT in "
                    f"{counts['present']}/{len(docs)} reports"
                )
            elif counts["absent"] > 0 and total_mentions >= 2:
                additional.append(
                    f"{finding.title()} ABSENT in "
                    f"{counts['absent']}/{len(docs)} reports"
                )

        logger.info(
            f"  Full-scan: {len(all_findings)} findings across "
            f"{len(finding_counts)} categories from {len(docs)} reports"
        )

        return all_findings, additional[:10]

    # ------------------------------------------------------------------ #
    #  Additional findings                                                 #
    # ------------------------------------------------------------------ #

    def _extract_additional_findings(
        self,
        docs: List[RetrievedDocument],
        topic: str,
    ) -> List[str]:
        """Extract notable findings NOT about the question topic."""
        additional = []
        topic_lower = topic.lower()

        # Common clinical findings to look for
        key_findings = [
            "cardiomegaly", "pleural effusion", "pneumothorax",
            "consolidation", "atelectasis", "edema", "opacity",
            "pneumonia", "mass", "nodule", "fracture", "emphysema",
        ]

        finding_counts: Dict[str, int] = {}

        for doc in docs:
            report = self._get_report_text(doc)
            report_lower = report.lower()

            for finding in key_findings:
                if finding == topic_lower:
                    continue
                if finding in report_lower:
                    finding_counts[finding] = (
                        finding_counts.get(finding, 0) + 1
                    )

        for finding, count in sorted(
            finding_counts.items(), key=lambda x: -x[1]
        ):
            if count >= 2:
                additional.append(
                    f"{finding.title()} mentioned in {count}/{len(docs)} reports"
                )

        return additional[:5]  # limit to top 5

    # ------------------------------------------------------------------ #
    #  Format summary for VLM                                              #
    # ------------------------------------------------------------------ #

    def _format_summary(self, summary: EvidenceSummary) -> str:
        """
        Format the evidence summary as clean text for the VLM.

        This is the text that replaces the raw report dump in the
        prompt. It is designed to be maximally clear and unambiguous
        for the VLM to ground its answer on.
        """
        parts = [
            f"EVIDENCE SUMMARY ({summary.total_reports} reports analyzed):\n"
        ]

        # Question-relevant findings
        if summary.relevant_findings:
            parts.append(f"Regarding '{summary.question_topic}':")
            for f in summary.relevant_findings:
                direction = "ABSENT" if f.is_negated else "PRESENT"
                parts.append(
                    f"  - Report {f.doc_id} (relevance {f.score:.2f}): "
                    f"\"{f.text}\" → {direction}"
                )

            # Consensus line
            parts.append("")
            consensus_map = {
                "UNANIMOUS_ABSENT": (
                    f"CONSENSUS: {summary.num_absent}/{summary.total_reports} "
                    f"reports indicate ABSENCE of {summary.question_topic}. "
                    f"Agreement: UNANIMOUS."
                ),
                "UNANIMOUS_PRESENT": (
                    f"CONSENSUS: {summary.num_present}/{summary.total_reports} "
                    f"reports indicate PRESENCE of {summary.question_topic}. "
                    f"Agreement: UNANIMOUS."
                ),
                "MAJORITY_ABSENT": (
                    f"CONSENSUS: {summary.num_absent}/{summary.total_reports} "
                    f"reports indicate ABSENCE of {summary.question_topic}. "
                    f"Agreement: MAJORITY."
                ),
                "MAJORITY_PRESENT": (
                    f"CONSENSUS: {summary.num_present}/{summary.total_reports} "
                    f"reports indicate PRESENCE of {summary.question_topic}. "
                    f"Agreement: MAJORITY."
                ),
                "MIXED_LEAN_ABSENT": (
                    f"CONSENSUS: MIXED. "
                    f"{summary.num_absent} reports say absent, "
                    f"{summary.num_present} say present. "
                    f"Leans toward ABSENCE."
                ),
                "MIXED_LEAN_PRESENT": (
                    f"CONSENSUS: MIXED. "
                    f"{summary.num_present} reports say present, "
                    f"{summary.num_absent} say absent. "
                    f"Leans toward PRESENCE."
                ),
                "INSUFFICIENT": (
                    "CONSENSUS: INSUFFICIENT EVIDENCE. "
                    "Retrieved reports do not clearly address this topic."
                ),
            }
            parts.append(
                consensus_map.get(summary.consensus, "CONSENSUS: UNKNOWN")
            )
        else:
            parts.append(
                f"No specific findings about '{summary.question_topic}' "
                f"found in retrieved reports."
            )
            parts.append("CONSENSUS: INSUFFICIENT EVIDENCE.")

        # Additional findings
        if summary.additional_findings:
            parts.append("\nOther notable findings:")
            for f in summary.additional_findings:
                parts.append(f"  - {f}")

        # Raw evidence reference (abbreviated)
        parts.append("\nSource reports (abbreviated):")
        # Only include the first few for reference
        for i, f in enumerate(summary.relevant_findings[:3]):
            findings_text = ""
            impression_text = ""
            # We stored these in metadata during retrieval
            parts.append(f"  Report {f.doc_id}: \"{f.text}\"")

        return "\n".join(parts)
