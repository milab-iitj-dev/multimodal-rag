"""
RAG-augmented generation engine — Phase 4 (Grounded).

Orchestrates the full grounded retrieval-augmented generation pipeline:
    1. Receive question (+ optional query image)
    2. Classify query type (binary / descriptive / text-only / mixed)
    3. Retrieve relevant evidence from ColQwen2 index
    4. Aggregate evidence (consensus for binary, full-scan for descriptive)
    5. Build query-type-aware prompt
    6. Generate answer with Qwen2-VL
    7. Verify answer against evidence (grounding check)
    8. Score confidence
    9. Return answer with full provenance

Pipeline:
    Classify → Retrieve → Aggregate → Prompt → VLM → Verify →
    Confidence → Final Output
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from PIL import Image

from src.domains.healthcare.generation.base_generator import BaseVLM, VLMOutput
from src.domains.healthcare.retrieval.base_retriever import BaseRetriever, RetrievedDocument
from src.domains.healthcare.context.context_builder import ContextBuilder
from src.domains.healthcare.context.evidence_aggregator import EvidenceAggregator, EvidenceSummary
from src.domains.healthcare.context.query_classifier import (
    QueryClassifier, QueryType, QueryClassification,
)
from src.domains.healthcare.generation.grounding import GroundingVerifier, GroundingResult
from src.domains.healthcare.generation.confidence import ConfidenceEstimator, ConfidenceResult
from src.shared.logging_utils import setup_logger

logger = setup_logger("generation.rag")


@dataclass
class RAGOutput:
    """
    Full output from the grounded RAG pipeline.

    Contains the generated answer plus all intermediate results
    for traceability, evaluation, and debugging.
    """
    answer: str                                       # final verified answer
    retrieved_docs: List[RetrievedDocument] = field(default_factory=list)
    context_text: str = ""                            # assembled context string
    evidence_summary: Optional[EvidenceSummary] = None
    grounding_result: Optional[GroundingResult] = None
    confidence: Optional[ConfidenceResult] = None
    query: str = ""                                   # original query
    vlm_output: Optional[VLMOutput] = None            # raw VLM output
    retrieval_time_sec: float = 0.0
    generation_time_sec: float = 0.0
    total_time_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGGenerator:
    """
    Grounded RAG generation engine (Phase 4).

    Full pipeline:
      1. Retriever: find relevant cases (ColQwen2 + RRF fusion)
      2. EvidenceAggregator: condense reports into structured summary
      3. ContextBuilder: format evidence for VLM prompt
      4. VLM (Qwen2.5-VL): generate grounded answer
      5. GroundingVerifier: check answer vs evidence consistency
      6. ConfidenceEstimator: score answer confidence

    Usage:
        generator = RAGGenerator(vlm, retriever)
        output = generator.generate("Is there pleural effusion?", image)
        print(output.answer)
        print(output.confidence.level)
        print(output.grounding_result.was_corrected)
    """

    def __init__(
        self,
        vlm: BaseVLM,
        retriever: BaseRetriever,
        context_builder: Optional[ContextBuilder] = None,
        evidence_aggregator: Optional[EvidenceAggregator] = None,
        grounding_verifier: Optional[GroundingVerifier] = None,
        confidence_estimator: Optional[ConfidenceEstimator] = None,
        top_k: int = 3,
    ):
        """
        Args:
            vlm:                  Loaded VLM for answer generation.
            retriever:            Loaded retriever with index.
            context_builder:      For formatting evidence (fallback).
            evidence_aggregator:  For structured evidence summaries.
            grounding_verifier:   For post-generation verification.
            confidence_estimator: For confidence scoring.
            top_k:                Number of documents to retrieve.
        """
        self.vlm = vlm
        self.retriever = retriever
        self.context_builder = context_builder or ContextBuilder()
        self.evidence_aggregator = (
            evidence_aggregator or EvidenceAggregator()
        )
        self.grounding_verifier = (
            grounding_verifier or GroundingVerifier()
        )
        self.confidence_estimator = (
            confidence_estimator or ConfidenceEstimator()
        )
        self.query_classifier = QueryClassifier()
        self.top_k = top_k

    # ------------------------------------------------------------------ #
    #  Full grounded RAG pipeline                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        query: str,
        query_image: Optional[Image.Image] = None,
        top_k: Optional[int] = None,
        max_new_tokens: int = 512,
    ) -> RAGOutput:
        """
        Run the full grounded RAG pipeline.

        Flow:
            retrieve → aggregate → prompt → generate → verify → score

        Args:
            query:          The clinical question.
            query_image:    Optional query image (patient's X-ray).
            top_k:          Override default top_k.
            max_new_tokens: Max tokens for VLM generation.

        Returns:
            RAGOutput with verified answer, confidence, and provenance.
        """
        total_start = time.time()
        k = top_k or self.top_k

        logger.info(f"Grounded RAG: query='{query[:80]}' top_k={k}")

        # ============================================================
        # Step 1: Classify query type
        # ============================================================
        classification = self.query_classifier.classify(
            query=query,
            has_image=query_image is not None,
        )
        query_type = classification.query_type
        logger.info(
            f"  [1/7] Query type: {query_type.value} "
            f"(topic={classification.detected_topic})"
        )

        # ============================================================
        # Step 2: Retrieve relevant documents
        # ============================================================
        retrieval_start = time.time()
        retrieved_docs = self.retriever.retrieve(
            query=query,
            query_image=query_image,
            top_k=k,
        )
        retrieval_time = time.time() - retrieval_start
        logger.info(
            f"  [2/7] Retrieval: {len(retrieved_docs)} docs "
            f"in {retrieval_time:.2f}s"
        )

        # ============================================================
        # Step 3: Aggregate evidence (query-type aware)
        # ============================================================
        evidence_summary = self.evidence_aggregator.aggregate(
            retrieved_docs, query, query_type=query_type
        )
        logger.info(
            f"  [3/7] Evidence: consensus={evidence_summary.consensus}, "
            f"findings={len(evidence_summary.relevant_findings)}"
        )

        # ============================================================
        # Step 4: Build context for VLM
        # ============================================================
        context_text = evidence_summary.formatted_text

        # ============================================================
        # Step 5: Select image and generate answer
        # ============================================================
        if query_image is not None:
            llava_image = query_image
            image_source = "query"
        else:
            llava_image = self.context_builder.get_best_image(
                retrieved_docs
            )
            image_source = "retrieved"

        if llava_image is None:
            # ── TEXT-ONLY PATH ──
            logger.info("  [4/7] No image — using text-only evidence path")

            text_answer = self._generate_text_only_answer(
                evidence_summary, query, query_type
            )
            generation_time = 0.0
            vlm_output = VLMOutput(
                answer=text_answer,
                raw_output=text_answer,
                generation_time_sec=0.0,
                input_token_count=0,
                output_token_count=len(text_answer.split()),
                metadata={
                    "model": "text-only-evidence",
                    "path": "text_only",
                    "query_type": query_type.value,
                },
            )
            image_source = "none"
        else:
            # ── IMAGE PATH: Generate answer with VLM ──
            generation_start = time.time()
            vlm_output = self.vlm.generate(
                image=llava_image,
                question=query,
                context=context_text,
                max_new_tokens=max_new_tokens,
                query_type=query_type,
            )
            generation_time = time.time() - generation_start
            logger.info(
                f"  [4/7] Generated: '{vlm_output.answer[:100]}...' "
                f"in {generation_time:.2f}s"
            )

        # ============================================================
        # Step 6: Verify grounding (query-type aware)
        # ============================================================
        grounding_result = self.grounding_verifier.verify(
            answer=vlm_output.answer,
            evidence_summary=evidence_summary,
            question=query,
            query_type=query_type,
        )

        if grounding_result.was_corrected:
            logger.info(
                f"  [5/7] CORRECTED: {grounding_result.correction_reason}"
            )
            final_answer = grounding_result.verified_answer
        else:
            logger.info(
                f"  [5/7] Grounding: "
                f"{'PASS' if grounding_result.is_grounded else 'FLAG'}"
            )
            final_answer = grounding_result.verified_answer

        # ============================================================
        # Step 7: Score confidence
        # ============================================================
        confidence = self.confidence_estimator.estimate(
            evidence_summary=evidence_summary,
            grounding_result=grounding_result,
            retrieved_docs=retrieved_docs,
        )
        logger.info(
            f"  [6/7] Confidence: {confidence.level} ({confidence.score})"
        )

        # ============================================================
        # Step 8: Assemble final output
        # ============================================================
        total_time = time.time() - total_start
        logger.info(f"  [7/7] Total: {total_time:.2f}s")

        return RAGOutput(
            answer=final_answer,
            retrieved_docs=retrieved_docs,
            context_text=context_text,
            evidence_summary=evidence_summary,
            grounding_result=grounding_result,
            confidence=confidence,
            query=query,
            vlm_output=vlm_output,
            retrieval_time_sec=round(retrieval_time, 2),
            generation_time_sec=round(generation_time, 2),
            total_time_sec=round(total_time, 2),
            metadata={
                "top_k": k,
                "num_retrieved": len(retrieved_docs),
                "context_length": len(context_text),
                "query_type": query_type.value,
                "query_has_image": query_image is not None,
                "image_source": image_source,
                "detected_topic": classification.detected_topic,
                "consensus": evidence_summary.consensus,
                "was_corrected": grounding_result.was_corrected,
                "confidence_level": confidence.level,
                "confidence_score": confidence.score,
                "generation_path": (
                    "text_only" if llava_image is None else "vlm"
                ),
                # Explicit retrieval mode from HybridRetriever
                # Values: "hybrid", "image_only", "text_only", "none"
                "retrieval_mode": getattr(
                    self.retriever, "_last_retrieval_mode", "unknown"
                ),
            },
        )

    # ------------------------------------------------------------------ #
    #  Text-only answer generation (no VLM needed)                         #
    # ------------------------------------------------------------------ #

    def _generate_text_only_answer(
        self,
        evidence_summary: EvidenceSummary,
        question: str,
        query_type: QueryType = QueryType.TEXT_ONLY,
    ) -> str:
        """
        Generate an answer from evidence alone (no image, no VLM).

        Uses the QueryType to determine answer format:
          - BINARY_CLINICAL → direct YES/NO with cited evidence
          - DESCRIPTIVE / MIXED → structured evidence listing
          - Insufficient evidence → explicit statement

        Args:
            evidence_summary: Structured evidence from aggregator.
            question:         The clinical question.
            query_type:       Classification from QueryClassifier.

        Returns:
            Natural language answer string.
        """
        topic = evidence_summary.question_topic
        consensus = evidence_summary.consensus

        # ── BINARY CLINICAL: YES/NO answer ──
        is_binary = query_type == QueryType.BINARY_CLINICAL

        if is_binary and consensus in (
            "UNANIMOUS_ABSENT", "MAJORITY_ABSENT"
        ):
            answer = (
                f"NO. Based on {evidence_summary.num_absent}/"
                f"{evidence_summary.total_reports} retrieved reports, "
                f"{topic} is absent."
            )
            if evidence_summary.relevant_findings:
                sample = evidence_summary.relevant_findings[0]
                answer += f' Evidence: "{sample.text}"'
            return answer

        if is_binary and consensus in (
            "UNANIMOUS_PRESENT", "MAJORITY_PRESENT"
        ):
            answer = (
                f"YES. Based on {evidence_summary.num_present}/"
                f"{evidence_summary.total_reports} retrieved reports, "
                f"{topic} is present."
            )
            if evidence_summary.relevant_findings:
                sample = evidence_summary.relevant_findings[0]
                answer += f' Evidence: "{sample.text}"'
            return answer

        if is_binary and "MIXED" in consensus:
            return (
                f"UNCERTAIN. Evidence is mixed — "
                f"{evidence_summary.num_present} reports indicate presence, "
                f"{evidence_summary.num_absent} indicate absence of {topic}."
            )

        # ── DESCRIPTIVE / MIXED / GENERAL: evidence listing ──
        if evidence_summary.relevant_findings:
            parts = [
                f"Based on {evidence_summary.total_reports} "
                f"retrieved reports:"
            ]
            seen = set()
            for f in evidence_summary.relevant_findings:
                if f.text not in seen:
                    status = "ABSENT" if f.is_negated else "PRESENT"
                    parts.append(
                        f"- {f.text} ({status}, report {f.doc_id})"
                    )
                    seen.add(f.text)
            if evidence_summary.additional_findings:
                parts.append("\nAdditional findings:")
                for af in evidence_summary.additional_findings:
                    parts.append(f"- {af}")
            return "\n".join(parts)

        # ── Insufficient evidence ──
        return (
            f"Insufficient evidence. {evidence_summary.total_reports} "
            f"reports were retrieved but none clearly address "
            f"'{topic}'. An image may be needed for visual assessment."
        )

    # ------------------------------------------------------------------ #
    #  Simple generation (no retrieval, backward compat)                   #
    # ------------------------------------------------------------------ #

    def generate_simple(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 512,
    ) -> VLMOutput:
        """
        Direct VQA without retrieval (backward compatibility).

        Bypasses the retrieval pipeline and generates directly
        from the image and question.

        Args:
            image:          Input image.
            question:       The question.
            max_new_tokens: Max tokens to generate.

        Returns:
            VLMOutput from the VLM.
        """
        return self.vlm.generate(
            image=image,
            question=question,
            max_new_tokens=max_new_tokens,
        )
