"""
Context builder for assembling retrieved evidence into VLM prompts.

Takes retrieval results (documents, images, scores) and builds a
structured context string that the VLM can use to generate grounded,
evidence-based answers. Controls how much context to include,
de-duplicates overlapping information, and formats it cleanly.

The context builder is the bridge between retrieval and generation:
  Retrieved cases → ContextBuilder → formatted text → LLaVA prompt
"""

from typing import List, Optional, Dict, Any

from PIL import Image

from src.domains.healthcare.retrieval.base_retriever import RetrievedDocument
from src.domains.healthcare.context.prompt_templates import RAG_VQA_PROMPT, SIMPLE_VQA_PROMPT
from src.domains.healthcare.ingestion.preprocessing import truncate_text
from src.shared.logging_utils import setup_logger

logger = setup_logger("context.builder")


class ContextBuilder:
    """
    Build structured context from retrieved evidence for LLaVA.

    Converts the top-k retrieved documents into a clean text block
    that can be injected into the VLM prompt. Also selects the best
    reference image from the retrieved results.

    Usage:
        builder = ContextBuilder(max_context_tokens=1024)
        context = builder.build_context(retrieved_docs)
        prompt = builder.build_prompt(question, context)
        best_image = builder.get_best_image(retrieved_docs)
    """

    def __init__(
        self,
        max_context_chars: int = 3000,
        max_evidence_chars: int = 800,
    ):
        """
        Args:
            max_context_chars: Maximum character budget for the entire context block.
            max_evidence_chars: Maximum character budget per evidence piece.
        """
        self.max_context_chars = max_context_chars
        self.max_evidence_chars = max_evidence_chars

    # ------------------------------------------------------------------ #
    #  Build context from retrieved documents                              #
    # ------------------------------------------------------------------ #

    def build_context(
        self,
        retrieved_docs: List[RetrievedDocument],
        include_scores: bool = True,
    ) -> str:
        """
        Build a structured text context from retrieved documents.

        Each retrieved case is formatted as a numbered evidence block
        with findings, impression, and metadata. The total context
        is kept within the token budget.

        Args:
            retrieved_docs: List of RetrievedDocument from the retriever.
            include_scores: Whether to include relevance scores.

        Returns:
            Formatted context string ready for prompt injection.
        """
        if not retrieved_docs:
            return ""

        context_parts = ["=== Retrieved Medical Evidence ===\n"]
        total_chars = len(context_parts[0])

        for i, doc in enumerate(retrieved_docs):
            evidence = self._format_evidence(
                doc,
                rank=i + 1,
                include_score=include_scores,
            )

            # Token budget check
            if total_chars + len(evidence) > self.max_context_chars:
                logger.info(
                    f"  Context budget reached at evidence #{i + 1} "
                    f"({total_chars} chars)"
                )
                break

            context_parts.append(evidence)
            total_chars += len(evidence)

        context_parts.append("=== End of Retrieved Evidence ===")

        full_context = "\n".join(context_parts)

        # Hard truncation safety net
        if len(full_context) > self.max_context_chars:
            full_context = full_context[:self.max_context_chars] + "\n[...truncated for token budget]"

        logger.info(
            f"Context built from {len(retrieved_docs)} documents "
            f"({len(full_context)} chars, ~{len(full_context.split())} words)"
        )

        return full_context

    # ------------------------------------------------------------------ #
    #  Format individual evidence blocks                                   #
    # ------------------------------------------------------------------ #

    def _format_evidence(
        self,
        doc: RetrievedDocument,
        rank: int,
        include_score: bool = True,
    ) -> str:
        """
        Format a single retrieved document as an evidence block.

        Args:
            doc:           The retrieved document.
            rank:          The rank position (1-based).
            include_score: Whether to include the relevance score.

        Returns:
            Formatted evidence string.
        """
        parts = [f"--- Evidence #{rank} ---"]

        # Case ID
        parts.append(f"Case ID: {doc.doc_id}")

        # Relevance score
        if include_score:
            parts.append(f"Relevance Score: {doc.score:.4f}")

        # Findings
        findings = doc.metadata.get("findings")
        if findings:
            parts.append(f"Findings: {findings[:self.max_evidence_chars]}")

        # Impression
        impression = doc.metadata.get("impression")
        if impression:
            parts.append(f"Impression: {impression[:self.max_evidence_chars]}")

        # If neither findings nor impression, use full text
        if not findings and not impression and doc.text:
            parts.append(f"Report: {doc.text[:self.max_evidence_chars]}")

        # Clinical metadata (if available)
        mesh_terms = doc.metadata.get("mesh_terms")
        if mesh_terms:
            parts.append(f"MeSH Terms: {'; '.join(mesh_terms)}")

        problems = doc.metadata.get("problems")
        if problems:
            parts.append(f"Problems: {'; '.join(problems)}")

        parts.append("")  # blank line separator
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Build full prompt                                                   #
    # ------------------------------------------------------------------ #

    def build_prompt(
        self,
        question: str,
        context: str,
        use_rag: bool = True,
    ) -> str:
        """
        Build the final prompt string for LLaVA.

        Args:
            question: The user's clinical question.
            context:  The assembled context from retrieved evidence.
            use_rag:  Whether to use RAG prompt template (True) or simple VQA (False).

        Returns:
            Complete prompt string formatted for LLaVA.
        """
        if use_rag and context:
            return RAG_VQA_PROMPT.format(
                context=context,
                question=question,
            )
        else:
            return SIMPLE_VQA_PROMPT.format(question=question)

    # ------------------------------------------------------------------ #
    #  Image selection                                                     #
    # ------------------------------------------------------------------ #

    def get_best_image(
        self,
        retrieved_docs: List[RetrievedDocument],
    ) -> Optional[Image.Image]:
        """
        Get the image from the highest-scoring retrieved document.

        Used when the user query doesn't include an image — the best
        retrieved image serves as the visual input for LLaVA.

        Args:
            retrieved_docs: Ranked list of retrieved documents.

        Returns:
            PIL Image from the top-ranked document, or None if no image available.
        """
        for doc in retrieved_docs:
            if doc.image is not None:
                return doc.image
        return None

    def get_all_images(
        self,
        retrieved_docs: List[RetrievedDocument],
    ) -> List[Image.Image]:
        """
        Get all images from retrieved documents.

        Returns:
            List of PIL Images (may be shorter than retrieved_docs
            if some documents lack images).
        """
        return [doc.image for doc in retrieved_docs if doc.image is not None]
