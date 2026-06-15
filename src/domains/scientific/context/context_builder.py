"""
Context Builder — Assembles VLM Input from Retrieved Documents.

Transforms a list of :class:`~src.retrieval.base_retriever.RetrievedDocument`
objects into a structured :class:`ContextObject` that can be passed
directly to the VLM for answer generation.  The context builder is
responsible for:

1. **Assembling the system prompt** — Defines the VLM's role and rules.
2. **Formatting the user prompt** — Injects the query and text context.
3. **Collecting page images** — Gathers all relevant page images for
   multi-modal input.
4. **Building text context** — Concatenates extracted text with
   citation markers.
5. **Collecting citations** — Aggregates source metadata for
   attribution.
6. **Token budget management** — Truncates text context if the total
   token count exceeds ``max_tokens``.

Example:
    >>> from src.domains.scientific.context.context_builder import ContextBuilder
    >>> from src.domains.scientific.context.prompt_templates import PromptTemplates
    >>> builder = ContextBuilder(max_tokens=4000)
    >>> ctx = builder.build(
    ...     query="What is the attention mechanism?",
    ...     retrieved_docs=results,
    ...     prompt_templates=PromptTemplates,
    ... )
    >>> print(ctx.system_prompt)
    >>> print(ctx.token_count)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type

import PIL.Image

from src.domains.scientific.retrieval.base_retriever import RetrievedDocument, SourceCitation
from src.shared.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ContextObject dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContextObject:
    """Structured context ready for VLM consumption.

    This is the final output of the context-building pipeline.  It
    bundles everything the VLM needs to generate a grounded,
    well-cited answer:

    Attributes:
        system_prompt: The system-level instruction that sets the
            VLM's role, behaviour rules, and citation requirements.
        user_prompt: The user-facing prompt that contains the query
            and any injected text context.
        page_images: A list of ``PIL.Image.Image`` objects representing
            the most relevant document pages.  These are passed as
            multi-modal input to the VLM.
        text_context: Concatenated text from retrieved documents,
            annotated with citation markers (e.g.
            ``[Source: Paper Title, Page 3]``).
        citations: A deduplicated list of :class:`SourceCitation`
            objects for all referenced sources.
        token_count: Estimated total token count (text + images)
            for the entire context, used for budget enforcement.
    """

    system_prompt: str
    user_prompt: str
    page_images: List[PIL.Image.Image]
    text_context: str
    citations: List[SourceCitation]
    token_count: int


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """Assembles VLM input from retrieved documents.

    Takes a user query and a list of retrieved documents, then builds
    a :class:`ContextObject` containing the system prompt, user prompt
    (with injected text context), page images, citations, and an
    estimated token count.

    If the total token count exceeds ``max_tokens``, the text context
    is truncated to fit within the budget while preserving the system
    and user prompt structure.

    Args:
        max_tokens: Maximum estimated token count for the assembled
            context.  Defaults to 4000 (suitable for most VLMs with
            8k context windows, leaving room for the generated answer).

    Example:
        >>> builder = ContextBuilder(max_tokens=4000)
        >>> ctx = builder.build(
        ...     query="Explain the transformer architecture.",
        ...     retrieved_docs=retrieved_docs,
        ...     prompt_templates=PromptTemplates,
        ... )
    """

    # Rough token estimation constants.
    # Average English word → ~1.3 tokens (GPT-style tokenisers).
    _CHARS_PER_TOKEN: float = 4.0  # ~4 chars per token on average.
    _TOKENS_PER_IMAGE: int = 1000  # Conservative estimate per image.

    def __init__(self, max_tokens: int = 4000) -> None:
        self.max_tokens = max_tokens

        logger.info(
            "ContextBuilder initialised — max_tokens=%d",
            self.max_tokens,
        )

    # -----------------------------------------------------------------
    # build
    # -----------------------------------------------------------------

    def build(
        self,
        query: str,
        retrieved_docs: List[RetrievedDocument],
        prompt_templates: Type,
    ) -> ContextObject:
        """Assemble a ContextObject from a query and retrieved documents.

        The build process proceeds in the following order:

        1. Extract the system prompt from *prompt_templates*.
        2. Collect page images from retrieved documents.
        3. Build text context by concatenating document text with
           citation markers.
        4. Collect and deduplicate citations.
        5. Estimate total token count (text + images).
        6. If over budget, truncate text context to fit.
        7. Format the user prompt with query and text context.
        8. Return the assembled :class:`ContextObject`.

        Args:
            query: The user's question or search query.
            retrieved_docs: A list of :class:`RetrievedDocument`
                objects from the retrieval pipeline.
            prompt_templates: A class (typically
                :class:`~src.context.prompt_templates.PromptTemplates`)
                that provides ``SYSTEM_PROMPT``,
                ``USER_PROMPT_TEMPLATE``, ``format_user_prompt()``,
                and ``format_citation()``.

        Returns:
            A fully assembled :class:`ContextObject` ready for VLM
            consumption.

        Raises:
            ValueError: If *query* is empty or *retrieved_docs* is
                empty.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if not retrieved_docs:
            raise ValueError(
                "retrieved_docs must contain at least one document."
            )

        logger.info(
            "Building context — query: '%s…' (%d chars), %d docs",
            query[:50],
            len(query),
            len(retrieved_docs),
        )

        # Step 1: System prompt.
        system_prompt = prompt_templates.SYSTEM_PROMPT
        logger.debug("System prompt length: %d chars", len(system_prompt))

        # Step 2: Collect page images.
        page_images: List[PIL.Image.Image] = []
        for doc in retrieved_docs:
            if doc.image is not None:
                page_images.append(doc.image)
        logger.info("Collected %d page images.", len(page_images))

        # Step 3: Build text context with citations.
        text_parts: List[str] = []
        citations: List[SourceCitation] = []
        seen_citation_keys: set = set()

        for doc in retrieved_docs:
            # Add citation marker.
            citation_marker = prompt_templates.format_citation(
                doc.source_citation.paper_title,
                doc.page_num,
            )

            if doc.text and doc.text.strip():
                text_block = f"{citation_marker}\n{doc.text.strip()}"
                text_parts.append(text_block)

            # Collect citation (deduplicated by paper_id + page_num).
            citation_key = (
                doc.source_citation.paper_id,
                doc.page_num,
            )
            if citation_key not in seen_citation_keys:
                seen_citation_keys.add(citation_key)
                citations.append(doc.source_citation)

        text_context = "\n\n---\n\n".join(text_parts) if text_parts else ""
        logger.info(
            "Built text context — %d chars, %d unique citations.",
            len(text_context),
            len(citations),
        )

        # Step 4: Estimate token count.
        # Reserve tokens for the user prompt template (query placeholder
        # + text_context placeholder + template overhead).
        template_overhead = 50  # Rough estimate for template formatting.
        system_tokens = self._estimate_text_tokens(system_prompt)
        image_tokens = len(page_images) * self._TOKENS_PER_IMAGE
        text_tokens = self._estimate_text_tokens(text_context)
        query_tokens = self._estimate_text_tokens(query)

        total_tokens = (
            system_tokens
            + query_tokens
            + text_tokens
            + image_tokens
            + template_overhead
        )
        logger.info(
            "Token estimate — system: %d, query: %d, text: %d, "
            "images: %d (%d images × %d), overhead: %d, total: %d",
            system_tokens,
            query_tokens,
            text_tokens,
            image_tokens,
            len(page_images),
            self._TOKENS_PER_IMAGE,
            template_overhead,
            total_tokens,
        )

        # Step 5: Truncate text context if over budget.
        budget_for_text = self.max_tokens - system_tokens - query_tokens - image_tokens - template_overhead
        if budget_for_text < 0:
            logger.warning(
                "Image and system tokens already exceed max_tokens (%d).  "
                "Reducing image count to fit budget.",
                self.max_tokens,
            )
            # Trim images to fit.
            max_images = max(
                0,
                (self.max_tokens - system_tokens - query_tokens - template_overhead)
                // self._TOKENS_PER_IMAGE,
            )
            if max_images < len(page_images):
                logger.warning(
                    "Trimming page images from %d to %d to fit token budget.",
                    len(page_images),
                    max_images,
                )
                page_images = page_images[:max_images]
                image_tokens = len(page_images) * self._TOKENS_PER_IMAGE
                budget_for_text = (
                    self.max_tokens
                    - system_tokens
                    - query_tokens
                    - image_tokens
                    - template_overhead
                )

        if budget_for_text > 0 and text_tokens > budget_for_text:
            logger.info(
                "Text context (%d tokens) exceeds budget (%d tokens).  "
                "Truncating…",
                text_tokens,
                budget_for_text,
            )
            text_context = self.truncate_text(text_context, budget_for_text)
            text_tokens = self._estimate_text_tokens(text_context)
            total_tokens = (
                system_tokens
                + query_tokens
                + text_tokens
                + image_tokens
                + template_overhead
            )

        # Step 6: Format user prompt.
        user_prompt = prompt_templates.format_user_prompt(query, text_context)

        logger.info(
            "Context built — total estimated tokens: %d (max: %d), "
            "images: %d, citations: %d",
            total_tokens,
            self.max_tokens,
            len(page_images),
            len(citations),
        )

        return ContextObject(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            page_images=page_images,
            text_context=text_context,
            citations=citations,
            token_count=total_tokens,
        )

    # -----------------------------------------------------------------
    # estimate_tokens
    # -----------------------------------------------------------------

    def estimate_tokens(
        self,
        text: str,
        images: Optional[List[PIL.Image.Image]] = None,
    ) -> int:
        """Estimate the total token count for text and images.

        Uses a character-based heuristic for text (approximately 4
        characters per token for GPT-style tokenisers) and a fixed
        estimate per image (1000 tokens, conservative for high-res
        page images).

        Note:
            This is a rough estimate.  Actual token counts depend on
            the specific tokeniser used by the VLM.  For precise
            counting, use the model's tokeniser directly.

        Args:
            text: The text content to estimate tokens for.
            images: Optional list of images.  Each image is estimated
                at 1000 tokens.

        Returns:
            Estimated total token count as an integer.
        """
        text_tokens = self._estimate_text_tokens(text)
        image_tokens = (
            len(images) * self._TOKENS_PER_IMAGE
            if images
            else 0
        )
        total = text_tokens + image_tokens
        logger.debug(
            "Token estimate — text: %d, images: %d, total: %d",
            text_tokens,
            image_tokens,
            total,
        )
        return total

    # -----------------------------------------------------------------
    # truncate_text
    # -----------------------------------------------------------------

    @staticmethod
    def truncate_text(text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token budget.

        Uses a character-based approximation (4 characters ≈ 1 token)
        to determine the maximum number of characters that fit within
        *max_tokens*.  Truncation is performed at the last complete
        sentence or paragraph boundary before the limit, with an
        ellipsis appended to indicate truncation.

        Args:
            text: The text to potentially truncate.
            max_tokens: Maximum number of tokens the text should
                occupy.

        Returns:
            The original text if it fits within the budget, or a
            truncated version ending with ``"… [truncated]"``.
        """
        if not text:
            return ""

        max_chars = int(max_tokens * ContextBuilder._CHARS_PER_TOKEN)

        if len(text) <= max_chars:
            return text

        # Truncate at character limit, then back up to the last
        # sentence boundary (period, exclamation, or question mark
        # followed by whitespace or end-of-string).
        truncated = text[:max_chars]

        # Try to find a sentence boundary within the last 20% of
        # the truncated text.
        search_start = max(0, int(len(truncated) * 0.8))
        boundary_chars = {".", "!", "?"}

        best_pos = -1
        for i in range(len(truncated) - 1, search_start - 1, -1):
            if truncated[i] in boundary_chars:
                # Check if followed by whitespace or end.
                if i + 1 >= len(truncated) or truncated[i + 1].isspace():
                    best_pos = i + 1
                    break

        if best_pos > 0:
            truncated = truncated[:best_pos]
        # else: no sentence boundary found, hard-truncate.

        result = truncated.rstrip() + " … [truncated]"

        logger.debug(
            "Truncated text from %d to %d chars (~%d tokens).",
            len(text),
            len(result),
            max_tokens,
        )

        return result

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        """Estimate token count for a text string.

        Uses the heuristic of approximately 4 characters per token,
        which is a reasonable approximation for English text processed
        by GPT-style byte-pair encoding tokenisers.

        Args:
            text: The text to estimate.

        Returns:
            Estimated token count as an integer.
        """
        if not text:
            return 0
        return max(1, int(len(text) / ContextBuilder._CHARS_PER_TOKEN))
