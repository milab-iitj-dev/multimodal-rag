"""
Prompt Templates for the Scientific Multimodal RAG Pipeline.

Defines the system prompt, user prompt template, self-check prompt,
and citation format used throughout the VLM generation stage.  All
templates are class-level constants on :class:`PromptTemplates` for
easy access and consistent formatting.

Design Principles
-----------------
1. **Role definition**: The system prompt establishes the VLM as a
   scientific research assistant, setting expectations for accuracy,
   citation, and honesty.
2. **Citation enforcement**: Every factual claim must be attributed
   to a specific source page, enabling users to verify answers.
3. **No hallucination**: Explicit instructions prohibit fabrication
   of information not present in the provided context.
4. **Self-check**: A post-generation verification prompt that asks
   the model to assess attribution, faithfulness, and confidence.

Example:
    >>> from src.domains.scientific.context.prompt_templates import PromptTemplates
    >>> user_msg = PromptTemplates.format_user_prompt(
    ...     query="What is the attention mechanism?",
    ...     text_context="The attention mechanism computes…",
    ... )
    >>> citation = PromptTemplates.format_citation(
    ...     paper_title="Attention Is All You Need",
    ...     page_num=3,
    ... )
"""

from __future__ import annotations

from src.shared.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PromptTemplates
# ---------------------------------------------------------------------------

class PromptTemplates:
    """Collection of prompt templates for VLM generation.

    All templates are class-level constants.  Instance methods are
    provided for convenience formatting of the user prompt and
    citation strings.

    Attributes:
        SYSTEM_PROMPT: The system-level instruction that defines the
            VLM's role and behavioural rules.
        USER_PROMPT_TEMPLATE: Template for the user-facing prompt
            with ``{query}`` and ``{text_context}`` placeholders.
        SELF_CHECK_PROMPT: Post-generation verification prompt for
            attribution, faithfulness, and confidence assessment.
        CITATION_FORMAT: Format string for inline citation markers,
            with ``{paper_title}`` and ``{page_num}`` placeholders.
    """

    SYSTEM_PROMPT: str = (
        "You are a scientific research assistant specializing in "
        "analyzing and explaining academic papers. Your role is to "
        "provide accurate, well-sourced answers based strictly on the "
        "provided context.\n\n"
        "Rules:\n"
        "1. **Cite your sources**: Every factual claim must include an "
        "inline citation in the format [Source: Paper Title, Page N]. "
        "Do not make claims without attribution.\n"
        "2. **No hallucination**: Do not fabricate information that is "
        "not present in the provided context (page images or text). If "
        "the context does not contain enough information to answer the "
        "question, say so explicitly.\n"
        "3. **Be precise**: When referencing figures, tables, or "
        "equations, include the specific label (e.g., 'Figure 3', "
        "'Table 2', 'Equation 5').\n"
        "4. **Be structured**: Organize your answer with clear "
        "headings or bullet points when appropriate. Use markdown "
        "formatting.\n"
        "5. **Acknowledge uncertainty**: If you are not confident about "
        "an answer, state your confidence level and explain why.\n"
        "6. **Multi-modal reasoning**: You may receive both page images "
        "and text. Use both sources of information. If the text "
        "contradicts the image, prioritize the image as it is the "
        "primary source.\n"
        "7. **Language**: Respond in the same language as the user's "
        "query. Default to English if ambiguous."
    )

    USER_PROMPT_TEMPLATE: str = (
        "## Query\n{query}\n\n"
        "## Retrieved Context\n{text_context}\n\n"
        "---\n\n"
        "Based on the retrieved context above (both the page images "
        "and the text excerpts), provide a comprehensive answer to "
        "the query. Remember to cite your sources using the format "
        "[Source: Paper Title, Page N] for every factual claim."
    )

    SELF_CHECK_PROMPT: str = (
        "Before finalizing your answer, perform a self-check:\n\n"
        "**Attribution Check**: Is every factual claim in your answer "
        "attributed to a specific source page?  Verify that each "
        "citation [Source: Paper Title, Page N] corresponds to content "
        "that actually appears on that page.\n\n"
        "**Faithfulness Check**: Does your answer faithfully reflect "
        "the information in the provided context, or does it introduce "
        "external knowledge or speculation?  Remove any claims not "
        "directly supported by the context.\n\n"
        "**Confidence Check**: Rate your overall confidence in the "
        "answer on a scale of 1 (low) to 5 (high).  If confidence is "
        "below 3, identify the specific gaps and state them explicitly.\n\n"
        "Revise your answer if needed based on this self-check."
    )

    CITATION_FORMAT: str = "[Source: {paper_title}, Page {page_num}]"

    # -----------------------------------------------------------------
    # format_user_prompt
    # -----------------------------------------------------------------

    @classmethod
    def format_user_prompt(cls, query: str, text_context: str) -> str:
        """Format the user prompt by injecting the query and text context.

        Replaces the ``{query}`` and ``{text_context}`` placeholders
        in :attr:`USER_PROMPT_TEMPLATE` with the provided values.

        Args:
            query: The user's question or search query.
            text_context: The concatenated text from retrieved
                documents, with citation markers.

        Returns:
            The formatted user prompt string.

        Example:
            >>> prompt = PromptTemplates.format_user_prompt(
            ...     query="What is the attention mechanism?",
            ...     text_context="[Source: Attention Is All You Need, Page 3]\\n…",
            ... )
        """
        formatted = cls.USER_PROMPT_TEMPLATE.format(
            query=query,
            text_context=text_context if text_context else "(No text context available.)",
        )

        logger.debug(
            "Formatted user prompt — length: %d chars", len(formatted)
        )

        return formatted

    # -----------------------------------------------------------------
    # format_citation
    # -----------------------------------------------------------------

    @classmethod
    def format_citation(cls, paper_title: str, page_num: int) -> str:
        """Format an inline citation string.

        Uses :attr:`CITATION_FORMAT` to produce a citation marker
        like ``[Source: Attention Is All You Need, Page 3]``.

        Args:
            paper_title: Title of the source paper.
            page_num: Page number within the paper (1-indexed).

        Returns:
            The formatted citation string.

        Example:
            >>> PromptTemplates.format_citation("Attention Is All You Need", 3)
            '[Source: Attention Is All You Need, Page 3]'
        """
        return cls.CITATION_FORMAT.format(
            paper_title=paper_title,
            page_num=page_num,
        )
