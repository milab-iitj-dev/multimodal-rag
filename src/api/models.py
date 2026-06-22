"""
Pydantic Models — Frozen API Contract for MMRAG Unified.

These models define the EXACT request/response schema for the
/query endpoint. The contract is FROZEN — field names, types,
and nesting must not change.

Request:
    QueryRequest with query, domain, top_k, include_images

Response:
    QueryResponse with answer, confidence, sources,
    retrieval_metadata, verification, latency_ms
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ── Request ─────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for POST /query.

    Attributes:
        query:          The user's natural-language question.
        domain:         Target domain or "auto" for automatic routing.
        top_k:          Number of evidence documents to retrieve.
        include_images: Whether to include image data in the response.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="The user's question.",
        examples=["What is cardiomegaly?"],
    )
    domain: Literal["healthcare", "scientific", "auto"] = Field(
        default="auto",
        description=(
            'Target domain. Use "auto" for automatic routing '
            "based on query content."
        ),
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of evidence documents to retrieve.",
    )
    include_images: bool = Field(
        default=True,
        description="Whether to include image data in the response.",
    )


# ── Response components ─────────────────────────────────────


class SourceItemResponse(BaseModel):
    """A single source/citation in the query response.

    Maps from UnifiedResponse.SourceItem with the frozen field names.
    """

    doc_id: str = Field(default="", description="Document identifier.")
    page: int = Field(default=0, description="Page number (0 = N/A).")
    title: str = Field(default="", description="Document or case title.")
    relevance_score: float = Field(
        default=0.0, description="Retrieval relevance score."
    )
    snippet: str = Field(
        default="", description="Text snippet from the document."
    )


class RetrievalScores(BaseModel):
    """Retrieval scores from different embedding methods.

    Healthcare mapping:
        colpali = ColQwen2 image retrieval score
        scincl  = ColQwen2 text retrieval score
        fused   = RRF fused score

    Scientific mapping:
        colpali = ColPali visual retrieval score
        scincl  = SciNCL text retrieval score
        fused   = Weighted fusion score
    """

    colpali: float = Field(
        default=0.0,
        description="Visual/image retrieval score.",
    )
    scincl: float = Field(
        default=0.0,
        description="Text retrieval score.",
    )
    fused: float = Field(
        default=0.0,
        description="Fused retrieval score.",
    )


class RetrievalMetadata(BaseModel):
    """Metadata about the retrieval method and scores."""

    method: Literal["fused", "colpali_only", "scincl_only"] = Field(
        default="fused",
        description="Retrieval method used.",
    )
    scores: RetrievalScores = Field(
        default_factory=RetrievalScores,
        description="Retrieval scores from each method.",
    )


class VerificationResult(BaseModel):
    """Verification/self-check results.

    Healthcare mapping:
        attribution     = grounding_passed
        faithfulness    = confidence >= threshold (0.5)
        confidence_pass = confidence_level != "LOW"

    Scientific mapping:
        attribution     = attribution_passed (SelfCheck)
        faithfulness    = faithfulness_passed (SelfCheck)
        confidence_pass = overall check passed
    """

    attribution: bool = Field(
        default=True,
        description="Whether the answer is attributed to evidence.",
    )
    faithfulness: bool = Field(
        default=True,
        description="Whether the answer is faithful to the evidence.",
    )
    confidence_pass: bool = Field(
        default=True,
        description="Whether the confidence score passes the threshold.",
    )


# ── Top-level response ──────────────────────────────────────


class QueryResponse(BaseModel):
    """Response body for POST /query.

    This is the FROZEN API contract. All fields must be present
    in every response regardless of domain.
    """

    answer: str = Field(
        ..., description="The generated answer text."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0–1.0).",
    )
    sources: List[SourceItemResponse] = Field(
        default_factory=list,
        description="List of evidence sources/citations.",
    )
    retrieval_metadata: RetrievalMetadata = Field(
        default_factory=RetrievalMetadata,
        description="Retrieval method and scores.",
    )
    verification: VerificationResult = Field(
        default_factory=VerificationResult,
        description="Answer verification results.",
    )
    latency_ms: int = Field(
        default=0,
        ge=0,
        description="End-to-end latency in milliseconds.",
    )


# ── Health / Ready ──────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "healthy"
    service: str = "mmrag-unified"
    version: str = "2.0.0"


class ReadyResponse(BaseModel):
    """Response for GET /ready."""

    ready: bool = True
    domains: List[str] = Field(default_factory=list)
    detail: str = ""
