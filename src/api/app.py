"""
Unified MMRAG API — FastAPI endpoint for all domains.

Uses the domain-agnostic DomainRouter. The API has ZERO knowledge
of healthcare or scientific internals. It calls router.route()
and serializes the UnifiedResponse.

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /query     — Run a query through the appropriate pipeline
    GET  /health    — Health check
    GET  /domains   — List available domains
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="MMRAG Unified API",
    description=(
        "Multimodal Retrieval-Augmented Generation for "
        "Healthcare (chest X-ray VQA) and Scientific (paper QA) domains."
    ),
    version="2.0.0",
)


# ── Pydantic models (API contract) ──────────────────────────────

class SourceItemModel(BaseModel):
    """A single source/citation in the API response."""
    title: str = ""
    score: float = 0.0
    snippet: str = ""
    url: str = ""
    page_numbers: List[int] = []
    metadata: Dict[str, Any] = {}


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    query: str
    domain: Optional[str] = None   # "healthcare", "scientific", or None (auto)
    image_path: Optional[str] = None
    top_k: int = 3


class QueryResponse(BaseModel):
    """
    Unified response from ANY domain pipeline.

    This matches UnifiedResponse exactly.
    """
    domain: str
    answer: str
    confidence: float = 0.0
    sources: List[SourceItemModel] = []
    metadata: Dict[str, Any] = {}


# ── Global router (initialized at startup) ──────────────────────

_router = None


def _get_router():
    """Lazy-init the domain router with placeholder pipelines."""
    global _router
    if _router is not None:
        return _router

    from src.router.domain_router import DomainRouter
    from pipelines.healthcare.adapter import HealthcarePipeline
    from pipelines.scientific.adapter import ScientificPipeline

    _router = DomainRouter()
    _router.register("healthcare", HealthcarePipeline(inner_pipeline=None))
    _router.register("scientific", ScientificPipeline(inner_pipeline=None))
    return _router


# ── Endpoints ───────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mmrag-unified", "version": "2.0.0"}


@app.get("/domains")
async def list_domains():
    """List available domains and their status."""
    return {
        "domains": [
            {
                "name": "healthcare",
                "description": "Chest X-ray VQA with ColQwen2 + grounding verification",
                "status": "available",
            },
            {
                "name": "scientific",
                "description": "Scientific paper QA with ColPali + SciNCL + self-check",
                "status": "available",
            },
        ]
    }


@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """
    Run a query through the appropriate domain pipeline.

    Domain detection:
      1. Explicit 'domain' parameter (if provided)
      2. Auto-detection from query keywords
      3. Config default (healthcare)

    Returns:
      QueryResponse (identical schema for ALL domains).
    """
    router = _get_router()

    # Load image if path provided
    image = None
    if request.image_path:
        try:
            from src.shared.image_utils import load_image
            image = load_image(request.image_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load image: {e}",
            )

    try:
        result = router.route(
            query=request.query,
            domain_hint=request.domain,
            image=image,
            top_k=request.top_k,
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Convert UnifiedResponse → QueryResponse (Pydantic)
    return QueryResponse(
        domain=result.domain,
        answer=result.answer,
        confidence=result.confidence,
        sources=[
            SourceItemModel(
                title=s.title,
                score=s.score,
                snippet=s.snippet,
                url=s.url,
                page_numbers=s.page_numbers,
                metadata=s.metadata,
            )
            for s in result.sources
        ],
        metadata=result.metadata,
    )
