"""
Unified MMRAG API — FastAPI endpoint for both domains.

Provides a single REST API that routes queries to either the
healthcare or scientific pipeline based on the 'domain' parameter.

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /query     — Run a query through the appropriate pipeline
    GET  /health    — Health check
    GET  /domains   — List available domains
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="MMRAG Unified API",
    description=(
        "Multimodal Retrieval-Augmented Generation for "
        "Healthcare (chest X-ray VQA) and Scientific (paper QA) domains."
    ),
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    query: str
    domain: Optional[str] = None  # "healthcare" or "scientific"
    image_path: Optional[str] = None
    top_k: int = 3


class QueryResponse(BaseModel):
    """Response body from the /query endpoint."""
    domain: str
    answer: str
    confidence: Optional[str] = None
    num_retrieved: int = 0
    metadata: dict = {}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mmrag-unified"}


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

    The domain is determined by:
      1. Explicit 'domain' parameter (if provided)
      2. Auto-detection from query keywords
      3. Config default (healthcare)
    """
    from src.router.domain_router import DomainRouter

    router = DomainRouter()
    domain = router.detect_domain(
        query=request.query,
        domain_hint=request.domain,
    )

    if domain == "healthcare":
        return await _run_healthcare(request, domain)
    elif domain == "scientific":
        return await _run_scientific(request, domain)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain: {domain}",
        )


async def _run_healthcare(request: QueryRequest, domain: str) -> QueryResponse:
    """
    Placeholder for healthcare pipeline integration.

    In production, this would:
      1. Load the RAGVQAPipeline
      2. Run request.query through it
      3. Return the grounded answer
    """
    return QueryResponse(
        domain=domain,
        answer=(
            f"[Healthcare] Pipeline ready for query: '{request.query}'. "
            f"Initialize RAGVQAPipeline with configs/healthcare/ to enable."
        ),
        confidence="N/A",
        num_retrieved=0,
        metadata={"pipeline": "healthcare", "status": "placeholder"},
    )


async def _run_scientific(request: QueryRequest, domain: str) -> QueryResponse:
    """
    Placeholder for scientific pipeline integration.

    In production, this would:
      1. Load the OnlinePipeline
      2. Run request.query through it
      3. Return the self-checked answer
    """
    return QueryResponse(
        domain=domain,
        answer=(
            f"[Scientific] Pipeline ready for query: '{request.query}'. "
            f"Initialize OnlinePipeline with configs/scientific/ to enable."
        ),
        confidence="N/A",
        num_retrieved=0,
        metadata={"pipeline": "scientific", "status": "placeholder"},
    )
