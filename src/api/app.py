"""
Unified MMRAG API — FastAPI application with frozen contract.

Endpoints:
    GET  /health  — Liveness check (always 200)
    GET  /ready   — Readiness check (are pipelines loaded?)
    POST /query   — Run a query through the appropriate pipeline

The API has ZERO knowledge of healthcare or scientific internals.
It calls router.route() and maps UnifiedResponse → QueryResponse.

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

OpenAPI docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import time
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    QueryRequest,
    QueryResponse,
    SourceItemResponse,
    RetrievalMetadata,
    RetrievalScores,
    VerificationResult,
    HealthResponse,
    ReadyResponse,
)

logger = logging.getLogger("mmrag.api")


# ── FastAPI app ─────────────────────────────────────────────

app = FastAPI(
    title="MMRAG Unified API",
    description=(
        "Multimodal Retrieval-Augmented Generation for "
        "Healthcare (chest X-ray VQA) and Scientific (paper QA) domains.\n\n"
        "## Domains\n"
        "- **healthcare** — ColQwen2 dual-index retrieval + Qwen2-VL generation\n"
        "- **scientific** — ColPali + SciNCL retrieval + Qwen2-VL generation\n"
        "- **auto** — Automatic domain routing based on query content\n\n"
        "## Endpoints\n"
        "- `GET /health` — Liveness probe\n"
        "- `GET /ready` — Readiness probe (pipeline status)\n"
        "- `POST /query` — Execute a RAG query"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global router (lazy init) ──────────────────────────────

_router = None


def _get_router():
    """Lazy-initialize the DomainRouter with real pipeline adapters.

    Uses PipelineFactory to load actual RAGVQAPipeline / OnlinePipeline
    when GPU and indices are available. Falls back to placeholder mode
    (inner_pipeline=None) when resources are unavailable.
    """
    global _router
    if _router is not None:
        return _router

    from src.router.domain_router import DomainRouter
    from pipelines.healthcare.adapter import HealthcarePipeline
    from pipelines.scientific.adapter import ScientificPipeline
    from src.api.pipeline_factory import (
        create_healthcare_pipeline,
        create_scientific_pipeline,
    )

    # Attempt to load real pipelines (returns None if unavailable)
    logger.info("Initializing pipelines via PipelineFactory...")
    health_inner = create_healthcare_pipeline()
    sci_inner = create_scientific_pipeline()

    _router = DomainRouter()
    _router.register("healthcare", HealthcarePipeline(inner_pipeline=health_inner))
    _router.register("scientific", ScientificPipeline(inner_pipeline=sci_inner))

    status_h = "LIVE" if health_inner else "placeholder"
    status_s = "LIVE" if sci_inner else "placeholder"
    logger.info(
        f"DomainRouter initialized: "
        f"healthcare={status_h}, scientific={status_s}"
    )
    return _router


# ── Response mapping helpers ────────────────────────────────

CONFIDENCE_THRESHOLD = 0.5


def _map_sources(unified_response) -> list[SourceItemResponse]:
    """Map UnifiedResponse.sources → List[SourceItemResponse]."""
    sources = []
    for s in unified_response.sources:
        doc_id = s.metadata.get("doc_id", "") or s.metadata.get("paper_id", "")
        page = 0
        if s.page_numbers:
            page = s.page_numbers[0]

        sources.append(
            SourceItemResponse(
                doc_id=str(doc_id),
                page=page,
                title=s.title,
                relevance_score=round(s.score, 4),
                snippet=s.snippet,
            )
        )
    return sources


def _map_retrieval_metadata(unified_response) -> RetrievalMetadata:
    """Map UnifiedResponse.metadata → RetrievalMetadata.

    Healthcare:
        colpali = image retrieval score (ColQwen2 image)
        scincl  = text retrieval score (ColQwen2 text)
        fused   = RRF fused score
        method  = "fused"

    Scientific:
        colpali = ColPali visual score
        scincl  = SciNCL text score
        fused   = weighted fusion score
        method  = from pipeline metadata
    """
    meta = unified_response.metadata
    domain = unified_response.domain

    # Extract scores from metadata (pipelines populate these)
    colpali_score = meta.get("colpali_score", 0.0)
    scincl_score = meta.get("scincl_score", 0.0)
    fused_score = meta.get("fused_score", 0.0)

    # Healthcare: extract from top source if available
    if domain == "healthcare" and unified_response.sources:
        top = unified_response.sources[0]
        top_meta = top.metadata
        colpali_score = colpali_score or top_meta.get("image_score", top.score)
        scincl_score = scincl_score or top_meta.get("text_score", 0.0)
        fused_score = fused_score or top_meta.get("rrf_score", top.score)

    # Scientific: extract from metadata
    if domain == "scientific":
        colpali_score = colpali_score or meta.get("visual_score", 0.0)
        scincl_score = scincl_score or meta.get("text_score", 0.0)
        fused_score = fused_score or meta.get("fusion_score", 0.0)

    # Determine method
    method = meta.get("retrieval_method", "fused")
    if method not in ("fused", "colpali_only", "scincl_only"):
        method = "fused"

    return RetrievalMetadata(
        method=method,
        scores=RetrievalScores(
            colpali=round(float(colpali_score), 4),
            scincl=round(float(scincl_score), 4),
            fused=round(float(fused_score), 4),
        ),
    )


def _map_verification(unified_response) -> VerificationResult:
    """Map UnifiedResponse.metadata → VerificationResult.

    Healthcare:
        attribution     = grounding_passed (from GroundingVerifier)
        faithfulness    = confidence >= 0.5
        confidence_pass = confidence_level != "LOW"

    Scientific:
        attribution     = attribution_passed (from SelfCheck)
        faithfulness    = faithfulness_passed (from SelfCheck)
        confidence_pass = self_check_passed
    """
    meta = unified_response.metadata
    domain = unified_response.domain
    confidence = unified_response.confidence

    if domain == "healthcare":
        attribution = meta.get("grounding_passed", True)
        faithfulness = confidence >= CONFIDENCE_THRESHOLD
        conf_level = meta.get("confidence_level", "UNKNOWN")
        confidence_pass = conf_level.upper() != "LOW"
    elif domain == "scientific":
        attribution = meta.get("attribution_passed", True)
        faithfulness = meta.get("faithfulness_passed", True)
        confidence_pass = meta.get("self_check_passed", True)
    else:
        # Unknown domain — default to True
        attribution = True
        faithfulness = True
        confidence_pass = True

    return VerificationResult(
        attribution=bool(attribution),
        faithfulness=bool(faithfulness),
        confidence_pass=bool(confidence_pass),
    )


# ── Endpoints ───────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness probe",
)
async def health_check():
    """Health check — always returns 200 if the service is running."""
    return HealthResponse()


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["System"],
    summary="Readiness probe",
)
async def readiness_check():
    """Readiness check — reports whether pipelines are loaded (not just registered)."""
    try:
        router = _get_router()
        domains = list(router._pipelines.keys())

        # Check which pipelines have real inner_pipeline loaded
        live_domains = []
        for name, pipe in router._pipelines.items():
            if hasattr(pipe, 'inner') and pipe.inner is not None:
                live_domains.append(name)

        if live_domains:
            detail = f"{len(live_domains)} LIVE pipeline(s): {', '.join(live_domains)}"
        else:
            detail = (
                f"{len(domains)} domain(s) registered in placeholder mode. "
                f"GPU/indices required for live pipelines."
            )

        return ReadyResponse(
            ready=len(live_domains) > 0,
            domains=domains,
            detail=detail,
        )
    except Exception as e:
        return ReadyResponse(
            ready=False,
            domains=[],
            detail=f"Router initialization failed: {e}",
        )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Execute a RAG query",
    description=(
        "Submit a natural-language query. The system retrieves "
        "relevant evidence documents and generates an answer.\n\n"
        'Set `domain="auto"` to let the system automatically '
        "detect the appropriate pipeline based on query content."
    ),
)
async def run_query(request: QueryRequest):
    """Execute a query through the appropriate domain pipeline.

    1. Route query to the correct domain pipeline (or auto-detect)
    2. Execute the pipeline
    3. Map UnifiedResponse → QueryResponse (frozen contract)
    4. Return with timing information
    """
    router = _get_router()

    # Load image if needed
    image = None
    # (image loading from path is not part of the frozen contract,
    #  but the pipeline adapters accept Optional[Image])

    # Map domain: "auto" → None (triggers auto-detection in router)
    domain_hint = request.domain if request.domain != "auto" else None

    # Execute with timing
    start_ms = time.monotonic_ns() // 1_000_000

    try:
        result = router.route(
            query=request.query,
            domain_hint=domain_hint,
            image=image,
            top_k=request.top_k,
        )
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid domain: {e}",
        )
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {e}",
        )

    end_ms = time.monotonic_ns() // 1_000_000
    latency = int(end_ms - start_ms)

    # Map UnifiedResponse → QueryResponse (frozen contract)
    return QueryResponse(
        answer=result.answer,
        confidence=round(result.confidence, 4),
        sources=_map_sources(result),
        retrieval_metadata=_map_retrieval_metadata(result),
        verification=_map_verification(result),
        latency_ms=latency,
    )
