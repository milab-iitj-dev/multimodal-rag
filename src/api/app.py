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
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

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


# ── Global router (singleton) ──────────────────────────────

_router = None
_router_init_error = None


def _get_router():
    """Initialize the DomainRouter with real pipeline adapters.

    Uses PipelineFactory to load actual RAGVQAPipeline / OnlinePipeline
    when GPU and indices are available. Falls back to placeholder mode
    (inner_pipeline=None) when resources are unavailable.

    Called once at startup (via lifespan) or on first request (fallback).
    """
    global _router, _router_init_error
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

    # ── Diagnostic logging ──
    status_h = "LIVE" if health_inner else "placeholder"
    status_s = "LIVE" if sci_inner else "placeholder"
    logger.info(
        f"DomainRouter initialized: "
        f"healthcare={status_h}, scientific={status_s}"
    )
    logger.info(f"  Router id:     {id(_router)}")

    for name, pipe in _router._pipelines.items():
        inner = getattr(pipe, 'inner', None)
        logger.info(
            f"  Pipeline '{name}': "
            f"adapter={type(pipe).__name__} id={id(pipe)}, "
            f"inner={type(inner).__name__ if inner else 'None'} "
            f"id={id(inner) if inner else 'N/A'}"
        )

    if not health_inner and not sci_inner:
        _router_init_error = "Both pipelines in placeholder mode"
        logger.warning(
            "WARNING: No live pipelines — all queries will return placeholders"
        )

    return _router


# ── Lifespan: eager startup ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly initialize DomainRouter at startup.

    Runs heavy model loading in a thread pool so it doesn't
    block the async event loop. This ensures:
      - Models load BEFORE any requests arrive
      - /health returns 200 while models load (separate endpoint)
      - /ready accurately reflects load state
      - SLURM readiness polling works correctly
    """
    logger.info("=" * 60)
    logger.info("STARTUP: Initializing DomainRouter (eager)...")
    logger.info("=" * 60)

    t0 = time.monotonic()
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _get_router)
        elapsed = time.monotonic() - t0
        logger.info(f"STARTUP: DomainRouter ready in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error(
            f"STARTUP: DomainRouter init FAILED after {elapsed:.1f}s: {e}",
            exc_info=True,
        )

    logger.info("=" * 60)
    logger.info("STARTUP COMPLETE — Server accepting requests")
    logger.info("=" * 60)

    yield

    logger.info("SHUTDOWN: Cleaning up...")


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
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
                title=s.title or "",
                relevance_score=round(s.score, 4),
                snippet=s.snippet or "",
            )
        )
    return sources


def _map_retrieval_metadata(unified_response) -> RetrievalMetadata:
    """Map UnifiedResponse.metadata → RetrievalMetadata.

    Score semantics (same contract for both domains):
        colpali = image/visual retrieval score of the top-ranked source.
                  Healthcare: ColQwen2 image MaxSim score.
                  Scientific: normalized ColPali score (0.0–1.0).
        scincl  = text retrieval score of the top-ranked source.
                  Healthcare: ColQwen2 text MaxSim score.
                  Scientific: normalized SciNCL score (0.0–1.0).
        fused   = final fusion/RRF score of the top-ranked source.
                  Healthcare: RRF fused score.
                  Scientific: weighted fusion score (0.0–1.0).

    Method-driven score assignment:
        scincl_only  → only scincl populated; colpali=0.0, fused=0.0
        colpali_only → only colpali populated; scincl=0.0, fused=0.0
        fused        → all three populated with real scores
    """
    meta = unified_response.metadata
    domain = unified_response.domain

    # Determine retrieval method (populated by adapters)
    method = meta.get("retrieval_method", "fused")
    if method not in ("fused", "colpali_only", "scincl_only"):
        method = "fused"

    # Extract domain-specific score keys from adapter metadata
    if domain == "healthcare":
        image_score = float(meta.get("image_score", 0.0))
        text_score = float(meta.get("text_score", 0.0))
        rrf_score = float(meta.get("rrf_score", 0.0))
    elif domain == "scientific":
        image_score = float(meta.get("visual_score", 0.0))
        text_score = float(meta.get("text_score", 0.0))
        rrf_score = float(meta.get("fusion_score", 0.0))
    else:
        image_score = 0.0
        text_score = 0.0
        rrf_score = 0.0

    # Assign scores based on method — unused paths stay 0.0
    if method == "scincl_only":
        colpali_val = 0.0
        scincl_val = text_score
        fused_val = 0.0
    elif method == "colpali_only":
        colpali_val = image_score
        scincl_val = 0.0
        fused_val = 0.0
    else:
        # "fused" — all three populated
        colpali_val = image_score
        scincl_val = text_score
        fused_val = rrf_score

    return RetrievalMetadata(
        method=method,
        scores=RetrievalScores(
            colpali=round(colpali_val, 4),
            scincl=round(scincl_val, 4),
            fused=round(fused_val, 4),
        ),
    )


def _map_verification(unified_response) -> VerificationResult:
    """Map UnifiedResponse.metadata → VerificationResult.

    Healthcare:
        attribution     = grounding_passed (GroundingVerifier checks
                          answer consistency with retrieved evidence).
        faithfulness    = PROXY: confidence >= 0.5. This is NOT true
                          NLI-based entailment verification.
        confidence_pass = confidence_level != "LOW".

    Scientific:
        attribution     = citation-presence check (whether the answer
                          text references source papers by title/page).
        faithfulness    = PROXY: is_from_docs flag (absence of
                          NOT_IN_DOCUMENTS marker). Not true NLI.
        confidence_pass = blended confidence >= 0.35.

    NOTE: Both faithfulness fields are documented proxies, not actual
    evidence-entailment verification.
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

        # Diagnostic: log router identity on every /ready call
        logger.debug(f"/ready: router id={id(router)}")

        # Check which pipelines have real inner_pipeline loaded
        live_domains = []
        for name, pipe in router._pipelines.items():
            inner = getattr(pipe, 'inner', None)
            is_live = inner is not None
            logger.debug(
                f"/ready: {name} → "
                f"{type(pipe).__name__} id={id(pipe)}, "
                f"inner={'LIVE' if is_live else 'None'}"
            )
            if is_live:
                live_domains.append(name)

        if live_domains:
            # Format: "Healthcare pipeline LIVE" or
            #         "Healthcare, Scientific pipelines LIVE"
            names = ", ".join(d.capitalize() for d in live_domains)
            suffix = "pipeline" if len(live_domains) == 1 else "pipelines"
            detail = f"{names} {suffix} LIVE"
        else:
            all_domains = list(router._pipelines.keys())
            detail = (
                f"{len(all_domains)} domain(s) registered in placeholder mode. "
                f"GPU/indices required for live pipelines."
            )

        return ReadyResponse(
            ready=len(live_domains) > 0,
            domains=live_domains,
            detail=detail,
        )
    except Exception as e:
        logger.error(f"/ready failed: {e}", exc_info=True)
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

    # Diagnostic: log router identity
    logger.debug(f"/query: router id={id(router)}, query='{request.query[:80]}'")

    # Load image if path provided (for true multimodal retrieval)
    image = None
    # ── Image path diagnostic logging ──
    logger.info(
        f"/query: image_path={request.image_path!r}, "
        f"query='{request.query[:60]}', domain={request.domain}"
    )
    if request.image_path:
        import os
        from src.shared.image_utils import load_image
        resolved = request.image_path
        if not os.path.isabs(resolved):
            resolved = os.path.join(os.getcwd(), resolved)
        logger.info(f"/query: resolved image path: {resolved}")
        logger.info(f"/query: file exists: {os.path.exists(resolved)}")
        if os.path.islink(resolved) or os.path.islink(os.path.dirname(resolved)):
            real_path = os.path.realpath(resolved)
            logger.info(f"/query: real path (symlink resolved): {real_path}")
            logger.info(f"/query: real path exists: {os.path.exists(real_path)}")
        try:
            image = load_image(resolved)
            logger.info(
                f"/query: ✓ loaded image from {resolved} "
                f"(size={image.size}, mode={image.mode})"
            )
        except FileNotFoundError:
            logger.error(f"/query: ✗ FileNotFoundError: {resolved}")
            raise HTTPException(
                status_code=400,
                detail=f"Image not found: {resolved}",
            )
        except ValueError as e:
            logger.error(f"/query: ✗ ValueError loading image: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Cannot open image: {e}",
            )
    else:
        logger.info("/query: no image_path provided → text-only mode")

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
