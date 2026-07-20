"""
Online Q&A RAG Pipeline Orchestrator
===================================
Orchestrates Domain Guard checks, hybrid retrieval, fusion, strict multimodal answer
generation with Qwen2-VL, and blended confidence scoring.
"""

from __future__ import annotations

import os
import json
import time
from src.domains.scientific.utils.helpers import clean_vram
from src.domains.scientific.models.loader import load_colpali, load_scincl, load_qwen2vl
from src.domains.scientific.retrieval.colpali_retriever import ColPaliRetriever
from src.domains.scientific.retrieval.text_retriever import TextRetriever
from src.domains.scientific.retrieval.fusion_retriever import FusionRetriever
from src.domains.scientific.generation.self_check import DomainGuard
from src.domains.scientific.generation.rag_generator import RAGGenerator

import logging
logger = logging.getLogger("mmrag.scientific.pipeline")

class CheckResult:
    """Scientific pipeline verification results.

    Verification semantics:
        passed      = answer was generated from documents (not "NOT_IN_DOCUMENTS").
        attribution = lightweight citation-presence check: whether the answer
                      text references at least one source paper by title or
                      page number. NOT full NLI-based attribution.
        faithfulness = PROXY — uses the is_from_docs flag as a documented stand-in.
                       Does NOT verify that claims are entailed by source evidence.
                       True NLI-based faithfulness would require an entailment model.
        confidence  = blended confidence score passes the threshold (>= 0.35).
    """
    def __init__(self, passed=True, attribution=True, faithfulness=True, confidence=True):
        self.passed = passed
        self.attribution_passed = attribution
        self.faithfulness_passed = faithfulness
        self.confidence_passed = confidence

class SourceCitation:
    """Source citation object to satisfy Streamlit UI contract."""
    def __init__(self, paper_title: str, paper_id: str, arxiv_url: str, page_numbers: list[int], relevance_score: float, text_snippet: str = ""):
        self.paper_title = paper_title
        self.paper_id = paper_id
        self.arxiv_url = arxiv_url
        self.page_numbers = page_numbers
        self.relevance_score = relevance_score
        self.text_snippet = text_snippet

class RAGResult:
    """Return result object to satisfy Streamlit UI contract."""
    def __init__(self, answer: str, confidence: float, sources: list[SourceCitation], check_result: CheckResult, total_time: float, retries: int = 0):
        self.answer = answer
        self.confidence = confidence
        self.sources = sources
        self.check_result = check_result
        self.total_time = total_time
        self.retries = retries

class OnlinePipeline:
    """Online query execution orchestrator."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        
        # Load indexing files
        indices_dir = self.paths.get("indices", "data/indices")
        
        # Resolve paths with RAG_BASE_DIR override
        base = os.getenv("RAG_BASE_DIR", "")
        if base:
            indices_dir = os.path.join(base, indices_dir)

        metadata_path = os.path.join(indices_dir, "page_metadata.json")
        doc_map_path = os.path.join(indices_dir, "doc_mapping.json")

        with open(metadata_path) as f:
            self.page_metadata = json.load(f)
        with open(doc_map_path) as f:
            self.doc_mapping = json.load(f)

        # Build colpali multivector paths map
        npy_dir = self.paths.get("multivectors", "data/indices/multivectors")
        if base:
            npy_dir = os.path.join(base, npy_dir)

        self.npy_index = {}
        if os.path.exists(npy_dir):
            for fname in os.listdir(npy_dir):
                if fname.endswith(".npy") and not fname.endswith(".meta.npy"):
                    page_key = fname.replace(".npy", "")
                    self.npy_index[page_key] = os.path.join(npy_dir, fname)

    def query(self, question: str, status_callback: callable = None) -> RAGResult:
        """Executes the strict hybrid RAG query pipeline sequentially to conserve VRAM."""
        t_start = time.time()
        top_k = self.cfg.get("retrieval", {}).get("top_k", 3)

        # ── Step 1: Domain Guard ──
        if status_callback:
            status_callback("colpali_encode", "Checking query relevance with Domain Guard...", 10)
        
        relevant, guard_conf, matched_kws = DomainGuard.is_relevant(question)

        _debug = os.environ.get("MMRAG_DEBUG", "") == "1"
        if _debug:
            logger.info("[DEBUG] DomainGuard: relevant=%s, conf=%.2f, kws=%s",
                        relevant, guard_conf, matched_kws)

        if not relevant:
            t_end = time.time()
            chk = CheckResult(passed=False, attribution=False, faithfulness=False, confidence=False)
            return RAGResult(
                answer="This question is not covered in the provided research papers.",
                confidence=0.0,
                sources=[],
                check_result=chk,
                total_time=round(t_end - t_start, 1)
            )

        # ── Step 2: ColPali Retrieval ──
        if status_callback:
            status_callback("colpali_encode", "Encoding query and running ColPali visual scoring...", 30)
        
        colpali_cfg = self.cfg.get("models", {}).get("colpali", {})
        colpali_model, colpali_processor = load_colpali(colpali_cfg, device="cuda")
        
        colpali_results = ColPaliRetriever.retrieve(
            question,
            self.npy_index,
            self.page_metadata,
            colpali_model,
            colpali_processor,
            top_k=10
        )
        if _debug:
            logger.info("[DEBUG] ColPali candidates: %d",
                        len(colpali_results))
            for i, r in enumerate(colpali_results[:5]):
                logger.info("  [%d] %s score=%.4f",
                            i, r.get('page_key', '?'), r.get('score', 0))
        
        # Unload ColPali
        del colpali_model, colpali_processor
        clean_vram()

        # ── Step 3: SciNCL Retrieval ──
        if status_callback:
            status_callback("scincl_encode", "Encoding query and searching ChromaDB with SciNCL...", 50)
        
        scincl_cfg = self.cfg.get("models", {}).get("scincl", {})
        scincl_model = load_scincl(scincl_cfg, device="cuda")
        scincl_embedding = scincl_model.encode(question, convert_to_numpy=True).tolist()
        
        # Unload SciNCL
        del scincl_model
        clean_vram()

        chroma_dir = self.paths.get("chroma_index", "data/indices/chroma_index")
        base = os.getenv("RAG_BASE_DIR", "")
        if base:
            chroma_dir = os.path.join(base, chroma_dir)
            
        collection_name = self.cfg.get("retrieval", {}).get("chroma_collection", "sci_rag_pages")
        
        scincl_results = TextRetriever.retrieve(
            scincl_embedding,
            chroma_dir,
            collection_name,
            self.page_metadata,
            top_k=10
        )
        if _debug:
            logger.info("[DEBUG] SciNCL candidates: %d",
                        len(scincl_results))
            for i, r in enumerate(scincl_results[:5]):
                logger.info("  [%d] %s score=%.4f",
                            i, r.get('page_key', '?'), r.get('score', 0))

        # ── Step 4: Score Fusion ──
        if status_callback:
            status_callback("fusion", "Fusing ColPali and SciNCL scores...", 70)
        
        if not colpali_results or not scincl_results:
            t_end = time.time()
            chk = CheckResult(passed=False, attribution=False, faithfulness=False, confidence=False)
            return RAGResult(
                answer="No sufficiently relevant pages found in the documents for this query.",
                confidence=0.0,
                sources=[],
                check_result=chk,
                total_time=round(t_end - t_start, 1)
            )

        fused_results = FusionRetriever.fuse(
            colpali_results,
            scincl_results,
            colpali_weight=self.cfg.get("retrieval", {}).get("colpali_weight", 0.7),
            scincl_weight=self.cfg.get("retrieval", {}).get("scincl_weight", 0.3),
            top_k=top_k
        )
        retrieval_conf = fused_results[0].get("fused_score", 0.0) if fused_results else 0.0
        if _debug:
            logger.info("[DEBUG] Fused results: %d, top conf=%.4f",
                        len(fused_results), retrieval_conf)
            for i, r in enumerate(fused_results[:3]):
                logger.info("  [%d] %s fused=%.4f colpali=%.4f scincl=%.4f",
                            i, r.get('page_key', '?'),
                            r.get('fused_score', 0),
                            r.get('colpali_norm_score', 0),
                            r.get('scincl_norm_score', 0))

        if not fused_results or retrieval_conf < 0.08:
            t_end = time.time()
            chk = CheckResult(passed=False, attribution=False, faithfulness=False, confidence=False)
            return RAGResult(
                answer="No sufficiently relevant pages found in the documents for this query.",
                confidence=0.0,
                sources=[],
                check_result=chk,
                total_time=round(t_end - t_start, 1)
            )

        # ── Step 5: Strict Qwen2-VL Generation ──
        if status_callback:
            status_callback("qwen_generate", "Generating strict answer with Qwen2-VL (on GPU)...", 85)
        
        qwen_cfg = self.cfg.get("models", {}).get("qwen2vl", {})
        qwen_model, qwen_processor = load_qwen2vl(qwen_cfg, device="cuda")
        
        gen_result = RAGGenerator.generate_strict(question, fused_results, qwen_model, qwen_processor)
        
        # Unload Qwen2-VL
        del qwen_model, qwen_processor
        clean_vram()

        if status_callback:
            status_callback("self_check", "Finalizing RAG results...", 95)

        t_end = time.time()
        total_time = t_end - t_start

        # Build citations with component scores
        sources = []
        top_colpali_score = 0.0
        top_scincl_score = 0.0
        top_fused_score = 0.0

        for i, r in enumerate(gen_result["sources"]):
            citation = SourceCitation(
                paper_title=r["paper_title"],
                paper_id=r["arxiv_id"],
                arxiv_url=r["arxiv_url"],
                page_numbers=[int(r["page_num"])],
                relevance_score=r["fused_score"],
                text_snippet=r["text_snippet"]
            )
            # Attach component scores from the fused results
            citation.colpali_norm_score = r.get("colpali_norm_score", 0.0)
            citation.scincl_norm_score = r.get("scincl_norm_score", 0.0)
            sources.append(citation)

            # Track best component scores across ALL returned sources.
            # The top-fused page may only appear in one retriever's results
            # (e.g., ColPali-only), giving it scincl_norm_score=0.0. Using
            # the max across all sources correctly reflects each retriever's
            # contribution to the fused ranking.
            if citation.colpali_norm_score > top_colpali_score:
                top_colpali_score = citation.colpali_norm_score
            if citation.scincl_norm_score > top_scincl_score:
                top_scincl_score = citation.scincl_norm_score
            if r["fused_score"] > top_fused_score:
                top_fused_score = r["fused_score"]

        # Calculate blended final confidence
        is_from_docs = gen_result["is_from_docs"]
        final_conf = DomainGuard.calculate_blended_confidence(guard_conf, retrieval_conf, is_from_docs)

        # ── Verification: attribution ──
        # Check whether the answer actually references any of the source
        # papers by name or page number. This is a lightweight citation-
        # presence check, not full NLI attribution.
        answer_lower = gen_result["answer"].lower()
        attribution_passed = False
        if is_from_docs and sources:
            for src in sources:
                title = getattr(src, "paper_title", "").lower()
                # Accept if answer mentions paper title (at least first 15 chars)
                # or any page number from citations
                if title and len(title) >= 5 and title[:15] in answer_lower:
                    attribution_passed = True
                    break
                for pn in getattr(src, "page_numbers", []):
                    if f"page {pn}" in answer_lower:
                        attribution_passed = True
                        break
                if attribution_passed:
                    break

        # ── Verification: faithfulness ──
        # PROXY ONLY — NOT actual NLI/entailment verification.
        # Uses the is_from_docs flag (absence of "NOT_IN_DOCUMENTS" marker)
        # as a documented proxy. This does NOT verify that every claim in
        # the answer is entailed by the source evidence.
        faithfulness_proxy = is_from_docs

        # ── Verification: confidence ──
        confidence_passed = is_from_docs and (final_conf >= 0.35)

        chk = CheckResult(
            passed=is_from_docs,
            attribution=attribution_passed,
            faithfulness=faithfulness_proxy,
            confidence=confidence_passed
        )

        result = RAGResult(
            answer=gen_result["answer"],
            confidence=final_conf,
            sources=sources,
            check_result=chk,
            total_time=total_time
        )
        # Attach top-ranked component scores for the adapter
        result.top_colpali_score = top_colpali_score
        result.top_scincl_score = top_scincl_score
        result.top_fused_score = top_fused_score

        return result
