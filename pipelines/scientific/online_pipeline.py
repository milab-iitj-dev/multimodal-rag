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

class CheckResult:
    """Mock verification checks to satisfy Streamlit UI contract."""
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

        # Build citations
        sources = []
        for r in gen_result["sources"]:
            citation = SourceCitation(
                paper_title=r["paper_title"],
                paper_id=r["arxiv_id"],
                arxiv_url=r["arxiv_url"],
                page_numbers=[int(r["page_num"])],
                relevance_score=r["fused_score"],
                text_snippet=r["text_snippet"]
            )
            sources.append(citation)

        # Calculate blended final confidence
        is_from_docs = gen_result["is_from_docs"]
        final_conf = DomainGuard.calculate_blended_confidence(guard_conf, retrieval_conf, is_from_docs)

        chk = CheckResult(
            passed=is_from_docs,
            attribution=is_from_docs,
            faithfulness=is_from_docs,
            confidence=is_from_docs and (final_conf >= 0.35)
        )

        return RAGResult(
            answer=gen_result["answer"],
            confidence=final_conf,
            sources=sources,
            check_result=chk,
            total_time=total_time
        )
