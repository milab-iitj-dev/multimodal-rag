"""
Hybrid retriever using Reciprocal Rank Fusion (RRF).

Orchestrates dual-index ColQwen2 retrieval for image+text queries
by running both retrieval paths and fusing results using RRF.
Also applies question-aware reranking to improve question sensitivity.

Architecture:
    Image query  → ColQwen2 image index  → image_results
    Text query   → ColQwen2 text index   → text_results
                                         ↓
                               RRF Fusion (rank-based)
                                         ↓
                          Question-Aware Reranking
                                         ↓
                               Final top-K results

RRF formula:
    RRF_score(d) = Σ  weight_i / (k + rank_i(d))
                   i
where k is a constant (typically 60) and rank_i(d) is the rank of
document d in the i-th retriever's result list.

Why RRF instead of weighted score fusion:
    ColQwen2 image MaxSim scores (~600-900) and text MaxSim scores
    (~50-200) are on completely different scales due to different
    numbers of patches vs tokens. RRF uses ranks, not raw scores,
    making it scale-agnostic and robust.
"""

import re
from typing import List, Optional, Dict, Any, Tuple

from PIL import Image

from src.domains.healthcare.retrieval.base_retriever import BaseRetriever, RetrievedDocument
from src.domains.healthcare.retrieval.colqwen2_retriever import ColQwen2Retriever
from src.shared.logging_utils import setup_logger

logger = setup_logger("retrieval.hybrid")

# Common medical stopwords that should not influence reranking
# (they appear in almost every report, so matching on them is noise)
MEDICAL_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "this", "that", "these", "those", "it", "its", "of", "in",
    "on", "at", "to", "for", "with", "by", "from", "as", "or",
    "and", "but", "not", "no", "nor", "if", "then", "than",
    "there", "here", "what", "which", "who", "whom", "how",
    "when", "where", "why", "any", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "so", "very", "just", "because",
    "patient", "history", "study", "image", "exam", "finding",
    "impression", "report", "clinical", "medical", "chest",
    "radiograph", "x-ray", "xray", "ray", "pa", "ap", "lateral",
    "view", "views", "comparison", "prior", "previous", "stable",
    "unchanged", "within", "normal", "limits", "seen", "noted",
    "suggest", "suggests", "suggestive", "consistent", "appears",
    "appear", "likely", "possible", "probable", "evidence",
    "indicate", "indicates", "demonstrate", "demonstrates",
    "show", "shows", "revealed", "reveals", "present", "absent",
    "negative", "positive", "bilateral", "right", "left",
    "upper", "lower", "mild", "moderate", "severe", "small",
    "large", "acute", "chronic",
}


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that fuses ColQwen2 image and text retrieval
    using Reciprocal Rank Fusion (RRF) and question-aware reranking.

    This is the primary retriever for the Healthcare MRAG pipeline.
    It wraps ColQwen2Retriever and routes queries to the appropriate
    retrieval path(s) based on available inputs.

    Query routing:
      - Image only  → ColQwen2 image index only
      - Text only   → ColQwen2 text index only
      - Image + text → both indexes → RRF fusion → reranking

    Usage:
        retriever = HybridRetriever(colqwen2_retriever, config)
        results = retriever.retrieve(
            query="Is there pneumonia?",
            query_image=xray_image,
            top_k=3,
        )
    """

    def __init__(
        self,
        colqwen2_retriever: ColQwen2Retriever,
        config: Optional[dict] = None,
    ):
        """
        Args:
            colqwen2_retriever: Loaded ColQwen2Retriever with dual index.
            config:             Retrieval config dict with fusion settings.
        """
        self.colqwen2 = colqwen2_retriever
        self.config = config or {}

        # Tracks which retrieval path was used in the last retrieve() call.
        # Values: "hybrid", "image_only", "text_only", "none"
        self._last_retrieval_mode: str = "none"

        # Fusion settings
        fusion_cfg = (
            self.config
            .get("retrieval", {})
            .get("colqwen2", {})
            .get("fusion", {})
        )
        self.rrf_k = fusion_cfg.get("rrf_k", 60)
        self.over_retrieve_k = fusion_cfg.get("over_retrieve_k", 15)
        self.w_image = fusion_cfg.get("weights", {}).get("image", 1.0)
        self.w_text = fusion_cfg.get("weights", {}).get("text", 1.0)

        # Reranking settings
        rerank_cfg = (
            self.config
            .get("retrieval", {})
            .get("colqwen2", {})
            .get("reranking", {})
        )
        self.reranking_enabled = rerank_cfg.get("enabled", True)
        self.rerank_boost = rerank_cfg.get("boost_weight", 0.15)
        self.use_stopwords = rerank_cfg.get("medical_stopwords", True)

        logger.info(
            f"HybridRetriever initialized: "
            f"rrf_k={self.rrf_k}, "
            f"over_retrieve_k={self.over_retrieve_k}, "
            f"w_image={self.w_image}, w_text={self.w_text}, "
            f"reranking={'on' if self.reranking_enabled else 'off'}"
        )

    # ------------------------------------------------------------------ #
    #  BaseRetriever: retrieve() — main entry point                        #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        query_image: Optional[Image.Image] = None,
        top_k: int = 3,
    ) -> List[RetrievedDocument]:
        """
        Retrieve the top-k most relevant documents.

        Routes to the appropriate path based on inputs:
          - Image + text → dual retrieval + RRF + reranking
          - Image only   → ColQwen2 image retrieval
          - Text only    → ColQwen2 text retrieval

        Args:
            query:       Text query string.
            query_image: Optional query image.
            top_k:       Number of final documents to return.

        Returns:
            List of RetrievedDocument sorted by relevance.
        """
        has_image = query_image is not None
        has_text = bool(query and query.strip())
        has_text_index = self.colqwen2.has_text_index

        if has_image and has_text and has_text_index:
            # Mode 3: Image + Text → dual retrieval + fusion
            logger.info("Hybrid mode: image + text → dual retrieval + RRF")
            self._last_retrieval_mode = "hybrid"
            return self._retrieve_dual(query, query_image, top_k)

        elif has_image:
            # Mode 1: Image only → image retrieval
            logger.info("Hybrid mode: image only → image index")
            self._last_retrieval_mode = "image_only"
            return self.colqwen2.retrieve_by_image(
                query_image, query=query, top_k=top_k
            )

        elif has_text:
            # Mode 2: Text only → text retrieval
            self._last_retrieval_mode = "text_only"
            if has_text_index:
                logger.info("Hybrid mode: text only → text index")
                results = self.colqwen2.retrieve_by_text(
                    query, top_k=top_k
                )
            else:
                logger.info(
                    "Hybrid mode: text only → cross-modal (no text index)"
                )
                results = self.colqwen2.retrieve(query, top_k=top_k)

            # Apply reranking for text-only mode too
            if self.reranking_enabled and results:
                results = self._rerank_by_question(query, results)
                results = results[:top_k]

            return results

        else:
            logger.warning("No query text or image provided")
            self._last_retrieval_mode = "none"
            return []

    # ------------------------------------------------------------------ #
    #  Dual retrieval: image + text → RRF fusion → reranking               #
    # ------------------------------------------------------------------ #

    def _retrieve_dual(
        self,
        query: str,
        query_image: Image.Image,
        top_k: int,
    ) -> List[RetrievedDocument]:
        """
        Run both image and text retrieval, fuse with RRF, rerank.

        Steps:
            1. Image → image index → top over_retrieve_k results
            2. Text → text index → top over_retrieve_k results
            3. RRF fusion of both result lists
            4. Question-aware reranking
            5. Return top-K final results

        Args:
            query:       Text query.
            query_image: Query image.
            top_k:       Final number of results.

        Returns:
            Fused and reranked list of RetrievedDocument.
        """
        # Step 1: Image retrieval (over-retrieve for fusion)
        image_results = self.colqwen2.retrieve_by_image(
            query_image, query=query, top_k=self.over_retrieve_k
        )
        logger.info(
            f"  Image path: {len(image_results)} results "
            f"(top score: {image_results[0].score:.4f})"
            if image_results else "  Image path: 0 results"
        )

        # Step 2: Text retrieval (over-retrieve for fusion)
        text_results = self.colqwen2.retrieve_by_text(
            query, top_k=self.over_retrieve_k
        )
        logger.info(
            f"  Text path: {len(text_results)} results "
            f"(top score: {text_results[0].score:.4f})"
            if text_results else "  Text path: 0 results"
        )

        # Step 3: RRF fusion
        fused_results = self._rrf_fuse(image_results, text_results)
        logger.info(
            f"  RRF fusion: {len(fused_results)} unique documents"
        )

        # Step 4: Question-aware reranking
        if self.reranking_enabled:
            fused_results = self._rerank_by_question(query, fused_results)
            logger.info("  Question-aware reranking applied")

        # Step 5: Return top-K
        final = fused_results[:top_k]

        logger.info(
            f"  Final results: {len(final)} documents "
            f"(scores: {[f'{r.score:.4f}' for r in final]})"
        )
        return final

    # ------------------------------------------------------------------ #
    #  RRF Fusion                                                          #
    # ------------------------------------------------------------------ #

    def _rrf_fuse(
        self,
        image_results: List[RetrievedDocument],
        text_results: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """
        Fuse two ranked lists using Reciprocal Rank Fusion.

        RRF_score(d) = w_img / (k + rank_img(d)) + w_txt / (k + rank_txt(d))

        Documents appearing in only one list get score from that
        list only (the other term is 0).

        Args:
            image_results: Ranked results from image retrieval.
            text_results:  Ranked results from text retrieval.

        Returns:
            Fused list sorted by RRF score (highest first).
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, RetrievedDocument] = {}
        source_info: Dict[str, Dict[str, Any]] = {}

        # Score from image results
        for rank, doc in enumerate(image_results):
            rrf_score = self.w_image / (self.rrf_k + rank + 1)
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0) + rrf_score
            doc_map[doc.doc_id] = doc
            source_info[doc.doc_id] = {
                "image_rank": rank + 1,
                "image_score": doc.score,
            }

        # Score from text results
        for rank, doc in enumerate(text_results):
            rrf_score = self.w_text / (self.rrf_k + rank + 1)
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0) + rrf_score

            if doc.doc_id not in doc_map:
                doc_map[doc.doc_id] = doc

            if doc.doc_id not in source_info:
                source_info[doc.doc_id] = {}
            source_info[doc.doc_id]["text_rank"] = rank + 1
            source_info[doc.doc_id]["text_score"] = doc.score

        # Sort by fused score
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda did: rrf_scores[did],
            reverse=True,
        )

        # Build result list with updated scores and metadata
        results = []
        for fused_rank, doc_id in enumerate(sorted_ids):
            doc = doc_map[doc_id]
            doc.score = rrf_scores[doc_id]
            doc.source = "hybrid_rrf"
            doc.metadata["rrf_rank"] = fused_rank + 1
            doc.metadata["rrf_score"] = rrf_scores[doc_id]
            doc.metadata.update(source_info.get(doc_id, {}))
            results.append(doc)

        return results

    # ------------------------------------------------------------------ #
    #  Question-Aware Reranking                                            #
    # ------------------------------------------------------------------ #

    def _rerank_by_question(
        self,
        question: str,
        results: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """
        Rerank results by keyword overlap between the question and
        each document's clinical text (findings + impression).

        This makes the question matter: "Is there cardiomegaly?"
        boosts documents whose reports mention cardiomegaly-related
        terms, regardless of visual similarity.

        The reranking score is additive — it boosts the existing
        score rather than replacing it, so visually similar cases
        are not penalized, just question-relevant ones are promoted.

        Args:
            question: The user's question text.
            results:  List of RetrievedDocument to rerank.

        Returns:
            Reranked list sorted by boosted score.
        """
        q_terms = self._extract_terms(question)

        if not q_terms:
            return results

        for doc in results:
            # Get document clinical text
            doc_text = ""
            findings = doc.metadata.get("findings", "") or ""
            impression = doc.metadata.get("impression", "") or ""
            doc_text = findings + " " + impression

            d_terms = self._extract_terms(doc_text)

            if not d_terms:
                continue

            # Compute overlap: what fraction of question terms
            # appear in the document text
            overlap_terms = q_terms & d_terms
            overlap_score = len(overlap_terms) / len(q_terms)

            # Additive boost to existing score
            boost = self.rerank_boost * overlap_score
            doc.score += boost

            # Store reranking metadata for debugging
            doc.metadata["rerank_overlap"] = list(overlap_terms)
            doc.metadata["rerank_overlap_score"] = round(overlap_score, 4)
            doc.metadata["rerank_boost"] = round(boost, 4)

        # Re-sort by boosted score
        results.sort(key=lambda d: d.score, reverse=True)
        return results

    def _extract_terms(self, text: str) -> set:
        """
        Extract meaningful terms from text for keyword matching.

        Lowercases, removes punctuation, filters stopwords.

        Args:
            text: Raw text string.

        Returns:
            Set of cleaned terms.
        """
        if not text:
            return set()

        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r"[a-z0-9]+", text.lower())

        # Filter short tokens and stopwords
        stopwords = MEDICAL_STOPWORDS if self.use_stopwords else set()
        terms = {
            t for t in tokens
            if len(t) > 2 and t not in stopwords
        }

        return terms

    # ------------------------------------------------------------------ #
    #  BaseRetriever interface (delegated to ColQwen2Retriever)             #
    # ------------------------------------------------------------------ #

    def index(self, documents: List[Dict[str, Any]]) -> None:
        """Delegate to ColQwen2Retriever."""
        self.colqwen2.index(documents)

    def save_index(self, path: str) -> None:
        """Delegate to ColQwen2Retriever."""
        self.colqwen2.save_index(path)

    def load_index(self, path: str) -> None:
        """Delegate to ColQwen2Retriever."""
        self.colqwen2.load_index(path)

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    @property
    def is_index_loaded(self) -> bool:
        """Whether the underlying index is loaded."""
        return self.colqwen2.is_index_loaded

    def summary(self) -> Dict[str, Any]:
        """Summary of the hybrid retriever state."""
        return {
            "retriever": "HybridRetriever",
            "underlying": self.colqwen2.summary(),
            "fusion": {
                "method": "rrf",
                "rrf_k": self.rrf_k,
                "over_retrieve_k": self.over_retrieve_k,
                "w_image": self.w_image,
                "w_text": self.w_text,
            },
            "reranking": {
                "enabled": self.reranking_enabled,
                "boost_weight": self.rerank_boost,
            },
        }
