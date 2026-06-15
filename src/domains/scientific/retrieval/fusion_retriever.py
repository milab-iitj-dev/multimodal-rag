"""
Fusion Retriever
================
Fuses scores from visual (ColPali) and textual (SciNCL) retrieval using min-max normalization.
"""

from src.shared.logging_utils import get_logger

logger = get_logger(__name__)

class FusionRetriever:
    """Combines and ranks scores from visual and text retrievals."""

    @staticmethod
    def fuse(
        colpali_results: list[dict],
        scincl_results: list[dict],
        colpali_weight: float = 0.7,
        scincl_weight: float = 0.3,
        top_k: int = 3
    ) -> list[dict]:
        """Performs min-max normalization and returns fused ranking list."""
        logger.info("Fusing visual (ColPali) and textual (SciNCL) retrieval scores...")
        if not colpali_results and not scincl_results:
            logger.warning("No results to fuse.")
            return []

        # 1. Normalize ColPali scores
        c_scores = {r["page_key"]: r["score"] for r in colpali_results}
        c_vals = list(c_scores.values())
        if c_vals:
            c_min, c_max = min(c_vals), max(c_vals)
            c_range = c_max - c_min + 1e-8
            logger.debug("ColPali score bounds: Min=%.4f, Max=%.4f (Weight=%.2f)", c_min, c_max, colpali_weight)
        else:
            c_min, c_max, c_range = 0, 0, 1e-8

        # 2. Normalize SciNCL scores
        s_scores = {r["page_key"]: r["score"] for r in scincl_results}
        s_vals = list(s_scores.values())
        if s_vals:
            s_min, s_max = min(s_vals), max(s_vals)
            s_range = s_max - s_min + 1e-8
            logger.debug("SciNCL score bounds: Min=%.4f, Max=%.4f (Weight=%.2f)", s_min, s_max, scincl_weight)
        else:
            s_min, s_max, s_range = 0, 0, 1e-8

        # 3. Fuse scores
        fused = {}
        for pk, sc in c_scores.items():
            norm_c = (sc - c_min) / c_range
            fused[pk] = colpali_weight * norm_c

        for pk, sc in s_scores.items():
            norm_s = (sc - s_min) / s_range
            if pk in fused:
                fused[pk] += scincl_weight * norm_s
            else:
                fused[pk] = scincl_weight * norm_s

        # 4. Sort and select top k
        fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 5. Build unified output objects
        all_results = {r["page_key"]: r for r in colpali_results}
        all_results.update({r["page_key"]: r for r in scincl_results})

        final = []
        for i, (page_key, fused_score) in enumerate(fused_sorted):
            r = all_results.get(page_key, {})
            # Make sure we copy to prevent altering original
            r_copy = dict(r)
            r_copy["fused_score"] = fused_score
            final.append(r_copy)
            logger.info("Fused Rank [%d]: Page %s (Fused Score: %.4f)", i+1, page_key, fused_score)

        return final
