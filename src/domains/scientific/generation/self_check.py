"""
Domain Guard and Self-Check
===========================
Validates query relevance to scientific paper domains using keyword lists
and computes blended confidence metrics.
"""

class DomainGuard:
    """Blocks out-of-scope questions and measures response alignment."""

    @staticmethod
    def is_relevant(query: str) -> tuple[bool, float, list[str]]:
        """Determines if query contains scientific research domain keywords."""
        domain_keywords = [
            "transformer", "vit", "vision", "attention", "patch",
            "image", "model", "paper", "architecture", "dataset",
            "training", "accuracy", "classification", "embedding",
            "token", "layer", "encoder", "pretrain", "finetune",
            "swin", "deit", "cnn", "resnet", "efficientformer",
            "multi-head", "self-attention", "positional encoding",
            "benchmark", "imagenet", "inference", "parameter",
            "deep learning", "neural", "weight", "gradient",
            "convolution", "pooling", "softmax", "mlp",
            "scale", "efficient", "distillation", "segmentation",
            "colapali", "scincl", "qwen", "rag", "retrieval",
            "figure", "table", "results", "experiment", "baseline",
            "performance", "memory", "latency", "throughput",
            "head", "block", "depth", "width", "resolution",
            "class", "label", "feature", "representation",
        ]

        query_lower = query.lower()
        matched = [kw for kw in domain_keywords if kw in query_lower]
        match_count = len(matched)

        if match_count == 0:
            confidence = 0.0
            relevant = False
        elif match_count == 1:
            confidence = 0.35
            relevant = True
        else:
            confidence = min(0.6 + (match_count - 2) * 0.08, 1.0)
            relevant = True

        return relevant, round(confidence, 2), matched

    @staticmethod
    def calculate_blended_confidence(guard_conf: float, retrieval_conf: float, is_from_docs: bool) -> float:
        """Fuses guard and retrieval scores to get overall confidence."""
        if is_from_docs:
            final_conf = 0.3 * guard_conf + 0.7 * retrieval_conf
        else:
            final_conf = guard_conf * 0.2
        return round(final_conf, 2)
