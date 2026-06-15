"""
Generation Metrics -- EM, F1, BLEU, ROUGE-L, BERTScore.

Evaluates generated answer quality against ground-truth answers.
Supports both closed (yes/no) and open-ended question types.

Dependencies (install on HPC before running):
    pip install rouge-score nltk
    pip install bert-score    (optional, for BERTScore)
"""

import re
import string
from collections import Counter
from typing import List, Dict, Any, Optional


# ------------------------------------------------------------------ #
#  Text normalization                                                  #
# ------------------------------------------------------------------ #

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    return _normalize(text).split()


# ------------------------------------------------------------------ #
#  Individual metrics                                                  #
# ------------------------------------------------------------------ #

def exact_match(prediction: str, reference: str) -> float:
    """Exact match after normalization. Returns 1.0 or 0.0."""
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bleu(prediction: str, reference: str) -> float:
    """
    BLEU-1 score (unigram precision with brevity penalty).

    Uses a simple implementation to avoid external dependencies.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens:
        return 0.0
    if not ref_tokens:
        return 0.0

    # Unigram precision
    ref_counts = Counter(ref_tokens)
    clipped = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            clipped += 1
            ref_counts[token] -= 1

    precision = clipped / len(pred_tokens)

    # Brevity penalty
    bp = min(1.0, len(pred_tokens) / len(ref_tokens))

    return round(bp * precision, 4)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    ROUGE-L F1 score (longest common subsequence).
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    # LCS length via dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    return round(2 * precision * recall / (precision + recall), 4)


def compute_bertscore(
    predictions: List[str],
    references: List[str],
) -> Optional[Dict[str, float]]:
    """
    BERTScore (batch computation). Returns mean P/R/F1.

    Requires: pip install bert-score
    Returns None if bert-score is not installed.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return None

    P, R, F1 = bert_score_fn(
        predictions, references,
        lang="en",
        verbose=False,
        rescale_with_baseline=True,
    )

    return {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
    }


# ------------------------------------------------------------------ #
#  Aggregate computation                                               #
# ------------------------------------------------------------------ #

def compute_generation_metrics(
    results: List[Dict[str, Any]],
    compute_bert: bool = True,
) -> Dict[str, Any]:
    """
    Compute aggregate generation metrics.

    Args:
        results: List of dicts, each with:
            - "prediction": str
            - "reference": str
            - "question_type": str ("closed" or "open")
        compute_bert: Whether to compute BERTScore (slow).

    Returns:
        Aggregate and per-type metrics.
    """
    if not results:
        return {"aggregate": {}, "per_type": {}}

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    def _compute_for_group(group):
        ems = [exact_match(r["prediction"], r["reference"]) for r in group]
        f1s = [token_f1(r["prediction"], r["reference"]) for r in group]
        bleus = [compute_bleu(r["prediction"], r["reference"]) for r in group]
        rouges = [compute_rouge_l(r["prediction"], r["reference"]) for r in group]

        metrics = {
            "num_samples": len(group),
            "exact_match": _mean(ems),
            "f1": _mean(f1s),
            "bleu_1": _mean(bleus),
            "rouge_l": _mean(rouges),
        }

        return metrics

    # Aggregate
    aggregate = _compute_for_group(results)

    # BERTScore (batch, aggregate only)
    if compute_bert:
        preds = [r["prediction"] for r in results]
        refs = [r["reference"] for r in results]
        bert = compute_bertscore(preds, refs)
        if bert:
            aggregate["bertscore_f1"] = bert["f1"]
            aggregate["bertscore_precision"] = bert["precision"]
            aggregate["bertscore_recall"] = bert["recall"]
        else:
            aggregate["bertscore_f1"] = None
            aggregate["bertscore_note"] = "bert-score not installed"

    # Per question type
    type_groups: Dict[str, List] = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        type_groups.setdefault(qt, []).append(r)

    per_type = {}
    for qt, group in type_groups.items():
        per_type[qt] = _compute_for_group(group)

        # BERTScore per type
        if compute_bert:
            preds = [r["prediction"] for r in group]
            refs = [r["reference"] for r in group]
            bert = compute_bertscore(preds, refs)
            if bert:
                per_type[qt]["bertscore_f1"] = bert["f1"]

    return {
        "aggregate": aggregate,
        "per_type": per_type,
    }
