"""
Evaluation Metrics — BLEU, ROUGE-L, ANLS, and F1 for RAG Assessment.

Provides standard evaluation metrics for comparing generated answers
against reference answers in scientific document RAG systems.  All
metrics operate on plain text strings and return float scores in
``[0, 1]`` (except BLEU-4 which is in ``[0, 100]`` by convention
from NLTK, but is normalised to ``[0, 1]`` here for consistency).

Metrics Overview
----------------
* **BLEU-4**: N-gram precision metric (4-gram) from machine
  translation evaluation.  Sensitive to exact word overlap.
* **ROUGE-L**: Longest common subsequence based metric from
  summarisation evaluation.  Captures structural similarity.
* **ANLS**: Average Normalised Levenshtein Similarity from
  document understanding benchmarks (DocVQA).  Robust to minor
  wording differences.
* **F1**: Token-level F1 score from SQuAD-style evaluation.
  Balances precision and recall at the token level.

Example:
    >>> from src.domains.scientific.utils.metrics import compute_bleu4, compute_rouge_l, compute_f1
    >>> pred = "The model uses attention mechanisms"
    >>> ref = "The model uses self-attention"
    >>> print(f"BLEU-4: {compute_bleu4(pred, ref):.4f}")
    >>> print(f"ROUGE-L: {compute_rouge_l(pred, ref):.4f}")
    >>> print(f"F1: {compute_f1(pred, ref):.4f}")
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd

from src.shared.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize text for metric computation.

    Lowercases, strips punctuation, and collapses whitespace.

    Args:
        text: Input text string.

    Returns:
        Normalized text string.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def _tokenize(text: str) -> List[str]:
    """Tokenize text into words after normalization.

    Args:
        text: Input text string.

    Returns:
        List of word tokens.
    """
    return _normalize_text(text).split()


# ---------------------------------------------------------------------------
# compute_bleu4
# ---------------------------------------------------------------------------

def compute_bleu4(prediction: str, reference: str) -> float:
    """Compute BLEU-4 score between prediction and reference.

    Uses NLTK's sentence-level BLEU implementation with SmoothingFunction
    (method 1) to handle short sentences gracefully.  The score is
    normalised to the ``[0, 1]`` range.

    Args:
        prediction: The generated answer text.
        reference: The ground-truth reference answer.

    Returns:
        BLEU-4 score as a float in ``[0, 1]``.  Returns 0.0 if either
        input is empty or if NLTK is not available.

    Example:
        >>> score = compute_bleu4("The cat sat on the mat", "The cat sat on the mat")
        >>> print(f"BLEU-4: {score:.4f}")
        1.0000
    """
    if not prediction or not reference:
        return 0.0

    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        logger.warning(
            "nltk not available — returning 0.0 for BLEU-4.  "
            "Install with: pip install nltk"
        )
        return 0.0

    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # NLTK expects references as a list of reference token lists
    smoothing = SmoothingFunction().method1

    try:
        score = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        )
        return float(score)
    except Exception as exc:
        logger.warning("BLEU-4 computation failed: %s — returning 0.0", exc)
        return 0.0


# ---------------------------------------------------------------------------
# compute_rouge_l
# ---------------------------------------------------------------------------

def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between prediction and reference.

    Uses the ``rouge_score`` library's ROUGE-L implementation, which
    measures the longest common subsequence (LCS) between the two
    texts and computes an F1 score based on precision and recall.

    Args:
        prediction: The generated answer text.
        reference: The ground-truth reference answer.

    Returns:
        ROUGE-L F1 score as a float in ``[0, 1]``.  Returns 0.0 if
        either input is empty or if ``rouge_score`` is not available.

    Example:
        >>> score = compute_rouge_l("The cat sat", "The cat sat on the mat")
        >>> print(f"ROUGE-L: {score:.4f}")
    """
    if not prediction or not reference:
        return 0.0

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning(
            "rouge_score not available — returning 0.0 for ROUGE-L.  "
            "Install with: pip install rouge-score"
        )
        return 0.0

    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return float(scores["rougeL"].fmeasure)
    except Exception as exc:
        logger.warning("ROUGE-L computation failed: %s — returning 0.0", exc)
        return 0.0


# ---------------------------------------------------------------------------
# compute_anls
# ---------------------------------------------------------------------------

def compute_anls(prediction: str, reference: str) -> float:
    """Compute Average Normalized Levenshtein Similarity (ANLS).

    ANLS is the standard metric used in document understanding
    benchmarks (DocVQA, InfographicVQA).  It measures character-level
    similarity between the prediction and reference using the
    Levenshtein (edit) distance, normalised to ``[0, 1]``.

    The formula is::

        ANLS = 1 - (edit_distance(pred, ref) / max(len(pred), len(ref)))

    A threshold of 0.5 is applied: if ANLS < 0.5, the score is set
    to 0.0, following the DocVQA convention that very dissimilar
    answers should receive zero credit.

    Args:
        prediction: The generated answer text.
        reference: The ground-truth reference answer.

    Returns:
        ANLS score as a float in ``[0, 1]``.  Returns 0.0 if either
        input is empty.

    Example:
        >>> score = compute_anls("attention mechanism", "attention mechanisms")
        >>> print(f"ANLS: {score:.4f}")
    """
    if not prediction or not reference:
        return 0.0

    pred_norm = _normalize_text(prediction)
    ref_norm = _normalize_text(reference)

    if not pred_norm or not ref_norm:
        return 0.0

    # Compute Levenshtein distance
    distance = _levenshtein_distance(pred_norm, ref_norm)
    max_len = max(len(pred_norm), len(ref_norm))

    if max_len == 0:
        return 1.0

    anls = 1.0 - (distance / max_len)

    # Apply DocVQA threshold: scores below 0.5 are set to 0.0
    if anls < 0.5:
        anls = 0.0

    return float(anls)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings.

    Uses the classic dynamic programming algorithm with O(min(m,n))
    space optimization.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        The edit distance as an integer.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# ---------------------------------------------------------------------------
# compute_f1
# ---------------------------------------------------------------------------

def compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score between prediction and reference.

    Tokenizes both texts into word sets, then computes the F1 score
    based on the overlap of token sets.  This follows the SQuAD
    evaluation convention where F1 balances precision (fraction of
    predicted tokens that are correct) and recall (fraction of
    reference tokens that are predicted).

    Args:
        prediction: The generated answer text.
        reference: The ground-truth reference answer.

    Returns:
        Token-level F1 score as a float in ``[0, 1]``.  Returns 0.0
        if either input is empty or contains no tokens after
        normalization.

    Example:
        >>> score = compute_f1("cat sat on mat", "the cat sat on the mat")
        >>> print(f"F1: {score:.4f}")
    """
    if not prediction or not reference:
        return 0.0

    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # Count overlapping tokens (intersection of multisets)
    common = sum((pred_counter & ref_counter).values())

    if common == 0:
        return 0.0

    precision = common / sum(pred_counter.values())
    recall = common / sum(ref_counter.values())

    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


# ---------------------------------------------------------------------------
# batch_compute
# ---------------------------------------------------------------------------

def batch_compute(results: List[Dict[str, str]]) -> pd.DataFrame:
    """Compute all metrics for a list of prediction/reference pairs.

    For each dictionary in *results*, computes BLEU-4, ROUGE-L,
    ANLS, and F1, and assembles the results into a pandas DataFrame.

    Args:
        results: A list of dictionaries, each containing at least:

            * ``"prediction"`` — The generated answer text.
            * ``"reference"`` — The ground-truth reference answer.

            Additional keys are preserved in the output DataFrame.

    Returns:
        A ``pd.DataFrame`` with columns for each metric plus any
        additional keys from the input dictionaries.

    Example:
        >>> data = [
        ...     {"prediction": "attention mechanism", "reference": "attention mechanisms", "question": "Q1"},
        ...     {"prediction": "transformer model", "reference": "transformer architecture", "question": "Q2"},
        ... ]
        >>> df = batch_compute(data)
        >>> print(df[["question", "bleu4", "rouge_l", "anls", "f1"]])
    """
    if not results:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for i, item in enumerate(results):
        pred = item.get("prediction", "")
        ref = item.get("reference", "")

        bleu4 = compute_bleu4(pred, ref)
        rouge_l = compute_rouge_l(pred, ref)
        anls = compute_anls(pred, ref)
        f1 = compute_f1(pred, ref)

        row = {
            "index": i,
            "prediction": pred,
            "reference": ref,
            "bleu4": bleu4,
            "rouge_l": rouge_l,
            "anls": anls,
            "f1": f1,
        }

        # Preserve additional keys
        for key, value in item.items():
            if key not in row:
                row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    # Log summary statistics
    if not df.empty:
        logger.info(
            "Batch metrics computed — %d items.  "
            "Mean BLEU-4: %.4f, ROUGE-L: %.4f, ANLS: %.4f, F1: %.4f",
            len(df),
            df["bleu4"].mean(),
            df["rouge_l"].mean(),
            df["anls"].mean(),
            df["f1"].mean(),
        )

    return df


# ---------------------------------------------------------------------------
# compare_with_baseline
# ---------------------------------------------------------------------------

def compare_with_baseline(
    ours: Dict[str, float],
    baseline: Dict[str, float],
) -> pd.DataFrame:
    """Create a side-by-side comparison of our metrics vs. a baseline.

    Takes two dictionaries of metric scores and produces a DataFrame
    showing both values and the difference (ours - baseline) for
    each metric.

    Args:
        ours: Dictionary of our system's metric scores, e.g.::

            {"bleu4": 0.45, "rouge_l": 0.62, "anls": 0.78, "f1": 0.65}

        baseline: Dictionary of the baseline system's metric scores,
            with the same keys as *ours*.

    Returns:
        A ``pd.DataFrame`` with columns: ``"metric"``, ``"ours"``,
        ``"baseline"``, ``"difference"``, and ``"pct_improvement"``.

    Example:
        >>> ours = {"bleu4": 0.45, "rouge_l": 0.62, "anls": 0.78, "f1": 0.65}
        >>> baseline = {"bleu4": 0.30, "rouge_l": 0.50, "anls": 0.65, "f1": 0.52}
        >>> df = compare_with_baseline(ours, baseline)
        >>> print(df.to_string(index=False))
    """
    rows: List[Dict[str, Any]] = []

    all_keys = sorted(set(ours.keys()) | set(baseline.keys()))

    for key in all_keys:
        our_val = ours.get(key, 0.0)
        base_val = baseline.get(key, 0.0)
        diff = our_val - base_val

        # Percentage improvement relative to baseline
        if base_val > 0:
            pct_improvement = (diff / base_val) * 100
        else:
            pct_improvement = float("inf") if diff > 0 else 0.0

        rows.append({
            "metric": key,
            "ours": our_val,
            "baseline": base_val,
            "difference": diff,
            "pct_improvement": pct_improvement,
        })

    df = pd.DataFrame(rows)

    logger.info("Baseline comparison computed — %d metrics.", len(df))

    return df
