"""
Evaluation runner for medical VQA.

Computes metrics on model predictions against ground truth answers.
Supports BLEU, ROUGE, BERTScore, and token-level F1.

Usage:
    evaluator = Evaluator(metrics=["bleu", "rouge", "f1"])
    results = evaluator.run(predictions, references)
    evaluator.save_report(results, "outputs/evaluation/report.json")
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.shared.logging_utils import setup_logger

logger = setup_logger("evaluation")


class Evaluator:
    """
    Evaluation runner for medical VQA predictions.

    Supported metrics:
        - bleu: BLEU-1,2,3,4 scores
        - rouge: ROUGE-L score
        - f1: Token-level F1 score
        - exact_match: Exact string match
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics or ["bleu", "rouge", "f1", "exact_match"]
        logger.info(f"Evaluator initialized with metrics: {self.metrics}")

    def run(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Compute metrics on predictions vs references."""
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references "
                f"({len(references)}) must have the same length"
            )

        logger.info(f"Evaluating {len(predictions)} samples...")
        start_time = time.time()
        results = {"num_samples": len(predictions), "metrics": {}}

        if "exact_match" in self.metrics:
            results["metrics"]["exact_match"] = self._exact_match(predictions, references)
        if "f1" in self.metrics:
            results["metrics"]["f1"] = self._token_f1(predictions, references)
        if "bleu" in self.metrics:
            try:
                results["metrics"]["bleu"] = self._bleu(predictions, references)
            except ImportError:
                logger.warning("BLEU requires nltk. pip install nltk")
                results["metrics"]["bleu"] = None
        if "rouge" in self.metrics:
            try:
                results["metrics"]["rouge"] = self._rouge(predictions, references)
            except ImportError:
                logger.warning("ROUGE requires rouge-score. pip install rouge-score")
                results["metrics"]["rouge"] = None

        elapsed = time.time() - start_time
        results["eval_time_sec"] = round(elapsed, 2)
        logger.info(f"Evaluation complete in {elapsed:.2f}s")
        return results

    def save_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to: {path}")

    def _exact_match(self, predictions: List[str], references: List[str]) -> float:
        matches = sum(1 for p, r in zip(predictions, references)
                      if p.strip().lower() == r.strip().lower())
        return round(matches / len(predictions), 4) if predictions else 0.0

    def _token_f1(self, predictions: List[str], references: List[str]) -> float:
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            if not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue
            common = pred_tokens & ref_tokens
            if not common:
                f1_scores.append(0.0)
                continue
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
        return round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0

    def _bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoother = SmoothingFunction().method1
        bleu_scores = {f"bleu_{i}": [] for i in range(1, 5)}
        for pred, ref in zip(predictions, references):
            ref_tokens = ref.lower().split()
            pred_tokens = pred.lower().split()
            for n in range(1, 5):
                weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
                score = sentence_bleu([ref_tokens], pred_tokens,
                                      weights=weights, smoothing_function=smoother)
                bleu_scores[f"bleu_{n}"].append(score)
        return {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in bleu_scores.items()}

    def _rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            scores.append(result["rougeL"].fmeasure)
        return {"rougeL": round(sum(scores) / len(scores), 4) if scores else 0.0}
