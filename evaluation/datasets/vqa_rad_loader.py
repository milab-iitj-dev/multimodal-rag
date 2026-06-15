"""
VQA-RAD Dataset Loader -- load VQA-RAD for generation benchmarking.

VQA-RAD is a dataset of 3,515 question-answer pairs on 315 radiology
images. Questions are either "closed" (yes/no) or "open" (free-text).

Expected layout:
    data/vqa_rad/
    +-- VQA_RAD Dataset Public.json    (official release)
    +-- images/                        (radiology images)

The JSON file contains a list of entries, each with:
    - "qid":          question ID
    - "image_name":   filename of the image
    - "question":     the question text
    - "answer":       the ground-truth answer
    - "answer_type":  "CLOSED" or "OPEN"
    - "question_type": category of the question
    - "phrase_type":  "freeform", "para", etc.

Usage:
    loader = VQARADLoader(data_dir="data/vqa_rad")
    loader.load()
    test_samples = loader.get_test_split()
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.utils.logging_utils import setup_logger

logger = setup_logger("evaluation.vqa_rad")


class VQARADLoader:
    """
    Load VQA-RAD dataset for generation benchmarking.

    Splits the dataset into train/test using the official split
    or a random 80/20 split if no official split markers exist.
    """

    def __init__(
        self,
        data_dir: str,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.seed = seed
        self._samples: List[Dict[str, Any]] = []

    def load(self) -> None:
        """Load the VQA-RAD JSON dataset."""
        # Find the JSON file
        json_path = self._find_json()
        if json_path is None:
            raise FileNotFoundError(
                f"Could not find VQA-RAD JSON in {self.data_dir}. "
                f"Expected: 'VQA_RAD Dataset Public.json' or similar."
            )

        logger.info(f"Loading VQA-RAD from {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Parse entries
        for entry in raw_data:
            qid = str(entry.get("qid", ""))
            question = str(entry.get("question", "") or "").strip()
            answer = str(entry.get("answer", "") or "").strip()
            image_name = str(entry.get("image_name", "") or "").strip()
            answer_type = str(entry.get("answer_type", "") or "").strip().upper()

            if not question or not answer:
                continue

            # Resolve image path
            image_path = None
            if image_name:
                candidate = self.data_dir / "images" / image_name
                if candidate.exists():
                    image_path = str(candidate)
                else:
                    # Try without subdirectory
                    candidate2 = self.data_dir / image_name
                    if candidate2.exists():
                        image_path = str(candidate2)

            # Map answer_type to our convention
            if answer_type == "CLOSED":
                question_type = "closed"
            elif answer_type == "OPEN":
                question_type = "open"
            else:
                # Infer from answer content
                if answer.lower() in ("yes", "no"):
                    question_type = "closed"
                else:
                    question_type = "open"

            self._samples.append({
                "qid": qid,
                "question": question,
                "answer": answer,
                "image_name": image_name,
                "image_path": image_path,
                "question_type": question_type,
                "answer_type": answer_type,
                "metadata": {
                    k: v for k, v in entry.items()
                    if k not in ("qid", "question", "answer", "image_name",
                                 "answer_type")
                },
            })

        logger.info(
            f"Loaded {len(self._samples)} QA pairs "
            f"({sum(1 for s in self._samples if s['question_type'] == 'closed')} closed, "
            f"{sum(1 for s in self._samples if s['question_type'] == 'open')} open)"
        )

    def _find_json(self) -> Optional[Path]:
        """Find the VQA-RAD JSON file."""
        candidates = [
            "VQA_RAD Dataset Public.json",
            "vqa_rad.json",
            "VQA_RAD.json",
            "dataset.json",
        ]
        for name in candidates:
            path = self.data_dir / name
            if path.exists():
                return path

        # Glob for any JSON
        jsons = list(self.data_dir.glob("*.json"))
        if jsons:
            return jsons[0]

        return None

    def get_test_split(
        self,
        test_ratio: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Get the test split.

        If the dataset has a 'phrase_type' field, uses entries with
        phrase_type != 'para' as test (following common practice).
        Otherwise falls back to a random split.
        """
        import random

        # Check if there's a phrase_type-based split
        has_phrase = any(
            s["metadata"].get("phrase_type") for s in self._samples
        )

        if has_phrase:
            # Use 'freeform' entries as test (standard VQA-RAD practice)
            test = [
                s for s in self._samples
                if str(s["metadata"].get("phrase_type", "")).lower() == "freeform"
            ]
            if test:
                logger.info(
                    f"Using phrase_type='freeform' as test split: "
                    f"{len(test)} samples"
                )
                return test

        # Fallback: random split
        rng = random.Random(self.seed)
        indices = list(range(len(self._samples)))
        rng.shuffle(indices)
        split_point = int(len(indices) * (1 - test_ratio))
        test_indices = indices[split_point:]

        test = [self._samples[i] for i in test_indices]
        logger.info(f"Using random test split: {len(test)} samples")
        return test

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """Return all loaded samples."""
        return self._samples

    def get_closed_samples(self) -> List[Dict[str, Any]]:
        """Return only closed (yes/no) samples."""
        return [s for s in self._samples if s["question_type"] == "closed"]

    def get_open_samples(self) -> List[Dict[str, Any]]:
        """Return only open-ended samples."""
        return [s for s in self._samples if s["question_type"] == "open"]
