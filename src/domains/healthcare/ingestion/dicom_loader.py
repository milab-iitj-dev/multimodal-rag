"""
OpenI (Open-i) Chest X-ray Dataset Loader — CSV Format.

Parses the Indiana University Chest X-ray collection from CSV files:
  - indiana_reports.csv:      uid, MeSH, Problems, image, indication,
                              comparison, findings, impression
  - indiana_projections.csv:  uid, filename, projection

Matches report UIDs to image filenames via the projections table,
then produces MedicalSample instances with auto-generated VQA pairs
(findings → question, impression → answer).

Expected dataset layout:
    data/openi/
    ├── images/                          # .dcm.png chest X-ray images
    └── reports/
        ├── indiana_reports.csv
        └── indiana_projections.csv
"""

import csv
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from src.domains.healthcare.ingestion.base_loader import BaseDataset, MedicalSample
from src.shared.image_utils import load_image
from src.shared.logging_utils import setup_logger

logger = setup_logger("data.openi")


# ------------------------------------------------------------------ #
#  VQA question templates for auto-generating training pairs          #
# ------------------------------------------------------------------ #
VQA_TEMPLATES = [
    "What are the findings in this chest X-ray?",
    "Describe the abnormalities visible in this radiograph.",
    "What does this chest X-ray show?",
    "Provide a clinical interpretation of this chest radiograph.",
    "What is the radiological impression of this image?",
    "Are there any significant findings in this X-ray?",
    "Summarize the key observations in this chest X-ray.",
    "What clinical conditions can be identified from this radiograph?",
]


class OpenIDataset(BaseDataset):
    """
    OpenI Chest X-ray dataset loader (CSV format).

    Loads indiana_reports.csv and indiana_projections.csv, matches
    report UIDs to image files, and produces MedicalSample instances
    with auto-generated VQA pairs for training and evaluation.
    """

    def __init__(
        self,
        images_dir: str,
        reports_dir: str,
        max_samples: Optional[int] = None,
        load_images: bool = True,
        prefer_frontal: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            images_dir:      Path to directory containing X-ray images.
            reports_dir:     Path to directory containing CSV report files.
            max_samples:     Cap on number of samples (None = all).
            load_images:     Whether to load images when accessing samples.
            prefer_frontal:  If True, prefer frontal projection images.
            seed:            Random seed for VQA template selection.
        """
        self.images_dir = Path(images_dir)
        self.reports_dir = Path(reports_dir)
        self.max_samples = max_samples
        self.load_images_flag = load_images
        self.prefer_frontal = prefer_frontal
        self.seed = seed

        self._samples: List[Dict[str, Any]] = []
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    #  Core interface implementation                                       #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Parse CSV reports and match them with images."""
        logger.info("Loading OpenI dataset from:")
        logger.info(f"  Images:  {self.images_dir}")
        logger.info(f"  Reports: {self.reports_dir}")

        if not self.reports_dir.exists():
            raise FileNotFoundError(f"Reports directory not found: {self.reports_dir}")

        # Locate the CSV files
        reports_csv = self._find_csv("indiana_reports")
        projections_csv = self._find_csv("indiana_projections")

        if reports_csv is None:
            raise FileNotFoundError(
                f"Could not find indiana_reports.csv in {self.reports_dir}. "
                f"Files present: {[f.name for f in self.reports_dir.iterdir()]}"
            )

        # Step 1: Build UID → image filename mapping from projections
        uid_to_images = self._load_projections(projections_csv)

        # Step 2: Parse reports and match images
        self._load_reports(reports_csv, uid_to_images)

        logger.info(f"Dataset ready: {len(self._samples)} total samples")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> MedicalSample:
        raw = self._samples[idx]

        # Lazy-load image only when accessed
        image = None
        if self.load_images_flag and raw.get("image_path"):
            try:
                image = load_image(raw["image_path"])
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Cannot load image for sample {raw['sample_id']}: {e}")

        # Pick a VQA question template (deterministic per index)
        question = self._rng.choice(VQA_TEMPLATES)

        # Use impression as the answer; fall back to findings
        answer = raw.get("impression") or raw.get("findings") or ""

        return MedicalSample(
            sample_id=raw["sample_id"],
            image=image,
            image_path=raw.get("image_path", ""),
            report=raw.get("report", ""),
            findings=raw.get("findings"),
            impression=raw.get("impression"),
            question=question,
            answer=answer.strip(),
            metadata=raw.get("metadata", {}),
        )

    def get_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, List[int]]:
        """Split dataset indices into train/val/test."""
        n = len(self._samples)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }

    # ------------------------------------------------------------------ #
    #  CSV parsing                                                         #
    # ------------------------------------------------------------------ #

    def _find_csv(self, name_prefix: str) -> Optional[Path]:
        """Find a CSV file matching a name prefix in the reports directory."""
        for f in self.reports_dir.iterdir():
            if f.suffix.lower() == ".csv" and name_prefix in f.stem.lower():
                return f
        return None

    def _load_projections(
        self, projections_csv: Optional[Path]
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Parse indiana_projections.csv → {uid: [{filename, projection}, ...]}.

        Each UID can have multiple images (frontal, lateral, etc.).
        """
        uid_to_images: Dict[str, List[Dict[str, str]]] = defaultdict(list)

        if projections_csv is None or not projections_csv.exists():
            logger.warning("Projections CSV not found — will use filename matching")
            return uid_to_images

        with open(projections_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("uid", "").strip()
                filename = row.get("filename", "").strip()
                projection = row.get("projection", "").strip()
                if uid and filename:
                    uid_to_images[uid].append({
                        "filename": filename,
                        "projection": projection,
                    })

        logger.info(f"Loaded projections for {len(uid_to_images)} UIDs")
        return uid_to_images

    def _load_reports(
        self,
        reports_csv: Path,
        uid_to_images: Dict[str, List[Dict[str, str]]],
    ) -> None:
        """
        Parse indiana_reports.csv and match with images.

        CSV columns: uid, MeSH, Problems, image, indication,
                     comparison, findings, impression
        """
        parsed = 0
        skipped_no_text = 0
        skipped_no_image = 0

        with open(reports_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if self.max_samples and parsed >= self.max_samples:
                    break

                uid = row.get("uid", "").strip()
                findings = self._clean_text(row.get("findings", ""))
                impression = self._clean_text(row.get("impression", ""))

                # Skip samples with no usable text
                if not findings and not impression:
                    skipped_no_text += 1
                    continue

                # Find matching image
                image_path = self._resolve_image(uid, uid_to_images)
                if image_path is None:
                    skipped_no_image += 1
                    continue

                # Build full report text
                full_report = ""
                if findings:
                    full_report += f"FINDINGS: {findings} "
                if impression:
                    full_report += f"IMPRESSION: {impression}"

                # Metadata
                metadata = {}
                mesh = row.get("MeSH", "").strip()
                if mesh:
                    metadata["mesh_terms"] = [t.strip() for t in mesh.split(";") if t.strip()]
                problems = row.get("Problems", "").strip()
                if problems:
                    metadata["problems"] = [p.strip() for p in problems.split(";") if p.strip()]
                indication = row.get("indication", "").strip()
                if indication:
                    metadata["indication"] = indication

                self._samples.append({
                    "sample_id": uid,
                    "findings": findings,
                    "impression": impression,
                    "report": full_report.strip(),
                    "image_path": str(image_path),
                    "metadata": metadata,
                })
                parsed += 1

        logger.info(f"Loaded {parsed} samples")
        logger.info(f"Skipped: {skipped_no_text} (no text), {skipped_no_image} (no image)")

    def _resolve_image(
        self,
        uid: str,
        uid_to_images: Dict[str, List[Dict[str, str]]],
    ) -> Optional[Path]:
        """
        Resolve a UID to an image file path.

        Strategy:
          1. Use projections table (prefer frontal if available)
          2. Fall back to glob matching on UID prefix
        """
        # Strategy 1: projections table
        if uid in uid_to_images:
            images = uid_to_images[uid]

            # Prefer frontal projection
            if self.prefer_frontal:
                frontal = [
                    img for img in images
                    if img["projection"].lower() == "frontal"
                ]
                if frontal:
                    images = frontal

            # Try each candidate
            for img in images:
                candidate = self.images_dir / img["filename"]
                if candidate.exists():
                    return candidate

        # Strategy 2: glob filename matching (uid_*.png)
        matches = sorted(self.images_dir.glob(f"{uid}_*"))
        if matches:
            # Prefer the first match (typically frontal)
            return matches[0]

        return None

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Clean and validate report text."""
        if not text or not text.strip():
            return None
        text = text.strip()
        # Skip very short or placeholder text
        if len(text) < 5:
            return None
        return text

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #

    def summary(self) -> Dict[str, Any]:
        """Extended summary with dataset-specific statistics."""
        base = super().summary()

        has_findings = sum(1 for s in self._samples if s.get("findings"))
        has_impression = sum(1 for s in self._samples if s.get("impression"))
        has_image = sum(1 for s in self._samples if s.get("image_path"))

        base.update({
            "with_findings": has_findings,
            "with_impression": has_impression,
            "with_image": has_image,
        })
        return base
