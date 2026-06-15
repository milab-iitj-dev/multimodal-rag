"""
Abstract base class for all medical datasets.

Every dataset (OpenI, MIMIC-CXR, PathVQA, etc.) implements this interface.
This guarantees that pipelines, training scripts, and evaluation scripts
work with ANY dataset without code changes — only config changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Iterator, List, Dict, Any

from PIL import Image


@dataclass
class MedicalSample:
    """
    A single medical image-report-QA sample.

    This is the universal data format that flows between all modules.
    Every dataset loader must produce MedicalSample instances.
    """
    sample_id: str                                # unique identifier
    image: Optional[Image.Image] = None           # loaded PIL image
    image_path: str = ""                          # path to image file
    report: Optional[str] = None                  # full radiology report text
    findings: Optional[str] = None                # findings section
    impression: Optional[str] = None              # impression / conclusion section
    question: Optional[str] = None                # VQA question (generated or manual)
    answer: Optional[str] = None                  # VQA answer (generated or manual)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDataset(ABC):
    """
    Abstract base for all medical datasets.

    Subclasses must implement:
        - load()         → parse raw files into internal storage
        - __len__()      → number of samples
        - __getitem__()  → single sample by index
        - get_splits()   → train/val/test split
    """

    @abstractmethod
    def load(self) -> None:
        """Load and parse the raw dataset files."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> MedicalSample:
        """Return a single sample by index."""
        ...

    def iterate(self) -> Iterator[MedicalSample]:
        """Iterate over all samples."""
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def get_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, List[int]]:
        """
        Return index lists for train/val/test splits.

        Returns:
            {"train": [indices], "val": [indices], "test": [indices]}
        """
        ...

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the loaded dataset."""
        return {
            "name": self.__class__.__name__,
            "total_samples": len(self),
        }
