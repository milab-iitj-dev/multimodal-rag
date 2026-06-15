"""
OpenI Test Builder -- construct gold retrieval labels from OpenI metadata.

Builds benchmark queries from the OpenI test split with gold-standard
relevant document IDs derived from MeSH terms and Problems fields.

Gold label strategy:
  For a query about finding X (e.g., "Is there cardiomegaly?"),
  a document is relevant if:
    1. Its MeSH terms contain X, OR
    2. Its Problems field contains X, OR
    3. Its findings/impression text mentions X (not negated)

This module creates three types of test queries:
  - text_only:  clinical text question, no image
  - image_only: query image from test set, no text
  - hybrid:     both text question and query image
"""

import csv
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.utils.logging_utils import setup_logger

logger = setup_logger("evaluation.openi_test")


# Common clinical findings to generate queries about
CLINICAL_FINDINGS = [
    "cardiomegaly",
    "pleural effusion",
    "pulmonary edema",
    "atelectasis",
    "pneumothorax",
    "consolidation",
    "pneumonia",
    "emphysema",
    "nodule",
    "mass",
    "fracture",
    "opacity",
    "fibrosis",
    "calcification",
    "scoliosis",
]

# Negation patterns (finding is explicitly absent)
_NEGATION_RE = re.compile(
    r"(?:no|without|absence of|negative for|denies|"
    r"not?\s+(?:evidence|signs?|indication))\s+(?:of\s+)?",
    re.IGNORECASE,
)


class OpenITestBuilder:
    """
    Build gold-labeled retrieval queries from the OpenI dataset.

    Usage:
        builder = OpenITestBuilder(
            reports_csv="data/openi/reports/indiana_reports.csv",
            images_dir="data/openi/images/",
        )
        builder.load()

        # Get test queries with gold labels
        queries = builder.build_test_queries(
            split_indices=test_indices,
            query_modes=["text_only", "image_only", "hybrid"],
        )
    """

    def __init__(
        self,
        reports_csv: str,
        projections_csv: str = "",
        images_dir: str = "",
        seed: int = 42,
    ):
        self.reports_csv = Path(reports_csv)
        self.projections_csv = Path(projections_csv) if projections_csv else None
        self.images_dir = Path(images_dir) if images_dir else None
        self.seed = seed
        self.rng = random.Random(seed)

        # Internal stores
        self._docs: List[Dict[str, Any]] = []
        self._finding_to_docs: Dict[str, List[str]] = {}  # finding -> [doc_ids]

    def load(self) -> None:
        """Load OpenI reports and build the finding-to-document index."""
        logger.info(f"Loading OpenI reports from {self.reports_csv}")

        with open(self.reports_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("uid", "").strip()
                if not uid:
                    continue

                findings = (row.get("findings", "") or "").strip()
                impression = (row.get("impression", "") or "").strip()
                mesh = (row.get("MeSH", "") or "").strip()
                problems = (row.get("Problems", "") or "").strip()

                if not findings and not impression:
                    continue

                full_text = f"{findings} {impression}".lower()

                # Parse MeSH and Problems into sets
                mesh_terms = set()
                if mesh:
                    mesh_terms = {
                        t.strip().lower() for t in mesh.split(";")
                        if t.strip()
                    }
                problem_terms = set()
                if problems:
                    problem_terms = {
                        p.strip().lower() for p in problems.split(";")
                        if p.strip()
                    }

                self._docs.append({
                    "doc_id": uid,
                    "findings": findings,
                    "impression": impression,
                    "full_text": full_text,
                    "mesh_terms": mesh_terms,
                    "problem_terms": problem_terms,
                })

        logger.info(f"Loaded {len(self._docs)} documents")

        # Build finding-to-document index
        self._build_finding_index()

    def _build_finding_index(self) -> None:
        """Map each clinical finding to the document IDs that contain it."""
        for finding in CLINICAL_FINDINGS:
            matching_docs = []
            finding_lower = finding.lower()

            for doc in self._docs:
                # Check MeSH terms
                if any(finding_lower in m for m in doc["mesh_terms"]):
                    matching_docs.append(doc["doc_id"])
                    continue

                # Check Problems
                if any(finding_lower in p for p in doc["problem_terms"]):
                    matching_docs.append(doc["doc_id"])
                    continue

                # Check report text (skip negated mentions)
                if self._text_mentions_finding(
                    doc["full_text"], finding_lower
                ):
                    matching_docs.append(doc["doc_id"])

            self._finding_to_docs[finding] = matching_docs

        for f, docs in self._finding_to_docs.items():
            logger.info(f"  Finding '{f}': {len(docs)} relevant docs")

    def _text_mentions_finding(
        self, text: str, finding: str
    ) -> bool:
        """Check if text mentions a finding positively (not negated)."""
        if finding not in text:
            return False

        # Check for negation in the surrounding context
        # Look at the 60 chars before the finding
        idx = text.find(finding)
        context_start = max(0, idx - 60)
        context = text[context_start:idx]

        if _NEGATION_RE.search(context):
            return False

        return True

    def build_test_queries(
        self,
        split_indices: Optional[List[int]] = None,
        max_queries_per_finding: int = 10,
        query_modes: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build benchmark queries with gold relevance labels.

        Args:
            split_indices: Indices of documents in the test split.
                          If None, uses all documents.
            max_queries_per_finding: Cap queries per finding.
            query_modes: Which modes to generate.

        Returns:
            List of query dicts with gold labels.
        """
        if query_modes is None:
            query_modes = ["text_only", "image_only", "hybrid"]

        # Use test split documents as query sources
        if split_indices is not None:
            test_doc_ids = {
                self._docs[i]["doc_id"]
                for i in split_indices
                if i < len(self._docs)
            }
        else:
            test_doc_ids = {d["doc_id"] for d in self._docs}

        queries = []

        for finding in CLINICAL_FINDINGS:
            gold_ids = self._finding_to_docs.get(finding, [])
            if not gold_ids:
                continue

            # Find test-split documents that mention this finding
            test_relevant = [
                did for did in gold_ids if did in test_doc_ids
            ]
            if not test_relevant:
                continue

            # Cap the number of queries
            source_docs = test_relevant[:max_queries_per_finding]

            for doc_id in source_docs:
                doc = next(
                    (d for d in self._docs if d["doc_id"] == doc_id), None
                )
                if doc is None:
                    continue

                # Determine image path
                image_path = self._resolve_image_path(doc_id)

                for mode in query_modes:
                    query = self._make_query(
                        finding=finding,
                        doc_id=doc_id,
                        gold_ids=gold_ids,
                        image_path=image_path,
                        mode=mode,
                    )
                    if query is not None:
                        queries.append(query)

        self.rng.shuffle(queries)
        logger.info(
            f"Built {len(queries)} test queries across "
            f"{len(CLINICAL_FINDINGS)} findings"
        )
        return queries

    def _make_query(
        self,
        finding: str,
        doc_id: str,
        gold_ids: List[str],
        image_path: Optional[str],
        mode: str,
    ) -> Optional[Dict[str, Any]]:
        """Create a single query dict for a given mode."""
        query_text = f"Is there {finding}?"

        if mode == "text_only":
            return {
                "query_id": f"{doc_id}_{finding}_text",
                "query_text": query_text,
                "query_image_path": None,
                "finding": finding,
                "source_doc_id": doc_id,
                "gold_ids": gold_ids,
                "query_mode": "text_only",
            }

        elif mode == "image_only":
            if not image_path:
                return None
            return {
                "query_id": f"{doc_id}_{finding}_img",
                "query_text": f"Is there {finding}?",
                "query_image_path": image_path,
                "finding": finding,
                "source_doc_id": doc_id,
                "gold_ids": gold_ids,
                "query_mode": "image_only",
            }

        elif mode == "hybrid":
            if not image_path:
                return None
            return {
                "query_id": f"{doc_id}_{finding}_hybrid",
                "query_text": query_text,
                "query_image_path": image_path,
                "finding": finding,
                "source_doc_id": doc_id,
                "gold_ids": gold_ids,
                "query_mode": "hybrid",
            }

        return None

    def _resolve_image_path(self, uid: str) -> Optional[str]:
        """Try to find an image file for this UID."""
        if self.images_dir is None or not self.images_dir.exists():
            return None

        # Try common patterns
        patterns = [
            f"{uid}_IM-*.dcm.png",
            f"{uid}_*.png",
            f"{uid}*.png",
        ]
        for pattern in patterns:
            matches = list(self.images_dir.glob(pattern))
            if matches:
                return str(matches[0])

        return None

    def get_all_doc_ids(self) -> List[str]:
        """Return all document IDs."""
        return [d["doc_id"] for d in self._docs]

    def get_doc_count(self) -> int:
        """Return total document count."""
        return len(self._docs)
