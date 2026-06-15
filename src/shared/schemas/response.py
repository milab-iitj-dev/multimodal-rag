"""
Unified Response Schema for all MMRAG domain pipelines.

Every pipeline (Healthcare, Scientific, or any future domain)
returns a UnifiedResponse. This is the ONLY public contract
between the pipeline layer and the API/UI layer.

Internal domain-specific data (EvidenceSummary, GroundingResult,
CheckResult, etc.) stays internal. Only the unified schema
crosses the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class SourceItem:
    """
    A single source/citation returned by any pipeline.

    Healthcare: maps from a RetrievedDocument (case_id, score, findings).
    Scientific: maps from a SourceCitation (paper, page, arxiv link).
    """
    title: str = ""
    score: float = 0.0
    snippet: str = ""
    url: str = ""
    page_numbers: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedResponse:
    """
    The single public output schema for all MMRAG pipelines.

    Every pipeline.run() MUST return this.
    The API serializes this.
    The UI renders this.
    No exceptions.

    Fields:
        domain:     "healthcare" or "scientific"
        answer:     The generated answer text
        confidence: Confidence score (0.0–1.0)
        sources:    List of SourceItem citations
        metadata:   Any extra domain-specific info (timing, etc.)
    """
    domain: str
    answer: str
    confidence: float = 0.0
    sources: List[SourceItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
