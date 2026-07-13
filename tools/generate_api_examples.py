"""
Permanent API Example Generator — generates verified query responses
from the LIVE MMRAG Unified API.

Generates real, validated API responses across all retrieval modes
(text, image, hybrid) and saves them as production artifacts.

Usage:
    python tools/generate_api_examples.py
    python tools/generate_api_examples.py --config configs/api_examples.yaml
    python tools/generate_api_examples.py --server http://localhost:8847

Requirements:
    - MMRAG API server running and ready (GET /ready → ready=true)
    - Dataset images available in data/openi/images/ (for image/hybrid)
    - requests library installed

Output:
    outputs/api_examples/
    ├── text_01.json ... text_03.json
    ├── image_01.json, image_02.json
    ├── hybrid_01.json, hybrid_02.json
    ├── summary.json
    └── README.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml


# ═══════════════════════════════════════════════════════════════════════
#  Response Validator
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ValidationResult:
    """Result of validating a single API response."""

    http_status: int = 0
    has_answer: bool = False
    is_placeholder: bool = True
    has_sources: bool = False
    has_scores: bool = False
    has_verification: bool = False
    has_latency: bool = False
    has_confidence: bool = False
    all_passed: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "http_status": self.http_status,
            "has_answer": self.has_answer,
            "is_placeholder": self.is_placeholder,
            "has_sources": self.has_sources,
            "has_scores": self.has_scores,
            "has_verification": self.has_verification,
            "has_latency": self.has_latency,
            "has_confidence": self.has_confidence,
            "all_passed": self.all_passed,
            "errors": self.errors,
        }


class ResponseValidator:
    """Validates API responses against the frozen contract.

    Checks:
        - HTTP 200
        - answer exists and is not empty
        - answer is not a placeholder ("Pipeline not loaded")
        - sources is non-empty
        - retrieval_metadata.scores populated
        - verification fields present
        - latency_ms > 0
        - confidence >= 0
    """

    PLACEHOLDER_MARKERS = [
        "Pipeline not loaded",
        "placeholder",
        "[Healthcare]",
        "[Scientific]",
    ]

    def validate(
        self,
        http_status: int,
        response_data: Optional[dict],
    ) -> ValidationResult:
        """Validate a single API response.

        Args:
            http_status:   HTTP status code from the request.
            response_data: Parsed JSON response body, or None on failure.

        Returns:
            ValidationResult with pass/fail for each check.
        """
        result = ValidationResult(http_status=http_status)

        if http_status != 200:
            result.errors.append(f"HTTP {http_status} (expected 200)")
            return result

        if response_data is None:
            result.errors.append("Response body is None")
            return result

        # Answer
        answer = response_data.get("answer", "")
        result.has_answer = bool(answer and len(answer.strip()) > 0)
        if not result.has_answer:
            result.errors.append("Empty answer")

        # Placeholder check
        result.is_placeholder = any(
            marker in answer for marker in self.PLACEHOLDER_MARKERS
        )
        if result.is_placeholder:
            result.errors.append(
                f"Placeholder answer detected: '{answer[:80]}'"
            )

        # Sources
        sources = response_data.get("sources", [])
        result.has_sources = len(sources) > 0
        if not result.has_sources:
            result.errors.append("No sources returned")

        # Retrieval scores
        rm = response_data.get("retrieval_metadata", {})
        scores = rm.get("scores", {})
        result.has_scores = bool(scores)
        if not result.has_scores:
            result.errors.append("No retrieval scores")

        # Verification
        verification = response_data.get("verification", {})
        required_keys = ["attribution", "faithfulness", "confidence_pass"]
        result.has_verification = all(
            k in verification for k in required_keys
        )
        if not result.has_verification:
            missing = [k for k in required_keys if k not in verification]
            result.errors.append(
                f"Missing verification fields: {missing}"
            )

        # Latency
        latency = response_data.get("latency_ms", 0)
        result.has_latency = isinstance(latency, (int, float)) and latency > 0
        if not result.has_latency:
            result.errors.append(f"Invalid latency: {latency}")

        # Confidence
        confidence = response_data.get("confidence", -1)
        result.has_confidence = (
            isinstance(confidence, (int, float)) and confidence >= 0
        )
        if not result.has_confidence:
            result.errors.append(f"Invalid confidence: {confidence}")

        # Overall
        result.all_passed = (
            result.has_answer
            and not result.is_placeholder
            and result.has_sources
            and result.has_scores
            and result.has_verification
            and result.has_latency
            and result.has_confidence
        )

        return result


# ═══════════════════════════════════════════════════════════════════════
#  API Example Generator
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ExampleResult:
    """A single generated and validated API example."""

    id: str
    mode: str  # "text", "image", "hybrid"
    query: str
    domain: str
    top_k: int
    description: str
    image_path: Optional[str]
    timestamp: str
    response: dict
    validation: dict
    passed: bool


class APIExampleGenerator:
    """Generates, validates, and saves API example responses.

    Connects to a LIVE MMRAG API server, runs configured queries,
    validates every response, and saves production artifacts.
    """

    def __init__(
        self,
        config: dict,
        server_url: Optional[str] = None,
    ):
        """
        Args:
            config:     Loaded api_examples.yaml configuration.
            server_url: Override server URL (e.g. http://localhost:8847).
        """
        server_cfg = config.get("server", {})
        if server_url:
            self.base_url = server_url.rstrip("/")
        else:
            host = server_cfg.get("host", "http://localhost")
            port = server_cfg.get("port", 8847)
            self.base_url = f"{host}:{port}"

        self.timeout = server_cfg.get("timeout_seconds", 180)
        self.config = config
        self.validator = ResponseValidator()

        output_cfg = config.get("output", {})
        self.output_dir = Path(
            output_cfg.get("directory", "outputs/api_examples")
        )

        dataset_cfg = config.get("dataset", {})
        self.image_dir = dataset_cfg.get("image_dir", "data/openi/images")
        self.image_pattern = dataset_cfg.get("image_pattern", "*.dcm.png")
        self.max_images = dataset_cfg.get("max_images", 3)

        self.results: List[ExampleResult] = []

    # ── Server checks ─────────────────────────────────────────

    def wait_for_ready(self, timeout: int = 300) -> bool:
        """Wait for the API server to be ready.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if ready, False if timed out.
        """
        print(f"  Waiting for server at {self.base_url} ...")
        waited = 0
        while waited < timeout:
            try:
                resp = requests.get(
                    f"{self.base_url}/ready",
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ready"):
                        print(
                            f"  ✓ Server ready: {data.get('detail', '')}"
                        )
                        return True
            except requests.ConnectionError:
                pass
            except Exception as e:
                print(f"  ⚠ Ready check error: {e}")

            time.sleep(5)
            waited += 5
            if waited % 30 == 0:
                print(f"  ... {waited}s elapsed")

        print(f"  ✗ Server not ready after {timeout}s")
        return False

    def check_health(self) -> bool:
        """Quick health check."""
        try:
            resp = requests.get(
                f"{self.base_url}/health", timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                print(
                    f"  ✓ Health: {data.get('status')} "
                    f"(v{data.get('version')})"
                )
                return True
        except Exception as e:
            print(f"  ✗ Health check failed: {e}")
        return False

    # ── Image discovery ───────────────────────────────────────

    def discover_images(self) -> List[str]:
        """Find real dataset images from the configured directory.

        Returns:
            List of image paths (relative to project root).
        """
        pattern = os.path.join(self.image_dir, self.image_pattern)
        images = sorted(glob.glob(pattern))

        if not images:
            # Try absolute path resolution
            abs_pattern = os.path.join(
                os.getcwd(), self.image_dir, self.image_pattern
            )
            images = sorted(glob.glob(abs_pattern))

        if not images:
            print(f"  ⚠ No images found matching {pattern}")
            return []

        # Take up to max_images
        selected = images[: self.max_images]
        print(f"  ✓ Found {len(images)} dataset images, using {len(selected)}")
        for img in selected:
            print(f"    {os.path.basename(img)}")

        return selected

    # ── Query execution ───────────────────────────────────────

    def run_query(
        self,
        query_id: str,
        mode: str,
        query: str,
        domain: str,
        top_k: int,
        description: str,
        image_path: Optional[str] = None,
    ) -> ExampleResult:
        """Execute a single query against the API and validate.

        Args:
            query_id:    Identifier for this example (e.g. "text_01").
            mode:        Retrieval mode ("text", "image", "hybrid").
            query:       Query text.
            domain:      Target domain.
            top_k:       Number of documents to retrieve.
            description: Human-readable description.
            image_path:  Optional path to query image.

        Returns:
            ExampleResult with response and validation.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        payload: Dict[str, Any] = {
            "query": query,
            "domain": domain,
            "top_k": top_k,
            "include_images": True,
        }
        if image_path:
            payload["image_path"] = image_path

        print(f"  [{query_id}] {mode:6s} | {query[:60]}")
        if image_path:
            print(f"           image: {os.path.basename(image_path)}")

        try:
            resp = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=self.timeout,
            )
            http_status = resp.status_code
            response_data = resp.json() if http_status == 200 else None
        except requests.Timeout:
            print(f"           ✗ TIMEOUT ({self.timeout}s)")
            http_status = 0
            response_data = None
        except Exception as e:
            print(f"           ✗ ERROR: {e}")
            http_status = 0
            response_data = None

        # Validate
        validation = self.validator.validate(http_status, response_data)

        if validation.all_passed:
            latency = response_data.get("latency_ms", 0)
            confidence = response_data.get("confidence", 0)
            n_sources = len(response_data.get("sources", []))
            method = (
                response_data
                .get("retrieval_metadata", {})
                .get("method", "?")
            )
            scores = (
                response_data
                .get("retrieval_metadata", {})
                .get("scores", {})
            )
            fused = scores.get("fused", 0)
            colpali = scores.get("colpali", 0)
            scincl = scores.get("scincl", 0)
            print(
                f"           ✓ PASS | "
                f"{latency}ms | conf={confidence:.4f} | "
                f"sources={n_sources} | {method} | "
                f"fused={fused:.4f} col={colpali:.4f} sci={scincl:.4f}"
            )
        else:
            print(f"           ✗ FAIL | {validation.errors}")

        result = ExampleResult(
            id=query_id,
            mode=mode,
            query=query,
            domain=domain,
            top_k=top_k,
            description=description,
            image_path=image_path,
            timestamp=timestamp,
            response=response_data or {},
            validation=validation.to_dict(),
            passed=validation.all_passed,
        )
        self.results.append(result)
        return result

    # ── Run all queries ───────────────────────────────────────

    def run_all(self) -> bool:
        """Execute all configured queries.

        Returns:
            True if ALL queries passed, False otherwise.
        """
        queries_cfg = self.config.get("queries", {})

        # Discover images for image/hybrid queries
        images = self.discover_images()

        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  A. Text Retrieval")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        for q in queries_cfg.get("text", []):
            self.run_query(
                query_id=q["id"],
                mode="text",
                query=q["query"],
                domain=q.get("domain", "healthcare"),
                top_k=q.get("top_k", 3),
                description=q.get("description", ""),
            )

        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  B. Image Retrieval")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        image_queries = queries_cfg.get("image", [])
        for i, q in enumerate(image_queries):
            img_path = images[i % len(images)] if images else None
            if not img_path:
                print(
                    f"  [{q['id']}] ✗ SKIP — no dataset images available"
                )
                continue

            self.run_query(
                query_id=q["id"],
                mode="image",
                query=q["query"],
                domain=q.get("domain", "healthcare"),
                top_k=q.get("top_k", 3),
                description=q.get("description", ""),
                image_path=img_path,
            )

        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  C. Hybrid Retrieval (Image + Text)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        hybrid_queries = queries_cfg.get("hybrid", [])
        for i, q in enumerate(hybrid_queries):
            # Use different images for hybrid to show answer variation
            img_idx = (i + 1) % len(images) if images else 0
            img_path = images[img_idx] if images else None
            if not img_path:
                print(
                    f"  [{q['id']}] ✗ SKIP — no dataset images available"
                )
                continue

            self.run_query(
                query_id=q["id"],
                mode="hybrid",
                query=q["query"],
                domain=q.get("domain", "healthcare"),
                top_k=q.get("top_k", 3),
                description=q.get("description", ""),
                image_path=img_path,
            )

        return all(r.passed for r in self.results)

    # ── Save artifacts ────────────────────────────────────────

    def save_all(self) -> bool:
        """Save all results as production artifacts.

        Creates:
            - Individual JSON files per query
            - summary.json with aggregate statistics
            - README.md with formatted table

        Returns:
            True if all files saved successfully.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]

        if not passed:
            print("\n  ✗ No passed results to save!")
            return False

        # ── Individual JSON files ──
        print("\n  Saving individual JSON files...")
        for r in passed:
            filepath = self.output_dir / f"{r.id}.json"
            data = {
                "id": r.id,
                "mode": r.mode,
                "query": r.query,
                "domain": r.domain,
                "top_k": r.top_k,
                "description": r.description,
                "image_path": r.image_path,
                "timestamp": r.timestamp,
                "response": r.response,
                "validation": r.validation,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"    ✓ {filepath}")

        # ── summary.json ──
        print("  Saving summary.json...")
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "server": self.base_url,
            "total_queries": len(self.results),
            "passed": len(passed),
            "failed": len(failed),
            "by_mode": {},
            "queries": [],
        }

        for mode in ("text", "image", "hybrid"):
            mode_results = [r for r in self.results if r.mode == mode]
            mode_passed = [r for r in mode_results if r.passed]
            summary["by_mode"][mode] = {
                "total": len(mode_results),
                "passed": len(mode_passed),
            }

        for r in self.results:
            q_summary = {
                "id": r.id,
                "mode": r.mode,
                "query": r.query,
                "image_path": r.image_path,
                "passed": r.passed,
            }
            if r.passed:
                q_summary.update({
                    "latency_ms": r.response.get("latency_ms", 0),
                    "confidence": r.response.get("confidence", 0),
                    "num_sources": len(r.response.get("sources", [])),
                    "retrieval_method": (
                        r.response
                        .get("retrieval_metadata", {})
                        .get("method", "?")
                    ),
                    "scores": (
                        r.response
                        .get("retrieval_metadata", {})
                        .get("scores", {})
                    ),
                })
            else:
                q_summary["errors"] = r.validation.get("errors", [])
            summary["queries"].append(q_summary)

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"    ✓ {summary_path}")

        # ── README.md ──
        print("  Saving README.md...")
        self._generate_readme(passed, failed, summary)

        return True

    def _generate_readme(
        self,
        passed: List[ExampleResult],
        failed: List[ExampleResult],
        summary: dict,
    ) -> None:
        """Generate a formatted README.md for the API examples."""
        lines = [
            "# MMRAG Unified — API Example Responses",
            "",
            f"**Generated:** {summary['generated_at']}",
            f"**Server:** {summary['server']}",
            f"**Total:** {summary['passed']}/{summary['total_queries']} passed",
            "",
            "---",
            "",
            "## Results",
            "",
            "| ID | Mode | Query | Image | Latency | Confidence | Sources | Method | Fused Score |",
            "|-----|------|-------|-------|---------|------------|---------|--------|-------------|",
        ]

        for r in passed:
            latency = r.response.get("latency_ms", 0)
            confidence = r.response.get("confidence", 0)
            n_sources = len(r.response.get("sources", []))
            rm = r.response.get("retrieval_metadata", {})
            method = rm.get("method", "?")
            fused = rm.get("scores", {}).get("fused", 0)
            img = os.path.basename(r.image_path) if r.image_path else "—"

            lines.append(
                f"| {r.id} | {r.mode} | {r.query[:40]}… | "
                f"{img} | {latency}ms | {confidence:.4f} | "
                f"{n_sources} | {method} | {fused:.4f} |"
            )

        if failed:
            lines.extend([
                "",
                "## Failed Queries",
                "",
            ])
            for r in failed:
                errors = r.validation.get("errors", [])
                lines.append(f"- **{r.id}** ({r.mode}): {errors}")

        lines.extend([
            "",
            "---",
            "",
            "## Retrieval Modes",
            "",
            "| Mode | Description | Pipeline Path |",
            "|------|-------------|---------------|",
            "| text | Text-only query | ColQwen2 text index → reranking |",
            "| image | Query image + text | ColQwen2 image encoding → image index |",
            "| hybrid | Image + text | Dual-index → RRF fusion → reranking → Qwen2-VL |",
            "",
            "## Files",
            "",
        ])

        for r in passed:
            lines.append(f"- `{r.id}.json` — {r.description}")

        lines.extend([
            "- `summary.json` — Aggregate statistics",
            "- `README.md` — This file",
            "",
            "---",
            "",
            "## Verification",
            "",
            "Every response was validated against the frozen API contract:",
            "",
            "- ✅ HTTP 200",
            "- ✅ Non-empty answer (not placeholder)",
            "- ✅ Sources populated",
            "- ✅ Retrieval scores populated",
            "- ✅ Verification fields present (attribution, faithfulness, confidence_pass)",
            "- ✅ Latency > 0",
            "- ✅ Confidence ≥ 0",
            "",
        ])

        readme_path = self.output_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"    ✓ {readme_path}")

    # ── Final verification ────────────────────────────────────

    def final_verification(self) -> bool:
        """Re-read and re-validate every saved JSON file.

        This is the final safety check before declaring success.

        Returns:
            True if all files are valid.
        """
        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  Final Verification")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        json_files = sorted(self.output_dir.glob("*.json"))
        json_files = [
            f for f in json_files if f.name != "summary.json"
        ]

        if not json_files:
            print("  ✗ No JSON files found!")
            return False

        all_valid = True
        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Verify structure
                assert "id" in data, "missing 'id'"
                assert "response" in data, "missing 'response'"
                assert "validation" in data, "missing 'validation'"

                # Verify response content
                response = data["response"]
                validation = self.validator.validate(200, response)

                if validation.all_passed:
                    print(f"  ✓ {filepath.name}")
                else:
                    print(
                        f"  ✗ {filepath.name}: {validation.errors}"
                    )
                    all_valid = False

                # Verify image path exists (if specified)
                img_path = data.get("image_path")
                if img_path:
                    if not os.path.exists(img_path):
                        # Try relative to CWD
                        abs_path = os.path.join(os.getcwd(), img_path)
                        if not os.path.exists(abs_path):
                            print(
                                f"  ⚠ {filepath.name}: image not "
                                f"found: {img_path}"
                            )
                            # Not a hard failure — image may be on HPC only

            except json.JSONDecodeError as e:
                print(f"  ✗ {filepath.name}: invalid JSON: {e}")
                all_valid = False
            except AssertionError as e:
                print(f"  ✗ {filepath.name}: {e}")
                all_valid = False
            except Exception as e:
                print(f"  ✗ {filepath.name}: {e}")
                all_valid = False

        # Verify summary.json
        summary_path = self.output_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                assert summary.get("passed", 0) > 0, "no passed queries"
                print(f"  ✓ summary.json ({summary['passed']} passed)")
            except Exception as e:
                print(f"  ✗ summary.json: {e}")
                all_valid = False
        else:
            print("  ✗ summary.json not found")
            all_valid = False

        # Verify README.md
        readme_path = self.output_dir / "README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            assert len(content) > 100, "README too short"
            print(f"  ✓ README.md ({len(content)} chars)")
        else:
            print("  ✗ README.md not found")
            all_valid = False

        return all_valid


# ═══════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════


def main() -> int:
    """Run the API example generator.

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Generate verified API examples from the LIVE MMRAG API",
    )
    parser.add_argument(
        "--config",
        default="configs/api_examples.yaml",
        help="Path to api_examples.yaml config",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Override server URL (e.g. http://localhost:8847)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Skip waiting for server readiness (assume already running)",
    )
    args = parser.parse_args()

    # ── Banner ──
    print("")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  MMRAG Unified — API Example Generator                  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Config: {args.config:<47s}║")
    print(f"║  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<47s}║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("")

    # ── Load config ──
    config_path = args.config
    if not os.path.exists(config_path):
        # Try from project root
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        config_path = os.path.join(project_root, args.config)

    if not os.path.exists(config_path):
        print(f"  ✗ Config not found: {args.config}")
        return 1

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"  ✓ Config loaded: {config_path}")

    # ── Initialize generator ──
    generator = APIExampleGenerator(config, server_url=args.server)

    # ── Wait for server ──
    if not args.no_wait:
        if not generator.wait_for_ready(timeout=300):
            print("\n  ✗ FATAL: Server not ready. Aborting.")
            return 1
    else:
        if not generator.check_health():
            print("\n  ✗ FATAL: Server not healthy. Aborting.")
            return 1

    # ── Run all queries ──
    print("")
    print("════════════════════════════════════════════════════════════")
    print("  Running API queries")
    print("════════════════════════════════════════════════════════════")

    all_passed = generator.run_all()

    # ── Summary ──
    passed = sum(1 for r in generator.results if r.passed)
    total = len(generator.results)
    failed = total - passed

    print("")
    print("════════════════════════════════════════════════════════════")
    print(f"  Query Results: {passed}/{total} passed")
    if failed > 0:
        print(f"  ⚠ {failed} query(ies) failed")
    print("════════════════════════════════════════════════════════════")

    if passed == 0:
        print("\n  ✗ FATAL: No queries passed. Nothing to save.")
        return 1

    # ── Save artifacts ──
    print("")
    print("════════════════════════════════════════════════════════════")
    print("  Saving production artifacts")
    print("════════════════════════════════════════════════════════════")

    if not generator.save_all():
        print("\n  ✗ FATAL: Failed to save artifacts.")
        return 1

    # ── Final verification ──
    if not generator.final_verification():
        print("\n  ✗ FATAL: Final verification failed.")
        return 1

    # ── Done ──
    print("")
    print("╔══════════════════════════════════════════════════════════╗")
    if all_passed:
        print("║  ✅ ALL QUERIES PASSED — ARTIFACTS VERIFIED              ║")
    else:
        print("║  ⚠  PARTIAL SUCCESS — some queries failed               ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Passed:  {passed}/{total:<44s}║")
    print(f"║  Output:  {str(generator.output_dir):<44s}║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
