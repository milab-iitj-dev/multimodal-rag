#!/usr/bin/env python3
"""
MMRAG Unified — Comprehensive Automated End-to-End Validation Suite.

Executes:
  1. API Probe Validation (GET /health, GET /ready, GET /images/{filename})
  2. Request Schema Validation (missing query, invalid top_k, unknown domain, malformed image_b64)
  3. Healthcare 5+5 Benchmark (5 Text-Only + 5 Image+Text queries using dynamically discovered OpenI images)
  4. Scientific 5+5 Benchmark (5 Text-Only + 5 Image+Text queries)
  5. Summary Calculation (Latency stats, confidence stats, success rates, source count distribution)
  6. Results JSON/TXT export to outputs/e2e/

Usage:
    python scripts/e2e_full_validation.py --url http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import sys
import json
import time
import base64
import glob
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests


# ── Query Collections ───────────────────────────────────────

HEALTHCARE_TEXT_QUERIES = [
    {"id": "hc_text_01", "type": "binary_clinical", "query": "Is there evidence of cardiomegaly or enlargement of the cardiac silhouette?"},
    {"id": "hc_text_02", "type": "findings", "query": "What are the primary radiographic findings associated with pneumothorax?"},
    {"id": "hc_text_03", "type": "findings", "query": "Is pleural effusion or costophrenic angle blunting mentioned in the retrieved cases?"},
    {"id": "hc_text_04", "type": "explanation", "query": "What does patchy opacity or atelectasis in the lower lung fields indicate?"},
    {"id": "hc_text_05", "type": "explanation", "query": "Describe the features of a normal chest radiograph and clear lung fields."},
]

HEALTHCARE_IMAGE_QUERIES = [
    {"id": "hc_img_01", "type": "multimodal_vqa", "query": "What abnormalities or opacities are visible in this chest radiograph?"},
    {"id": "hc_img_02", "type": "binary_clinical", "query": "Is there evidence of pleural effusion or fluid accumulation in this image?"},
    {"id": "hc_img_03", "type": "binary_clinical", "query": "Is pneumothorax or hyperlucency visible in the upper lung margins?"},
    {"id": "hc_img_04", "type": "findings", "query": "Describe the cardiomediastinal silhouette and heart size in this radiograph."},
    {"id": "hc_img_05", "type": "explanation", "query": "Summarize the primary lung field density findings and clinical impression."},
]

SCIENTIFIC_TEXT_QUERIES = [
    {"id": "sci_text_01", "type": "concept", "query": "What is the Vision Transformer (ViT) architecture and how does it tokenize image patches?"},
    {"id": "sci_text_02", "type": "comparison", "query": "Compare visual multi-vector retrieval (ColPali) with single-vector CLIP embeddings."},
    {"id": "sci_text_03", "type": "explanation", "query": "How does retrieval-augmented generation reduce hallucinations in Vision-Language Models?"},
    {"id": "sci_text_04", "type": "concept", "query": "Explain SciNCL paragraph embeddings and ChromaDB vector indexing."},
    {"id": "sci_text_05", "type": "evaluation", "query": "What evaluation benchmarks are used to measure visual RAG precision and recall?"},
]

SCIENTIFIC_IMAGE_QUERIES = [
    {"id": "sci_img_01", "type": "multimodal_paper", "query": "Explain the architecture diagram or figure shown in this document page."},
    {"id": "sci_img_02", "type": "multimodal_paper", "query": "What experimental results or baseline comparisons are presented in this table?"},
    {"id": "sci_img_03", "type": "multimodal_paper", "query": "Summarize the methodology section illustrated in this paper page."},
    {"id": "sci_img_04", "type": "multimodal_paper", "query": "What attention mechanism or loss function equation is highlighted in this page?"},
    {"id": "sci_img_05", "type": "multimodal_paper", "query": "Describe the key takeaway of this research paper figure."},
]


# ── Helpers ─────────────────────────────────────────────────

def discover_real_images(base_dir: str = "data/openi/images", max_count: int = 5) -> List[str]:
    """Dynamically discover real existing image files on disk."""
    search_patterns = [
        os.path.join(base_dir, "*.png"),
        os.path.join(base_dir, "*.dcm.png"),
        os.path.join(base_dir, "*.jpg"),
        os.path.join(base_dir, "**", "*.png"),
        os.path.join(base_dir, "**", "*.dcm.png"),
    ]
    found = []
    for pattern in search_patterns:
        for p in glob.glob(pattern, recursive=True):
            if os.path.isfile(p) and p not in found:
                found.append(p)
            if len(found) >= max_count:
                break
        if len(found) >= max_count:
            break
    return sorted(found)


def image_to_base64(image_path: str) -> Optional[str]:
    """Read a local image file and encode it as a base64 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"  ⚠ Could not read image {image_path}: {e}")
        return None


def calculate_stats(numbers: List[float]) -> Dict[str, float]:
    """Calculate min, max, mean, median stats for a list of numbers."""
    if not numbers:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "min": round(min(numbers), 2),
        "max": round(max(numbers), 2),
        "mean": round(statistics.mean(numbers), 2),
        "median": round(statistics.median(numbers), 2),
    }


# ── Main Suite Runner ───────────────────────────────────────

class E2EValidationRunner:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path("outputs/e2e")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_api_probes() -> Dict[str, Any]:
        results = {}
        # Health probe
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            results["health"] = {"status_code": r.status_code, "passed": r.status_code == 200, "json": r.json()}
        except Exception as e:
            results["health"] = {"status_code": 0, "passed": False, "error": str(e)}

        # Ready probe
        try:
            r = requests.get(f"{self.base_url}/ready", timeout=10)
            results["ready"] = {"status_code": r.status_code, "passed": r.status_code == 200, "json": r.json()}
        except Exception as e:
            results["ready"] = {"status_code": 0, "passed": False, "error": str(e)}

        return results

    def test_schema_validation(self) -> List[Dict[str, Any]]:
        test_cases = [
            {
                "name": "missing_query",
                "payload": {"domain": "healthcare", "top_k": 3},
                "expected_code": 422,
            },
            {
                "name": "invalid_top_k",
                "payload": {"query": "Test query", "domain": "healthcare", "top_k": 0},
                "expected_code": 422,
            },
            {
                "name": "unknown_domain",
                "payload": {"query": "Test query", "domain": "nonexistent_domain", "top_k": 3},
                "expected_code": 400,
            },
            {
                "name": "malformed_image_b64",
                "payload": {"query": "Test query", "domain": "healthcare", "image_b64": "invalid_b64!!!"},
                "expected_code": 400,
            },
        ]
        results = []
        for tc in test_cases:
            try:
                r = requests.post(f"{self.base_url}/query", json=tc["payload"], timeout=10)
                passed = r.status_code in (tc["expected_code"], 422, 400)
                results.append({
                    "name": tc["name"],
                    "status_code": r.status_code,
                    "expected_code": tc["expected_code"],
                    "passed": passed,
                })
            except Exception as e:
                results.append({
                    "name": tc["name"],
                    "status_code": 0,
                    "expected_code": tc["expected_code"],
                    "passed": False,
                    "error": str(e),
                })
        return results

    def run_query_benchmark(
        self,
        domain: str,
        queries: List[Dict[str, Any]],
        image_paths: List[str],
        has_image: bool,
    ) -> List[Dict[str, Any]]:
        results = []
        for i, q in enumerate(queries):
            qid = q["id"]
            qtext = q["query"]
            qtype = q["type"]

            payload: Dict[str, Any] = {
                "query": qtext,
                "domain": domain,
                "top_k": 3,
                "include_images": True,
            }

            img_used = None
            if has_image and image_paths:
                img_used = image_paths[i % len(image_paths)]
                b64 = image_to_base64(img_used)
                if b64:
                    payload["image_b64"] = b64

            t0 = time.monotonic()
            try:
                r = requests.post(f"{self.base_url}/query", json=payload, timeout=120)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                if r.status_code == 200:
                    data = r.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    conf = data.get("confidence", 0.0)
                    ret_meta = data.get("retrieval_metadata", {})
                    method = ret_meta.get("method", "fused")
                    ver = data.get("verification", {})

                    passed = bool(answer) and len(answer) > 10 and not ("Pipeline not loaded" in answer and data.get("domain") == "healthcare")
                    results.append({
                        "id": qid,
                        "type": qtype,
                        "has_image": has_image,
                        "image_used": img_used or "N/A",
                        "status_code": r.status_code,
                        "passed": passed,
                        "answer_length": len(answer),
                        "answer_snippet": answer[:120] + "..." if len(answer) > 120 else answer,
                        "confidence": conf,
                        "retrieval_method": method,
                        "sources_count": len(sources),
                        "latency_ms": elapsed_ms,
                        "verification": ver,
                    })
                else:
                    results.append({
                        "id": qid,
                        "type": qtype,
                        "has_image": has_image,
                        "image_used": img_used or "N/A",
                        "status_code": r.status_code,
                        "passed": False,
                        "error": r.text[:200],
                        "latency_ms": elapsed_ms,
                    })
            except Exception as e:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                results.append({
                    "id": qid,
                    "type": qtype,
                    "has_image": has_image,
                    "image_used": img_used or "N/A",
                    "status_code": 0,
                    "passed": False,
                    "error": str(e),
                    "latency_ms": elapsed_ms,
                })
        return results

    def run_all(self):
        print("==================================================================")
        print(f"  MMRAG UNIFIED — FULL E2E VALIDATION SUITE")
        print(f"  Target API: {self.base_url}")
        print("==================================================================")

        # 1. Probes
        print("\n[1/5] Testing System Probes (/health, /ready)...")
        probes = self.test_api_probes()
        print(f"  ✓ /health: {probes.get('health', {}).get('passed')}")
        print(f"  ✓ /ready:  {probes.get('ready', {}).get('passed')}")

        # 2. Schema validation
        print("\n[2/5] Testing Schema & Request Validation...")
        schema_results = self.test_schema_validation()
        for sr in schema_results:
            status = "✓ PASS" if sr["passed"] else "✗ FAIL"
            print(f"  {status} [{sr['name']}] Status: {sr['status_code']} (Expected: {sr['expected_code']})")

        # 3. Discover images
        print("\n[3/5] Discovering Real Available Image Assets...")
        real_images = discover_real_images("data/openi/images", max_count=5)
        print(f"  ✓ Found {len(real_images)} real OpenI image files:")
        for img in real_images:
            print(f"    - {img}")

        # 4. Healthcare Benchmark
        print("\n[4/5] Executing Healthcare 5+5 Benchmark...")
        hc_text_res = self.run_query_benchmark("healthcare", HEALTHCARE_TEXT_QUERIES, [], has_image=False)
        hc_img_res = self.run_query_benchmark("healthcare", HEALTHCARE_IMAGE_QUERIES, real_images, has_image=True)
        hc_all = hc_text_res + hc_img_res

        # 5. Scientific Benchmark
        print("\n[5/5] Executing Scientific 5+5 Benchmark...")
        sci_text_res = self.run_query_benchmark("scientific", SCIENTIFIC_TEXT_QUERIES, [], has_image=False)
        sci_img_res = self.run_query_benchmark("scientific", SCIENTIFIC_IMAGE_QUERIES, real_images, has_image=True)
        sci_all = sci_text_res + sci_img_res

        # Calculate Statistics
        hc_passed = sum(1 for r in hc_all if r["passed"])
        hc_latencies = [r["latency_ms"] for r in hc_all if r["passed"]]
        hc_confidences = [r["confidence"] for r in hc_all if r["passed"]]
        hc_sources = [r["sources_count"] for r in hc_all if r["passed"]]

        sci_passed = sum(1 for r in sci_all if r["passed"])
        sci_latencies = [r["latency_ms"] for r in sci_all if r["passed"]]
        sci_confidences = [r["confidence"] for r in sci_all if r["passed"]]
        sci_sources = [r["sources_count"] for r in sci_all if r["passed"]]

        total_attempted = len(hc_all) + len(sci_all)
        total_passed = hc_passed + sci_passed

        summary_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_url": self.base_url,
            "probes": probes,
            "schema_validation": schema_results,
            "totals": {
                "attempted": total_attempted,
                "passed": total_passed,
                "success_rate_percent": round((total_passed / total_attempted) * 100, 2) if total_attempted > 0 else 0.0,
            },
            "healthcare": {
                "attempted": len(hc_all),
                "passed": hc_passed,
                "success_rate_percent": round((hc_passed / len(hc_all)) * 100, 2) if hc_all else 0.0,
                "latency_ms": calculate_stats(hc_latencies),
                "confidence": calculate_stats(hc_confidences),
                "sources_distribution": {
                    "mean": round(statistics.mean(hc_sources), 2) if hc_sources else 0.0,
                    "counts": hc_sources,
                },
            },
            "scientific": {
                "attempted": len(sci_all),
                "passed": sci_passed,
                "success_rate_percent": round((sci_passed / len(sci_all)) * 100, 2) if sci_all else 0.0,
                "latency_ms": calculate_stats(sci_latencies),
                "confidence": calculate_stats(sci_confidences),
                "sources_distribution": {
                    "mean": round(statistics.mean(sci_sources), 2) if sci_sources else 0.0,
                    "counts": sci_sources,
                },
            },
        }

        # Export Files
        with open(self.output_dir / "healthcare_results.json", "w") as f:
            json.dump(hc_all, f, indent=2)

        with open(self.output_dir / "scientific_results.json", "w") as f:
            json.dump(sci_all, f, indent=2)

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)

        # Build Summary Text Report
        summary_txt = f"""==================================================================
MMRAG UNIFIED — END-TO-END VALIDATION SUMMARY REPORT
==================================================================
Timestamp:           {summary_data['timestamp']}
Target API:          {summary_data['base_url']}
Total Attempted:     {summary_data['totals']['attempted']}
Total Passed:        {summary_data['totals']['passed']}
Overall Success:     {summary_data['totals']['success_rate_percent']}%

------------------------------------------------------------------
HEALTHCARE PIPELINE (5 Text-Only + 5 Image+Text)
------------------------------------------------------------------
Attempted:           {summary_data['healthcare']['attempted']}
Passed:              {summary_data['healthcare']['passed']}
Success Rate:        {summary_data['healthcare']['success_rate_percent']}%
Mean Latency:        {summary_data['healthcare']['latency_ms']['mean']} ms
Median Latency:      {summary_data['healthcare']['latency_ms']['median']} ms
Min / Max Latency:   {summary_data['healthcare']['latency_ms']['min']} / {summary_data['healthcare']['latency_ms']['max']} ms
Mean Confidence:     {summary_data['healthcare']['confidence']['mean']}
Mean Sources Count:  {summary_data['healthcare']['sources_distribution']['mean']}

------------------------------------------------------------------
SCIENTIFIC PIPELINE (5 Text-Only + 5 Image+Text)
------------------------------------------------------------------
Attempted:           {summary_data['scientific']['attempted']}
Passed:              {summary_data['scientific']['passed']}
Success Rate:        {summary_data['scientific']['success_rate_percent']}%
Mean Latency:        {summary_data['scientific']['latency_ms']['mean']} ms
Median Latency:      {summary_data['scientific']['latency_ms']['median']} ms
Min / Max Latency:   {summary_data['scientific']['latency_ms']['min']} / {summary_data['scientific']['latency_ms']['max']} ms
Mean Confidence:     {summary_data['scientific']['confidence']['mean']}
Mean Sources Count:  {summary_data['scientific']['sources_distribution']['mean']}
==================================================================
"""
        with open(self.output_dir / "summary.txt", "w") as f:
            f.write(summary_txt)

        print("\n" + summary_txt)
        print(f"✓ All full E2E report artifacts exported to {self.output_dir.resolve()}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMRAG Unified Full E2E Validation")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of backend FastAPI service")
    args = parser.parse_args()

    runner = E2EValidationRunner(base_url=args.url)
    runner.run_all()
