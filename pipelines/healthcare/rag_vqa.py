"""
Grounded RAG VQA Pipeline — Phase 4

End-to-end online pipeline with evidence grounding:
    1. Load saved ColQwen2 dual index
    2. Load VLM (Qwen2.5-VL-7B or LLaVA)
    3. Accept user query (text-only or image + text)
    4. Retrieve top-k relevant cases via ColQwen2 + RRF fusion
    5. Aggregate evidence into structured summary
    6. Generate grounded answer with VLM
    7. Verify answer against evidence (grounding check)
    8. Score confidence
    9. Return verified answer with full provenance

Usage:
    python -m pipelines.rag_vqa --query "What does this chest X-ray show?"
    python -m pipelines.rag_vqa --query "Describe the findings" --query-image path/to/image.png
    python -m pipelines.rag_vqa --eval --max-samples 10
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from PIL import Image

from src.domains.healthcare.embeddings.colqwen2_embedder import ColQwen2Embedder
from src.domains.healthcare.retrieval.colqwen2_retriever import ColQwen2Retriever
from src.domains.healthcare.retrieval.hybrid_retriever import HybridRetriever
from src.domains.healthcare.context.context_builder import ContextBuilder
from src.domains.healthcare.generation.rag_generator import RAGGenerator, RAGOutput
from src.domains.healthcare.generation.base_generator import BaseVLM
from src.domains.healthcare.ingestion.base_loader import BaseDataset
from src.domains.healthcare.ingestion.preprocessing import MedicalImagePreprocessor
from src.shared.logging_utils import setup_logger

logger = setup_logger("pipeline.rag_vqa")


class RAGVQAPipeline:
    """
    Phase 4 grounded pipeline: Query → Retrieve → Aggregate → Generate → Verify.

    Loads a pre-built ColQwen2 dual index and VLM, then processes
    user queries through the full grounded RAG pipeline.

    Pipeline:
      Retrieve → Evidence Aggregation → VLM Generation →
      Grounding Verification → Confidence Scoring → Output
    """

    def __init__(
        self,
        vlm: BaseVLM,
        retrieval_config: dict,
        index_dir: str = "data/indexes/colqwen2_index",
        top_k: int = 3,
        max_context_chars: int = 3000,
        max_evidence_chars: int = 800,
        output_dir: str = "outputs/rag_results",
    ):
        """
        Args:
            vlm:                Loaded LLaVA model.
            retrieval_config:   Retrieval configuration dict.
            index_dir:          Path to saved ColQwen2 index.
            top_k:              Number of documents to retrieve.
            max_context_chars:  Token budget for context (chars).
            max_evidence_chars: Token budget per evidence piece (chars).
            output_dir:         Directory for saving results.
        """
        self.vlm = vlm
        self.retrieval_config = retrieval_config
        self.top_k = top_k
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = MedicalImagePreprocessor()

        # Initialize ColQwen2 embedder and base retriever
        logger.info("Initializing ColQwen2 retriever...")
        self.embedder = ColQwen2Embedder()
        self.embedder.load(retrieval_config)

        colqwen2_retriever = ColQwen2Retriever(
            self.embedder, config=retrieval_config
        )
        colqwen2_retriever.load_index(index_dir)
        logger.info(
            f"Index loaded: {colqwen2_retriever.num_indexed} documents "
            f"from {index_dir}"
        )

        # Wrap with HybridRetriever if method is 'hybrid'
        retrieval_method = (
            retrieval_config
            .get("retrieval", {})
            .get("method", "colqwen2")
        )

        if retrieval_method == "hybrid" and colqwen2_retriever.has_text_index:
            logger.info(
                "Using HybridRetriever (dual-index + RRF + reranking)"
            )
            self.retriever = HybridRetriever(
                colqwen2_retriever=colqwen2_retriever,
                config=retrieval_config,
            )
        else:
            if retrieval_method == "hybrid":
                logger.warning(
                    "Hybrid mode requested but no text index found. "
                    "Falling back to ColQwen2-only retrieval. "
                    "Run: python -m pipelines.offline_indexing --text-only"
                )
            self.retriever = colqwen2_retriever

        # Context builder (used for image selection fallback)
        self.context_builder = ContextBuilder(
            max_context_chars=max_context_chars,
            max_evidence_chars=max_evidence_chars,
        )

        # Grounded RAG generator — auto-creates:
        #   - EvidenceAggregator
        #   - GroundingVerifier
        #   - ConfidenceEstimator
        self.generator = RAGGenerator(
            vlm=self.vlm,
            retriever=self.retriever,
            context_builder=self.context_builder,
            top_k=self.top_k,
        )

        logger.info(
            f"Grounded RAG pipeline ready "
            f"(retriever: {type(self.retriever).__name__}, "
            f"vlm: {vlm.model_name})"
        )

    # ------------------------------------------------------------------ #
    #  Single query                                                        #
    # ------------------------------------------------------------------ #

    def run_single(
        self,
        query: str,
        query_image: Optional[Image.Image] = None,
        query_image_path: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RAGOutput:
        """
        Process a single query through the RAG pipeline.

        Args:
            query:            Text query / clinical question.
            query_image:      Optional PIL image for multimodal query.
            query_image_path: Optional path to query image (alternative
                              to passing PIL image directly).
            top_k:            Override default top_k.

        Returns:
            RAGOutput with answer, retrieved docs, and provenance.
        """
        # Load image from path if provided
        if query_image is None and query_image_path is not None:
            from src.shared.image_utils import load_image
            query_image = load_image(query_image_path)
            query_image = self.preprocessor(query_image)

        logger.info(f"Query: '{query}'")
        logger.info(f"Query has image: {query_image is not None}")

        output = self.generator.generate(
            query=query,
            query_image=query_image,
            top_k=top_k or self.top_k,
        )

        return output

    # ------------------------------------------------------------------ #
    #  Batch evaluation on dataset                                         #
    # ------------------------------------------------------------------ #

    def run_eval(
        self,
        dataset: BaseDataset,
        sample_indices: Optional[List[int]] = None,
        max_samples: int = 10,
        custom_question: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run RAG VQA on dataset samples for evaluation.

        For each sample, uses the sample's image as the query image
        and runs the full retrieve → context → generate pipeline.

        Args:
            dataset:         Loaded dataset.
            sample_indices:  Specific indices to evaluate.
            max_samples:     Cap on samples (if indices not given).
            custom_question: Override question for all samples.

        Returns:
            List of result dicts.
        """
        if sample_indices is None:
            sample_indices = list(range(min(max_samples, len(dataset))))

        logger.info(f"Running RAG evaluation on {len(sample_indices)} samples")
        results = []

        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]
            logger.info(
                f"\n  [{i+1}/{len(sample_indices)}] Sample: {sample.sample_id}"
            )

            if sample.image is None:
                logger.warning(f"  Skipping {sample.sample_id}: no image")
                continue

            question = custom_question or sample.question
            image = self.preprocessor(sample.image)

            try:
                output = self.generator.generate(
                    query=question,
                    query_image=image,
                    top_k=self.top_k,
                )

                result = {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "question": question,
                    "generated_answer": output.answer,
                    "ground_truth": sample.answer,
                    "num_retrieved": len(output.retrieved_docs),
                    "retrieved_doc_ids": [
                        d.doc_id for d in output.retrieved_docs
                    ],
                    "retrieved_scores": [
                        d.score for d in output.retrieved_docs
                    ],
                    "evidence_consensus": (
                        output.evidence_summary.consensus
                        if output.evidence_summary else None
                    ),
                    "was_corrected": (
                        output.grounding_result.was_corrected
                        if output.grounding_result else False
                    ),
                    "confidence_level": (
                        output.confidence.level
                        if output.confidence else None
                    ),
                    "confidence_score": (
                        output.confidence.score
                        if output.confidence else None
                    ),
                    "retrieval_time_sec": output.retrieval_time_sec,
                    "generation_time_sec": output.generation_time_sec,
                    "total_time_sec": output.total_time_sec,
                }
                results.append(result)

                logger.info(f"  Question:    {question}")
                logger.info(f"  Answer:      {output.answer[:200]}")
                logger.info(
                    f"  Confidence:  "
                    f"{output.confidence.level if output.confidence else '?'}"
                )
                logger.info(f"  Time:        {output.total_time_sec}s")

            except Exception as e:
                logger.error(f"  Error on {sample.sample_id}: {e}")
                results.append({
                    "sample_id": sample.sample_id,
                    "error": str(e),
                })

        # Save results
        self._save_results(results, prefix="rag_eval")
        return results

    # ------------------------------------------------------------------ #
    #  Save results                                                        #
    # ------------------------------------------------------------------ #

    def _save_results(
        self,
        results: List[Dict[str, Any]],
        prefix: str = "rag_results",
    ) -> None:
        """Save results to a timestamped JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"{prefix}_{timestamp}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Results saved to: {out_path}")


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #

def main():
    """Run the RAG VQA pipeline from command line."""
    import argparse
    import yaml
    from src.domains.healthcare.generation.model_factory import create_model
    from src.shared.device import print_gpu_status

    parser = argparse.ArgumentParser(
        description="Phase 4: Grounded RAG VQA Pipeline"
    )
    parser.add_argument(
        "--model-config",
        default="configs/model_config.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--retrieval-config",
        default="configs/retrieval_config.yaml",
        help="Path to retrieval config YAML",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data_config.yaml",
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--index-dir",
        default="data/indexes/colqwen2_index",
        help="Path to saved ColQwen2 index",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Text query for single-query mode",
    )
    parser.add_argument(
        "--query-image",
        type=str,
        default=None,
        help="Path to query image (for image+text query mode)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation on dataset samples",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Max samples for evaluation mode",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rag_results",
        help="Directory for saving results",
    )
    args = parser.parse_args()

    # Print GPU status
    print_gpu_status()

    # Load configs
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.retrieval_config) as f:
        retrieval_config = yaml.safe_load(f)

    # Load VLM (Qwen2.5-VL or LLaVA based on config)
    model_name = model_config["model"]["name"]
    logger.info(f"Loading VLM: {model_name}...")
    model = create_model(model_config)
    model.load(model_config)

    pipeline = RAGVQAPipeline(
        vlm=model,
        retrieval_config=retrieval_config,
        index_dir=args.index_dir,
        top_k=args.top_k,
        max_context_chars=3000,
        max_evidence_chars=800,
        output_dir=args.output_dir,
    )

    if args.query:
        # Single query mode
        output = pipeline.run_single(
            query=args.query,
            query_image_path=args.query_image,
        )

        print("\n" + "=" * 60)
        print("GROUNDED RAG VQA RESULT")
        print("=" * 60)
        print(f"Query:       {args.query}")
        print(f"Answer:      {output.answer}")

        # Grounding info
        if output.grounding_result:
            gr = output.grounding_result
            if gr.was_corrected:
                print(f"\n⚠ ANSWER WAS CORRECTED")
                print(f"  Original: {gr.original_answer[:200]}")
                print(f"  Reason:   {gr.correction_reason}")
            elif gr.contradiction_detected:
                print(f"\n⚠ CONTRADICTION DETECTED (not corrected)")
                print(f"  Reason:   {gr.correction_reason}")

        # Confidence
        if output.confidence:
            print(f"\nConfidence:  {output.confidence.level} "
                  f"({output.confidence.score})")
            for k, v in output.confidence.factors.items():
                print(f"  {k}: {v}")

        # Evidence summary
        if output.evidence_summary:
            es = output.evidence_summary
            print(f"\nEvidence:    {es.consensus} "
                  f"({len(es.relevant_findings)} findings)")

        # Retrieved docs
        print(f"\nRetrieved {len(output.retrieved_docs)} documents:")
        for doc in output.retrieved_docs:
            print(f"  [{doc.metadata.get('rank', '?')}] {doc.doc_id} "
                  f"(score: {doc.score:.4f})")
        print(f"\nRetrieval:   {output.retrieval_time_sec}s")
        print(f"Generation:  {output.generation_time_sec}s")
        print(f"Total:       {output.total_time_sec}s")
        print("=" * 60)

    elif args.eval:
        # Evaluation mode
        with open(args.data_config) as f:
            data_config = yaml.safe_load(f)

        from src.shared.config_loader import resolve_data_paths
        data_config = resolve_data_paths(data_config)

        from src.domains.healthcare.ingestion.dicom_loader import OpenIDataset
        ds_cfg = data_config["dataset"]
        dataset = OpenIDataset(
            images_dir=ds_cfg["images_dir"],
            reports_dir=ds_cfg["reports_dir"],
            max_samples=args.max_samples,
        )
        dataset.load()

        results = pipeline.run_eval(
            dataset=dataset,
            max_samples=args.max_samples,
        )

        logger.info(f"Evaluation complete: {len(results)} results")

    else:
        print("Specify --query for single query or --eval for evaluation.")
        print("Examples:")
        print('  python -m pipelines.rag_vqa --query "What abnormalities are visible?"')
        print('  python -m pipelines.rag_vqa --query "Describe findings" --query-image image.png')
        print("  python -m pipelines.rag_vqa --eval --max-samples 5")


if __name__ == "__main__":
    main()
