"""
ColQwen2 multi-vector embedding model.

Produces per-token embeddings for late-interaction retrieval (MaxSim).
Unlike CLIP's single-vector approach, ColQwen2 retains spatial information
by generating one embedding per image patch / text token.

Uses ColQwen2 + ColQwen2Processor from colpali-engine for stable
ColPali-family model loading (transformers 4.47 does not expose
ColQwen2ForRetrieval natively, so we rely on colpali-engine).

Key design decisions:
  - Does NOT extend BaseEmbedder because ColQwen2 produces multi-vector
    torch.Tensor outputs (shape [n_tokens, embed_dim]), not single-vector
    np.ndarray outputs. Keeping a separate interface is more honest than
    forcing incompatible shapes through the same API.
  - Supports both image encoding (for offline indexing) and text/image
    query encoding (for online retrieval).
  - Batch processing with configurable batch size for VRAM management.
"""

import torch
from typing import List, Optional, Dict, Any
from PIL import Image

from src.shared.logging_utils import setup_logger

logger = setup_logger("embeddings.colqwen2")


def _extract_embeddings(outputs, context=""):
    """
    Safely extract embedding tensor from ColQwen2 model outputs.

    Handles version differences across transformers and colpali-engine:
      - HF native:      outputs.embeddings
      - colpali-engine:  outputs.reps or raw tensor
      - fallback:        outputs.last_hidden_state
    """
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "embeddings") and outputs.embeddings is not None:
        return outputs.embeddings
    if hasattr(outputs, "reps") and outputs.reps is not None:
        return outputs.reps
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        if isinstance(outputs[0], torch.Tensor):
            return outputs[0]
    raise ValueError(
        f"Unknown ColQwen2 output format in {context}: "
        f"type={type(outputs)}"
    )


class ColQwen2Embedder:
    """
    ColQwen2 multi-vector encoder for document retrieval.

    Produces per-token embeddings for late-interaction (MaxSim) scoring.
    Used in two places:
      1. Offline indexing: encode OpenI images → stored embeddings
      2. Online retrieval: encode user queries → query embeddings

    Usage:
        embedder = ColQwen2Embedder()
        embedder.load(config)

        # Offline: encode document images
        doc_embeddings = embedder.encode_images([pil_image_1, pil_image_2])

        # Online: encode text query
        query_embeddings = embedder.encode_queries(["What does this X-ray show?"])

        # Online: encode image + text query
        query_embeddings = embedder.encode_image_queries(
            images=[query_image],
            queries=["What abnormalities are visible?"]
        )

        # Score
        scores = embedder.score(query_embeddings, doc_embeddings)
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False
        self._model_name = "colqwen2"

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #

    def load(self, config: dict) -> None:
        """
        Load ColQwen2 model and processor.

        Args:
            config: Config dict. Expected structure:
                config["retrieval"]["colqwen2"]["model_name"] = "vidore/colqwen2-v1.0"
                config["retrieval"]["colqwen2"]["batch_size"] = 4
        """
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        # HuggingFace authentication for gated models
        # Set HF_TOKEN environment variable or use `huggingface-cli login`
        try:
            import os
            from huggingface_hub import login
            hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN"))
            if hf_token:
                login(token=hf_token)
                logger.info("  HuggingFace authentication successful")
            else:
                logger.info("  No HF_TOKEN set — using cached credentials or public models")
        except Exception as e:
            logger.warning(f"  HuggingFace login skipped: {e}")

        colqwen2_cfg = config.get("retrieval", {}).get("colqwen2", {})
        model_id = colqwen2_cfg.get("model_name", "vidore/colqwen2-v1.0")
        self._batch_size = colqwen2_cfg.get("batch_size", 4)

        logger.info(f"Loading ColQwen2 model: {model_id}")

        # Load processor
        self._processor = ColQwen2Processor.from_pretrained(model_id)
        logger.info("  Processor loaded")

        # Load model in bfloat16 for memory efficiency
        self._model = ColQwen2.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        self._device = self._model.device
        self._loaded = True

        logger.info(f"  Model loaded on device: {self._device}")
        logger.info(f"  Model dtype: {self._model.dtype}")
        logger.info(f"  Batch size: {self._batch_size}")

    # ------------------------------------------------------------------ #
    #  Document image encoding (offline indexing)                           #
    # ------------------------------------------------------------------ #

    def encode_images(
        self,
        images: List[Image.Image],
        batch_size: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Encode document images into multi-vector embeddings.

        Each image produces a tensor of shape [n_patches, embed_dim],
        where n_patches depends on the image resolution.

        Args:
            images:     List of PIL images (document pages / X-rays).
            batch_size: Override default batch size if needed.

        Returns:
            List of torch.Tensor, one per image. Each tensor has shape
            [n_patches, embed_dim] and is stored on CPU for persistence.
        """
        self._check_loaded()
        batch_size = batch_size or self._batch_size
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            logger.info(
                f"  Encoding image batch {i // batch_size + 1}"
                f"/{(len(images) + batch_size - 1) // batch_size}"
                f" ({len(batch_images)} images)"
            )

            # Use colpali_engine's dedicated process_images() API.
            # The generic __call__ (inherited from Qwen2VLProcessor)
            # does NOT correctly generate image_grid_thw for ColQwen2,
            # causing attention shape mismatches in the vision tower.
            inputs = self._process_images_safe(batch_images)

            # Log shapes for debugging
            self._log_input_shapes(inputs, "encode_images")

            try:
                with torch.no_grad():
                    outputs = self._model(**inputs)
            except RuntimeError as e:
                logger.error(f"ColQwen2 forward pass failed in encode_images: {e}")
                self._log_input_shapes(inputs, "encode_images [FAILED]")
                raise

            # Extract embeddings tensor from model output object
            embeddings = _extract_embeddings(outputs, context="encode_images")
            for j in range(embeddings.shape[0]):
                all_embeddings.append(embeddings[j].cpu())

        logger.info(f"  Encoded {len(all_embeddings)} images total")
        return all_embeddings

    # ------------------------------------------------------------------ #
    #  Text query encoding (online retrieval — text only)                   #
    # ------------------------------------------------------------------ #

    def encode_queries(
        self,
        queries: List[str],
        batch_size: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Encode text queries into multi-vector embeddings.

        Each query produces a tensor of shape [n_tokens, embed_dim].

        Args:
            queries:    List of query strings.
            batch_size: Override default batch size if needed.

        Returns:
            List of torch.Tensor, one per query. Each tensor has shape
            [n_tokens, embed_dim] and is stored on CPU.
        """
        self._check_loaded()
        batch_size = batch_size or self._batch_size
        all_embeddings = []

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]

            # Use colpali_engine's dedicated process_queries() API
            inputs = self._process_queries_safe(batch_queries)

            # Log shapes for debugging
            self._log_input_shapes(inputs, "encode_queries")

            try:
                with torch.no_grad():
                    outputs = self._model(**inputs)
            except RuntimeError as e:
                logger.error(f"ColQwen2 forward pass failed in encode_queries: {e}")
                self._log_input_shapes(inputs, "encode_queries [FAILED]")
                raise

            embeddings = _extract_embeddings(outputs, context="encode_queries")
            for j in range(embeddings.shape[0]):
                all_embeddings.append(embeddings[j].cpu())

        return all_embeddings

    # ------------------------------------------------------------------ #
    #  Document text encoding (offline indexing — text index)               #
    # ------------------------------------------------------------------ #

    def encode_document_text(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        max_length: int = 256,
    ) -> List[torch.Tensor]:
        """
        Encode document text (findings + impression) into multi-vector
        embeddings for the text retrieval index.

        Uses the same process_queries() encoder as online query encoding,
        but with a higher max_length to accommodate longer report text.
        Both document text and query text embeddings live in the same
        ColQwen2 embedding space, enabling text-to-text MaxSim matching.

        Args:
            texts:      List of document text strings (one per document).
            batch_size: Override default batch size if needed.
            max_length: Maximum token length for text encoding.
                        Default 256 fits most OpenI reports
                        (findings + impression ≈ 70-280 tokens).

        Returns:
            List of torch.Tensor, one per document. Each tensor has shape
            [n_tokens, embed_dim] and is stored on CPU.
        """
        self._check_loaded()
        batch_size = batch_size or self._batch_size
        all_embeddings = []

        # Filter out empty texts — replace with a minimal placeholder
        # to maintain alignment with doc_ids
        texts = [t if t and t.strip() else "no report available" for t in texts]

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(
                f"  Encoding text batch {i // batch_size + 1}"
                f"/{(len(texts) + batch_size - 1) // batch_size}"
                f" ({len(batch_texts)} documents)"
            )

            # Use process_queries with extended max_length for document text.
            # Both document text and online queries go through the same
            # encoder, ensuring they live in the same embedding space.
            try:
                if hasattr(self._processor, "process_queries"):
                    inputs = self._processor.process_queries(
                        batch_texts,
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                    )
                else:
                    # Fallback for older colpali_engine versions
                    logger.warning(
                        "ColQwen2Processor.process_queries() not found, "
                        "falling back to generic __call__."
                    )
                    inputs = self._processor(
                        text=batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    )
                inputs = inputs.to(self._model.device)
            except TypeError:
                # Some versions of process_queries may not accept max_length
                # as a kwarg — fall back to default processing
                logger.warning(
                    "process_queries() does not accept max_length, "
                    "using default tokenization settings."
                )
                inputs = self._process_queries_safe(batch_texts)

            # Log shapes for debugging
            self._log_input_shapes(inputs, "encode_document_text")

            try:
                with torch.no_grad():
                    outputs = self._model(**inputs)
            except RuntimeError as e:
                logger.error(
                    f"ColQwen2 forward pass failed in "
                    f"encode_document_text: {e}"
                )
                self._log_input_shapes(
                    inputs, "encode_document_text [FAILED]"
                )
                raise

            embeddings = _extract_embeddings(
                outputs, context="encode_document_text"
            )
            for j in range(embeddings.shape[0]):
                all_embeddings.append(embeddings[j].cpu())

        logger.info(f"  Encoded {len(all_embeddings)} document texts total")
        return all_embeddings

    # ------------------------------------------------------------------ #
    #  Image + text query encoding (online retrieval — multimodal)          #
    # ------------------------------------------------------------------ #

    def encode_image_queries(
        self,
        images: List[Image.Image],
        queries: List[str],
        batch_size: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Encode image queries for retrieval against the document index.

        ColQwen2 is a visual retriever — it matches image patches against
        indexed document patches via MaxSim. When a user provides both an
        image and text, we encode the image using process_images() so the
        embeddings live in the same space as the indexed documents.

        The text query is not encoded here — it is used separately by the
        RAG generator (LLaVA) for answer generation.

        Args:
            images:     List of query images (one per query).
            queries:    List of query strings (same length as images).
            batch_size: Override default batch size if needed.

        Returns:
            List of torch.Tensor, one per query image.
        """
        self._check_loaded()
        batch_size = batch_size or self._batch_size

        if len(images) != len(queries):
            raise ValueError(
                f"images and queries must have the same length, "
                f"got {len(images)} images and {len(queries)} queries"
            )

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # ColQwen2 retrieval encodes images into the document embedding
            # space. The text query doesn't participate in retrieval encoding;
            # it's used downstream by LLaVA for answer generation.
            #
            # Using process_images() ensures image_grid_thw and pixel_values
            # are correctly generated for the Qwen2-VL vision tower.
            inputs = self._process_images_safe(batch_images)

            # Log shapes for debugging
            self._log_input_shapes(inputs, "encode_image_queries")

            try:
                with torch.no_grad():
                    outputs = self._model(**inputs)
            except RuntimeError as e:
                logger.error(f"ColQwen2 forward pass failed in encode_image_queries: {e}")
                self._log_input_shapes(inputs, "encode_image_queries [FAILED]")
                raise

            embeddings = _extract_embeddings(outputs, context="encode_image_queries")
            for j in range(embeddings.shape[0]):
                all_embeddings.append(embeddings[j].cpu())

        return all_embeddings

    # ------------------------------------------------------------------ #
    #  Processor helpers (colpali_engine API)                               #
    # ------------------------------------------------------------------ #

    def _process_images_safe(self, images: List[Image.Image]) -> dict:
        """
        Process images using colpali_engine's dedicated API.

        Falls back to the generic __call__ only if process_images()
        is not available (older colpali_engine versions).
        """
        # Ensure all images are RGB
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

        if hasattr(self._processor, "process_images"):
            inputs = self._processor.process_images(images)
        else:
            # Fallback for older colpali_engine versions
            logger.warning(
                "ColQwen2Processor.process_images() not found, "
                "falling back to generic __call__. This may cause "
                "image_grid_thw shape mismatches."
            )
            inputs = self._processor(
                images=images,
                return_tensors="pt",
            )

        return inputs.to(self._model.device)

    def _process_queries_safe(self, queries: List[str]) -> dict:
        """
        Process text queries using colpali_engine's dedicated API.

        Falls back to the generic __call__ only if process_queries()
        is not available (older colpali_engine versions).
        """
        if hasattr(self._processor, "process_queries"):
            inputs = self._processor.process_queries(queries)
        else:
            # Fallback for older colpali_engine versions
            logger.warning(
                "ColQwen2Processor.process_queries() not found, "
                "falling back to generic __call__."
            )
            inputs = self._processor(
                text=queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

        return inputs.to(self._model.device)

    def _log_input_shapes(self, inputs: dict, context: str = "") -> None:
        """Log tensor shapes for debugging shape-mismatch errors."""
        parts = [f"  [{context}] Input shapes:"]
        for k, v in inputs.items():
            if hasattr(v, "shape"):
                parts.append(f"    {k}: {v.shape} (dtype={v.dtype})")
            else:
                parts.append(f"    {k}: {type(v).__name__}")
        logger.info("\n".join(parts))

    # ------------------------------------------------------------------ #
    #  Scoring (MaxSim late interaction)                                    #
    # ------------------------------------------------------------------ #

    def score(
        self,
        query_embeddings: List[torch.Tensor],
        doc_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute MaxSim late-interaction scores between queries and documents.

        MaxSim (ColBERT-style):
          For each query token, find its max cosine similarity to any
          document token, then sum across all query tokens.

          score(q, d) = sum_i( max_j( cos_sim(q_i, d_j) ) )

        This preserves the multi-vector retrieval behavior of ColQwen2.

        Args:
            query_embeddings: List of query tensors [n_tokens, embed_dim].
            doc_embeddings:   List of document tensors [n_patches, embed_dim].

        Returns:
            torch.Tensor of shape [n_queries, n_docs] with similarity scores.
        """
        self._check_loaded()

        n_queries = len(query_embeddings)
        n_docs = len(doc_embeddings)
        scores = torch.zeros(n_queries, n_docs)

        for qi in range(n_queries):
            q = query_embeddings[qi].to(self._model.device).float()
            # L2 normalize query tokens
            q = torch.nn.functional.normalize(q, p=2, dim=-1)

            for di in range(n_docs):
                d = doc_embeddings[di].to(self._model.device).float()
                # L2 normalize document tokens
                d = torch.nn.functional.normalize(d, p=2, dim=-1)

                # [n_q_tokens, n_d_tokens] cosine similarity matrix
                sim_matrix = torch.matmul(q, d.transpose(0, 1))

                # MaxSim: for each query token, take max similarity across doc tokens
                max_sim_per_token = sim_matrix.max(dim=-1).values  # [n_q_tokens]

                # Sum over query tokens to get final score
                scores[qi, di] = max_sim_per_token.sum().item()

        logger.info(
            f"  MaxSim scoring: {n_queries} queries x {n_docs} docs, "
            f"scores range [{scores.min():.2f}, {scores.max():.2f}]"
        )

        return scores

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _pad_and_stack(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Pad a list of variable-length tensors and stack into a batch.

        Each tensor has shape [seq_len, embed_dim]. We pad seq_len to
        the maximum across all tensors, using zeros.

        Returns:
            Batched tensor of shape [batch_size, max_seq_len, embed_dim].
        """
        max_len = max(t.shape[0] for t in tensors)
        embed_dim = tensors[0].shape[1]

        padded = torch.zeros(len(tensors), max_len, embed_dim, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0], :] = t

        return padded

    def _check_loaded(self) -> None:
        """Raise if model not loaded."""
        if not self._loaded:
            raise RuntimeError(
                "ColQwen2 model not loaded. Call load(config) first."
            )

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self):
        """The device the model is loaded on."""
        return self._device

    def unload(self) -> None:
        """
        Unload the model to free VRAM.

        Useful when switching between ColQwen2 and LLaVA
        on memory-constrained GPUs.
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False

        # Force garbage collection and CUDA cache clearing
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("ColQwen2 model unloaded, VRAM freed")
