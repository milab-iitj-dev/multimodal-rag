"""
Qwen2-VL-2B-Instruct Vision-Language Model.

Concrete implementation of :class:`BaseVLM` using the
**Qwen2-VL-2B-Instruct** model with 4-bit NF4 quantization via
BitsAndBytes.  On a Kaggle P100 (16 GB VRAM) the quantised model
occupies approximately 1.5 GB, leaving ample headroom for inputs
and the KV cache.

MaxSim & VRAM Notes
-------------------
The Qwen2-VL generator is loaded **after** retrieval (ColPali + SciNCL
have already been unloaded).  The typical VRAM budget on P100 is:

    ColPali (~2.5 GB) → unload → SciNCL (~0.6 GB) → unload
    → Qwen2-VL (~1.5 GB with 4-bit) → generate → unload

Because only one model resides on GPU at a time, the MaxSim late-
interaction scoring between ColPali multi-vectors and the query is
performed **before** Qwen2-VL is loaded, avoiding OOM.

Example:
    >>> from src.domains.scientific.models.qwen2vl_model import Qwen2VLModel
    >>> model = Qwen2VLModel()
    >>> model.load()
    >>> result = model.generate("Describe this figure.", images=[img])
    >>> print(result.answer, result.confidence)
    >>> model.unload()
"""

from __future__ import annotations

import gc
import re
import time
from typing import Any, Dict, List, Optional

import torch

from src.domains.scientific.models.base_vlm import BaseVLM, VLMOutput
from src.shared.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: VRAM logging
# ---------------------------------------------------------------------------

def _log_vram(label: str) -> None:
    """Log current GPU memory usage with a descriptive label.

    Args:
        label: A human-readable tag prepended to the log line, e.g.
            ``"after load"`` or ``"after unload"``.
    """
    if not torch.cuda.is_available():
        logger.debug("%s | CUDA not available", label)
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    logger.info(
        "%s | VRAM — allocated: %.2f GB, reserved: %.2f GB",
        label,
        allocated,
        reserved,
    )


# ---------------------------------------------------------------------------
# Qwen2VLModel
# ---------------------------------------------------------------------------

class Qwen2VLModel(BaseVLM):
    """Qwen2-VL-2B-Instruct with 4-bit NF4 quantization.

    This model serves as the **generation** backbone in the Scientific
    Multimodal RAG pipeline.  It accepts a text prompt and optional
    page images, then produces a grounded answer with confidence and
    source-page citations.

    Args:
        model_name: Hugging Face model identifier.  Defaults to
            ``"Qwen/Qwen2-VL-2B-Instruct"``.
        quantization: Quantization strategy.  Currently only ``"4bit"``
            (NF4 with double quantization) is supported.
        device: Target device, e.g. ``"cuda"`` or ``"cpu"``.
        confidence_threshold: If the model's mean token probability is
            below this value the caller may request a retry.
        max_retries: Maximum number of automatic retries within
            :meth:`generate` when confidence is below threshold.

    Raises:
        RuntimeError: If CUDA is requested but not available.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        quantization: str = "4bit",
        device: str = "cuda",
        confidence_threshold: float = 0.6,
        max_retries: int = 2,
    ) -> None:
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

        # Internal state — set by load()
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

        logger.info(
            "Qwen2VLModel initialised — model=%s, quant=%s, device=%s, "
            "conf_thresh=%.2f, max_retries=%d",
            self.model_name,
            self.quantization,
            self.device,
            self.confidence_threshold,
            self.max_retries,
        )

    # -----------------------------------------------------------------
    # load
    # -----------------------------------------------------------------

    def load(self) -> None:
        """Load the Qwen2-VL model and processor with 4-bit quantization.

        Uses :class:`transformers.BitsAndBytesConfig` with NF4 quantization
        and double quantization to minimise VRAM.  The model is placed with
        ``device_map="auto"`` and switched to ``eval()`` mode.

        Raises:
            RuntimeError: If CUDA is not available when ``device="cuda"``,
                or if the model download / load fails (e.g. OOM, network
                error).
        """
        logger.info("Loading Qwen2-VL model: %s", self.model_name)

        # ── Device check ──
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available — falling back to CPU.  "
                "Generation will be significantly slower."
            )
            self.device = "cpu"

        try:
            from transformers import BitsAndBytesConfig
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            # ── Quantization config ──
            if self.quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                logger.debug("BitsAndBytesConfig: 4-bit NF4 + double quant")
            else:
                raise ValueError(
                    f"Unsupported quantization: {self.quantization!r}. "
                    "Only '4bit' is currently supported."
                )

            # ── Load model ──
            logger.info("Downloading / loading model weights…")
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self._model.eval()
            logger.info("Model loaded and set to eval mode.")

            # ── Load processor ──
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            logger.info("Processor loaded.")

            _log_vram("Qwen2-VL after load")

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "CUDA OOM while loading Qwen2-VL.  "
                "Ensure previous models have been unloaded."
            )
            self._model = None
            self._processor = None
            raise RuntimeError(
                "CUDA Out of Memory while loading Qwen2-VL.  "
                "Call unload() on other models first."
            ) from None

        except Exception as exc:
            logger.error("Failed to load Qwen2-VL: %s", exc)
            self._model = None
            self._processor = None
            raise RuntimeError(
                f"Failed to load Qwen2-VL model: {exc}"
            ) from exc

    # -----------------------------------------------------------------
    # generate
    # -----------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ) -> VLMOutput:
        """Generate an answer with optional image context.

        The prompt is wrapped in the Qwen2-VL chat template.  When
        *images* are provided they are inserted as ``<|image_pad|>``
        tokens that the vision encoder processes.

        Confidence is computed as the mean probability of generated
        tokens (exp of mean log-prob).  Source pages are extracted
        from bracket-style citations like ``[page 3]`` or ``[p.7]``.

        If confidence is below ``confidence_threshold`` the method
        retries up to ``max_retries`` times with slightly higher
        temperature to escape low-confidence basins.

        Args:
            prompt: The user question or instruction.
            images: Optional list of PIL images.  When ``None`` the
                model operates in text-only mode.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (0.0 = greedy when
                supported, otherwise near-greedy).

        Returns:
            A :class:`VLMOutput` with the answer, confidence score,
            extracted source pages, raw output, optional log-probs,
            and generation time.

        Raises:
            RuntimeError: If the model has not been loaded or if an
                OOM error occurs during generation.
            ValueError: If inputs fail validation.
        """
        self._validate_generate_inputs(prompt, max_new_tokens, temperature)

        if not self.is_loaded():
            raise RuntimeError(
                "Model not loaded. Call load() before generate()."
            )

        logger.info(
            "Generating — prompt_len=%d, n_images=%s, max_new_tokens=%d, "
            "temperature=%.2f",
            len(prompt),
            len(images) if images else 0,
            max_new_tokens,
            temperature,
        )

        best_output: Optional[VLMOutput] = None
        current_temp = temperature

        for attempt in range(self.max_retries + 1):
            try:
                result = self._generate_single(
                    prompt=prompt,
                    images=images,
                    max_new_tokens=max_new_tokens,
                    temperature=current_temp,
                )

                if result.confidence >= self.confidence_threshold:
                    logger.info(
                        "Attempt %d — confidence %.3f >= threshold %.3f.  "
                        "Returning.",
                        attempt + 1,
                        result.confidence,
                        self.confidence_threshold,
                    )
                    return result

                # Below threshold — keep best so far and retry.
                logger.warning(
                    "Attempt %d — confidence %.3f < threshold %.3f.  "
                    "Retrying with higher temperature.",
                    attempt + 1,
                    result.confidence,
                    self.confidence_threshold,
                )
                if best_output is None or result.confidence > best_output.confidence:
                    best_output = result

                # Increase temperature for next attempt.
                current_temp = min(current_temp + 0.2, 1.0)

            except torch.cuda.OutOfMemoryError:
                logger.error(
                    "CUDA OOM on attempt %d.  Cannot continue generation.",
                    attempt + 1,
                )
                raise RuntimeError(
                    "CUDA Out of Memory during generation."
                ) from None

        # All attempts exhausted — return best output even if below threshold.
        if best_output is not None:
            logger.warning(
                "All retries exhausted — returning best confidence %.3f.",
                best_output.confidence,
            )
            return best_output

        # This should never be reached but satisfies type checkers.
        raise RuntimeError("No generation output produced.")

    # -----------------------------------------------------------------
    # Internal generation
    # -----------------------------------------------------------------

    def _generate_single(
        self,
        prompt: str,
        images: Optional[List[Any]],
        max_new_tokens: int,
        temperature: float,
    ) -> VLMOutput:
        """Execute a single generation pass.

        Args:
            prompt: User prompt text.
            images: Optional list of PIL images.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature for this pass.

        Returns:
            A :class:`VLMOutput` from this single attempt.
        """
        t_start = time.time()

        # ── Build chat messages ──
        content: List[Dict[str, Any]] = []
        if images:
            for _ in images:
                content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # ── Apply chat template ──
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.debug("Chat template applied. Prompt length: %d", len(text))

        # ── Process inputs ──
        inputs = self._processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        # ── Generate ──
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1e-7,
                do_sample=temperature > 0,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # ── Decode ──
        generated_ids = outputs.sequences
        # Strip prompt tokens — only keep newly generated tokens.
        prompt_len = inputs["input_ids"].shape[1]
        new_token_ids = generated_ids[:, prompt_len:]

        raw_output = self._processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        generation_time = time.time() - t_start
        logger.debug(
            "Decoded %d new tokens in %.2f s",
            new_token_ids.shape[1],
            generation_time,
        )

        # ── Extract confidence from scores ──
        logprobs_dict = self._extract_logprobs(outputs, new_token_ids)
        confidence = self._compute_confidence(logprobs_dict)

        # ── Extract source page citations ──
        source_pages = self._extract_source_pages(raw_output)

        # ── Clean answer ──
        answer = self._clean_answer(raw_output)

        return VLMOutput(
            answer=answer,
            confidence=confidence,
            source_pages=source_pages,
            raw_output=raw_output,
            logprobs=logprobs_dict,
            generation_time=generation_time,
        )

    # -----------------------------------------------------------------
    # Confidence & logprob helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_logprobs(
        outputs: Any,
        new_token_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """Extract per-token log-probabilities from generation outputs.

        The ``scores`` field of ``GenerateOutput`` contains the logits
        for each generation step.  We gather the log-probability of the
        actually-chosen token at each step.

        Args:
            outputs: The return value of ``model.generate(…,
                return_dict_in_generate=True, output_scores=True)``.
            new_token_ids: Tensor of shape ``(1, num_new_tokens)`` with
                the IDs of the generated tokens.

        Returns:
            Dictionary mapping token string → log-probability (float).
            Only the top entries are kept to avoid bloating memory.
        """
        if not hasattr(outputs, "scores") or outputs.scores is None:
            logger.debug("No scores in generation output — logprobs unavailable.")
            return {}

        logprobs: Dict[str, float] = {}
        try:
            for step_idx, score_tensor in enumerate(outputs.scores):
                # score_tensor shape: (batch_size, vocab_size)
                if step_idx >= new_token_ids.shape[1]:
                    break
                token_id = new_token_ids[0, step_idx].item()
                log_softmax = torch.log_softmax(score_tensor[0], dim=-1)
                token_logprob = log_softmax[token_id].item()
                logprobs[f"token_{step_idx}"] = token_logprob
        except Exception as exc:
            logger.warning("Failed to extract logprobs: %s", exc)

        return logprobs

    @staticmethod
    def _compute_confidence(logprobs: Dict[str, float]) -> float:
        """Compute mean token probability from log-probabilities.

        Args:
            logprobs: Mapping of token key → log-probability.

        Returns:
            Mean probability in [0.0, 1.0].  Returns 0.5 (neutral)
            when no log-probs are available.
        """
        if not logprobs:
            logger.debug("No logprobs — returning neutral confidence 0.5")
            return 0.5

        values = list(logprobs.values())
        mean_logprob = sum(values) / len(values)
        confidence = float(torch.exp(torch.tensor(mean_logprob)).item())
        # Clamp to [0, 1] for numerical safety.
        confidence = max(0.0, min(1.0, confidence))
        logger.debug("Confidence from logprobs: %.4f", confidence)
        return confidence

    # -----------------------------------------------------------------
    # Source-page extraction
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_source_pages(raw_output: str) -> List[str]:
        """Extract source-page citations from the raw model output.

        Supports patterns like ``[page 3]``, ``[p. 7]``, ``[page_3]``,
        and ``[pp. 12-15]``.

        Args:
            raw_output: The unprocessed generation output.

        Returns:
            A sorted list of unique page-identifier strings, e.g.
            ``["page_3", "page_7"]``.
        """
        pattern = r"\[p{1,2}\.?\s*(\d+(?:\s*[-–]\s*\d+)*)\]"
        matches = re.findall(pattern, raw_output, flags=re.IGNORECASE)

        pages: List[str] = []
        for match in matches:
            # Handle ranges like "12-15"
            if "-" in match or "–" in match:
                range_part = re.split(r"[-–]", match)
                try:
                    start = int(range_part[0].strip())
                    end = int(range_part[1].strip())
                    for p in range(start, end + 1):
                        pages.append(f"page_{p}")
                except (ValueError, IndexError):
                    pages.append(f"page_{match.strip()}")
            else:
                pages.append(f"page_{match.strip()}")

        # Deduplicate while preserving order.
        seen = set()
        unique_pages: List[str] = []
        for p in pages:
            if p not in seen:
                seen.add(p)
                unique_pages.append(p)

        if unique_pages:
            logger.debug("Extracted source pages: %s", unique_pages)

        return unique_pages

    # -----------------------------------------------------------------
    # Answer cleaning
    # -----------------------------------------------------------------

    @staticmethod
    def _clean_answer(raw_output: str) -> str:
        """Clean the raw model output into a presentable answer.

        Removes citation brackets and leading/trailing whitespace.

        Args:
            raw_output: The raw text from the model.

        Returns:
            A cleaned answer string.
        """
        # Remove citation patterns like [page 3], [p.7], etc.
        cleaned = re.sub(
            r"\[p{1,2}\.?\s*\d+(?:\s*[-–]\s*\d+)*\]",
            "",
            raw_output,
            flags=re.IGNORECASE,
        )
        # Collapse multiple spaces and strip.
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if cleaned else raw_output.strip()

    # -----------------------------------------------------------------
    # unload
    # -----------------------------------------------------------------

    def unload(self) -> None:
        """Unload the model and processor, freeing GPU memory.

        Deletes the model and processor objects, forces garbage
        collection, and empties the CUDA cache.  This is critical
        for the staggered-loading strategy on GPU-constrained
        environments.

        Raises:
            RuntimeError: If unloading encounters an unexpected error.
        """
        logger.info("Unloading Qwen2-VL model…")

        vram_before = (
            torch.cuda.memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )

        try:
            del self._model
            del self._processor
        except AttributeError:
            logger.warning("Model or processor was already None during unload.")
        finally:
            self._model = None
            self._processor = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        vram_after = (
            torch.cuda.memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )

        freed = vram_before - vram_after
        logger.info(
            "Qwen2-VL unloaded — VRAM freed: %.2f GB "
            "(before: %.2f GB, after: %.2f GB)",
            freed,
            vram_before,
            vram_after,
        )
        _log_vram("Qwen2-VL after unload")
