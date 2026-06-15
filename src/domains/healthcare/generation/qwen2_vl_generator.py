"""
Qwen2-VL-7B Model Wrapper.

Implements BaseVLM for Qwen2-VL-7B-Instruct in pure BF16 precision.
Uses the Qwen2-VL chat template for structured medical VQA with
strong grounding support.

Loads in bfloat16 with device_map="auto" — uses ~15GB VRAM on A100 40GB.
No quantization needed.

Compatibility:
  - Works with transformers>=4.45.0 (tested at 4.47.1)
  - Compatible with colpali-engine 0.3.8 (ColQwen2 retrieval)
  - Uses Qwen2VLForConditionalGeneration (NOT Qwen2_5_VL)

Architecture notes:
  - Qwen2-VL uses a structured conversation format:
      <|im_start|>system\n{system_prompt}<|im_end|>
      <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>
      <|im_start|>assistant\n
  - The processor handles image token injection automatically
  - Qwen2-VL has much stronger instruction following than LLaVA-1.5
"""

import time
from typing import Optional, Dict, Any, List

import torch
from PIL import Image

from src.domains.healthcare.generation.base_generator import BaseVLM, VLMOutput
from src.shared.device import get_vram_usage_gb
from src.shared.logging_utils import setup_logger

logger = setup_logger("models.qwen2_vl")


# ------------------------------------------------------------------ #
#  System prompt for grounded medical VQA                              #
# ------------------------------------------------------------------ #

MEDICAL_SYSTEM_PROMPT = (
    "You are an expert radiologist analyzing chest X-ray images. "
    "You will be given retrieved clinical evidence from similar cases "
    "in a medical knowledge base.\n\n"
    "STRICT RULES:\n"
    "1. Base your answer PRIMARILY on the retrieved evidence and the image.\n"
    "2. Do NOT hallucinate findings that are not supported by evidence.\n"
    "3. If the evidence clearly states a finding is absent "
    "(e.g., 'no pleural effusion'), your answer must reflect that.\n"
    "4. If the evidence conflicts with what you observe in the image, "
    "explicitly state the discrepancy.\n"
    "5. If evidence is insufficient, say so explicitly.\n"
    "6. Always justify your answer by referencing specific evidence."
)


class Qwen2VLModel(BaseVLM):
    """
    Qwen2-VL-7B-Instruct wrapper for grounded medical VQA.

    Key advantages over LLaVA-1.5-7B:
      - Stronger instruction following (critical for grounding)
      - Better negation understanding
      - Native multi-turn chat template
      - Dynamic resolution support

    Compatible with transformers==4.47.1 and colpali-engine==0.3.8.

    Usage:
        model = Qwen2VLModel()
        model.load(config)
        output = model.generate(image, "Is there pneumonia?", context="...")
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._config = None
        self._device = None
        self._model_name = "qwen2-vl-7b"
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  BaseVLM interface                                                   #
    # ------------------------------------------------------------------ #

    def load(self, config: dict) -> None:
        """
        Load Qwen2-VL-7B in BF16 precision.

        Uses pure bfloat16 with device_map="auto" for A100 40GB.
        No quantization — BF16 Qwen2-VL-7B uses ~15GB VRAM,
        well within A100 40GB capacity.

        Requires: transformers>=4.45.0 (Qwen2VLForConditionalGeneration)
        Tested:   transformers==4.47.1 with colpali-engine==0.3.8
        """
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoProcessor,
        )

        self._config = config
        model_cfg = config["model"]
        model_id = model_cfg["model_id"]

        logger.info(f"Loading Qwen2-VL model: {model_id}")
        logger.info(f"  Precision: bfloat16 (no quantization)")

        # Load processor with pixel limits to prevent OOM.
        # Qwen2-VL uses dynamic resolution — without limits, large images
        # produce 3000+ visual tokens causing ~40GB attention memory.
        # 512 patches = ~400K pixels — sufficient for chest X-ray detail.
        # VRAM budget: ~16.6GB (model) + ~4.5GB (inference) ≈ 21GB on A100 40GB.
        min_pixels = 256 * 28 * 28    # 200,704 pixels
        max_pixels = 512 * 28 * 28    # 401,408 pixels

        self._processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        logger.info(
            f"  Processor loaded (pixels: {min_pixels}-{max_pixels}, "
            f"max ~512 visual tokens)"
        )

        # Load model in pure BF16
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        self._device = self._model.device
        self._loaded = True

        logger.info(f"  Model loaded on device: {self._device}")
        logger.info(f"  Model dtype: {self._model.dtype}")

        mem = self.get_memory_footprint()
        logger.info(f"  VRAM allocated: {mem['allocated_gb']} GB")

    def generate(
        self,
        image: Image.Image,
        question: str,
        context: Optional[str] = None,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> VLMOutput:
        """Generate answer from image + question using Qwen2-VL."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        gen_cfg = self._config["model"].get("generation", {})
        query_type = kwargs.get("query_type", None)

        # Build the chat messages (query-type aware)
        messages = self._build_messages(question, context, query_type)

        # Process inputs using the chat template
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        device = self._model.device
        inputs = {
            k: v.to(device) if hasattr(v, 'to') else v
            for k, v in inputs.items()
        }

        input_token_count = inputs["input_ids"].shape[-1]

        # Generate
        try:
            start_time = time.time()
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=gen_cfg.get("temperature", 0.1),
                    top_p=gen_cfg.get("top_p", 0.9),
                    do_sample=gen_cfg.get("do_sample", False),
                    repetition_penalty=gen_cfg.get(
                        "repetition_penalty", 1.1
                    ),
                )
            generation_time = time.time() - start_time
        except RuntimeError as e:
            logger.error(f"Generation failed: {e}")
            return VLMOutput(
                answer=f"[Generation error: {e}]",
                raw_output="",
                generation_time_sec=0.0,
                input_token_count=input_token_count,
                output_token_count=0,
                metadata={"error": str(e), "model": self._model_name},
            )

        # Decode only new tokens
        generated_ids = output_ids[0, input_token_count:]
        raw_output = self._processor.decode(
            generated_ids, skip_special_tokens=True
        )
        answer = raw_output.strip()

        return VLMOutput(
            answer=answer,
            raw_output=raw_output,
            generation_time_sec=round(generation_time, 2),
            input_token_count=input_token_count,
            output_token_count=len(generated_ids),
            metadata={
                "model": self._model_name,
                "prompt": prompt,
            },
        )

    def caption(self, image: Image.Image) -> str:
        """Generate a clinical caption for a medical image."""
        output = self.generate(
            image=image,
            question=(
                "Describe all clinically significant findings visible "
                "in this chest X-ray image."
            ),
            max_new_tokens=256,
        )
        return output.answer

    def get_memory_footprint(self) -> Dict[str, float]:
        """Report VRAM usage."""
        vram = get_vram_usage_gb()
        return {
            "allocated_gb": vram["allocated"],
            "reserved_gb": vram["reserved"],
            "total_gb": vram["total"],
        }

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------ #
    #  Internal: build chat messages                                       #
    # ------------------------------------------------------------------ #

    def _build_messages(
        self,
        question: str,
        context: Optional[str] = None,
        query_type=None,
    ) -> List[Dict[str, Any]]:
        """
        Build Qwen2-VL chat messages with query-type-aware prompting.

        Creates a structured conversation with:
          1. System message: grounding rules
          2. User message: image + evidence + question

        The instruction suffix changes based on query_type:
          - binary_clinical  → "Start with YES or NO"
          - descriptive      → "Describe all findings systematically"
          - mixed/default    → "Provide a detailed clinical answer"

        Args:
            question:   The clinical question.
            context:    Optional evidence summary from the aggregator.
            query_type: QueryType enum from the classifier.

        Returns:
            List of message dicts for the chat template.
        """
        messages = [
            {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
        ]

        # Build user message content (multimodal: image + text)
        user_content = []

        # Image placeholder (processor handles actual injection)
        user_content.append({"type": "image", "image": "placeholder"})

        # Select instruction suffix based on query type
        instruction = self._get_instruction_for_query_type(query_type)

        # Evidence block (if available)
        if context:
            text_parts = [
                "RETRIEVED EVIDENCE FROM SIMILAR CASES:\n",
                context,
                "\n\nQUESTION: " + question,
                "\n\n" + instruction,
            ]
        else:
            text_parts = [
                "QUESTION: " + question,
                "\n\n" + instruction,
            ]

        user_content.append({
            "type": "text",
            "text": "".join(text_parts),
        })

        messages.append({"role": "user", "content": user_content})

        return messages

    def _get_instruction_for_query_type(self, query_type) -> str:
        """
        Get the generation instruction based on query type.

        This is the key routing point — descriptive queries get
        a different instruction than binary clinical queries.
        """
        # Import here to avoid circular imports
        from src.domains.healthcare.context.query_classifier import QueryType

        if query_type == QueryType.BINARY_CLINICAL:
            return (
                "Provide your answer. Start with a direct YES or NO. "
                "Then explain your reasoning, citing specific evidence."
            )

        if query_type == QueryType.DESCRIPTIVE_IMAGE:
            return (
                "Describe all clinically significant findings visible "
                "in this image. Structure your response as:\n"
                "1. Primary findings (most significant abnormalities)\n"
                "2. Secondary findings\n"
                "3. Normal structures\n"
                "Reference the retrieved evidence where relevant."
            )

        if query_type == QueryType.MIXED_IMAGE_TEXT:
            return (
                "Provide a detailed clinical answer based on both "
                "the image and the retrieved evidence. "
                "Cite specific evidence to support your observations."
            )

        # Default / TEXT_ONLY / unknown
        return (
            "Provide a detailed clinical answer based on the image. "
            "Cite specific evidence to support your answer."
        )

    # ------------------------------------------------------------------ #
    #  Adapter support (future QLoRA fine-tuning)                          #
    # ------------------------------------------------------------------ #

    def _load_adapter(self, adapter_path: str) -> None:
        """Load a LoRA adapter on top of the base model."""
        from peft import PeftModel

        logger.info(f"  Loading LoRA adapter from: {adapter_path}")
        self._model = PeftModel.from_pretrained(
            self._model, adapter_path
        )
        logger.info("  LoRA adapter loaded successfully")

    @property
    def model(self):
        """Direct access to the underlying model."""
        return self._model

    @property
    def processor(self):
        """Direct access to the processor."""
        return self._processor
