"""
QLoRA Training Script — Local GPU Version.

Fine-tunes LLaVA-1.5-7B with QLoRA on OpenI medical VQA data.

Usage:
    python scripts/train_local.py \
        --model-config configs/model_config.yaml \
        --data-config configs/data_config.yaml \
        --training-config configs/training_config.yaml
"""

import argparse
import sys
import yaml
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.domains.healthcare.ingestion.dicom_loader import OpenIDataset
from src.domains.healthcare.ingestion.preprocessing import clean_report_text, MedicalImagePreprocessor
from src.domains.healthcare.generation.model_factory import create_model
from src.shared.device import print_gpu_status
from src.shared.logging_utils import setup_logger

logger = setup_logger("training")


class MedicalVQACollator:
    """Tokenizes image+question+answer into trainer-ready batches.

    IMPORTANT: Do NOT truncate tokenized sequences — LLaVA expands <image>
    to ~576 visual tokens. Truncation breaks image/text alignment.
    Pre-truncate the answer text instead.
    """

    def __init__(self, processor, max_answer_words=128):
        self.processor = processor
        self.max_answer_words = max_answer_words

    def _truncate_answer(self, answer):
        words = answer.split()
        if len(words) > self.max_answer_words:
            return " ".join(words[:self.max_answer_words])
        return answer

    def __call__(self, batch):
        images, texts = [], []
        for s in batch:
            images.append(s["image"])
            answer = self._truncate_answer(s["answer"])
            texts.append(f"USER: <image>\n{s['question']}\nASSISTANT: {answer}")

        enc = self.processor(text=texts, images=images, return_tensors="pt",
                             padding=True)
        labels = enc["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc


def build_hf_dataset(openi_dataset, indices):
    prep = MedicalImagePreprocessor()
    samples = []
    for idx in indices:
        s = openi_dataset[idx]
        if s.image is None or not s.answer:
            continue
        samples.append({"image": prep(s.image), "question": s.question,
                        "answer": clean_report_text(s.answer)})
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--data-config", default="configs/data_config.yaml")
    parser.add_argument("--training-config", default="configs/training_config.yaml")
    args = parser.parse_args()

    with open(args.model_config) as f:  model_config = yaml.safe_load(f)
    with open(args.data_config) as f:   data_config = yaml.safe_load(f)
    with open(args.training_config) as f: train_config = yaml.safe_load(f)
    tcfg = train_config["training"]

    info = print_gpu_status(logger)
    if not info["cuda_available"]:
        logger.error("No GPU. QLoRA requires CUDA."); sys.exit(1)

    from src.shared.config_loader import resolve_data_paths
    data_config = resolve_data_paths(data_config)

    ds = data_config["dataset"]
    dataset = OpenIDataset(ds["images_dir"], ds["reports_dir"], ds.get("max_samples"))
    dataset.load()
    splits = dataset.get_splits(ds["split"]["train"], ds["split"]["val"], ds.get("seed",42))
    train_data = build_hf_dataset(dataset, splits["train"])
    val_data   = build_hf_dataset(dataset, splits["val"])
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    model_wrapper = create_model(model_config)
    model_wrapper.load(model_config)
    model_wrapper.prepare_for_training()

    from transformers import TrainingArguments, Trainer
    training_args = TrainingArguments(
        output_dir=tcfg["output_dir"], num_train_epochs=tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"],
        warmup_ratio=tcfg["warmup_ratio"], lr_scheduler_type=tcfg["lr_scheduler_type"],
        logging_steps=tcfg["logging_steps"], save_steps=tcfg["save_steps"],
        eval_steps=tcfg["eval_steps"], eval_strategy="steps",
        save_total_limit=tcfg["save_total_limit"],
        fp16=tcfg["fp16"], bf16=tcfg["bf16"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        dataloader_num_workers=tcfg["dataloader_num_workers"],
        remove_unused_columns=False, report_to=tcfg["report_to"],
        seed=tcfg["seed"], load_best_model_at_end=True, metric_for_best_model="eval_loss",
    )
    collator = MedicalVQACollator(model_wrapper.processor)
    trainer = Trainer(model=model_wrapper.model, args=training_args,
                      train_dataset=train_data, eval_dataset=val_data, data_collator=collator)

    logger.info("=" * 50 + "\nSTARTING QLORA TRAINING\n" + "=" * 50)
    trainer.train()

    final_dir = f"{tcfg['output_dir']}/final_adapter"
    model_wrapper.save_adapter(final_dir)
    logger.info(f"Done. Adapter saved: {final_dir}")

if __name__ == "__main__":
    main()
