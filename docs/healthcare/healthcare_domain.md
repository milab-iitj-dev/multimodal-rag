# Healthcare Domain

## Dataset: OpenI Chest X-rays

- **Source**: Indiana University Open-i collection
- **Format**: CSV (indiana_reports.csv + indiana_projections.csv) + PNG images
- **Content**: ~3,600 frontal/lateral chest X-ray images with radiology reports
- **Loader**: `src/ingestion/dicom_loader.py` → `OpenIDataset`

## Model: LLaVA-1.5-7B

- **Architecture**: Vision Transformer (CLIP ViT-L/14) + Vicuna-7B language model
- **Quantization**: 4-bit NF4 via bitsandbytes (QLoRA-ready)
- **Fine-tuning**: QLoRA on q_proj, k_proj, v_proj, o_proj (language model only)
- **VRAM**: ~5.2 GB (4-bit quantized)
- **Wrapper**: `src/generation/llava_generator.py` → `LLaVAModel`

## Retrieval: ColQwen2

- **Model**: `vidore/colqwen2-v1.0-hf` (ColPali architecture + Qwen2-VL backbone)
- **Method**: Late-interaction MaxSim scoring (per-token multi-vector embeddings)
- **Index**: Pre-built offline, loaded at query time
- **Embedder**: `src/embeddings/colqwen2_embedder.py` → `ColQwen2Embedder`
- **Retriever**: `src/retrieval/colqwen2_retriever.py` → `ColQwen2Retriever`

## Training

```bash
python scripts/train_qlora.py \
    --model-config configs/model_config.yaml \
    --data-config configs/data_config.yaml \
    --training-config configs/training_config.yaml
```

## Inference

```bash
# Phase 1 (direct VQA)
python scripts/inference.py --image path/to/xray.png

# Phase 2 (RAG-augmented)
python -m pipelines.rag_vqa --query "What does this chest X-ray show?"
```
