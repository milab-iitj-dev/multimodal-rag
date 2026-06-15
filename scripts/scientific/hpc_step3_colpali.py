"""STEP 3 — ColPali visual embedding → .npy multivectors"""
import os, gc, json, time
import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

NPY_DIR = "data/indices/multivectors"

with open("data/indices/page_metadata.json") as f:
    page_metadata = json.load(f)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU available! This step requires CUDA.")

free_gb = (torch.cuda.get_device_properties(0).total_memory
           - torch.cuda.memory_allocated()) / 1024**3
print(f"  GPU      : {torch.cuda.get_device_name(0)}")
print(f"  VRAM free: {free_gb:.1f} GB")
print(f"  Pages    : {len(page_metadata)}")

print("\n  Loading ColPali model (vidore/colpali-v1.2)...")
colpali_model = ColPali.from_pretrained(
    "vidore/colpali-v1.2",
    torch_dtype=torch.float16,
    device_map="cuda",
    low_cpu_mem_usage=True,
)
colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
colpali_model.eval()
print(f"  ✅ Loaded — VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

existing_npy = {f.replace(".npy", "") for f in os.listdir(NPY_DIR) if f.endswith(".npy")}
to_embed = [
    (k, v["image_path"]) for k, v in page_metadata.items()
    if k not in existing_npy and v.get("image_path") and os.path.exists(v["image_path"])
]

print(f"\n  Already embedded : {len(existing_npy)}")
print(f"  To embed         : {len(to_embed)}")
print(f"  Starting...\n")

embedded, errors = 0, 0
t0 = time.time()

for i, (page_key, img_path) in enumerate(to_embed):
    try:
        img   = Image.open(img_path).convert("RGB")
        batch = colpali_processor.process_images(images=[img])
        batch = {k: v.to(colpali_model.device) for k, v in batch.items()}
        with torch.no_grad():
            emb = colpali_model(**batch)
        np.save(f"{NPY_DIR}/{page_key}.npy", emb[0].cpu().float().numpy())
        embedded += 1
        del batch, emb, img
        torch.cuda.empty_cache()

        if (i + 1) % 20 == 0 or (i + 1) == len(to_embed):
            elapsed = time.time() - t0
            rate    = embedded / max(elapsed, 1)
            eta     = (len(to_embed) - (i + 1)) / max(rate, 0.01) / 60
            vram    = torch.cuda.memory_allocated() / 1024**3
            print(f"  [{i+1:3d}/{len(to_embed)}] "
                  f"embedded:{embedded} rate:{rate:.2f}p/s "
                  f"ETA:{eta:.1f}min VRAM:{vram:.1f}GB")

    except torch.cuda.OutOfMemoryError:
        print(f"  ⚠️  OOM on {page_key} — skipping")
        torch.cuda.empty_cache()
        gc.collect()
        errors += 1
    except Exception as e:
        print(f"  ❌ {page_key}: {e}")
        errors += 1
        torch.cuda.empty_cache()

del colpali_model, colpali_processor
gc.collect()
torch.cuda.empty_cache()

npy_count = len([f for f in os.listdir(NPY_DIR) if f.endswith(".npy")])
print(f"\n  ✅ ColPali embedding done:")
print(f"     Embedded : {embedded}")
print(f"     Errors   : {errors}")
print(f"     .npy files: {npy_count}")
print(f"     Time      : {(time.time()-t0)/60:.1f} min")
