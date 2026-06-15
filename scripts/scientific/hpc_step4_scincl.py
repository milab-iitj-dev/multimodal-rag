"""STEP 4 — SciNCL text embedding → ChromaDB"""
import gc, json, time
import torch
from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_DIR = "data/indices/chroma_index"

with open("data/indices/page_metadata.json") as f:
    page_metadata = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")
print(f"  Loading SciNCL (malteos/scincl)...")
scincl = SentenceTransformer("malteos/scincl", device=device)
if device == "cuda":
    print(f"  VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"  ✅ SciNCL loaded")

client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    client.delete_collection("sci_rag_pages")
    print("  Deleted existing collection")
except Exception:
    pass

collection = client.create_collection("sci_rag_pages", metadata={"hnsw:space": "cosine"})
print(f"  ✅ ChromaDB collection created")

texts, metas, ids = [], [], []
for pk, meta in page_metadata.items():
    t = meta.get("text", "")
    if t and len(t.strip()) >= 20:
        texts.append(t[:512])
        metas.append({
            "doc_id": meta["doc_id"],
            "page_num": str(meta["page_num"]),
            "paper_title": meta.get("paper_title", meta["doc_id"]),
        })
        ids.append(pk)

print(f"\n  Pages with valid text: {len(texts)}")
print(f"  Pages skipped (empty): {len(page_metadata) - len(texts)}")
print(f"  Embedding...\n")

BATCH = 32
count = 0
t0 = time.time()

for i in range(0, len(texts), BATCH):
    bt = texts[i:i+BATCH]
    bi = ids[i:i+BATCH]
    bm = metas[i:i+BATCH]
    embs = scincl.encode(bt, show_progress_bar=False, batch_size=BATCH, convert_to_numpy=True)
    collection.upsert(ids=bi, embeddings=embs.tolist(),
                      documents=[t[:500] for t in bt], metadatas=bm)
    count += len(bi)
    rate = count / max(time.time() - t0, 1)
    eta  = (len(texts) - count) / max(rate, 0.1) / 60
    print(f"  [{count:3d}/{len(texts)}] rate:{rate:.1f}p/s ETA:{eta:.1f}min")

del scincl
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"\n  ✅ SciNCL + ChromaDB done:")
print(f"     Embedded  : {count} pages")
print(f"     Collection: {collection.count()} entries")
print(f"     Time      : {(time.time()-t0)/60:.1f} min")
