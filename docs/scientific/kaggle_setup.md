# Kaggle Setup Guide

Step-by-step instructions for deploying the Scientific Multimodal RAG system on Kaggle notebooks with GPU (P100, 16 GB VRAM).

---

## Prerequisites

- A [Kaggle](https://www.kaggle.com/) account
- Phone verification enabled (required for GPU access)
- The project code uploaded as a Kaggle dataset

---

## Session 1: Offline Pipeline (Index Building)

This session downloads arXiv papers, parses them into page images and text, and builds the embedding index. **Duration**: ~30-40 minutes.

### Step 1: Create a New Notebook

1. Go to [kaggle.com/notebooks](https://www.kaggle.com/notebooks)
2. Click **New Notebook**
3. In Settings → Accelerator, select **GPU P100**
4. In Settings → Language, select **Python**

### Step 2: Upload the Project Code

1. Create a new Kaggle Dataset:
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
   - Click **New Dataset**
   - Upload the entire `Scientific-Multimodal-RAG/` directory as a zip file
   - Name it `scientific-multimodal-rag`

2. Add the dataset to your notebook:
   - Click **+ Add Input** in the right panel
   - Search for `scientific-multimodal-rag`
   - Click **Add**

### Step 3: Configure the Notebook

1. Copy the contents of `kaggle/notebook-offline.py` into the notebook cells
2. Verify the paths resolve correctly — the code will auto-detect `/kaggle/working/`

### Step 4: Run the Offline Pipeline

Execute all cells in order:

1. **Install dependencies**: `pip install colpali-engine chromadb transformers bitsandbytes pdf2image marker-pdf`
2. **Set up paths**: The `resolve_paths()` function in `src/utils/config_loader.py` auto-detects Kaggle and sets the base directory to `/kaggle/working/`
3. **Run the pipeline**: This will:
   - Download 10 Vision Transformer papers from arXiv
   - Parse each PDF into page images and markdown
   - Embed all pages with ColPali (multi-vector, saved as `.npy`)
   - Embed all text chunks with SciNCL (dense, saved to ChromaDB)
   - Save metadata (`doc_mapping.json`, `page_metadata.json`)
   - Update `checkpoint.json` after each PDF

### Step 5: Save the Output

1. After the pipeline completes, verify the output:
   ```python
   import os
   for root, dirs, files in os.walk("/kaggle/working/data"):
       print(root, len(files))
   ```

2. Save the output as a new dataset version:
   - Go to **Output** tab in the notebook
   - Click **Save as Dataset**
   - Name it `scientific-multimodal-rag-index`
   - Set visibility to **Public** or **Private**

### Troubleshooting

| Issue | Solution |
|---|---|
| OOM during ColPali embedding | Reduce `max_pages_per_batch` to 2 in `configs/model_config.yaml` |
| pdf2image not found | Install poppler: `!apt-get install -y poppler-utils` |
| marker-pdf fails | Falls back to PyMuPDF automatically; install with `pip install pymupdf` |
| Session timeout | The pipeline checkpoints after each PDF; use `pipeline.resume()` to continue |

---

## Session 2: Online Pipeline (Query Demo)

This session loads the pre-built index and provides a Gradio demo for querying. **Duration**: ~5-12 seconds per query.

### Step 1: Create a New Notebook

1. Create a new Kaggle Notebook with **GPU P100**
2. Add the following as inputs:
   - The `scientific-multimodal-rag` code dataset (from Session 1 Step 2)
   - The `scientific-multimodal-rag-index` output dataset (from Session 1 Step 5)

### Step 2: Configure the Notebook

1. Copy the contents of `kaggle/notebook-online.py` into the notebook cells
2. The code will:
   - Copy the index data from the input dataset to `/kaggle/working/data/`
   - Initialise the OnlinePipeline with the pre-built index
   - Launch a Gradio demo with `share=True`

### Step 3: Run the Demo

1. Execute all cells
2. The Gradio interface will provide a public URL
3. Ask questions about Vision Transformer papers
4. Each query takes ~5-12 seconds (encode → retrieve → generate → check)

### Step 4: Interact

Example queries to try:
- "What is the Vision Transformer?"
- "How does the patch embedding work in ViT?"
- "What is the difference between ViT and DeiT?"
- "Explain the attention mechanism in Vision Transformers"
- "What does Figure 3 in the ViT paper show?"

---

## Path Resolution on Kaggle

The `resolve_paths()` function in `src/utils/config_loader.py` automatically detects the Kaggle environment:

```python
# If /kaggle/working/ exists → Kaggle environment
kaggle_working = Path("/kaggle/working/")
if kaggle_working.exists():
    base_dir = kaggle_working
else:
    base_dir = project_root
```

All data paths in `configs/data_config.yaml` are resolved relative to this base directory:

| Config Key | Local Path | Kaggle Path |
|---|---|---|
| `raw_pdfs` | `data/raw/` | `/kaggle/working/data/raw/` |
| `parsed_pages` | `data/parsed/pages/` | `/kaggle/working/data/parsed/pages/` |
| `chroma_index` | `data/indices/chroma_index/` | `/kaggle/working/data/indices/chroma_index/` |
| `multivectors` | `data/indices/multivectors/` | `/kaggle/working/data/indices/multivectors/` |

---

## Important Notes

1. **Session Timeout**: Kaggle notebooks have a 12-hour runtime limit. The offline pipeline takes ~30-40 minutes for 10 papers, well within limits.

2. **Output Persistence**: Kaggle notebook output is saved as a dataset. Make sure to save the `data/` directory as output before the session ends.

3. **GPU Quota**: Kaggle provides 30 hours/week of GPU time. The offline pipeline uses ~1 hour; the online demo uses GPU time proportional to the number of queries.

4. **Internet Access**: Kaggle notebooks have internet access by default. This is needed for downloading model weights from Hugging Face Hub on first use.

5. **Model Caching**: Hugging Face caches models in `~/.cache/huggingface/`. This does not persist across sessions, so models are re-downloaded each time.
