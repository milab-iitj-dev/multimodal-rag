# 💻 Running Scientific Multimodal RAG Locally

This guide explains how to configure, ingest papers, and run the Q&A Streamlit application on your local machine (Windows/Mac/Linux).

---

## 🛠️ Step 1: Environment Setup

### 1. Create a Python Virtual Environment
We isolate dependencies to prevent version conflicts with your global Python installations.
```bash
py -m venv .venv
```
* **Why `py`?** On Windows, `py` is the official Python Launcher that automatically detects your installed Python versions. If you are on Mac/Linux, replace `py` with `python3`.

### 2. Activate the Virtual Environment
```bash
# For Windows Git Bash:
source .venv/Scripts/activate

# For Windows Command Prompt:
.venv\Scripts\activate

# For Mac/Linux:
source .venv/bin/activate
```
* **Why activate?** This redirects your shell to use the Python interpreter and package binaries inside the `.venv` directory, keeping your global environment clean.

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```
* **Why?** This installs PyTorch, SentenceTransformers, ChromaDB, Streamlit, and other required packages.

---

## 🏃 Step 2: Running the System

We use `main.py` as a single orchestrator entry point.

### 📥 1. Ingestion Mode (Offline Ingestion)
Before asking questions, you must download the papers and create the vector database index.

```bash
py main.py --mode offline
```
* **What this does:**
  1. Downloads research papers from arXiv (defined under the `papers` list in `configs/config.yaml`).
  2. Parses the PDF pages into high-resolution images and extracts raw text lines.
  3. Uses **ColPali** to generate multi-vector visual embeddings (saved as `.npy` files).
  4. Uses **SciNCL** to generate text embeddings and indexes them inside a local **ChromaDB** database.
  5. Packages the database files into `.zip` archives.
* **Why run this first?** The online Q&A app cannot answer questions without pre-computed visual and textual vector representations of the research papers.

---

### 🌐 2. Web Interface Mode (Online Q&A)
Once offline ingestion completes, start the Streamlit browser application:

```bash
py main.py --mode online
```
* **What this does:**
  1. Detects your local hardware capability (GPU/CPU).
  2. Loads the pre-computed ColPali and SciNCL indexes from disk.
  3. Launches a local web server hosting the Streamlit interface.
* **Accessing the UI:** Visit **`http://localhost:8501`** in your browser.
* **Why run this?** This provides a clean interface to submit natural language queries, visualize retrieved page images, and view cited paper answers.

---

### ⚡ 3. Sequential Execution (Full Mode)
To run both the ingestion and start the app sequentially:
```bash
py main.py --mode full
```

---

## 💡 Hardware & Performance Notes

* **GPU vs CPU**: The code automatically checks `torch.cuda.is_available()`. If you have a CUDA-capable NVIDIA card, it will use GPU (fast). If not, it falls back to CPU (slow, but works).
* **VRAM Efficiency**: Models are loaded one-by-one and unloaded immediately after use. This staggering strategy keeps peak VRAM usage under **~2.5 GB**, letting the pipeline run on normal laptops.
