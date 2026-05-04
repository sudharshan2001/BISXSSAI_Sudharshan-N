**End-to-End BIS standards pipeline using Qwen3-VL with multilingual retrieval and generation for accurate, context-aware recommendations.**

**Models**:
- `Qwen3-VL-4B-Instruct-FP8` — visual extraction + LLM reranking (via vLLM)
- `BAAI/bge-small-en-v1.5` — semantic embeddings

---

## Repository Structure

```
BIS_submission/
├── src/
│   ├── config.py           # All paths, model IDs, hyperparameters
│   ├── pdf_to_images.py    # PDF → per-standard page groups
│   ├── visual_extractor.py # Qwen3-VL metadata extraction via vLLM
│   ├── indexer.py          # FAISS + BM25 index builder
│   ├── retriever.py        # Hybrid EnsembleRetriever
│   ├── reranker.py         # LLM reranker
│   └── pipeline.py         # End-to-end RAG pipeline
├── app/
│   └── gradio_ui.py        # Demo web UI
├── data/
│   └── public_results.json # Results on public test set
├── datum/                  # Put dataset.pdf and public_test_set.json here
├── inference.py            # Judge entry point
├── setup_index.py          # One-time ingestion pipeline
├── eval_script.py          # Official evaluation script
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

**Recommended: conda environment**

```bash
conda create -n bis_rag python=3.10 -y
conda activate bis_rag
pip install -r requirements.txt
```

**GPU requirements**: NVIDIA GPU with ≥8GB VRAM (tested on RTX 4080 Super 16GB, CUDA 12.4+)

> Note: vLLM installs its own compatible PyTorch version. If you see NCCL errors, run:
> `pip install "nvidia-nccl-cu12>=2.22"`

### 2. Place dataset

```
datum/
├── dataset.pdf          ← BIS SP 21:2005 PDF
└── public_test_set.json ← provided test queries
```

### 3. Start the vLLM server

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct-FP8 --port 8000 --trust-remote-code --gpu-memory-utilization 0.95 --dtype auto --max-model-len 16384
```

reduce max-model-len and gpu-memory-utilization based on the GPU you haev

Wait until you see `Application startup complete` in the terminal.

> If using a locally downloaded model, pass the folder path instead of the HuggingFace ID.

### 4. Build the indexes (one-time)

```bash
python setup_index.py --pdf datum/dataset.pdf
```

This runs visual extraction on all 545 standards via Qwen3-VL and builds FAISS + BM25 indexes. Takes ~3–5 minutes on an RTX 4080.

---

## Running Inference

### Fast Mode

```bash
python inference.py --input datum/public_test_set.json --output data/results_fast.json --mode fast
```

**Evaluate results:**

```bash
python datum/eval_script.py --results data/results_fast.json
```

### Accurate Mode

```bash
python inference.py --input datum/public_test_set.json --output data/results_accurate.json --mode accurate
```

**Evaluate results:**

```bash
python datum/eval_script.py --results data/results_accurate.json
```

---

## Demo UI

```bash
python app/gradio_ui.py --port 7860
```

---

## Evaluation Results (Public Test Set)

| Metric       | Score  | Target |
|-------------|--------|--------|
| Hit Rate @3 | 100%   | >80%   |
| MRR @5      | 1.0000 | >0.7   |
| Avg Latency | 0.59s  | <5s    |

---

## How It Works

1. **Ingestion** (`setup_index.py`): The BIS SP 21 PDF is parsed page by page. Pages starting with "SUMMARY OF" are identified as new standard entries. Pages are rendered as images and sent to Qwen3-VL to extract structured metadata: `standard_id`, `title`, `scope`, `keywords`, `material`.

2. **Indexing**: Extracted metadata is embedded with `BAAI/bge-small-en-v1.5` into a FAISS vector store. A BM25 index is built on the same text for keyword matching. Both are persisted to disk.

3. **Retrieval**: At query time, an `EnsembleRetriever` fuses FAISS (semantic) and BM25 (keyword) results. The top candidates are passed to Qwen3-VL via vLLM for LLM-based reranking.

4. **Output**: The top-5 ranked IS standard IDs are returned with per-query latency.

---

## Environment Notes

- Python 3.10
- CUDA 12.4+ (tested with driver 580.x / CUDA 13.0)
- vLLM serves Qwen3-VL on `http://localhost:8000/v1` (OpenAI-compatible)
- `VLLM_BASE_URL` and `VLLM_API_KEY` can be overridden via environment variables
- `langchain-classic` package provides `EnsembleRetriever` (not in langchain-community)
