import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datum"
INDEX_DIR = BASE_DIR / "indexes"
DATA_OUT_DIR = BASE_DIR / "data"

PDF_PATH = DATA_DIR / "dataset.pdf"
PUBLIC_TEST_PATH = DATA_DIR / "public_test_set.json"

FAISS_INDEX_PATH = INDEX_DIR / "faiss_index"
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"
METADATA_PATH = INDEX_DIR / "standards_metadata.json"

VLM_MODEL = "Qwen/Qwen3-VL-4B-Instruct-FP8"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "not-needed")

USE_RERANKER = True

FAISS_TOP_K = 7
BM25_TOP_K = 2
ENSEMBLE_WEIGHTS = [1, 0]
RERANK_TOP_N = 5

PDF_RENDER_DPI_SCALE = 2.0
EXTRACTION_BATCH_SIZE = 1
VLM_MAX_TOKENS = 512

INDEX_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
