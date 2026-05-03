from __future__ import annotations

import pickle
from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever

from src.config import (
    BM25_INDEX_PATH,
    BM25_TOP_K,
    EMBED_MODEL,
    ENSEMBLE_WEIGHTS,
    FAISS_INDEX_PATH,
    FAISS_TOP_K,
)


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_ensemble_retriever() -> EnsembleRetriever:
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. Run setup_index.py first."
        )
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_INDEX_PATH}. Run setup_index.py first."
        )

    embeddings = _get_embeddings()

    faiss_store = FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": FAISS_TOP_K})

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_retriever: BM25Retriever = pickle.load(f)
    bm25_retriever.k = BM25_TOP_K

    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=ENSEMBLE_WEIGHTS,
    )
