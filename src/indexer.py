from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    BM25_INDEX_PATH,
    EMBED_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)


def _build_page_content(meta: Dict[str, Any]) -> str:
    parts = [
        meta.get("standard_id", ""),
        meta.get("title", ""),
        meta.get("scope", ""),
        " ".join(meta.get("keywords", [])),
        meta.get("material", ""),
    ]
    return "\n".join(p for p in parts if p)


def _metadata_to_document(meta: Dict[str, Any]) -> Document:
    return Document(
        page_content=_build_page_content(meta),
        metadata={
            "standard_id": meta.get("standard_id", ""),
            "title": meta.get("title", ""),
            "material": meta.get("material", ""),
        },
    )


def build_index(
    extracted_metadata: List[Dict[str, Any]],
    verbose: bool = True,
) -> None:
    seen: Dict[str, Dict[str, Any]] = {}
    for meta in extracted_metadata:
        sid = meta.get("standard_id", "UNKNOWN")
        if sid != "UNKNOWN":
            seen[sid] = meta

    unique_meta = list(seen.values())

    if verbose:
        print(f"[indexer] Building index for {len(unique_meta)} unique standards")

    docs = [_metadata_to_document(m) for m in unique_meta]

    if verbose:
        print(f"[indexer] Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if verbose:
        print("[indexer] Building FAISS vectorstore...")
    faiss_store = FAISS.from_documents(docs, embeddings)
    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    faiss_store.save_local(str(FAISS_INDEX_PATH))
    if verbose:
        print(f"[indexer] FAISS saved → {FAISS_INDEX_PATH}")

    if verbose:
        print("[indexer] Building BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
    if verbose:
        print(f"[indexer] BM25 saved → {BM25_INDEX_PATH}")

    metadata_out = {m["standard_id"]: m for m in unique_meta}
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_out, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[indexer] Metadata saved → {METADATA_PATH}")
        print(f"[indexer] Indexing complete. {len(unique_meta)} standards indexed.")


def load_metadata() -> Dict[str, Dict[str, Any]]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found at {METADATA_PATH}. Run setup_index.py first."
        )
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
