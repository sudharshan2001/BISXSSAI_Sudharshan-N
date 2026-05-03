from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from openai import OpenAI

from src.retriever import get_ensemble_retriever
from src.reranker import Reranker
from src.config import USE_RERANKER, VLM_MODEL, VLLM_API_KEY, VLLM_BASE_URL


def _translate_to_english(query: str) -> Tuple[str, bool]:
    """Detect language and translate to English if needed.
    Returns (translated_query, was_translated).
    """
    try:
        from langdetect import detect, LangDetectException
        try:
            lang = detect(query)
        except LangDetectException:
            lang = "en"
    except ImportError:
        lang = "en"

    if lang == "en":
        return query, False

    try:
        client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Translate the following text to English. "
                    "Return only the translated text, no explanation.\n\n"
                    + query
                ),
            }],
            max_tokens=256,
            temperature=0.0,
        )
        translated = (response.choices[0].message.content or query).strip()
        return translated, True
    except Exception:
        return query, False


class BISPipeline:
    def __init__(self) -> None:
        self._retriever = get_ensemble_retriever()
        self._reranker = Reranker() if USE_RERANKER else None

    def run(self, query: str) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # 1. Translate to English if needed
        query, was_translated = _translate_to_english(query)
        if was_translated:
            print(f"[pipeline] Translated query: {query}")

        # 2. Retrieve hybrid candidates
        candidate_docs = self._retriever.invoke(query)

        # 3. Rerank with vLLM if enabled
        if self._reranker and candidate_docs:
            try:
                ranked_ids = self._reranker.rerank(query, candidate_docs[:7])
            except Exception:
                ranked_ids = [doc.metadata.get("standard_id", "") for doc in candidate_docs]
        else:
            ranked_ids = [doc.metadata.get("standard_id", "") for doc in candidate_docs]

        # 4. Normalize format
        ranked_ids = [sid.replace(" : ", ": ").replace("(PART", "(Part") for sid in ranked_ids]

        return {
            "retrieved_standards": ranked_ids,
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }
