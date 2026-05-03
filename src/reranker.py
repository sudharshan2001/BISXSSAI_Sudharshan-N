from __future__ import annotations

import json
import re
from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import (
    RERANK_TOP_N,
    VLM_MODEL,
    VLLM_API_KEY,
    VLLM_BASE_URL,
)

RERANK_SYSTEM = (
    "You are an expert on Bureau of Indian Standards (BIS). "
    "Given a product query and a list of candidate IS standards, "
    "return the IDs of the most relevant standards ranked from most to least relevant."
)

RERANK_TEMPLATE = """\
Query: {query}

Candidates:
{candidates}

Rank by relevance. Return JSON array of IDs only, no explanation.
Example: ["IS 269 : 1989", "IS 8112 : 1989"]
"""


def _format_candidates(docs: List[Document]) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        sid = doc.metadata.get("standard_id", "?")
        title = doc.metadata.get("title", "")
        lines.append(f"{i}. {sid} — {title}")
    return "\n".join(lines)


def _parse_ranked_ids(raw: str, fallback_docs: List[Document]) -> List[str]:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw.strip(), flags=re.MULTILINE)
    raw = raw.strip()

    try:
        ranked = json.loads(raw)
        if isinstance(ranked, list) and all(isinstance(x, str) for x in ranked):
            return ranked
    except json.JSONDecodeError:
        pass

    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            ranked = json.loads(m.group(0))
            if isinstance(ranked, list):
                return [str(x) for x in ranked]
        except json.JSONDecodeError:
            pass

    return [doc.metadata.get("standard_id", "") for doc in fallback_docs]


class Reranker:
    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            model=VLM_MODEL,
            max_tokens=512,
            temperature=0.0,
        )

    def rerank(self, query: str, docs: List[Document]) -> List[str]:
        if not docs:
            return []

        candidates_text = _format_candidates(docs)
        prompt = RERANK_TEMPLATE.format(query=query, candidates=candidates_text)
        messages = [("system", RERANK_SYSTEM), ("human", prompt)]

        try:
            response = self._llm.invoke(messages)
            raw = response.content or ""
        except Exception:
            return [doc.metadata.get("standard_id", "") for doc in docs][:RERANK_TOP_N]

        ranked_ids = _parse_ranked_ids(raw, fallback_docs=docs)
        return ranked_ids[:RERANK_TOP_N]
