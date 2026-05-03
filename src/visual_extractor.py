from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from PIL import Image
from tqdm.asyncio import tqdm as atqdm

from src.config import (
    EXTRACTION_BATCH_SIZE,
    VLM_MAX_TOKENS,
    VLM_MODEL,
    VLLM_API_KEY,
    VLLM_BASE_URL,
)
from src.pdf_to_images import StandardPageGroup

_IS_PATTERN = re.compile(
    r"IS\s+\d+(?:\s*\(\s*Part\s*\d+\s*\))?\s*:\s*\d{4}",
    re.IGNORECASE,
)

EXTRACTION_PROMPT = """\
You are analyzing a page from BIS SP 21:2005 (Summaries of Indian Standards for Building Materials).

Extract the following fields and return ONLY valid JSON — no markdown fences, no explanation:

{
  "standard_id": "IS XXXX : YYYY",
  "title": "Full official title of the standard",
  "scope": "1-3 sentence description of what this standard covers",
  "keywords": ["keyword1", "keyword2", ...],
  "material": "primary material category (e.g. cement, steel, concrete, aggregate, brick)"
}

Rules:
- standard_id must follow exactly "IS XXXX : YYYY" (spaces around colon).
- For multi-part standards use "IS XXXX (Part N) : YYYY".
- If multiple standards appear on the page, extract only the PRIMARY one (in the header).
- Respond with raw JSON only.
"""


def _pil_to_base64(img: Image.Image, max_width: int = 1600) -> str:
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_messages(images: List[Image.Image]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for img in images:
        b64 = _pil_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": EXTRACTION_PROMPT})
    return [{"role": "user", "content": content}]


def _parse_vlm_response(raw: str, fallback_id: str) -> Dict[str, Any]:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw.strip(), flags=re.MULTILINE)
    raw = raw.strip()

    try:
        data = json.loads(raw)
        data.setdefault("standard_id", fallback_id)
        data.setdefault("title", "")
        data.setdefault("scope", "")
        data.setdefault("keywords", [])
        data.setdefault("material", "")
        return data
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                data.setdefault("standard_id", fallback_id)
                data.setdefault("title", "")
                data.setdefault("scope", "")
                data.setdefault("keywords", [])
                data.setdefault("material", "")
                return data
            except json.JSONDecodeError:
                pass

    return {
        "standard_id": fallback_id,
        "title": "",
        "scope": "",
        "keywords": [],
        "material": "",
        "_extraction_failed": True,
    }


async def _extract_one(
    client: AsyncOpenAI,
    group: StandardPageGroup,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        messages = _build_messages(group.images)
        try:
            response = await client.chat.completions.create(
                model=VLM_MODEL,
                messages=messages,
                max_tokens=VLM_MAX_TOKENS,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            return {
                "standard_id": group.standard_id_guess,
                "title": group.title_guess,
                "scope": "",
                "keywords": [],
                "material": "",
                "_extraction_failed": True,
                "_error": str(exc),
            }

    result = _parse_vlm_response(raw, fallback_id=group.standard_id_guess)
    if not _IS_PATTERN.match(result.get("standard_id", "")):
        result["standard_id"] = group.standard_id_guess
    return result


async def extract_all_async(
    groups: List[StandardPageGroup],
    verbose: bool = True,
    batch_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    concurrency = batch_size if batch_size is not None else EXTRACTION_BATCH_SIZE
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [_extract_one(client, g, semaphore) for g in groups]

    if verbose:
        results = await atqdm.gather(*tasks, desc="Extracting standards")
    else:
        results = await asyncio.gather(*tasks)

    await client.close()

    failed = sum(1 for r in results if r.get("_extraction_failed"))
    if verbose and failed:
        print(f"[visual_extractor] {failed}/{len(results)} extractions used fallback")

    return list(results)


def extract_all(
    groups: List[StandardPageGroup],
    verbose: bool = True,
    batch_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    return asyncio.run(extract_all_async(groups, verbose=verbose, batch_size=batch_size))
