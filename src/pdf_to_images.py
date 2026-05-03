from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz
from PIL import Image
from tqdm import tqdm

_IS_PATTERN = re.compile(
    r"IS\s+(\d+(?:\s*\(\s*Part\s*\d+\s*\))?)\s*:\s*(\d{4})",
    re.IGNORECASE,
)


@dataclass
class StandardPageGroup:
    standard_id_guess: str
    title_guess: str
    page_indices: List[int]
    images: List[Image.Image] = field(default_factory=list)


def _normalize_standard_id(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"\s*:\s*", " : ", raw)
    raw = re.sub(r"IS\s+", "IS ", raw)
    raw = re.sub(r"\(\s*[Pp]art\s*(\d+)\s*\)", r"(Part \1)", raw)
    return raw


def _page_to_image(page: fitz.Page, scale: float = 2.0) -> Image.Image:
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _extract_title_from_text(text: str) -> str:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        if "SUMMARY OF" in ln.upper():
            if i + 1 < len(lines):
                return lines[i + 1]
    return ""


def load_pdf_as_standard_groups(
    pdf_path: str | Path,
    scale: float = 2.0,
    verbose: bool = True,
) -> List[StandardPageGroup]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    if verbose:
        print(f"[pdf_to_images] Opened '{pdf_path.name}' — {total_pages} pages")

    groups: List[StandardPageGroup] = []
    current_group: Optional[StandardPageGroup] = None

    page_iter = tqdm(range(total_pages), desc="Scanning pages", disable=not verbose)

    for page_idx in page_iter:
        page = doc[page_idx]
        text = page.get_text()
        is_new_standard = "SUMMARY OF" in text.upper()
        match = _IS_PATTERN.search(text)

        if is_new_standard and match:
            if current_group is not None:
                groups.append(current_group)
            raw_id = match.group(0)
            standard_id = _normalize_standard_id(raw_id)
            title = _extract_title_from_text(text)
            current_group = StandardPageGroup(
                standard_id_guess=standard_id,
                title_guess=title,
                page_indices=[page_idx],
            )
        elif current_group is not None:
            current_group.page_indices.append(page_idx)

    if current_group is not None:
        groups.append(current_group)

    for group in groups:
        page = doc[group.page_indices[0]]
        group.images = [_page_to_image(doc[i], scale) for i in group.page_indices]

    if verbose:
        print(f"[pdf_to_images] Found {len(groups)} standard groups")

    doc.close()
    return groups
