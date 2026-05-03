import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    EXTRACTION_BATCH_SIZE,
    INDEX_DIR,
    METADATA_PATH,
    PDF_PATH,
    PDF_RENDER_DPI_SCALE,
)
from src.indexer import build_index
from src.pdf_to_images import load_pdf_as_standard_groups
from src.visual_extractor import extract_all


def parse_args():
    p = argparse.ArgumentParser(description="Build BIS RAG indexes from dataset PDF")
    p.add_argument("--pdf", type=str, default=str(PDF_PATH))
    p.add_argument("--scale", type=float, default=PDF_RENDER_DPI_SCALE)
    p.add_argument("--batch", type=int, default=EXTRACTION_BATCH_SIZE)
    p.add_argument("--skip-extraction", action="store_true",
                   help="Re-index using existing metadata (skip VLM extraction)")
    p.add_argument("--save-metadata", type=str, default=None,
                   help="Save raw extracted metadata to this path for inspection")
    return p.parse_args()


def main():
    args = parse_args()

    if args.skip_extraction:
        if not METADATA_PATH.exists():
            print(f"[setup_index] ERROR: No metadata at {METADATA_PATH}")
            sys.exit(1)
        print(f"[setup_index] Loading existing metadata from {METADATA_PATH}")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)
        extracted = list(metadata_dict.values())
    else:
        print("=" * 60)
        print("Step 1/3  PDF → page image groups")
        print("=" * 60)
        groups = load_pdf_as_standard_groups(pdf_path=args.pdf, scale=args.scale, verbose=True)
        if not groups:
            print("[setup_index] ERROR: No standard groups extracted from PDF.")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("Step 2/3  Visual extraction via Qwen3-VL")
        print("=" * 60)
        extracted = extract_all(groups, verbose=True, batch_size=args.batch)

        if args.save_metadata:
            Path(args.save_metadata).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_metadata, "w", encoding="utf-8") as f:
                json.dump(extracted, f, indent=2, ensure_ascii=False)
            print(f"[setup_index] Raw metadata saved → {args.save_metadata}")

    print("\n" + "=" * 60)
    print("Step 3/3  Building FAISS + BM25 indexes")
    print("=" * 60)
    build_index(extracted, verbose=True)

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("  Run: python inference.py --input datum/public_test_set.json --output data/results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
