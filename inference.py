"""
inference.py
────────────
Judge entry point for BIS RAG pipeline.

Usage (as specified by hackathon rules):
    python inference.py --input hidden_private_dataset.json --output team_results.json

Input JSON schema:
    [{"id": "...", "query": "...", "expected_standards": [...]}, ...]

Output JSON schema (strictly enforced):
    [{"id": "...", "retrieved_standards": [...], "latency_seconds": 0.0}, ...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path when script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description="BIS Standards RAG — inference script")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file with queries",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write output JSON results",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["fast", "accurate"],
        default="accurate",
        help="",
    )
    return p.parse_args()


def main():
    args = parse_args()

    import src.config as config
    if args.mode == "fast":
        config.USE_RERANKER = False
        config.FAISS_TOP_K = 15
        config.BM25_TOP_K = 5
    else:
        config.USE_RERANKER = True
        config.FAISS_TOP_K = 7
        config.BM25_TOP_K = 2

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if not isinstance(queries, list):
        sys.exit(1)

    from src.pipeline import BISPipeline
    pipeline = BISPipeline()

    results = []
    total_start = time.perf_counter()

    for i, item in enumerate(queries, 1):
        qid = item.get("id", f"Q{i:03d}")
        query = item.get("query", "")
        expected = item.get("expected_standards", [])

        if not query:
            results.append({
                "id": qid,
                "retrieved_standards": [],
                "latency_seconds": 0.0,
            })
            continue

        result = pipeline.run(query)

        normalized_ids = [sid.replace(" : ", ": ").replace("(PART", "(Part") for sid in result["retrieved_standards"]]

        result_item = {
            "id": qid,
            "retrieved_standards": normalized_ids,
            "latency_seconds": result["latency_seconds"],
        }
        
        if expected:
            result_item["expected_standards"] = expected
        
        results.append(result_item)

    total_elapsed = time.perf_counter() - total_start
    avg_latency = total_elapsed / len(queries) if queries else 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
