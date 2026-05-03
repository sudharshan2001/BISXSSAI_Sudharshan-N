from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexer import load_metadata
from src.pipeline import BISPipeline

_pipeline: BISPipeline | None = None
_metadata: dict | None = None


def _get_pipeline() -> BISPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = BISPipeline()
    return _pipeline


def _get_metadata() -> dict:
    global _metadata
    if _metadata is None:
        _metadata = load_metadata()
    return _metadata


def query_standards(query: str) -> tuple[str, str, str]:
    query = query.strip()
    if not query:
        return "Please enter a product description.", "", ""

    pipeline = _get_pipeline()
    metadata = _get_metadata()

    from src.pipeline import _translate_to_english
    translated_query, was_translated = _translate_to_english(query)
    translation_notice = f"🌐 Translated to English: *{translated_query}*" if was_translated else ""

    result = pipeline.run(translated_query)
    ids = result["retrieved_standards"]
    latency = result["latency_seconds"]

    if not ids:
        return "No standards found for this query.", f"Latency: {latency:.2f}s", translation_notice

    lines = []
    for rank, sid in enumerate(ids, 1):
        meta = metadata.get(sid, {})
        title = meta.get("title", " ")
        scope = meta.get("scope", "")
        material = meta.get("material", "")
        lines.append(f"### {rank}. `{sid}`")
        lines.append(f"**{title}**")
        if material:
            lines.append(f"*Category: {material}*")
        if scope:
            lines.append(f"\n{scope}")
        lines.append("")

    return "\n".join(lines), f"Latency: {latency:.2f}s", translation_notice


EXAMPLES = [
    ["We manufacture 33 Grade Ordinary Portland Cement. Which BIS standard applies?"],
    ["Coarse and fine aggregates from natural sources for use in structural concrete."],
    ["Hollow and solid lightweight concrete masonry blocks — dimensions and physical requirements."],
    ["Portland slag cement — chemical and physical requirements."],
    ["Portland pozzolana cement (calcined clay based) — applicable standard."],
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="BIS Standards Recommendation Engine", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# BIS Standards Recommendation Engine\n"
            "**AI-powered BIS Standard discovery for Micro & Small Enterprises**\n\n"
            "Enter a product description to get relevant Bureau of Indian Standards (IS) recommendations."
        )

        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Product / Compliance Query",
                    placeholder="e.g. We manufacture 33 Grade Ordinary Portland Cement...",
                    lines=4,
                )
                with gr.Row():
                    submit_btn = gr.Button("Find Standards", variant="primary")
                    clear_btn = gr.Button("Clear")
                gr.Examples(examples=EXAMPLES, inputs=query_input, label="Example Queries")

            with gr.Column(scale=3):
                translation_output = gr.Markdown(label="")
                results_output = gr.Markdown(label="Recommended Standards")
                latency_output = gr.Textbox(label="", interactive=False, show_label=False, max_lines=1)

        submit_btn.click(fn=query_standards, inputs=query_input, outputs=[results_output, latency_output, translation_output])
        clear_btn.click(fn=lambda: ("", "", ""), inputs=None, outputs=[results_output, latency_output, translation_output])
        query_input.submit(fn=query_standards, inputs=query_input, outputs=[results_output, latency_output, translation_output])

    return demo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)
