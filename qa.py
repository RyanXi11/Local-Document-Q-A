"""CLI for asking questions about the vendored Alice in Wonderland text."""

from __future__ import annotations

import argparse
import sys

from src.generate import answer
from src.ingest import Chunk
from src.ollama_client import EMBED_MODEL, OllamaError
from src.retrieve import load_or_build_index, search

DOCUMENT_TITLE = "Alice's Adventures in Wonderland"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions about Alice's Adventures in Wonderland using local Ollama RAG.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Ask one question and exit. Omit to enter interactive mode.",
    )
    return parser.parse_args(argv)


def format_output(answer_text: str, hits: list[tuple[Chunk, float]]) -> str:
    lines = ["Answer:", answer_text.strip(), "", "Retrieved context:"]
    for i, (chunk, score) in enumerate(hits, start=1):
        lines.append(f"[{i}]  score={score:.3f}  {chunk.chapter}")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def ask(index, question: str) -> None:
    hits = search(index, question)
    answer_text = answer(question, hits)
    print(format_output(answer_text, hits))


def run_repl(index) -> None:
    print("Type a question, or 'exit' / 'quit' / Ctrl+C to quit.")
    while True:
        try:
            question = input("Question (or 'exit')> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return
        ask(index, question)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        index = load_or_build_index()
        print(
            f"Indexed {len(index.chunks)} chunks from {DOCUMENT_TITLE} "
            f"using {EMBED_MODEL}."
        )
        if args.question:
            ask(index, args.question)
        else:
            run_repl(index)
    except OllamaError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
