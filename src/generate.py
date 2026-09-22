"""Build a grounded prompt and generate an answer from retrieved chunks."""

from __future__ import annotations

from src.ingest import Chunk
from src.ollama_client import chat

SYSTEM_PROMPT = (
    "You are answering questions about a document using only the numbered excerpts below.\n"
    "Use those excerpts as the sole source of truth.\n"
    "If they do not contain the answer, say you do not know from the provided text. "
    "Do not invent plot or facts.\n"
    "When you use an excerpt, cite it as [1], [2], etc."
)


def format_excerpts(hits: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (chunk, _score) in enumerate(hits, start=1):
        blocks.append(f"[{i}] {chunk.chapter}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    excerpts = format_excerpts(hits)
    return f"Question: {question}\n\n{excerpts}"


def answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    return chat(SYSTEM_PROMPT, build_user_prompt(question, hits))
