"""Embed chunks, cache the index, and retrieve top-k matches by cosine similarity."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.ingest import DEFAULT_DOCUMENT, ROOT, Chunk, load_and_chunk
from src.ollama_client import EMBED_MODEL, embed

INDEX_PATH = ROOT / "data" / "index.npz"
CHUNKS_PATH = ROOT / "data" / "chunks.json"
TOP_K = 4


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def _source_mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def _cache_is_current(source: Path) -> bool:
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return False
    try:
        meta = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        meta.get("embed_model") == EMBED_MODEL
        and meta.get("source_mtime_ns") == _source_mtime_ns(source)
        and meta.get("source_path") == str(source.resolve())
    )


def _load_cache() -> Index:
    payload = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in payload["chunks"]]
    embeddings = np.load(INDEX_PATH)["embeddings"]
    return Index(chunks=chunks, embeddings=embeddings)


def _save_cache(index: Index, source: Path) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(INDEX_PATH, embeddings=index.embeddings)
    payload = {
        "embed_model": EMBED_MODEL,
        "source_mtime_ns": _source_mtime_ns(source),
        "source_path": str(source.resolve()),
        "chunks": [asdict(chunk) for chunk in index.chunks],
    }
    CHUNKS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _embed_chunks(chunks: list[Chunk]) -> np.ndarray:
    vectors: list[list[float]] = []
    total = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        vectors.append(embed(chunk.text))
        print(f"\rEmbedded {i}/{total}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return _normalize(np.asarray(vectors, dtype=np.float32))


def load_or_build_index(path: Path | str = DEFAULT_DOCUMENT) -> Index:
    source = Path(path)
    if _cache_is_current(source):
        return _load_cache()

    chunks = load_and_chunk(source)
    embeddings = _embed_chunks(chunks)
    index = Index(chunks=chunks, embeddings=embeddings)
    _save_cache(index, source)
    return index


def search(index: Index, query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
    query_vector = _normalize(np.asarray([embed(query)], dtype=np.float32))[0]
    scores = index.embeddings @ query_vector
    k = min(k, len(index.chunks))
    ranked = np.argsort(scores)[::-1][:k]
    return [(index.chunks[i], float(scores[i])) for i in ranked]
