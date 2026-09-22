"""Load the vendored Gutenberg text and split it into overlapping chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCUMENT = ROOT / "data" / "alice.txt"

_START_MARKER = re.compile(r"\*\*\*\s*START OF[^\n]*\*\*\*")
_END_MARKER = re.compile(r"\*\*\*\s*END OF[^\n]*\*\*\*")
_CHAPTER_SPLIT = re.compile(r"(?m)^(?=CHAPTER\s+[IVXLCDM]+\.)")


@dataclass(frozen=True)
class Chunk:
    text: str
    chapter: str
    index: int
    start_char: int
    end_char: int


def strip_gutenberg(text: str) -> str:
    """Keep the novel body between the START and END markers."""
    start = _START_MARKER.search(text)
    if start:
        text = text[start.end() :]
    end = _END_MARKER.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


def load_document(path: Path | str = DEFAULT_DOCUMENT) -> str:
    return strip_gutenberg(Path(path).read_text(encoding="utf-8"))


def split_chapters(text: str) -> list[tuple[str, str]]:
    """Split on CHAPTER headings. Text before the first chapter is Front matter."""
    sections = [part for part in _CHAPTER_SPLIT.split(text) if part.strip()]
    chapters: list[tuple[str, str]] = []
    for section in sections:
        chapters.append(_parse_chapter_section(section.strip()))
    return chapters


def _parse_chapter_section(section: str) -> tuple[str, str]:
    if not section.startswith("CHAPTER"):
        return "Front matter", section

    lines = section.splitlines()
    heading = lines[0].strip()
    body_start = 1
    if len(lines) > 1:
        maybe_title = lines[1].strip()
        if maybe_title and len(maybe_title) < 80:
            heading = f"{heading} {maybe_title}"
            body_start = 2
    body = "\n".join(lines[body_start:]).strip()
    return heading, body


def _find_break(text: str, start: int, hard_end: int) -> int:
    """Prefer a paragraph break, then a sentence, then a word boundary."""
    window = text[start:hard_end]
    paragraph = window.rfind("\n\n")
    if paragraph != -1 and paragraph >= len(window) // 4:
        return start + paragraph
    sentence = window.rfind(". ")
    if sentence != -1 and sentence >= len(window) // 4:
        return start + sentence + 1
    space = window.rfind(" ")
    if space != -1 and space >= len(window) // 4:
        return start + space
    return hard_end


def _snap_to_word_start(text: str, pos: int, not_before: int) -> int:
    """If pos is inside a word, move back to that word's first character."""
    if pos <= not_before:
        return pos
    if pos >= len(text):
        return pos
    while pos > not_before and not text[pos - 1].isspace():
        pos -= 1
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def chunk_text(
    text: str,
    chapter: str,
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    origin: int = 0,
    start_index: int = 0,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    length = len(text)
    index = start_index

    while start < length:
        while start < length and text[start].isspace():
            start += 1
        if start >= length:
            break

        hard_end = min(start + size, length)
        end = hard_end if hard_end >= length else _find_break(text, start, hard_end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    text=piece,
                    chapter=chapter,
                    index=index,
                    start_char=origin + start,
                    end_char=origin + end,
                )
            )
            index += 1

        if end >= length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        else:
            next_start = _snap_to_word_start(text, next_start, start + 1)
            if next_start <= start:
                next_start = end
        start = next_start

    return chunks


def load_and_chunk(path: Path | str = DEFAULT_DOCUMENT) -> list[Chunk]:
    text = load_document(path)
    chunks: list[Chunk] = []
    cursor = 0
    for chapter, body in split_chapters(text):
        chapter_origin = text.find(body, cursor)
        if chapter_origin == -1:
            chapter_origin = cursor
        chunks.extend(
            chunk_text(
                body,
                chapter,
                origin=chapter_origin,
                start_index=len(chunks),
            )
        )
        cursor = chapter_origin + len(body)
    return chunks
