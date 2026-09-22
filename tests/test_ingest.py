import unittest

from src.ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_DOCUMENT,
    chunk_text,
    load_and_chunk,
    split_chapters,
    strip_gutenberg,
)

SAMPLE = """License preamble. Do not keep this.

*** START OF THE PROJECT GUTENBERG EBOOK 11 ***

Title page and contents.

CHAPTER I.
Down the Rabbit-Hole

Alice was beginning to get very tired of sitting by her sister on the bank.

CHAPTER II.
The Pool of Tears

Curiouser and curiouser, cried Alice as she swam about.

*** END OF THE PROJECT GUTENBERG EBOOK 11 ***

Legal footer. Do not keep this.
"""


class StripGutenbergTests(unittest.TestCase):
    def test_keeps_text_between_markers(self):
        body = strip_gutenberg(SAMPLE)
        self.assertIn("Down the Rabbit-Hole", body)
        self.assertIn("CHAPTER II.", body)
        self.assertNotIn("License preamble", body)
        self.assertNotIn("Legal footer", body)
        self.assertNotIn("START OF", body)
        self.assertNotIn("END OF", body)

    def test_unchanged_when_markers_missing(self):
        self.assertEqual(strip_gutenberg("  just a book  "), "just a book")


class SplitChapterTests(unittest.TestCase):
    def test_front_matter_and_chapter_titles(self):
        chapters = split_chapters(strip_gutenberg(SAMPLE))
        labels = [label for label, _ in chapters]
        self.assertEqual(
            labels,
            [
                "Front matter",
                "CHAPTER I. Down the Rabbit-Hole",
                "CHAPTER II. The Pool of Tears",
            ],
        )
        self.assertIn("Title page", chapters[0][1])
        self.assertIn("Alice was beginning", chapters[1][1])
        self.assertNotIn("CHAPTER I.", chapters[1][1])


class ChunkTextTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        chunks = chunk_text("short", "CHAPTER I.")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "short")
        self.assertEqual(chunks[0].chapter, "CHAPTER I.")
        self.assertEqual(chunks[0].index, 0)

    def test_windows_overlap_and_stay_near_size(self):
        text = ("Alice followed the rabbit. " * 60).strip()
        chunks = chunk_text(text, "CHAPTER I.", size=80, overlap=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 80)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(current.index, previous.index + 1)
            self.assertLess(
                current.start_char,
                previous.end_char,
                "consecutive chunks should overlap in the source text",
            )


class LoadAndChunkTests(unittest.TestCase):
    def test_alice_has_twelve_chapters_and_no_license(self):
        self.assertTrue(DEFAULT_DOCUMENT.exists())
        chunks = load_and_chunk(DEFAULT_DOCUMENT)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].index, 0)
        self.assertEqual(chunks[-1].index, len(chunks) - 1)

        chapters = {chunk.chapter for chunk in chunks}
        numbered = sorted(
            name for name in chapters if name.startswith("CHAPTER ")
        )
        self.assertEqual(len(numbered), 12)

        joined = " ".join(chunk.text for chunk in chunks)
        self.assertNotIn("START OF THE PROJECT GUTENBERG", joined)
        self.assertNotIn("END OF THE PROJECT GUTENBERG", joined)

        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), CHUNK_SIZE)
            self.assertGreaterEqual(CHUNK_OVERLAP, 0)


if __name__ == "__main__":
    unittest.main()
