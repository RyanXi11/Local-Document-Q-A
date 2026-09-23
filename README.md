# Local Document Q&A

A command-line RAG app that answers questions about *Alice’s Adventures in Wonderland* (public domain, [Project Gutenberg #11](https://www.gutenberg.org/ebooks/11)). It chunks the book, retrieves the most similar passages with local embeddings, and asks a local LLM to answer **only** from those passages. Retrieved text is always printed with the answer.

No API key. Review uses [Ollama](https://ollama.com/).

## Run

**Requirements:** Python 3.10+, Ollama installed and running.

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
pip install -r requirements.txt
python qa.py
```

Interactive mode:

```text
Type a question, or 'exit' / 'quit' / Ctrl+C to quit.
Question (or 'exit')>
```

One shot:

```bash
python qa.py "Who says Off with her head?"
```

Ingest tests (no Ollama):

```bash
python -m unittest tests.test_ingest -v
```

The embedding index is committed (`data/index.npz`, `data/chunks.json`), so the first run should **not** re-embed the whole book. It still calls Ollama once to embed the question and once to generate the answer. If you change `data/alice.txt` or the embed model, the index rebuilds automatically (several minutes on CPU).

## How it works

1. **Ingest** — Strip the Gutenberg `START`/`END` markers. Split on `CHAPTER` headings, then ~800-character windows with 150-character overlap, breaking on paragraphs/sentences when possible.
2. **Index** — Embed each chunk with Ollama `nomic-embed-text`. Store vectors in numpy; cache on disk keyed by embed model + SHA-256 of the source file.
3. **Retrieve** — Embed the question, rank chunks by cosine similarity, take top 4.
4. **Generate** — Send those four numbered excerpts to Ollama `llama3.2` with a grounded prompt. Print the answer, then the same excerpts with chapter and similarity score.

```
alice.txt → chunks → embeddings → top-4 → llama3.2 → answer + citations
```

## Design decisions

- **CLI, not a web UI.** The spec grades the retrieval/LLM pipeline, not layout.
- **From scratch (no LangChain / LlamaIndex).** The steps above stay visible in a few small modules.
- **Ollama only.** No evaluator API key, and the document never leaves the machine.
- **`nomic-embed-text` for retrieval, `llama3.2` for answers.** Embedding models and chat models are trained for different jobs.
- **Numpy cosine similarity, not a vector database.** ~280 chunks fit in memory; this is the same math FAISS would run at this scale.
- **Chapter-aware overlapping chunks.** Overlap keeps a sentence that straddles a cut. Windows snap to word boundaries so citations do not start mid-word.
- **Grounded prompt.** If the excerpts do not contain the answer, the model is told to say it does not know rather than invent plot.
- **Always show retrieved context**, including “I don’t know” answers. Those four chunks are what the model saw; the spec asks to show that text.
- **Vendored cache.** Embedding the book one chunk at a time is slow on CPU; shipping the index keeps review fast.

## Limitations

- **Retrieval can miss.** Cosine similarity matches topic, not “the sentence that answers *why*.” “Why did Alice follow the White Rabbit?” often retrieves other White Rabbit scenes instead of the “burning with curiosity” passage, which *is* in the index. The model then correctly refuses, because the excerpts it received do not contain the reason. Top-k is 4; raising it would help some questions and add noise to others.
- **Small local models** may ignore citation numbers or paraphrase loosely even when the right chunk is present.
- **Similarity scores are not probabilities.** Unrelated questions still get a top-4 (often weakly related “miles” / geography lines). There is no score cutoff.
- **Single document, no chat memory.** Each question is independent.
- This is a take-home RAG pipeline, not production search.

## Example questions

- Who says “Off with her head!”?
- What happens at the Mad Tea-Party?
- Why did Alice follow the White Rabbit?
- How long is the Mississippi River? (not in the book; should refuse)

## Layout

- `qa.py` — argparse CLI
- `src/ingest.py` — load, strip, chunk
- `src/retrieve.py` — embed, cache, top-4
- `src/ollama_client.py` — HTTP to `localhost:11434`
- `src/generate.py` — grounded prompt
- `data/alice.txt` — source document
- `tests/test_ingest.py` — unittest for ingest
