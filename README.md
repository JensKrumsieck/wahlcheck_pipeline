# Wahlcheck AI

RAG pipeline that checks Braunschweig local-election party manifestos against a
fixed set of theses ([`input/fragen.json`](input/fragen.json)) and outputs a
structured, cited position for each party/thesis pair, matching
[`input/antwort.schema.json`](input/antwort.schema.json).

## Pipeline

1. **Query expansion** – an LLM derives a glossary, key concepts, and an
   opposing thesis for each question, used to build richer retrieval queries.
2. **Extract** – party PDFs (`documents/*.pdf`) are converted to text via
   `pymupdf`/`pymupdf4llm`, normalized, split into sentences, and grouped into
   overlapping sliding-window blocks.
3. **Index** – each party gets a hybrid index: dense embeddings
   (`BAAI/bge-m3`) + BM25 (German stemmer).
4. **Retrieve & rerank** – candidates are fused across query variants (RRF)
   and reranked with `BAAI/bge-reranker-v2-m3`.
5. **Rate** – an LLM rates the party's position (`-1`/`0`/`1`) with a cited
   quote. A second, "blind" LLM judge rates independently over the full
   candidate pool; disagreements go through arbitration/majority voting, with
   unresolved cases flagged for human review.
6. **Output** – quotes are re-verified against the source text (page number,
   containment check) and written to `build/antworten/<PARTY>.json`.

Intermediate artifacts for each stage are cached under `build/` and reused on
subsequent runs unless deleted.

## Requirements

- Python 3.12, [uv](https://docs.astral.sh/uv/)
- An LLM backend: either a local [Ollama](https://ollama.com) server, or an
  OpenAI-compatible endpoint (e.g. OpenWebUI) via `openwebui:<model>` model
  names — configure `OPENWEBUI_BASE_URL` / `OPENWEBUI_API_KEY` in `.env`
  (see `.env.example`).

## Usage

```bash
uv sync
cp .env.example .env  # only needed for openwebui:<model>
uv run wahlcheck_ai
```

This processes every PDF in `documents/` against the theses in
`input/fragen.json` and writes results to `build/antworten/`.
