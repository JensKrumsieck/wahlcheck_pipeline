import bisect
from dataclasses import asdict, dataclass
from itertools import accumulate, chain
import json
import os
from pathlib import Path
import ftfy
import pymupdf
import pymupdf4llm
from syntok import segmenter
from wahlcheck_ai.config import EXTRACTED_DIR


@dataclass
class Document:
    full_text: str
    pages: list[dict]
    sentences: list
    blocks: list[dict]


def extract(filename: Path, force: bool = False) -> Document:
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    stem = filename.stem

    extracted_filename = EXTRACTED_DIR / f"{stem}.json"

    if extracted_filename.exists() and not force:
        with open(extracted_filename, "r", encoding="utf-8") as f:
            raw = json.load(f)
            return Document(**raw)

    doc = read_document(filename)
    with open(extracted_filename, "w", encoding="utf-8") as f:
        json.dump(asdict(doc), f, indent=4, ensure_ascii=False)
    return doc


_GUILLEMETS = {
    "«": '"',
    "»": '"',
    "‹": "'",
    "›": "'",
}
_WORD_HYPHENS = {
    "‐": "-",
    "‑": "-",
    "‒": "-",
}

_SPACES = {
    " ": " ",  # NBSP
    " ": " ",  # THIN SPACE
    " ": " ",  # NARROW NBSP
}


def read_document(filename: Path) -> Document:
    doc = pymupdf.open(filename)

    # redact line numbers as fount in spd
    for page in doc:
        raw = page.get_text("dict")
        spans = [
            span
            for block in raw["blocks"]  # type: ignore
            for line in block.get("lines", [])  # type: ignore
            for span in line["spans"]
        ]
        if not spans:
            continue

        origins = [span["origin"] for span in spans]
        min_x = min([x for (x, _) in origins])
        min_x_spans = [span for span in spans if abs(span["origin"][0] - min_x) < 1.0]

        redacted = False
        for span in min_x_spans:
            if span["text"].strip().isdigit():
                page.add_redact_annot(span["bbox"])
                redacted = True
        if redacted:
            page.apply_redactions()

    pages: list[dict] = pymupdf4llm.to_markdown(
        doc, header=False, footer=False, page_chunks=True
    )  # type: ignore

    pages = [{**page, "text": _normalize(page["text"])} for page in pages]

    page_numbers = [page["metadata"]["page_number"] for page in pages]
    texts = [page["text"].strip() for page in pages]

    # merge documents keeping page offset information
    page_starts = list(accumulate([0] + [len(text) + 1 for text in texts]))[:-1]
    full_text = " ".join(texts)

    heading_starts = _heading_paths(pages, page_starts)
    heading_offsets = [offset for offset, _ in heading_starts]

    # build sentences
    sentences = []
    analyzed_sentences = chain.from_iterable(segmenter.analyze(full_text))

    for idx, s in enumerate(analyzed_sentences):
        start = s[0].offset
        end = s[-1].offset + len(s[-1].value)
        page_idx = bisect.bisect_right(page_starts, start) - 1

        heading_idx = bisect.bisect_right(heading_offsets, start) - 1
        heading_path = heading_starts[heading_idx][1] if heading_idx >= 0 else []

        sentences.append(
            {
                "id": idx,
                "text": full_text[start:end],
                "start": start,
                "end": end,
                "page": page_numbers[page_idx],
                "heading_path": heading_path,
            }
        )

    window_size = 10
    stride = 5
    blocks = []

    last_start = max(len(sentences) - window_size, 0)
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:      # don't drop the tail
        starts.append(last_start)

    for block_id, i in enumerate(starts):
        window = sentences[i : i + window_size]

        headings = []
        for sentence in window:
            heading = sentence["heading_path"]
            if heading and heading not in headings:
                headings.append(heading)

        text = " ".join(sentence["text"] for sentence in window)
        embedding_parts = [" > ".join(heading) for heading in headings]
        embedding_text = "\n".join(embedding_parts + [text])

        blocks.append({
            "id": block_id,
            "sentence_ids": list(range(i, i + len(window))),
            "pages": sorted({s["page"] for s in window}),
            "text": text,
            "headings": headings,
            "embedding_text": embedding_text,
        })
    return Document(full_text, pages, sentences, blocks)


def _normalize(text: str):
    text = ftfy.fix_text(
        text,
        config=ftfy.TextFixerConfig(
            normalization="NFC",
        ),
    )
    text = _translate(text, _GUILLEMETS)
    text = _translate(text, _WORD_HYPHENS)
    text = _translate(text, _SPACES)
    return text


def _translate(text: str, table: dict[str, str]) -> str:
    for src, dst in table.items():
        text = text.replace(src, dst)
    return text


def _heading_paths(pages, page_starts):
    headings = []
    path = []

    for page, page_start in zip(pages, page_starts):
        offset = page_start

        for line in page["text"].splitlines(keepends=True):
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                heading = line[level:].strip()

                path = path[: level - 1]
                path.append(heading)

                headings.append((offset, path.copy()))

            offset += len(line)

    return headings
