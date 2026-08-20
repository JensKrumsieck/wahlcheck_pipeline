import difflib
import json
import os
import re
from datetime import datetime
from pathlib import Path

from wahlcheck_ai.config import (
    ANTWORTEN_DIR,
    EXTRACTED_DIR,
    RATING_DIR,
    RETRIEVAL_DIR,
)

# internal party code (build/*/<code>.json, documents/<code>.pdf) -> antwort.schema.json enum
SCHEMA_PARTEI = {
    "AFD": "AfD",
    "BIBS": "BIBS",
    "BSW": "BSW",
    "CDU": "CDU",
    "FDP": "FDP",
    "GRUENE": "GRÜNE",
    "LINKE": "Linke",
    "SPD": "SPD",
    "Volt": "Volt",
}

_ELLIPSIS_RE = re.compile(r"\.\.\.|…")
_STRIP_CHARS = " .„“\"'‚‘’”«»"
_VERIFIED_THRESHOLD = 0.85

_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")
# boilerplate that shows up in nearly every thesis/programme sentence and
# would otherwise "match" almost any passage, regardless of topic
_STOPWORDS = {
    "sollen", "soll", "werden", "wird", "wurde", "eine", "einen", "einem",
    "einer", "eines", "diese", "dieser", "dieses", "durch", "nicht", "auch",
    "immer", "weiter", "weitere", "weiteren", "kommunale", "kommunalen",
    "kommunaler", "kommunal", "stadt", "städtischen", "städtische",
    "städtischer", "braunschweig", "politik", "bereits", "sowie", "zwischen",
    "innerhalb", "gegen", "ohne", "unter", "über", "zudem", "damit", "dabei",
    "dafür", "dass", "welche", "welcher", "welchen", "jeder", "jede", "jedes",
    "sollte", "können", "müssen", "sowohl", "sondern",
}
_GEGENPROBE_WINDOW = 2


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    return " ".join(text.replace("…", "...").split())


def _quote_segments(zitat: str) -> list[str]:
    """A cited quote may stitch together non-contiguous fragments with '...'
    (e.g. "Zusätzliche Belastungen ... für Bürger ... wir setzen auf ..."),
    sometimes wrapped in curly quotes ("„...für ... werden.“"). Each fragment
    has to be matched on its own - matching the whole string as one unit would
    never find a contiguous match in the source. Fragments below the length
    cutoff are dropped: splitting on '...' next to a leading/trailing "„"/"“"
    can leave a bare quote character behind, which would otherwise "match"
    almost anything and silently poison the verification score."""
    parts = [_normalize(p).strip(_STRIP_CHARS) for p in _ELLIPSIS_RE.split(zitat)]
    return [p for p in parts if len(p) >= 4]


def _containment_ratio(needle: str, haystack: str) -> float:
    """What fraction of `needle`'s characters are found, in order, inside
    `haystack` - robust to the source using slightly different whitespace,
    quote characters, or a paraphrase around the matched core."""
    if not needle:
        return 0.0
    matcher = difflib.SequenceMatcher(None, needle, haystack, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(needle)


def _best_sentence_for_segment(segment: str, sentences: list[dict]) -> tuple[dict | None, float]:
    """Finds which sentence (or adjacent sentence pair, for a fragment that
    straddles a sentence boundary) a quote segment actually sits in."""
    best_sentence = None
    best_score = -1.0

    for sentence in sentences:
        score = _containment_ratio(segment, _normalize(sentence["text"]))
        if score > best_score:
            best_score = score
            best_sentence = sentence

    for left, right in zip(sentences, sentences[1:]):
        joined = _normalize(left["text"]) + " " + _normalize(right["text"])
        score = _containment_ratio(segment, joined)
        if score > best_score:
            best_score = score
            best_sentence = left  # attribute a boundary-spanning quote to its first page

    return best_sentence, best_score


def _resolve_segments(segments: list[str], candidate_sentences: list[dict]) -> tuple[int | None, float]:
    pages, scores = [], []
    for segment in segments:
        sentence, score = _best_sentence_for_segment(segment, candidate_sentences)
        if sentence is not None:
            pages.append(sentence["page"])
            scores.append(score)

    if not pages:
        return None, 0.0
    # the quote is attributed to the page its first fragment starts on;
    # the weakest-matching fragment gates whether the whole quote counts as verified
    return pages[0], min(scores)


def resolve_page(zitat: str, zitat_nummer, blocks: list[dict], sentences: list[dict]) -> tuple[int | None, bool]:
    """Re-derives the actual page a cited quote sits on.

    `zitat_nummer` narrows the search down to one chunk (a ~10-sentence
    sliding window, which can span more than one page - about 40% of chunks
    in a typical programme do). `zitat` then pins down which sentence inside
    that chunk the quote actually came from, so the returned page is the
    quote's real page rather than just "one of the pages this chunk touches".

    If the cited chunk doesn't actually contain the quote - the rater cited
    the wrong id, which does happen - this falls back to searching the whole
    document by text alone, so a bad citation doesn't have to mean a bad page.
    """
    if zitat_nummer is None or not zitat:
        return None, False

    segments = _quote_segments(zitat) or [_normalize(zitat)]
    block = next((b for b in blocks if b["id"] == zitat_nummer), None)

    if block is not None:
        if len(block["pages"]) == 1:
            if _containment_ratio(_normalize(zitat), _normalize(block["text"])) >= _VERIFIED_THRESHOLD:
                return block["pages"][0], True
        else:
            block_sentences = sentences[block["sentence_ids"][0] : block["sentence_ids"][-1] + 1]
            page, score = _resolve_segments(segments, block_sentences)
            if score >= _VERIFIED_THRESHOLD:
                return page, True

    # the cited chunk didn't check out - search the full document by text alone
    page, score = _resolve_segments(segments, sentences)
    if score >= _VERIFIED_THRESHOLD:
        return page, True

    # nothing verified anywhere; prefer a full-document best guess, falling
    # back to the (mis-)cited chunk's own page if even that came up empty
    if page is not None:
        return page, False
    if block is not None:
        return block["pages"][0], False
    return None, False


def _retrieval_score(zitat_nummer, retrievals_for_thesis: list[dict]) -> float | None:
    if zitat_nummer is None:
        return None
    match = next((c for c in retrievals_for_thesis if str(c["id"]) == str(zitat_nummer)), None)
    return match["rerank_score"] if match else None


def _key_terms(text: str, top_n: int = 6) -> list[str]:
    """The thesis's most distinctive words. German capitalizes every noun
    (not just proper nouns), so keeping only capitalized words - other than
    the sentence-initial one, which is capitalized regardless of its part of
    speech - is a cheap, dependency-free way to land on the actual topical
    vocabulary ("Windkraft", "Klinikum", "Gesamtschulen") instead of generic
    long verbs/adjectives ("beschleunigt", "zusätzliche", "gezielt") that
    turned out to false-positive constantly by appearing in unrelated
    passages throughout the programme regardless of topic."""
    words = _WORD_RE.findall(text)
    candidates = words[1:] if len(words) > 1 else words
    nouns = {
        word.lower()
        for word in candidates
        if word[:1].isupper() and len(word) >= 6 and word.lower() not in _STOPWORDS
    }
    if not nouns:
        # fallback for theses too short/unusual to yield a capitalized noun
        nouns = {
            word.lower()
            for word in candidates
            if len(word) >= 6 and word.lower() not in _STOPWORDS
        }
    return sorted(nouns, key=len, reverse=True)[:top_n]


def _reviewed_sentence_ids(retrievals_for_thesis: list[dict]) -> set[int]:
    """Every sentence id already covered by a chunk the rater/judge had in
    front of them for this thesis - a hit inside this set isn't a miss, it's
    something that was already weighed and still (correctly) didn't clear the
    bar for `wertung != 0`."""
    ids: set[int] = set()
    for candidate in retrievals_for_thesis:
        ids.update(candidate.get("sentence_ids", []))
    return ids


def volltext_gegenprobe(
    these_text: str, wertung: int, sentences: list[dict], retrievals_for_thesis: list[dict]
) -> str:
    """Independent lexical cross-check for `wertung == 0`: does the RAW thesis
    vocabulary cluster anywhere in the document OUTSIDE what the RAG pipeline
    already retrieved and the rater already considered? This is deliberately
    a different, cruder method than the retrieval pipeline (plain substring
    search over the full document, no embeddings, no reranking) - the point
    is to catch a case where chunking/retrieval suppressed a passage a flat
    keyword scan would have found, not to re-derive the same evidence again.
    """
    if wertung != 0:
        return "n/a"

    key_terms = _key_terms(these_text)
    if len(key_terms) < 2:
        # not enough distinctive vocabulary in this thesis to run a
        # meaningful cross-check at all - honestly say so rather than guess
        return "n/a"

    reviewed = _reviewed_sentence_ids(retrievals_for_thesis)

    for i in range(len(sentences) - _GEGENPROBE_WINDOW + 1):
        window = sentences[i : i + _GEGENPROBE_WINDOW]
        if {s["id"] for s in window}.issubset(reviewed):
            continue  # already in front of the rater - not a new finding
        joined = " ".join(s["text"] for s in window).lower()
        if sum(1 for term in key_terms if term in joined) >= 2:
            return "abweichend"

    return "bestaetigt"


def _beleglage(wertung: int, verified: bool, sicherheit: str) -> str:
    if wertung == 0:
        return "keine_fundstelle"
    if verified and sicherheit == "hoch":
        return "eindeutig"
    return "schwach"


def _pruefung(rating: dict, verified: bool, retrieval_score: float | None, gegenprobe: str) -> dict:
    wertung = rating["wertung"]
    human_review = bool(rating.get("human_review"))
    consens = bool(rating.get("consens", True))

    flags = []
    if human_review:
        flags.append("human_review")
    if not consens:
        flags.append("kein_konsens")
    if wertung != 0 and not verified:
        flags.append("zitat_unverifiziert")
    if gegenprobe == "abweichend":
        flags.append("volltext_fund")

    prioritaet = int(human_review) + int(wertung != 0 and not verified) + int(gegenprobe == "abweichend")

    return {
        "zitat_verifiziert": verified,
        "beleglage": _beleglage(wertung, verified, rating["sicherheit"]),
        "retrieval_score": retrieval_score,
        "judge_dissens": not consens,
        "volltext_gegenprobe": gegenprobe,
        "flags": flags,
        "prioritaet": prioritaet,
    }


def build_antwort(
    these_text: str,
    partei: str,
    rating: dict,
    blocks: list[dict],
    sentences: list[dict],
    retrievals_for_thesis: list[dict],
    model: str,
    eingelesen: str,
) -> dict:
    """Builds one antwort.schema.json-conforming record from an internal
    rating entry (as produced by `wahlcheck_ai.rate`)."""
    zitat = rating.get("zitat") or ""
    zitat_nummer = rating.get("zitat_nummer")

    seite, verified = resolve_page(zitat, zitat_nummer, blocks, sentences)
    retrieval_score = _retrieval_score(zitat_nummer, retrievals_for_thesis)
    gegenprobe = volltext_gegenprobe(these_text, rating["wertung"], sentences, retrievals_for_thesis)

    return {
        "these": these_text,
        "partei": partei,
        "wertung": rating["wertung"],
        "sicherheit": rating["sicherheit"],
        "zitat": zitat,
        "seite": seite,
        "kommentar": None,
        "modell": model,
        "eingelesen": eingelesen,
        "pruefung": _pruefung(rating, verified, retrieval_score, gegenprobe),
    }


def write_antworten(party_file: Path, theses: list[dict], model: str, eingelesen: str | None = None) -> list[dict]:
    """Reads the already-built rating/retrieval/extract artifacts for one
    party and writes build/antworten/<PARTY>.json - one schema-conforming
    record per thesis, in the same order as `theses`."""
    party = party_file.stem
    partei = SCHEMA_PARTEI.get(party, party)

    ratings = _load(RATING_DIR / f"{party}.json")
    retrievals = _load(RETRIEVAL_DIR / f"{party}.json")
    document = _load(EXTRACTED_DIR / f"{party}.json")
    blocks = document["blocks"]
    sentences = document["sentences"]

    if eingelesen is None:
        mtime = (EXTRACTED_DIR / f"{party}.json").stat().st_mtime
        eingelesen = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    antworten = [
        build_antwort(
            thesis["these"]["these"],
            partei,
            ratings[thesis["these"]["id"]],
            blocks,
            sentences,
            retrievals.get(thesis["these"]["id"], []),
            model,
            eingelesen,
        )
        for thesis in theses
    ]

    os.makedirs(ANTWORTEN_DIR, exist_ok=True)
    with open(ANTWORTEN_DIR / f"{party}.json", "w", encoding="utf-8") as f:
        json.dump(antworten, f, ensure_ascii=False, indent=2)

    return antworten
