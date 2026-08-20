import json
import os
from pathlib import Path
from typing import Counter
from tqdm import tqdm
from wahlcheck_ai.config import BUILD_DIR, GLOSSARY_JSON, RATING_DIR
from wahlcheck_ai.prompts import judge, rate


def rating(filename: Path, theses, retrievals, model: str, force: bool = False):
    filename = RATING_DIR / f"{filename.stem}.json"
    os.makedirs(filename.parent, exist_ok=True)
    if not filename.exists() or force:
        print(f"Rating Theses for {filename.stem}")

        with open(BUILD_DIR / "glossar.json", "r") as f:
            glossary_to_theses = json.load(f)

        with open(GLOSSARY_JSON, "r") as f:
            glossary = json.load(f)

        ratings = {}
        for thesis in tqdm(theses):
            thesis_glossary = [
                x["terms"]
                for x in glossary_to_theses
                if x["these"]["id"] == thesis["these"]["id"]
            ][0]
            thesis_glossary = [g for g in glossary if g["term"] in thesis_glossary]

            ratings[thesis["these"]["id"]] = _rating_impl(
                thesis, retrievals, thesis_glossary, filename.stem, model
            )

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(ratings, f, ensure_ascii=False, indent=4)
    with open(filename, "r", encoding="utf-8") as f:
        ratings = json.load(f)
    return ratings


max_retries = 2


CONFIDENCE_RATIO = 0.5
MIN_CONFIDENT = 8
FALLBACK_N = 25


def _select_evidence(candidates: list) -> list:
    if not candidates:
        return candidates
    top_score = max(c["rerank_score"] for c in candidates)
    chosen = [
        c for c in candidates if c["rerank_score"] >= top_score * CONFIDENCE_RATIO
    ]
    if len(chosen) < MIN_CONFIDENT:
        chosen = sorted(candidates, key=lambda c: -c["rerank_score"])[:FALLBACK_N]
    return chosen


def _merge_evidence(chosen_ones: list, blind: dict, all_candidates: list) -> list:
    """Adds the source the blind judge cited to the rater's evidence set, if it
    wasn't already in there. This is how a genuine evidence gap (the rater's
    filtered set missed something) gets repaired, as opposed to re-arguing over
    the same text."""
    cited_id = blind.get("zitat_nummer")
    if cited_id is None:
        return chosen_ones
    if any(str(c["id"]) == str(cited_id) for c in chosen_ones):
        return chosen_ones
    match = next((c for c in all_candidates if str(c["id"]) == str(cited_id)), None)
    if match is None:
        return chosen_ones
    return chosen_ones + [match]


def _rating_impl(thesis, retrievals, glossary, party, model: str):
    thesis_id = thesis["these"]["id"]
    quote = thesis["these"]["these"]
    print(f"{thesis_id}: {quote}")
    all_candidates = retrievals[thesis_id]
    chosen_ones = _select_evidence(all_candidates)

    rating = rate.rate(quote, chosen_ones, glossary, model)
    if rating["wertung"] == 0 and len(chosen_ones) != len(all_candidates):
        # retry with all candidates
        chosen_ones = all_candidates
        rating = rate.rate(quote, chosen_ones, glossary, model)

    # One blind, independent second opinion over the FULL candidate pool -
    # deliberately never shown `rating`, and not limited to the rater's
    # filtered evidence, so it can catch both a misread and evidence the
    # rater's filter dropped. Computed once; the loop below only reconciles
    # `rating` against this fixed reference point.
    blind = judge.evaluate(quote, all_candidates, glossary, party, model)

    history = []
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}")
        verdict = judge.compare(quote, rating, blind, model)
        history.append({"attempt": attempt, "rating": rating, "verdict": verdict})
        print(
            f"Rating: {rating['wertung']} | Blind: {blind['wertung']} | "
            f"Konsens: {verdict['consens']}"
        )

        if verdict["consens"]:
            return {
                **rating,
                "consens": True,
                "judge_bewertung": blind["wertung"],
                "attempts": attempt,
                # even though they now agree, flag it if the rater had to
                # revise its first answer to get there
                "human_review": attempt > 1,
            }

        if attempt == max_retries:
            break

        cited_id = blind.get("zitat_nummer")
        already_had_it = cited_id is not None and any(
            str(c["id"]) == str(cited_id) for c in chosen_ones
        )
        if cited_id is not None and not already_had_it:
            # evidence gap: the blind pass found something the rater didn't have
            print(f"Judge cited new evidence: {cited_id}")
            chosen_ones = _merge_evidence(chosen_ones, blind, all_candidates)
            rating = rate.rate(quote, chosen_ones, glossary, model)
        else:
            # reasoning gap: same evidence, different read - ask the rater to
            # engage with the specific objection instead of re-rolling blind
            print("Judge disagrees on the same evidence, asking rater to reconsider")
            rating = rate.reconsider(
                quote, chosen_ones, glossary, model, blind["kommentar"]
            )

    # no consens reached after all retries = majority vote across attempts
    final_wertung = _majority_rating(history, blind)

    return {
        **rating,
        "wertung": final_wertung,
        "consens": False,
        "judge_bewertung": blind["wertung"],
        "attempts": max_retries,
        "human_review": True,
    }


def _majority_rating(history, blind):
    ratings = [x["rating"]["wertung"] for x in history]
    counts = Counter(ratings)

    max_votes = max(counts.values())
    winners = [rating for rating, votes in counts.items() if votes == max_votes]

    # Clear majority
    if len(winners) == 1:
        return winners[0]

    # Tie -> blind judge breaks it
    judge_vote = blind["wertung"]

    if judge_vote in winners:
        return judge_vote

    # Should normally not happen if judge_vote is -1/0/1.
    # Fall back to the latest rating.
    return history[-1]["rating"]["wertung"]
