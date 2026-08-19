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


def _rating_impl(thesis, retrievals, glossary, party, model: str):
    thesis_id = thesis["these"]["id"]
    quote = thesis["these"]["these"]
    print(f"{thesis_id}: {quote}")
    chosen_ones = _select_evidence(retrievals[thesis_id])

    rating = {}
    judge_rating = {}
    history = []
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}")
        rating = rate.rate(quote, chosen_ones, glossary, model)
        if rating["wertung"] == 0 and len(chosen_ones) != len(retrievals[thesis_id]):
            # retry with all candidates
            chosen_ones = retrievals[thesis_id]
            rating = rate.rate(quote, chosen_ones, glossary, model)

        judge_rating = judge.rate(quote, rating, chosen_ones, glossary, party, model)
        current_attempt = {"attempt": attempt, "rating": rating, "judge": judge_rating}
        history.append(current_attempt)
        print(f"Rating: {rating['wertung']} | " f"Judge: {judge_rating['consens']}")

        if not judge_rating["consens"]:
            print(f"Judge: {judge_rating['eigene_wertung']}")
            chosen_ones = retrievals[thesis_id]  # extend set if judge is not consensual

        if judge_rating["consens"]:
            return {
                **rating,
                "consens": True,
                "judge_bewertung": judge_rating["eigene_wertung"],
                "attempts": attempt,
                "human_review": False,
            }
    # no consens reached = majority vote
    final_wertung = _majority_rating(history, judge_rating)

    return {
        **rating,
        "wertung": final_wertung,
        "consens": False,
        "judge_bewertung": judge_rating["eigene_wertung"],
        "attempts": max_retries,
        "human_review": True,
    }


def _majority_rating(history, judge_rating):
    ratings = [x["rating"]["wertung"] for x in history]
    counts = Counter(ratings)

    max_votes = max(counts.values())
    winners = [rating for rating, votes in counts.items() if votes == max_votes]

    # Clear majority
    if len(winners) == 1:
        return winners[0]

    # Tie -> judge breaks it
    judge_vote = judge_rating["eigene_wertung"]

    if judge_vote in winners:
        return judge_vote

    # Should normally not happen if judge_vote is -1/0/1.
    # Fall back to the latest rating.
    return history[-1]["rating"]["wertung"]
