import json
import os
from pathlib import Path
from tqdm import tqdm
from wahlcheck_ai.config import BUILD_DIR, GLOSSARY_JSON, RATING_DIR
from wahlcheck_ai.prompts import judge, rate


def rating(filename: Path, theses, retrievals, model: str):
    filename = RATING_DIR / f"{filename.stem}.json"
    os.makedirs(filename.parent, exist_ok=True)
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
            thesis, retrievals, thesis_glossary, model
        )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(ratings, f, ensure_ascii=False, indent=4)

    return ratings


max_retries = 3


def _rating_impl(thesis, retrievals, glossary, model: str):
    thesis_id = thesis["these"]["id"]
    quote = thesis["these"]["these"]
    print(f"{thesis_id}: {quote}")
    chosen_ones = [
        retrieval
        for retrieval in retrievals[thesis_id]
        if retrieval["rerank_score"] > 0.05
    ]
    if len(chosen_ones) < 2:
        chosen_ones = retrievals[thesis_id][:10]

    rating = {}
    judge_rating = {}
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}")
        rating = rate.rate(quote, chosen_ones, glossary, model)
        judge_rating = judge.rate(quote, rating, chosen_ones, glossary, model)
        print(f"Rating: {rating['wertung']} | " f"Judge: {judge_rating['consens']}")

        if not judge_rating["consens"]:
            print(f"Judge: {judge_rating['eigene_wertung']}")

        if judge_rating["consens"]:
            return {
                **rating,
                "consens": True,
                "judge_bewertung": judge_rating["eigene_wertung"],
                "attempts": attempt,
                "human_review": False,
            }
    return {
        **rating,
        "consens": False,
        "judge_bewertung": judge_rating["bewertung"],
        "attempts": max_retries,
        "human_review": True,
    }
