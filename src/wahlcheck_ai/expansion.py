import json
from wahlcheck_ai.config import BUILD_DIR
from wahlcheck_ai.prompts.concept import concept_extraction
from wahlcheck_ai.prompts.glossary import glossary_extraction
from wahlcheck_ai.prompts.opposing import opposing_these


def expand_queries(model):
    expanded_file = BUILD_DIR / "expanded_queries.json"
    if not expanded_file.exists():
        glossarys = glossary(model)
        concepts = concept(model, glossarys)
        opposings = opposing(model)
        combined = combine_by_thesis(opposings, concepts, glossarys)

        with open(expanded_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=4)

    with open(expanded_file, "r", encoding="utf-8") as f:
        combined = json.load(f)
    return combined


def glossary(model: str):
    glossary_file = BUILD_DIR / "glossar.json"
    if not glossary_file.exists():
        glossary = glossary_extraction(model)
        with open(glossary_file, "w", encoding="utf-8") as f:
            json.dump(glossary, f, ensure_ascii=False, indent=4)
    with open(glossary_file, "r", encoding="utf-8") as f:
        glossary = json.load(f)
    return glossary


def concept(model: str, glossary):
    concepts_file = BUILD_DIR / "concepts.json"
    if not concepts_file.exists():
        concepts = concept_extraction(model, glossary)
        with open(concepts_file, "w", encoding="utf-8") as f:
            json.dump(concepts, f, ensure_ascii=False, indent=4)
    with open(concepts_file, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    return concepts


def opposing(model: str):
    opposing_theses_file = BUILD_DIR / "opposing_theses.json"
    if not opposing_theses_file.exists():
        opposing = opposing_these(model)
        with open(opposing_theses_file, "w", encoding="utf-8") as f:
            json.dump(opposing, f, ensure_ascii=False, indent=4)
    with open(opposing_theses_file, "r", encoding="utf-8") as f:
        opposing = json.load(f)
    return opposing


def combine_by_thesis(base, *others):
    by_id = {item["these"]["id"]: item for item in base}

    for data in others:
        for item in data:
            thesis_id = item["these"]["id"]
            by_id[thesis_id].update({k: v for k, v in item.items() if k != "these"})

    return list(by_id.values())
