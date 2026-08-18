from wahlcheck_ai.llm import chat_json
from wahlcheck_ai.theses import load_glossary, load_theses

SYSTEM_PROMPT = """You analyze political statements for information retrieval.

Decompose the given political thesis into concepts that may
appear in a party programme even when the exact wording of
the thesis is absent. You will also get additional information of local terms 
in Braunschweig for you to understand local concepts better.

Extract:

- topics: broader political topics or policy areas
- measures: concrete political measures, instruments, or interventions
- contexts: political, spatial, or factual contexts in which the thesis is relevant
- goals: political goals or intended effects
- implications: logically or politically implied concepts, formulations, or
  broader political concepts that may be associated with the thesis

Important:
- Do NOT merely generate synonyms or grammatical variations..
- Identify common political concepts and formulations that could express
   the same underlying political position at a more abstract level.
- Move from concrete measures to broader political concepts where there is
   a plausible political relationship.
- Also identify concepts that are commonly discussed together with the
   thesis when this relationship is politically meaningful and useful for
   retrieval.
- Prefer strong, semantically meaningful concepts over a long list of weak
   associations.
- Keep concepts short.
- Prefer German.
- Do not invent specific facts that are not implied by the thesis.

Answer as JSON

Additional Information:
"""


CONCEPTS_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {"type": "string"},
        },
        "measures": {
            "type": "array",
            "items": {"type": "string"},
        },
        "contexts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "goals": {
            "type": "array",
            "items": {"type": "string"},
        },
        "implications": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "topics",
        "measures",
        "contexts",
        "goals",
        "implications",
    ],
    "additionalProperties": False,
}


def concept_extraction(model: str, these_terms: list):
    theses = load_theses()
    glossary = load_glossary()
    glossary_by_term = {x["term"]: x for x in glossary}

    results = []
    for ix, thesis in enumerate(theses):
        explanations = []

        for term in these_terms[ix]["terms"]:
            glossary_item = glossary_by_term[term]
            ret = f"- {term}"
            if glossary_item["expands_to"]:
                ret += f" expands to {glossary_item['expands_to']}"
            if glossary_item["definition"]:
                ret += f" is defined as {glossary_item['definition']}"
            if glossary_item["synonym"]:
                ret += f" a synonym would be {glossary_item['synonym']}"
            explanations.append(ret)

        result = chat_json(
            model,
            [SYSTEM_PROMPT] + explanations,
            thesis["these"],
            CONCEPTS_SCHEMA,
        )
        results.append({"these": thesis, **result})
    return results
