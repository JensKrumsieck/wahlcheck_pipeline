import json
from wahlcheck_ai.llm import chat_json
from wahlcheck_ai.theses import load_glossary, load_theses

SYSTEM_PROMPT = """You analyze political statements for information retrieval.

You will get a political thesis as plain text and a glossary as JSON.

Extract:
- terms: Terms in the glossary that are relevant to the given political thesis.

Important:
- Do NOT merely generate synonyms.
- Do NOT invent new terms that are not in the glossary
- Use only the field `term` to identify the terms
- Do not invent specific facts that are not implied by the thesis.

Answer as JSON according to schema
___GLOSSARY___
"""


GLOSSARY_SCHEMA = {
    "type": "object",
    "properties": {
        "terms": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "terms",
    ],
    "additionalProperties": False,
}


def glossary_extraction(model: str):
    theses = load_theses()
    glossary = load_glossary()

    results = []
    for item in theses:
        result = chat_json(
            model,
            [SYSTEM_PROMPT + json.dumps(glossary)],
            item["these"],
            GLOSSARY_SCHEMA,
        )
        results.append({"these": item, **result})

    return results
