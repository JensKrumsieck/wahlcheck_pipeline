from wahlcheck_ai.llm import chat_json
from wahlcheck_ai.theses import load_theses

SYSTEM_PROMPT = """You analyze political statements for information retrieval.

You receive a political statement. Formulate ONE search query targeting potentially restrictive, 
opposing, or balancing statements within the same topic area that could contradict or limit the statement.

The goal is to also retrieve passages that oppose or qualify the approach expressed in the statement, 
not only passages that support it.

Important:
- Do NOT invent concrete facts. Only describe what should be searched for, not what an election program supposedly says.
- Preserve the subject matter of the original statement.
- Formulate a meaningful search phrase, NOT a question, with a maximum length of 350 characters.
- Prefer German.

Answer as JSON
"""


OPPOSING_SCHEMA = {
    "type": "object",
    "properties": {"opposing": {"type": "string", "maxLength": 350}},
    "required": ["opposing"],
}


def opposing_these(model: str):
    theses = load_theses()

    results = []
    for ix, thesis in enumerate(theses):
        result = chat_json(
            model,
            [SYSTEM_PROMPT],
            thesis["these"],
            OPPOSING_SCHEMA,
        )
        results.append({"these": thesis, **result})
    return results
