import json

from wahlcheck_ai.llm import chat_json

SYSTEM_PROMPT = """You rate political statements based on excerpts from election programmes.

You receive:
1. A political statement 
2. Several relevant excerpts from election programmes (all from the same programme)
3. A Glossary of terms you might not know

Your task is to determine whether the provided sources support the thesis, contradict it, or do not provide enough evidence to make a determination.

Rating:
- 1 = SUPPORT: The sources substantively support the political statement.
- -1 = OPPOSE: The sources substantively contradict the political statement.
- 0 = UNCLEAR: The sources do not contain a clear position on the statement or are insufficient to determine one.

if -1 or 1 MUST give a short explaination on how the decision was made. Optional but nice to have if 0

Important rules:
- Base your rating ONLY on the provided sources.
- Do NOT invent statements, positions, or facts that are not present in the sources.
- Pay particular attention to negations, limitations, conditions, exceptions, and opposing priorities.
- For a rating of 1 or -1, provide a short verbatim quote from the sources that supports the rating.
- For a rating of 0, the quote may be an empty string.
- Do NOT use your own political knowledge or external information.
- Ignore formatting, HTML-comments and artifacts in the text snippets
- In `kommentar` do not talk about the quotes index number, rather use wordings from the text in quotation marks
- `zitat_nummer` is the quote's index given to you at its start
- Prefer German!

Answer as JSON
"""


RATING_SCHEMA = {
    "type": "object",
    "properties": {
        "wertung": {"type": "integer", "enum": [-1, 0, 1]},
        "sicherheit": {"type": "string", "enum": ["hoch", "mittel", "niedrig"]},
        "zitat": {"type": ["string", "null"]},
        "zitat_nummer": {"type": ["integer", "null"]},
        "kommentar": {"type": "string"},
    },
    "required": ["wertung", "sicherheit", "zitat", "zitat_nummer"],
}


def rate(these, belege: list, glossary, model: str):
    user_prompt = f"""
    THESIS: 
    
    {these}
    
    
    SOURCES:
    
    {"\n - ".join([f"ID {beleg["id"]}: __{beleg["text"]}__" for beleg in belege])}
    
    GLOSSARY:
    {json.dumps(glossary)}
    """

    result = chat_json(
        model,
        [SYSTEM_PROMPT],
        user_prompt,
        RATING_SCHEMA,
    )
    return result
