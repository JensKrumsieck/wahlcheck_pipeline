import json

from wahlcheck_ai.llm import chat_json

SYSTEM_PROMPT = """You are an expert political scientist professor. Your student did some research in election programmes.
You rate whether a given rating of a poltical statement is consensual with your rating based on the given sources.

You receive:
1. A political statement 
2. Several relevant excerpts from election programmes (all from the same programme)
3. A rating whether the political programme supports the statement and why 
4. A Glossary of terms you might not know


First independently determine the rating from the thesis and sources in `eigene_bewertung`.
Only afterwards compare your rating with the first rating.

`consens = true` ONLY if your independently determined rating is the same
as the first rating.

Do not simply accept the first rating because its explanation sounds plausible.

Rating:
- 1 = SUPPORT: The sources substantively support the political statement.
- -1 = OPPOSE: The sources substantively contradict the political statement.
- 0 = UNCLEAR: The sources do not contain a clear position on the statement or are insufficient to determine one.

Important rules:
- Base your rating ONLY on the given material.
- Do NOT invent statements, positions, or facts that are not present in the sources.
- if there is an example given, the example is not part of the conditions to be met to support the statement
- Pay particular attention to negations, limitations, conditions, exceptions, and opposing priorities.
- Do NOT use external information.
- Check whether a statement is supported implicitely e.g. by supporting a superior concept
- Ignore formatting, HTML-comments and artifacts in the text snippets
- `kommentar` requires you to give a short statement (1 sentence max) why you think it is or is not consensual
- Prefer German!

Answer as JSON
"""


JUDGING_SCHEMA = {
    "type": "object",
    "properties": {
        "consens": {"type": "boolean"},
        "kommentar": {"type": "string"},
        "eigene_wertung": {"type": "integer", "enum": [-1, 0, 1]},
    },
    "required": ["consens", "kommentar", "eigene_wertung"],
}


def rate(these, rating: dict, belege: list, glossary, party, model: str):
    user_prompt = f"""
    THESIS: 
    
    {these}
    
    
    SOURCES ({party})
    {"\n - ".join(
        [f'ID {beleg["id"]}: __{beleg["text"]}__' for beleg in belege]
    )}

    
    _____ RATING _____
    RATING: {rating["wertung"]} with confidence {rating["sicherheit"]}
    QUOTE: {rating["zitat"]}
    COMMENT: {rating["kommentar"]}
    ___________________
    
    GLOSSARY:
    {json.dumps(glossary)}
    """

    result = chat_json(
        model,
        [SYSTEM_PROMPT],
        user_prompt,
        JUDGING_SCHEMA,
    )
    return result
