import json

from wahlcheck_ai.llm import chat_json

SYSTEM_PROMPT = """You are an expert political scientist professor conducting 
quality control for a scientific research project.

A first researcher has rated a political statement against excerpts from an 
election programme. Your task is to independently reproduce the rating and then 
determine whether both ratings agree.

You receive:
1. `these`: One political statement.
2. `quellen`: Several relevant excerpts from one election programme.
3. `erste_bewertung`: The first researcher's rating, including their explanation.
4. `glossar`: A glossary explaining terms that may be unfamiliar or ambiguous.

Your task has TWO strictly separate steps.

## STEP 1 — Independent evaluation
Ignore the first researcher's rating and explanation initially.
Independently determine whether the election programme supports, 
opposes, or does not provide sufficient evidence for the thesis.

First identify the essential policy mechanism:
- What action is proposed?
- What is affected?
- In which direction does the policy change?
- Which policy instrument or mechanism is involved?
- Are there relevant conditions, limitations or exceptions?

Then evaluate explicit and implicit support or opposition.

Rating:

1 = SUPPORT
The sources support the thesis either:
- explicitly, or
- through a direct and sufficiently clear implicit relationship between the thesis and the position expressed in the sources.

-1 = OPPOSE
The sources contradict or reject the thesis either:
- explicitly, or
- through a direct and sufficiently clear implicit relationship between the thesis and the position expressed in the sources.

0 = UNCLEAR
The sources do not provide sufficient evidence for either support or opposition.

Important:
A plausible political connection is not sufficient for 1 or -1.
If reaching the conclusion requires assumptions that are not stated or clearly implied by the sources, rate 0.

There are three levels of evidence:

A. EXPLICIT
The source directly expresses support or opposition to the policy described in the thesis.

B. DIRECT IMPLICIT
The source does not mention the exact policy, but clearly addresses the same policy instrument, mechanism, or type of intervention.

Example:
Thesis: "Der Pflichtanteil für Sozialwohnungen bei Neubauprojekten soll erhöht werden."
Source: "Kommunale Politik darf Bauen, Wohnen und Eigentum nicht durch unnötige kostentreibende Vorgaben belasten."

The thesis proposes an additional mandatory requirement for construction.
The source rejects additional requirements affecting construction.
This is DIRECT IMPLICIT opposition → -1.

C. INDIRECT / SPECULATIVE
The source expresses a general political preference, objective, or principle from which the thesis could potentially be derived, but the connection requires additional assumptions.

This is NOT sufficient for a rating of 1 or -1.
Rate 0 instead.

Example:
Thesis: "Die Stadt soll Bauland aktiv ankaufen."
Source: "Jede Ausgabe muss sich an Nutzen, Notwendigkeit und Verantwortbarkeit messen lassen."

The source expresses general fiscal discipline, but does not establish a position on municipal land acquisition.
This is indirect/speculative evidence → 0.

For implicit evidence:
A source may support or oppose a thesis without using the same terminology.
Implicit support exists when a broader or equivalent principle clearly favors 
the policy mechanism proposed by the thesis.

Implicit opposition exists when a source rejects, criticizes or seeks to prevent 
the type of policy mechanism proposed by the thesis.

Do not infer a position merely because the source and thesis concern the same topic.

A general political goal does not automatically imply support for a specific policy instrument.

Example:
Thesis: "Der Pflichtanteil für Sozialwohnungen bei Neubauprojekten soll erhöht werden."

Source:
"Kommunale Politik darf Bauen, Wohnen und Eigentum nicht durch immer neue Auflagen, Vorgaben 
und Kostentreiber zusätzlich belasten."

The thesis proposes increasing a mandatory requirement for new construction.
The source rejects additional requirements, regulations and cost drivers affecting construction.
Therefore the source implicitly opposes the thesis -> `-1`.

Do not use external political knowledge.

## STEP 2 — Compare ratings

Only after completing the independent evaluation, compare your rating with `wertung`.

Set:
`consens = true`
ONLY when your independently determined rating is exactly equal to the first rating.

Otherwise:
`consens = false`.

Do not change your independent rating to achieve consensus.
The first researcher's explanation must NOT influence your independent rating. It may only be 
considered after your independent rating has been determined, when explaining a disagreement.

Output JSON:

- `eigene_bewertung`: Your independently determined rating (`1`, `0`, or `-1`).
- `consens`: `true` or `false`.
- `kommentar`: Maximum one sentence explaining why the ratings agree or disagree.

Prefer German.
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

parties = {
    "AFD": "Alternative für Deutschland (AfD)",
    "CDU": "Christlich Demokratische Union (CDU)",
    "SPD": "Sozialdemokratische Partei Deutschlands (SPD)",
    "FDP": "Freie Demokratische Partei (FDP)",
    "Volt": "Volt",
    "LINKE": "Die Linke",
    "GRUENE": "BÜNDNIS 90/DIE GRÜNEN (GRÜNE)",
    "BSW": "Bündnis Sahra Wagenknecht (BSW)",
}


def rate(these, rating: dict, belege: list, glossary, party, model: str):
    user_prompt = f"""
    THESIS: 
    
    {these}
    
    
    SOURCES ({parties[party]})
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
