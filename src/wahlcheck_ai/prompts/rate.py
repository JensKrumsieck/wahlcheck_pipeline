import json

from wahlcheck_ai.llm import chat_json

SYSTEM_PROMPT = """You are an expert political scientist.

You evaluate political statements against excerpts from election programmes.

You receive:
1. `these`: One political statement that describes a proposed political position or policy.
2. `quellen`: Several relevant excerpts from one election programme.
3. `glossar`: A glossary explaining terms that may be unfamiliar or ambiguous.

Your task:
Determine whether the election programme supports, opposes, or does not provide sufficient 
evidence for the political statement.

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

Policy-mechanism test:

Before assigning 1 or -1 based on implicit evidence, ask:

"Would a reader who only knows the supplied source reasonably understand the source as taking a position on the policy mechanism proposed in the thesis?"

If YES → implicit support/opposition may be appropriate.

If NO → rate 0.

Do not supply missing intermediate political assumptions yourself.

Example:
Thesis: "Die Stadt soll Bauland aktiv ankaufen."
Source: "Jede Ausgabe muss sich an Nutzen, Notwendigkeit und Verantwortbarkeit messen lassen."

The source expresses general fiscal discipline, but does not establish a position on municipal land acquisition.
This is indirect/speculative evidence → 0.

A recurring trap is a GENERIC refrain that rejects "new burdens, requirements or bans
for citizens and businesses" (e.g. against extra costs, red tape or regulation).
This refrain is DIRECT IMPLICIT evidence only against theses that themselves propose
an economic or administrative burden (fees, mandatory quotas, extra requirements,
taxes). It is NOT sufficient evidence against a thesis that proposes a specific,
non-economic measure - such as a content-based ban, a symbolic policy, or a values
question - unless the source specifically addresses that measure.

Example:
Thesis: "Die Stadt soll Werbung für die Bundeswehr auf städtischen Bussen verbieten."
Source: "Braunschweig darf Bürger und Unternehmen nicht durch zusätzliche kommunale
Vorgaben, Verbote oder Belastungen weiter unter Druck setzen."

The source rejects new economic/administrative burdens on citizens and businesses.
An advertising ban on public transport is not an economic burden on citizens or
businesses - it is a content/values decision the source does not address.
This is INDIRECT/SPECULATIVE → 0, even though both mention "Verbote".

Reasoning procedure:

1. Identify the essential policy mechanism of the thesis:
   - What action is proposed?
   - What is affected?
   - In which direction does the policy change?
   - Which policy instrument or mechanism is involved?
   - Are there important conditions, limitations or exceptions?

2. Evaluate the sources for EXPLICIT evidence.
   The source explicitly supports or opposes the policy described by the thesis.

3. Evaluate the sources for IMPLICIT evidence.
   A source may support or oppose the thesis without using the same terminology.

   IMPLICIT SUPPORT exists when:
   - the source expresses a broader or equivalent policy principle that clearly favors the proposed policy mechanism; or
   - the source supports the same policy mechanism and direction using different terminology.

   IMPLICIT OPPOSITION exists when:
   - the source rejects, criticizes or seeks to prevent the type of policy mechanism proposed by the thesis; or
   - the source supports the same policy mechanism but in the opposite direction.

4. Do NOT infer a position merely because the source and thesis concern the same topic.
   A connection must follow clearly from the meaning of the source and the policy mechanism of the thesis.

5. A source supporting a general political goal does not automatically support a specific policy instrument.

   Example:
   Thesis: "Der Pflichtanteil für Sozialwohnungen bei Neubauprojekten soll erhöht werden."
   Source: "Kommunale Politik darf Bauen, Wohnen und Eigentum nicht durch immer neue Auflagen, 
   Vorgaben und Kostentreiber zusätzlich belasten."

   The thesis proposes increasing a mandatory requirement for new construction.
   The source rejects additional requirements, regulations and cost drivers affecting construction.
   Therefore, the source implicitly opposes the thesis → `-1`.

6. Do not use external political knowledge.
   All conclusions must be grounded in the supplied sources.

7. Pay particular attention to:
   - negations
   - limitations
   - conditions
   - exceptions
   - opposing priorities
   - changes in direction or magnitude

8. Examples introduced with terms such as "z.B." are illustrative and are not additional conditions that must be satisfied.

9. Ignore formatting, HTML comments and other textual artifacts.

Output:

Return JSON containing:
- `wertung`: `1`, `0`, or `-1`
- `sicherheit`: `"hoch"`, `"mittel"`, or `"niedrig"`
- `kommentar`: A short explanation of the decision.
- `zitat`: A short verbatim quote supporting the rating.
- `zitat_nummer`: The index of the quote used.

For `wertung = 1` or `-1`, a short verbatim quote is required.
For `wertung = 0`, `zitat` may be an empty string.
For `wertung = 0`, `sicherheit` must always be `"niedrig"`.

In `kommentar`, explain the political reasoning in language understandable to a reader who has not seen the supplied excerpts. 
Do not talk about "the sources", "the evidence", "the quote", "the programme", or quote indices. Instead, use relevant wording 
from the programme in quotation marks where useful.
 
Prefer German!
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


def reconsider(these, belege: list, glossary, model: str, objection: str):
    """Re-rates `these` given a specific objection raised by an independent
    second reviewer who read the same thesis and sources. Reuses the same
    rubric as `rate`, so the model may keep its original rating if the
    objection does not hold up."""
    user_prompt = f"""
    THESIS:

    {these}


    SOURCES:

    {"\n - ".join([f"ID {beleg["id"]}: __{beleg["text"]}__" for beleg in belege])}

    GLOSSARY:
    {json.dumps(glossary)}


    A second, independent reviewer read the same thesis and sources and reached a
    different reading:

    {objection}

    Reconsider your rating in light of this. If the objection identifies something
    you missed or a mechanism mismatch you overlooked, update your rating
    accordingly. If, after reconsidering, your original reading still holds, keep
    your rating and explain in `kommentar` why the objection does not change the
    conclusion.
    """

    result = chat_json(
        model,
        [SYSTEM_PROMPT],
        user_prompt,
        RATING_SCHEMA,
    )
    return result
