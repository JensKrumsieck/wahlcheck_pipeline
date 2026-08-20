import json

from wahlcheck_ai.llm import chat_json

EVALUATE_SYSTEM_PROMPT = """You are an expert political scientist professor conducting
independent quality control for a scientific research project.

A separate researcher is independently rating political statements against excerpts
from an election programme. You perform your OWN independent rating of the same
thesis and sources. You do NOT see the other researcher's rating - it is compared
to yours only after you have answered.

You receive:
1. `these`: One political statement.
2. `quellen`: Several relevant excerpts from one election programme.
3. `glossar`: A glossary explaining terms that may be unfamiliar or ambiguous.

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

A recurring trap is a GENERIC refrain that rejects "new burdens, requirements, bans
or costs for citizens, owners or businesses" (e.g. against extra costs, red tape or
regulation). Before treating such a refrain as DIRECT IMPLICIT evidence, BOTH of the
following must hold:

- AFFECTED PARTY: the thesis must place its burden on the SAME party the source
  protects. A source shielding private citizens, owners, tenants or businesses does
  NOT speak to the city's own spending, subsidies, or its own property/infrastructure
  decisions - those affect the municipal budget, not a private party.
- DOMAIN: the source's stated policy area must match, or clearly encompass, the
  thesis's domain. A refrain scoped to one section of the programme (e.g. housing,
  energy, migration) does not transfer to an unrelated one (e.g. school meals,
  digitalization, culture) just because both could be called a "new burden" or
  "new cost" in the abstract.

If EITHER check fails, this is INDIRECT/SPECULATIVE → 0, even if the vocabulary
overlaps (e.g. both mention "belasten", "Verbot", or "Kosten").

Example (party mismatch):
Thesis: "Auf möglichst allen geeigneten städtischen Dächern und Parkplätzen sollen Solaranlagen entstehen."
Source: "Wir lehnen eine kommunale Politik ab, die Bürger, Eigentümer, Mieter oder Unternehmen durch zusätzliche Vorgaben ... belastet."

The thesis proposes the CITY installing panels on its OWN property - not a new
obligation imposed on private citizens, owners or businesses. The source protects
private parties from new burdens; it says nothing about the city's own
infrastructure spending.
This is INDIRECT/SPECULATIVE → 0, even though both concern "Belastung".

Example (domain mismatch):
Thesis: "An allen Schulen und Kitas soll es kostenloses Mittagessen geben."
Source: "Kommunale Politik darf Bauen, Wohnen und Eigentum nicht durch immer neue Auflagen, Vorgaben und Kostentreiber zusätzlich belasten."

The source specifically objects to new requirements in construction/housing/property
policy. A free school-meal programme is a completely different policy domain
(education/social policy, not construction).
This is INDIRECT/SPECULATIVE → 0 - a refrain scoped to one domain does not transfer
to an unrelated domain merely because both could be called a "new burden".

Example (non-economic measure - a special case of the same check):
Thesis: "Die Stadt soll Werbung für die Bundeswehr auf städtischen Bussen verbieten."
Source: "Braunschweig darf Bürger und Unternehmen nicht durch zusätzliche kommunale
Vorgaben, Verbote oder Belastungen weiter unter Druck setzen."

The source rejects new economic/administrative burdens on citizens and businesses.
An advertising ban on public transport is not an economic burden on citizens or
businesses - it is a content/values decision the source does not address.
This is INDIRECT/SPECULATIVE → 0, even though both mention "Verbote".

For implicit evidence:
A source may support or oppose a thesis without using the same terminology.
Implicit support exists when a broader or equivalent principle clearly favors
the policy mechanism proposed by the thesis.

Implicit opposition exists when a source rejects, criticizes or seeks to prevent
the type of policy mechanism proposed by the thesis.

Do not infer a position merely because the source and thesis concern the same topic.

A general political goal does not automatically imply support for a specific policy instrument.

Do not use external political knowledge.
All conclusions must be grounded in the supplied sources.

Output JSON:

- `wertung`: Your independently determined rating (`1`, `0`, or `-1`).
- `sicherheit`: `"hoch"`, `"mittel"`, or `"niedrig"`.
- `zitat`: A short verbatim quote supporting your rating (empty string if `wertung` is 0).
- `zitat_nummer`: The ID of the quote used (null if `wertung` is 0).
- `kommentar`: A short explanation of your reasoning, understandable to a reader who has not seen the sources.

Prefer German.
"""

COMPARE_SYSTEM_PROMPT = """You are an expert political scientist professor writing a
short quality-control note.

You receive two independently produced ratings of the same political thesis against
the same election programme excerpts:
1. `erste_bewertung`: the first researcher's rating and explanation.
2. `zweite_bewertung`: a second researcher's rating and explanation, produced without
   seeing the first researcher's answer.

Both used the same three-point scale (`1` = support, `0` = unclear, `-1` = oppose).

Task:
Write ONE sentence in German explaining, for a reader who has not seen the excerpts,
why the two ratings agree or disagree. If they disagree, name the concrete
difference in interpretation (e.g. a different quote used, a different reading of
the same quote, or one rating treating a general principle as sufficient where the
other requires an exact mechanism match).

Do not change either rating. Do not decide who is right.

Output JSON:
- `kommentar`: One sentence.
"""


EVALUATE_SCHEMA = {
    "type": "object",
    "properties": {
        "wertung": {"type": "integer", "enum": [-1, 0, 1]},
        "sicherheit": {"type": "string", "enum": ["hoch", "mittel", "niedrig"]},
        "zitat": {"type": ["string", "null"]},
        "zitat_nummer": {"type": ["integer", "null"]},
        "kommentar": {"type": "string"},
    },
    "required": ["wertung", "sicherheit", "zitat", "zitat_nummer", "kommentar"],
}

COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "kommentar": {"type": "string"},
    },
    "required": ["kommentar"],
}

ARBITRATE_SYSTEM_PROMPT = """You are an expert political scientist professor
adjudicating a disagreement between two researchers.

Two independent researchers rated the same political thesis against the same
election programme excerpts and reached different ratings. Decide, strictly on the
evidentiary rubric below, which rating the sources actually support.

You receive:
1. `these`: One political statement.
2. `quellen`: Several relevant excerpts from one election programme.
3. `bewertung_a` / `bewertung_b`: Two independently produced ratings and their
   explanations. Their labeling as A or B carries NO meaning and does not indicate
   which was produced first - do not favor one because of its position, and do not
   average or split the difference between them.
4. `glossar`: A glossary explaining terms that may be unfamiliar or ambiguous.

Rating scale: `1` = SUPPORT, `-1` = OPPOSE, `0` = UNCLEAR.

There are three levels of evidence:

A. EXPLICIT - the source directly expresses support or opposition to the policy
   described in the thesis.
B. DIRECT IMPLICIT - the source does not mention the exact policy, but clearly
   addresses the same policy instrument, mechanism, or type of intervention.
C. INDIRECT / SPECULATIVE - the source expresses a general preference, objective or
   principle from which the thesis could potentially be derived, but the connection
   requires additional assumptions. This is NOT sufficient for `1` or `-1` - rate `0`.

A recurring trap is a GENERIC refrain that rejects "new burdens, requirements, bans
or costs for citizens, owners or businesses". Before treating such a refrain as
DIRECT IMPLICIT evidence, BOTH of the following must hold:

- AFFECTED PARTY: the thesis must place its burden on the SAME party the source
  protects. A source shielding private citizens, owners, tenants or businesses does
  NOT speak to the city's own spending, subsidies, or its own property/infrastructure
  decisions - those affect the municipal budget, not a private party.
- DOMAIN: the source's stated policy area must match, or clearly encompass, the
  thesis's domain. A refrain scoped to one section of the programme (e.g. housing,
  energy, migration) does not transfer to an unrelated one (e.g. school meals,
  digitalization, culture) just because both could be called a "new burden" or
  "new cost" in the abstract.

If EITHER check fails, this is INDIRECT/SPECULATIVE → 0, even if the vocabulary
overlaps (e.g. both mention "belasten", "Verbot", or "Kosten"). This also covers a
thesis proposing a specific, non-economic measure - such as a content-based ban, a
symbolic policy, or a values question - which a generic economic-burden refrain does
not address either.

Example (party mismatch): thesis = city installing solar panels on its OWN roofs and
parking lots; source = a refrain protecting "Bürger, Eigentümer, Mieter oder
Unternehmen" from new burdens. The city acting on its own property is not a burden on
private parties → 0, even though both concern "Belastung".

Example (domain mismatch): thesis = free school lunches; source = "Kommunale Politik
darf Bauen, Wohnen und Eigentum nicht durch immer neue Auflagen ... belasten." A
refrain scoped to construction/housing policy does not transfer to school meals → 0.

If a rating relies on this kind of generic refrain without both checks holding, that
rating is the flawed one, regardless of which letter it is labeled.

Decide independently, from the sources, which policy-mechanism match - if any - is
actually established. Two disagreeing ratings are not by themselves evidence for a
middle position: if NEITHER rating grounds its conclusion in a quote that clearly
addresses the thesis's specific policy mechanism, the correct rating is `0`,
regardless of what either researcher concluded. If exactly one rating grounds its
conclusion in a quote that is genuinely on-mechanism, prefer that one. Only agree
with a rating because its evidence and reasoning hold up - never merely because it
was presented as the objection, or because it was listed first or second.

Do not use external political knowledge. All conclusions must be grounded in the
supplied sources.

Output JSON:
- `wertung`: `1`, `0`, or `-1`.
- `sicherheit`: `"hoch"`, `"mittel"`, or `"niedrig"`.
- `zitat`: A short verbatim quote supporting your rating (empty string if `wertung` is 0).
- `zitat_nummer`: The ID of the quote used (null if `wertung` is 0).
- `kommentar`: A short explanation of your decision, understandable to a reader who has not seen the sources.

Prefer German.
"""

ARBITRATE_SCHEMA = EVALUATE_SCHEMA

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


def evaluate(these, belege: list, glossary, party, model: str) -> dict:
    """Blind, independent rating of `these` against `belege`. Never sees the
    first researcher's rating, so it cannot anchor on it."""
    user_prompt = f"""
    THESIS:

    {these}


    SOURCES ({parties[party]})
    {"\n - ".join(
        [f'ID {beleg["id"]}: __{beleg["text"]}__' for beleg in belege]
    )}


    GLOSSARY:
    {json.dumps(glossary)}
    """

    return chat_json(model, [EVALUATE_SYSTEM_PROMPT], user_prompt, EVALUATE_SCHEMA)


def compare(these, rating: dict, blind: dict, model: str) -> dict:
    """Compares an already-produced rating against a blind, independent one.
    `consens` is decided in code (exact match on `wertung`), not by the LLM -
    the model only supplies the human-readable explanation."""
    user_prompt = f"""
    THESIS:

    {these}


    ERSTE_BEWERTUNG:
    RATING: {rating["wertung"]} (Sicherheit: {rating["sicherheit"]})
    ZITAT: {rating["zitat"]}
    KOMMENTAR: {rating["kommentar"]}

    ZWEITE_BEWERTUNG:
    RATING: {blind["wertung"]} (Sicherheit: {blind["sicherheit"]})
    ZITAT: {blind["zitat"]}
    KOMMENTAR: {blind["kommentar"]}
    """

    result = chat_json(model, [COMPARE_SYSTEM_PROMPT], user_prompt, COMPARE_SCHEMA)
    return {
        "consens": rating["wertung"] == blind["wertung"],
        "eigene_wertung": blind["wertung"],
        "kommentar": result["kommentar"],
    }


def arbitrate(these, belege: list, verdict_a: dict, verdict_b: dict, glossary, model: str) -> dict:
    """Neutral third opinion for a reasoning disagreement (same evidence, different
    read). `verdict_a`/`verdict_b` should be passed in random order by the caller -
    this function does not know or care which one is the original rater's and which
    is the blind judge's, so it cannot default to either by position."""
    user_prompt = f"""
    THESIS:

    {these}


    SOURCES:
    {"\n - ".join(
        [f'ID {beleg["id"]}: __{beleg["text"]}__' for beleg in belege]
    )}


    BEWERTUNG_A:
    RATING: {verdict_a["wertung"]} (Sicherheit: {verdict_a["sicherheit"]})
    ZITAT: {verdict_a["zitat"]}
    KOMMENTAR: {verdict_a["kommentar"]}

    BEWERTUNG_B:
    RATING: {verdict_b["wertung"]} (Sicherheit: {verdict_b["sicherheit"]})
    ZITAT: {verdict_b["zitat"]}
    KOMMENTAR: {verdict_b["kommentar"]}


    GLOSSARY:
    {json.dumps(glossary)}
    """

    return chat_json(model, [ARBITRATE_SYSTEM_PROMPT], user_prompt, ARBITRATE_SCHEMA)
