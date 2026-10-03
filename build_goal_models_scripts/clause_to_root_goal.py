"""
Convert an ISO/IEC 27001 Annex A clause heading into the root goal (G0).

A clause heading is "<id> <title>", e.g.
    "A.5.1 Management direction for information security"

Two modes:
  * verbalize=False (default) -> keep the title as-is (an abstract anchor
    node, traceable to the ISO clause):
        "Management direction for information security"
  * verbalize=True -> turn the NOUN-PHRASE title into a verb-led goal:
        "Provide management direction for information security"
    The leading verb is a heuristic: a per-clause override when known,
    else the neutral, always-grammatical default "Ensure". Review the
    output, since a noun-phrase title does not carry its own verb.

A title that is ALREADY verb-led ("Protect secure areas",
"Manage information security responsibilities", "Ensure personnel security
before employment") is a goal already, so it is kept UNCHANGED. Without this
guard, verbalize=True produced a doubled verb:
    "Maintain protect secure areas", "Ensure ensure personnel security ..."
This check wins over ROOT_VERB, since the title's own verb is better
evidence than a per-clause override written for the old noun-phrase titles.
"""

import re

DEFAULT_VERB = "Ensure"

# Per-clause override of the leading verb (better-fitting than the default).
ROOT_VERB = {
    "A.5.1": "Provide",
    "A.6.1": "Establish",
    "A.6.2": "Secure",
    "A.7.3": "Manage",
    "A.8.1": "Assign",
    "A.8.3": "Control",
    "A.9.1": "Establish",
    "A.9.3": "Define",
    "A.10.1": "Apply",
    "A.11.1": "Maintain",
}

# Directive verbs that may OPEN an already verb-led title.
# Deliberately CURATED and narrow: a word that is far more often a NOUN at the
# start of a heading must stay out, otherwise a real noun-phrase title would
# never be verbalized. Kept OUT on purpose, with the title that motivates it:
#   access     "Access Control to Information Systems"   (NIST AC)
#   control    "Control of operational software"         (ISO A.12.5)
#   inventory  "Inventory and Control of Enterprise Assets" (CIS.1)
#   test       "Test data"                               (ISO A.14.3)
#   backup     "Backup"                                  (ISO A.12.3)
#   report / review / log / plan-ning / monitor-ing: same reason.
# Extend the set if your titles use other leading verbs -- and check first
# that no noun-phrase title starts with the same word.
TITLE_VERBS = {
    "ensure", "protect", "manage", "maintain", "provide", "establish",
    "define", "apply", "assign", "prevent", "limit", "restrict", "segregate",
    "classify", "handle", "dispose", "verify", "enforce", "implement",
    "govern", "secure", "mitigate", "authorize", "authorise", "authenticate",
    "acquire", "remove", "revoke", "safeguard", "preserve", "supervise",
    # --- added after scanning the titles of the five standards -------------
    # each of these produced a DOUBLED verb ("Ensure map AI systems ..."):
    "map",      # "Map AI Systems and Risks"          (AI-RMF MAP)
    "measure",  # "Measure AI Risks and Performance"  (AI-RMF MEASURE)
    "audit",    # "Audit and Maintain Accountability" (NIST AU)
    "assess",   # "Assess Security and Privacy Risks" (NIST RA)
    "plan",     # "Plan System Security and Privacy"  (NIST PL)
    # safe additions: their noun form is a different token
    # ("Response", "Recovery", "Monitoring", "Deployment", "Evaluation")
    "respond", "recover", "detect", "evaluate", "operate", "deploy",
    "oversee", "treat",
}


def lower_title_object(title):
    """
    Lower-case a Title-Cased heading so it can serve as the object of a verb,
    while PRESERVING acronyms.
        "Access Control"          -> "access control"
        "Audit and Accountability"-> "audit and accountability"
        "AI policy"               -> "AI policy"      (acronym kept)
    Without this, "Ensure " + "Access Control" read "Ensure aC Access Control"
    / "Ensure access Control".
    """
    out = []
    for w in title.split(" "):
        # An acronym (>=2 chars, all caps) keeps its case; so does any word
        # with internal capitals. Only plain Title-Case words are lowered.
        if len(w) >= 2 and w.isupper():
            out.append(w)
        elif w[:1].isupper() and w[1:].islower():
            out.append(w.lower())
        else:
            out.append(w)
    return " ".join(out)


def starts_with_verb(title):
    """
    True if the title is already a verb-led goal ("Protect secure areas").
    Requires at least two words, so a bare verb is not mistaken for a title.
    """
    words = title.strip().split()
    if len(words) < 2:
        return False
    return words[0].lower().strip(",:;") in TITLE_VERBS


# Titles beginning with one of these are temporal/scope qualifiers
# (a "when", not a "what"): no verb turns them into a sensible goal,
# so we keep the raw title and the verbal goal must come from the objective.
NON_VERBALIZABLE_PREFIXES = ("prior to", "during", "before", "after",
                             "while", "upon", "throughout", "in the event")


def is_verbalizable(title):
    """
    False if the title is a temporal/scope phrase (e.g. 'Prior to
    employment') that cannot be turned into a verb-led goal.
    """
    return not title.strip().lower().startswith(NON_VERBALIZABLE_PREFIXES)


def clause_to_root_goal(clause, verbalize=False):
    """
    Convert one clause heading into the root goal.

    clause_to_root_goal("A.5.1 Management direction for information security")
    -> {"goal_id": "G0", "clause_id": "A.5.1",
        "statement": "Management direction for information security"}

    clause_to_root_goal("A.5.1 Management direction...", verbalize=True)
    -> {..., "statement": "Provide management direction for information security"}
    """
    text = re.sub(r"\s+", " ", clause.strip())
    # Extract Clause ID and Title (Regex)
    # Accepts: "A.5.1" (ISO), "AC" / "AC-1" (NIST family), "CIS.1.1" (CIS),
    # "GOVERN" (AI-RMF), "5.1" (numeric). The alphabetic part must be
    # UPPER-CASE so an ordinary title word is never mistaken for an id.
    m = re.match(r"^([A-Z][A-Z0-9]*(?:[.\-][A-Za-z0-9]+)*|\d+(?:\.\d+)*)\s+(.*)$",
                 text)
    clause_id, title = (m.group(1), m.group(2)) if m else (None, text)
    # Clean the title
    title = title.rstrip(".")

    if verbalize and title and is_verbalizable(title) \
            and not starts_with_verb(title):
        # Noun-phrase title -> prepend a verb to make it a goal.
        verb = ROOT_VERB.get(clause_id, DEFAULT_VERB)
        obj = lower_title_object(title) # Title becomes the object
        statement = f"{verb} {obj}"
    elif verbalize and title and starts_with_verb(title):
        # Already a goal: keep the title's own verb, but normalize the case of
        # the rest so it reads like the other goals ("Map AI Systems and Risks"
        # -> "Map AI systems and risks"). Acronyms are preserved.
        normalized = lower_title_object(title)
        statement = normalized[0].upper() + normalized[1:]
    else:
        # Not verbalizable (temporal/scope title) -> keep the title.
        statement = title[0].upper() + title[1:] if title else title

    return {"goal_id": "G0", "clause_id": clause_id, "statement": statement}