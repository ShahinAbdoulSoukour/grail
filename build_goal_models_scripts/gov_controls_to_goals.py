"""
Transform each governance control into goals.

Patterns covered (in matching order):
P1  "There shall be a(n) <NP>" --> Establish <NP>
P2  "<subj> shall be <adj/state complement>" --> Ensure <subj> is/are <complement>
P3  "<subj> shall be <verb_ed>(, <verb_ed>)* ..." --> one leaf goal per participle
    - handles "by <agent>" attached to a specific participle
    - handles phrasal verbs (carried out, disposed of, drawn up)
    - optionally splits disjunctive triggers ("at planned intervals
      OR if significant changes occur") into separate leaf goals
P4  "<subj> shall <verb> ..." (active voice) --> goal with subj as agent
    "<subj> <verb>s ..." (active present)  --> goal, subj recorded as agent
P5  "<verb1>, <verb2>, and <verb3> <object>" (imperative, no "shall")
    --> one leaf goal per verb
    - for standards written as directives (NIST SP 800-53, CIS), which are
      already goal statements: only the leading verb chain needs splitting
      ("Develop, document, and disseminate policy" --> 3 goals)
    - a control with a single verb is kept as one leaf goal, unchanged
P6  "<subj> are/is <verb_ed>(, <verb_ed>)*" (declarative, no "shall")
    --> one leaf goal per participle, subject becomes the object
    - for frameworks written as DESIRED STATES rather than directives
      (NIST AI-RMF): "Roles and responsibilities ... are defined, documented
      and understood" --> "Define roles ...", "Document roles ...",
      "Understand roles ..."
    - reuses the P3 segment machinery ("are/is" instead of "shall be")
P7  anything none of the above can rewrite    --> Ensure <sentence>
    - last resort for copulas and PLURAL present indicatives, which offer
      neither a participle nor an -s marker to pivot on:
        "Evaluations involving human subjects meet applicable requirements
         and are representative of the relevant population."
      --> "Ensure evaluations involving human subjects meet applicable
           requirements and are representative of the relevant population"
    - returning the sentence verbatim instead would leave a NON-GOAL in the
      model; wrapping it keeps every control resolvable while inventing
      nothing. With P7, no control is ever left UNMATCHED.

MATCHING ORDER
P1 -> P2/P3 -> P4 -> {P5, P6} -> P4b -> P7
Matching ORDER matters: P5 is tried BEFORE P6, because a directive may
contain "are" in a subordinate clause (NIST CA-2: "Assess the controls ...
to determine if they are implemented correctly") and must not be read as a
declarative statement.
"""

import re

# --- Lexical resources ---
# Past participle --> base form, for verbs occurring in ISO 27001 Annex A.
# A dictionary is preferred over stemming heuristics: it is transparent,
# auditable, and trivially extensible when moving to another standard.
PARTICIPLE_LEMMAS = {
    "defined": "define", "approved": "approve", "published": "publish",
    "communicated": "communicate", "reviewed": "review",
    "allocated": "allocate", "segregated": "segregate",
    "maintained": "maintain", "addressed": "address", "adopted": "adopt",
    "implemented": "implement", "stated": "state", "required": "require",
    "established": "establish", "documented": "document",
    "identified": "identify", "owned": "own", "returned": "return",
    "classified": "classify", "developed": "develop",
    "protected": "protect", "restricted": "restrict",
    "controlled": "control", "provided": "provide", "removed": "remove",
    "adjusted": "adjust", "designed": "design", "applied": "apply",
    "isolated": "isolate", "enforced": "enforce", "used": "use",
    "screened": "screen", "managed": "manage", "labelled": "label",
    "handled": "handle", "encrypted": "encrypt", "verified": "verify",
    "monitored": "monitor", "tested": "test", "recorded": "record",
    "assigned": "assign", "revoked": "revoke", "limited": "limit",
    # --- verbs whose base form ends in -e (the -ed heuristic drops it:
    #     "determined" -> "determin"). Added after scanning ISO 27001,
    #     ISO 42001, NIST SP 800-53 and CIS.
    "determined": "determine", "acquired": "acquire", "associated": "associate",
    "authorized": "authorize", "authorised": "authorise", "automated": "automate",
    "based": "base", "dedicated": "dedicate", "designated": "designate",
    "disposed": "dispose", "escalated": "escalate", "exposed": "expose",
    "introduced": "introduce", "operated": "operate", "related": "relate",
    "sanitized": "sanitize", "sanitised": "sanitise", "stored": "store",
    "updated": "update", "validated": "validate", "generated": "generate",
    "categorized": "categorize", "separated": "separate", "terminated": "terminate",
    "allocated": "allocate", "activated": "activate", "initiated": "initiate",
    # --- verbs where the double-consonant rule misfires
    #     ("processed" -> "proces", "installed" -> "instal")
    "processed": "process", "accessed": "access", "installed": "install",
    "assessed": "assess", "addressed": "address", "discussed": "discuss",
    # --- verbs met in NIST AI-RMF (declarative statements, pattern P6)
    "understood": "understand", "integrated": "integrate", "elicited": "elicit",
    "considered": "consider", "examined": "examine", "specified": "specify",
    "tracked": "track", "selected": "select", "involved": "involve",
    "demonstrated": "demonstrate", "measured": "measure", "evaluated": "evaluate",
    "explained": "explain", "interpreted": "interpret", "informed": "inform",
    "prioritized": "prioritize", "aligned": "align", "exercised": "exercise",
    "improved": "improve", "accepted": "accept", "retained": "retain",
    "reported": "report", "mitigated": "mitigate", "remediated": "remediate",
    "deployed": "deploy", "trained": "train", "resourced": "resource",
    "disseminated": "disseminate", "corrected": "correct", "detected": "detect",
    "made": "make", "given": "give", "kept": "keep", "built": "build",
    "re-evaluated": "re-evaluate", "socialized": "socialize",
    "disabled": "disable", "enabled": "enable", "revoked": "revoke",
    # phrasal verbs
    "carried out": "carry out", "drawn up": "draw up",
    "disposed of": "dispose of", "set up": "set up", "put in": "put in",
}

# Words ending in "-ed" that are NOT action participles in these standards:
# adjectives/modifiers ("unauthorized access", "privileged users", "detailed
# inventory") and the plain noun/verb "need". They must never trigger a verb
# split, otherwise "employees and privileged users" would be cut in two.
NON_PARTICIPLES = {
    "need", "needed", "detailed", "privileged", "unauthorized", "unauthorised",
    "interested", "unintended", "intended", "skilled", "trusted", "advanced",
    "limited", "related", "based", "dedicated", "designated", "automated",
    "connected", "interconnected", "affected", "committed",
}

# Nouns that take a plural verb without a trailing "-s"
# ("personnel are aware", not "personnel is aware").
IRREGULAR_PLURALS = {"personnel", "media", "data", "staff", "people",
                     "criteria", "phenomena", "equipment"}

# Particles that may follow a participle to form a phrasal verb.
PHRASAL_PARTICLES = {"of", "out", "up"}

# Participles to be treated as state complements
# "<subject> shall be <adjective|noun_complement>" --> no real action verb.
STATE_COMPLEMENTS = {"owned", "accountable", "available", "interactive",
                     "aware", "responsible", "suitable", "adequate",
                     "valid", "reliable", "safe", "secure", "current",
                     "unique", "consistent", "traceable"}

# Prepositions/markers that end an agent noun phrase ("by management at ...").
AGENT_STOPWORDS = {"at", "to", "in", "for", "upon", "when", "if",
                   "during", "before", "after", "through", "against"}

# P5: base-form verbs that may OPEN a directive control (NIST, CIS). The
# check is exact-match on the first word, so a plural noun ("Processes",
# "Controls", "Roles") never counts as a verb and correctly falls through
# to P6. Built from the participle lemmas + directive-specific verbs.
IMPERATIVE_VERBS = set(PARTICIPLE_LEMMAS.values()) | {
    "ensure", "manage", "enforce", "employ", "display", "prevent", "respond",
    "conduct", "perform", "include", "utilize", "provide", "plan", "collect",
    "configure", "filter", "log", "secure", "segment", "scan", "patch",
    "install", "uninstall", "upgrade", "centralize", "block", "harden",
    "separate", "categorize", "analyze", "analyse", "disseminate", "correct",
    "detect", "delete", "disable", "enable", "revoke", "restrict", "protect",
    "monitor", "identify", "report", "train", "deploy", "retain", "mitigate",
    "remediate", "uniquely", "authenticate", "allowlist", "standardize",
    "standardise", "inventory", "encrypt", "decommission", "isolate",
    "restrict", "verify", "validate", "classify", "label", "dispose",
}

# Adverbs that may precede the verb chain ("Uniquely identify ...",
# "Actively manage ...") or the participle list ("are clearly defined ...").
LEADING_ADVERB_RE = re.compile(r"^(?P<adv>\w+ly)\s+(?=\w)", re.IGNORECASE)

# Disjunctive trigger markers ("... at planned intervals OR IF ... occur").
TRIGGER_SPLIT_RE = re.compile(r"\s+or\s+(if|when|upon)\s+", re.IGNORECASE)

# Purpose clause markers (shared by all split triggers).
PURPOSE_RE = re.compile(r"\s+to\s+(ensure|guarantee|avoid|prevent|enable)\b", re.IGNORECASE)


# --- Small helpers ---
def normalize(text):
    """
    Trim, collapse whitespace, drop a trailing period.

    :param text: a string (text) which may contain leading/trailing spaces, multiple spaces, newlines, tabs, or a final full stop.
    """
    # Trim and collapse whitespace
    text = re.sub(r"\s+", " ", text.strip())
    # a clean, single‑spaced string with no trailing dot.
    return text[:-1] if text.endswith(".") else text


def looks_like_participle(word):
    """
    True if `word` is plausibly a past participle (dictionary or -ed).

    :param word: the word to evaluate.
    """
    # Convert to lowercase AND strip surrounding punctuation. The lookahead in
    # split_verb_segments passes the RAW next token, so "escalated," kept its
    # comma and failed this test -- which is why an Oxford comma
    # ("reported, escalated, and addressed") broke pattern P3.
    w = word.lower().strip(",;:.()\"'")

    # Known non-participles (adjectives, "need") never count.
    if w in NON_PARTICIPLES:
        return False

    # It returns True if either condition is met:
    # 1. Exact match in PARTICIPLE_LEMMAS (irregular verbs)
    # 2. Suffix rule for regular verbs (-ed + stem length guard: the stem left
    #    after dropping "ed" must be >= 3 chars, so "need" -> "ne" is rejected)
    return w in PARTICIPLE_LEMMAS or (w.endswith("ed") and len(w) - 2 >= 3)


def to_base_form(participle):
    """
    This function is the inverse transformation of looks_like_participle.
    Once a word is flagged as a past participle, this function converts it back to its base (infinitive/imperative) form.

    Past participle -> base (imperative) form. Dictionary first,
    then conservative -ed heuristics as a fallback.

    :param participle: a string representing a single English verb --> past participle.
    """
    p = participle.lower().strip(",;:.()\"'") # Lowercase + strip punctuation

    # Dictionary lookup for irregulars
    # Irregulars first, because they don't follow suffix rules.
    # Ex: "taken" → "take", "built" → "build"
    # This avoids attempting nonsensical suffix stripping on irregular forms.
    if p in PARTICIPLE_LEMMAS:
        return PARTICIPLE_LEMMAS[p]

    # Handle ‑ied --> ‑y (Y‑to‑I Rule Reversal)
    # Verbs ending in consonant + y change y --> i before adding ‑ed (e.g., identify --> identif*ied*).
    # This rule simply strips "ied" and appends "y".
    # Ex: "identified" → "identif" + "y" = "identify".
    if p.endswith("ied"): # identified -> identify
        return p[:-3] + "y"

    # Handle Regular ‑ed (Two‑Step Heuristic)
    if p.endswith("ed"):
        stem = p[:-2] # # remove "ed" suffix to get a stem (e.g., "controlled" --> "controlle").
        # Double‑Consonant rule
        # Ex: control --> controlled (stem: controlle --> last two are "ll" --> remove one --> "control").
        if len(stem) > 2 and stem[-1] == stem[-2]: # controlled -> control
            return stem[:-1]
        return stem # review(ed) -> review

    # If the word doesn't match any rule (e.g., not in dictionary, doesn't end with ‑ied or ‑ed),
    # return it as‑is (lowercased). This prevents crashes on unexpected input.
    return p



def lower_subject(subject):
    """
    Normalize the capitalization of a noun phrase.

    This function lowercases the first word of a subject noun phrase
    when it is either:
      - a common determiner (e.g., "The", "A", "An"), or
      - a title-cased word (e.g., "User", "System").

    This normalization is useful when transforming passive requirements
    into active goals, where the original subject of the passive sentence
    becomes the object of the active sentence.
    Lowercasing the initial word helps the resulting phrase read naturally.

    :param subject: the subject noun phrase to normalize.
    """
    words = subject.split(" ", 1)
    first = words[0]
    # Checks if the first word:
    # - is a common determiner (a, an, the, etc.), regardless of capitalization, or
    # - is title-cased (first letter uppercase, rest lowercase), such as "User" or "System".
    if first.lower() in {"a", "an", "the", "all", "any", "each"} or first.istitle():
        words[0] = first.lower()
    return " ".join(words) # Reassembles the phrase and returns the normalized noun phrase.



def capitalize_first(text):
    """
    Capitalize the first word of a noun phrase.

    :param text: the noun phrase to capitalize.
    """
    # Returns the string with its first character converted to uppercase.
    # If the input string is empty, it is returned unchanged.
    return text[0].upper() + text[1:] if text else text



# ----- Segment handling for multi-verb passives (pattern P3) -----
def split_verb_segments(verb_part):
    """
    Split 'defined, approved by management, published and communicated to X'
    into ['defined', 'approved by management', 'published',
          'communicated to X'].

    A split only happens at ',' / 'and' / 'or' when the *next* token looks
    like a participle, so coordinations inside trailing modifiers
    ('employees and relevant external parties') are preserved.
    Returns a list of (segment, connector) pairs; the connector ('and'/'or')
    is the one *preceding* the segment (None for the first one).

    :param verb_part: the verb part to split.
    """
    # Split the input string into individual tokens (words)
    tokens = verb_part.split(" ")

    # segments : stores the completed verb segments
    # current : tokens currently being collected for one segment
    # connectors : separator preceding each segment (None, ',', 'and', 'or')
    # pending : separator to assign to the next segment
    segments, current, connectors, pending = [], [], [], None

    # Start scanning tokens from the beginning
    i = 0

    # Process all tokens one by one
    while i < len(tokens):
        tok = tokens[i] # Current token

        # Remove a trailing comma if present
        # Ex: "defined," -> "defined"
        bare = tok.rstrip(",")

        # True if token ends with a comma
        # Ex: "defined," --> True
        trailing_comma = tok.endswith(",")
        # Look ahead to the next token (if any)
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None

        # Case 1: Split on "and" or "or"
        # Only split if the next token looks like a participle.
        # Ex: published and communicated
        # becomes:
        # published
        # communicated
        # But:
        # employees and external parties
        # should NOT be split because "external" is not a participle.
        if bare.lower() in {"and", "or"} and nxt and looks_like_participle(nxt):
            # If we already collected tokens for a segment
            if current: # close previous segment,
                # # Remove any trailing comma left on the last token
                current[-1] = current[-1].rstrip(",")  # drop boundary comma
            # Join collected tokens into a complete segment
            segments.append(" ".join(current))
            # Store the connector that preceded this segment
            connectors.append(pending)
            # Start a new segment
            # Remember whether this boundary was "and" or "or"
            current, pending = [], bare.lower()
        # Case 2: Split on a comma
        # Only split if the next token looks like a participle.
        # Ex: defined, approved
        # becomes:
        # defined
        # approved
        elif trailing_comma and nxt and looks_like_participle(nxt):
            # Add the current word without the comma
            current.append(bare) # split here: drop the comma
            # Save the completed segment
            segments.append(" ".join(current))
            # Save the connector preceding this segment
            connectors.append(pending)
            # Start a new segment
            # Remember that the next segment follows a comma
            current, pending = [], ","
        # Case 3: Normal token
        # Keep collecting tokens into the current segment.
        else:
            current.append(tok) # keep internal commas as-is
        i += 1 # Move to the next token

    # After the loop, there may still be one unfinished segment.
    # Add it to the result.
    if current:
        segments.append(" ".join(current))
        connectors.append(pending)

    # Combine each segment with its corresponding connector
    # A list of (segment, connector) pairs. Each segment contains one
    # participial phrase, and connector indicates how it was connected
    # to the previous segment.
    return list(zip(segments, connectors))



def parse_segment(segment):
    """
    This function decomposes one segment into (participle, agent, rest).
    'approved by management'        --> ('approved', 'management', '')
    'communicated to employees ...' --> ('communicated', None, 'to employees ...')
    'carried out in accordance ...' --> ('carried out', None, 'in accordance ...')

    :param segment: the segment to parse.
    """
    # Split the segment into individual words
    words = segment.split(" ")
    participle = words[0]
    idx = 1 # Start scanning from the second word

    # Handle phrasal verbs (multi-word participles)
    # Ex: "carried out", "drawn up", "disposed of"
    # Condition:
    # - There is a second word
    # - The second word is in PHRASAL_PARTICLES (e.g., "out", "up", "off")
    # - The combined phrase exists in PARTICIPLE_LEMMAS
    if idx < len(words) and words[idx].lower() in PHRASAL_PARTICLES \
            and f"{participle.lower()} {words[idx].lower()}" in PARTICIPLE_LEMMAS:
        # Extend participle to include the particle
        participle = f"{participle} {words[idx]}"
        idx += 1

    # Detect agent introduced by "by"
    # Ex: "approved by management"
    agent = None
    if idx < len(words) and words[idx].lower() == "by":
        idx += 1 # Skip the word "by"
        agent_words = [] # Collect words that form the agent phrase

        # Continue until:
        # - end of segment, OR
        # - a stopword is encountered (e.g., "to", "in", "for")
        while idx < len(words) and words[idx].lower() not in AGENT_STOPWORDS:
            agent_words.append(words[idx].rstrip(",")) # Remove trailing commas from words
            # If a word ends with a comma, stop early (boundary marker)
            if words[idx].endswith(","):
                idx += 1
                break
            # Move to next word
            idx += 1
        # Combine collected words into a single agent string
        agent = " ".join(agent_words)

    # Remaining words after participle + agent form the "rest"
    # Ex: "to employees", "in accordance with policy"
    rest = " ".join(words[idx:])

    # Return structured decomposition
    return participle, agent, rest



def split_disjunctive_triggers(rest):
    """
    Split a trigger expression containing alternative conditions.

    Ex:
    'at planned intervals or if significant changes occur to ensure X'
    Become:
        --> ['at planned intervals to ensure X',
            'if significant changes occur to ensure X']

    The purpose clause ('to ensure ...') is factored out and re-attached
    to every trigger. Returns [rest] when there is nothing to split.

    :param rest: Remaining text after the participle/agent has been removed.
    """
    purpose = ""

    # Search for a purpose clause using PURPOSE_RE
    match = PURPOSE_RE.search(rest)
    if match:
        # Keep the purpose clause separately
        purpose = rest[match.start():]
        # Remove it from the trigger text
        rest = rest[:match.start()]

    # Split trigger conditions.
    # Ex: "at planned intervals or if significant changes occur"
    # may become:
    # [
    #  "at planned intervals",
    #  "if",
    #  "significant changes occur"
    # ]
    # depending on TRIGGER_SPLIT_RE.
    parts = TRIGGER_SPLIT_RE.split(rest)
    # No split occurred.
    # Return the original trigger plus any purpose clause.
    if len(parts) == 1:
        return [rest + purpose]
    # First trigger is always the text before the first marker.
    triggers = [parts[0].strip()]
    # re.split keeps captured groups.
    # Ex:
    # parts =
    # [
    #   "at planned intervals ",
    #   "if",
    #   " significant changes occur"
    # ]
    # marker = "if"
    # clause = " significant changes occur"
    # Result: "if significant changes occur"
    for marker, clause in zip(parts[1::2], parts[2::2]):
        triggers.append(f"{marker.lower()} {clause.strip()}") # Rebuild the trigger condition

    # Re-attach the purpose clause to every trigger.
    # Ex:
    #   trigger:
    #       "if significant changes occur"
    #   purpose:
    #       "to ensure effectiveness"
    # Result:
    #  "if significant changes occur
    #  to ensure effectiveness"
    return [f"{t}{purpose}" for t in triggers]


# ----- The four patterns -----
def match_there_shall_be(control_id, text):
    """
    P1: 'There shall be a(n) <NP>' --> 'Establish <NP>'.

    Match requirements of the form:
        "There shall be a <NP>"
        "There shall be an <NP>"
    and transform them into a goal of the form:
        "Establish <NP>"
    Ex:
        Input: "There shall be an information security policy"
        Output: "Establish an information security policy"

    :param control_id: Identifier of the control/requirement.
    :param text: Requirement text to analyze.
    """
    m = re.match(r"^There shall be (an?\s+.+)$", text, re.IGNORECASE)
    # If the requirement does not match the pattern,
    # this transformation is not applicable.
    if not m:
        return None
    # m.group(1) contains everything after: "There shall be "
    # A list containing one generated goal, or None if the pattern does not match.
    # control_id: source control identifier
    # "P1_there_shall_be": transformation pattern id
    # "establish": action verb (goal verb)
    # f"Establish {m.group(1)}": generated goal text
    # None: no additional metadata
    return [make_goal(control_id, "P1_there_shall_be", "establish",
                      f"Establish {m.group(1)}", None)]



def match_passive(control_id, text, split_triggers):
    """
    P3/P2: Match passive requirements of the form: <subject> shall be <past participle> ...
    Ex:
        "Information shall be classified."
        "Assets shall be owned."
        "Policies shall be defined, approved and communicated."
    The function generates:
      - P2 goals for state complements (e.g., "owned")
      - P3 goals for passive actions (e.g., "classified", "approved")

    :param control_id: Identifier of the control/requirement.
    :param text: Requirement text to analyze.
    :param split_triggers: Whether disjunctive triggers should be expanded.
    """
    # Match: <subject> shall be <verbs>
    # Ex: "Assets shall be owned"
    # subject = "Assets"
    # verbs = "owned"
    # Ex: Policies shall be defined, approved and communicated"
    # subject = "Policies"
    # verbs = "defined, approved and communicated"
    m = re.match(r"^(?P<subject>.+?) shall be (?P<verbs>.+)$", text)
    # Pattern does not match.
    if not m:
        return None
    # Normalize the subject so it can later become the object of an active goal.
    # Ex: "The policy" -> "the policy"
    subject = lower_subject(m.group("subject"))
    goals = [] # List of generated goals

    # Split coordinated participles.
    # Ex: "defined, approved and communicated"
    # becomes:
    #   ("defined", None)
    #   ("approved", ",")
    #   ("communicated", "and")
    for segment, connector in split_verb_segments(m.group("verbs")):
        # Extract:
        #   participle
        #   agent introduced by "by"
        #   remaining modifiers
        #
        # Ex: "approved by management annually"
        # becomes:
        #   participle = "approved"
        #   agent = "management"
        #   rest = "annually"
        participle, agent, rest = parse_segment(segment)

        # If the first word is neither a participle NOR a known state
        # complement, this is probably not a passive requirement.
        # NOTE: the state-complement test must be part of THIS guard. It used
        # to run only below, so any complement not ending in "-ed" ("aware",
        # "accountable", "available", "interactive") was rejected here and P2
        # could never fire for it.
        if (not looks_like_participle(participle)
                and participle.lower() not in STATE_COMPLEMENTS):
            return None  # not a passive after all -> try next pattern

        # --- P2: state complement ---
        # Ex:
        #   "Assets shall be owned"
        #   "Responsibilities shall be assigned"
        # These express a desired state rather than an action.
        # Generated goal --> "Ensure assets are owned"
        # ------
        if participle.lower() in STATE_COMPLEMENTS:
            # Find the HEAD noun of the subject to pick "is" vs "are".
            # English noun phrases are head-FINAL ("password management
            # systems" -> "systems"), so taking the first word gave the wrong
            # number. But a postmodifier flips that ("assets maintained in the
            # inventory" -> "assets"), so we stop at the first preposition or
            # participle and take the LAST word before it.
            words = [w for w in subject.split()
                     if w.lower() not in {"a", "an", "the", "all", "any"}]
            core = []
            for w in words:
                if w.lower() in {"of", "in", "on", "for", "to", "with", "by",
                                 "from", "at"} or looks_like_participle(w):
                    break
                core.append(w)
            head = (core[-1] if core else (words[0] if words else subject))
            head = head.strip(",;:").lower()

            # Plural detection: trailing -s (but not -ss), plus irregulars
            # that take a plural verb without an -s ("personnel are aware").
            verb_be = "are" if (head in IRREGULAR_PLURALS
                                or (head.endswith("s") and not head.endswith("ss"))) \
                      else "is"
            # Create the state goal.
            goals.append(make_goal(
                control_id, "P2_state_complement", participle.lower(),
                f"Ensure {subject} {verb_be} {participle.lower()}"
                + (f" {rest}" if rest else ""), agent, connector))
            continue # Skip the P3 transformation.

        # --- P3: action participle --> active leaf goal(s) ---
        # Ex: "Information shall be classified"
        # becomes: "Classify information"
        # ------
        # Convert participle to infinitive/base form.
        # classified --> classify
        # approved --> approve
        base = to_base_form(participle)

        # Optionally expand disjunctive triggers.
        # Ex: at planned intervals or if changes occur"
        # becomes:
        # [
        #   "at planned intervals",
        #   "if changes occur"
        # ]
        rests = split_disjunctive_triggers(rest) if split_triggers else [rest]
        # Generate one goal per trigger variant.
        for k, r in enumerate(rests):
            # Construct active-voice goal.
            # Ex: "Classify information"
            statement = f"{capitalize_first(base)} {subject}" + (f" {r}" if r else "")
            # Additional trigger alternatives are connected by OR.
            conn = "or" if k > 0 else connector
            goals.append(make_goal(control_id, "P3_passive_multi_verb",
                                   base, statement, agent, conn))
    # List of generated goals or None if the pattern does not match.
    return goals or None



def match_active(control_id, text):
    """
    P4: Match active-voice requirements of the form:
        <subject> shall <verb phrase>
    Ex:
        "Management shall review the policy."
        "The organization shall establish objectives."
    In this pattern, the grammatical subject is already the
    responsible actor (agent) of the goal.

    :param control_id: Identifier of the control/requirement.
    :param text: Requirement text to analyze.
    """
    # Match: <subject> shall <verb phrase>
    # The negative look-ahead (?!be\b) excludes: "shall be ..."
    # because passive requirements are handled separately
    # by the P2/P3 transformation patterns.
    #
    # Ex: "Management shall review the policy"
    # subject = "Management"
    # vp = "review the policy"
    m = re.match(r"^(?P<subject>.+?) shall (?P<vp>(?!be\b).+)$", text)
    # The requirement does not follow the active pattern.
    if not m:
        return None
    # Extract the verb phrase.
    # Ex: "review the policy"
    vp = m.group("vp")
    # Generate the goal.
    # vp.split(" ")[0]
    # extracts the main action verb:
    #   "review the policy" -> "review"
    #   "establish objectives" -> "establish"
    # capitalize_first(vp)
    # converts: "review the policy"
    # into: "Review the policy"
    # The original subject becomes the responsible
    # agent of the generated goal.
    return [make_goal(control_id, "P4_active", vp.split(" ")[0],
                      capitalize_first(vp), m.group("subject"))]




def match_imperative(control_id, text):
    """
    P5: directive control already written as a goal (NIST SP 800-53, CIS).
        "<verb1>, <verb2>, and <verb3> <object>" --> one leaf goal per verb.

    Ex:
        "Develop, document, and disseminate policy and procedures ..."
        --> "Develop policy and procedures ..."
            "Document policy and procedures ..."
            "Disseminate policy and procedures ..."
        "Address Unauthorized Assets"        --> unchanged (single verb)
        "Uniquely identify and authenticate users" --> 2 goals (adverb kept)

    The control is recognized ONLY when its first word is a known base-form
    verb (IMPERATIVE_VERBS), optionally preceded by an adverb. This exact
    match is what stops a plural noun subject ("Processes ... are established")
    from being mistaken for a verb, so such sentences fall through to P6.

    :param control_id: Identifier of the control/requirement.
    :param text: Requirement text to analyze.
    """
    # Optional leading adverb ("Uniquely identify ...", "Actively manage ...").
    adverb = ""
    m_adv = LEADING_ADVERB_RE.match(text)
    if m_adv:
        adverb = m_adv.group("adv")
        text = text[m_adv.end():]

    words = text.split(" ")
    if not words:
        return None

    # The first word must be a known verb, else this is not a directive.
    if words[0].lower().rstrip(",") not in IMPERATIVE_VERBS:
        return None

    # Collect the leading verb chain: verbs separated by "," / "and" / "or".
    # Stop at the first token that is not a verb or a connector: that token
    # opens the object ("policy and procedures ...").
    verbs, i = [], 0
    while i < len(words):
        bare = words[i].lower().rstrip(",")
        if bare in {"and", "or"}:
            # a connector only continues the chain if a verb follows
            nxt = words[i + 1].lower().rstrip(",") if i + 1 < len(words) else ""
            if nxt in IMPERATIVE_VERBS:
                i += 1
                continue
            break
        if bare in IMPERATIVE_VERBS:
            verbs.append(words[i].rstrip(","))
            had_comma = words[i].endswith(",")
            i += 1
            # the chain continues only through a comma or a connector
            if not had_comma and i < len(words) \
                    and words[i].lower().rstrip(",") not in {"and", "or"}:
                break
            continue
        break

    rest = " ".join(words[i:]).strip()
    if not verbs:
        return None

    # A single verb: the control IS already a goal, keep it unchanged.
    if len(verbs) == 1:
        statement = capitalize_first(
            (f"{adverb} " if adverb else "") + " ".join([verbs[0]] + words[i:]).strip())
        return [make_goal(control_id, "P5_imperative",
                          to_base_form(verbs[0]), statement, None)]

    # Several verbs: one leaf goal each, sharing the same object.
    goals = []
    for k, v in enumerate(verbs):
        head = f"{adverb} {v}" if adverb else v
        statement = capitalize_first(head + (f" {rest}" if rest else ""))
        goals.append(make_goal(control_id, "P5_imperative", v.lower(),
                               statement, None, None if k == 0 else "and"))
    return goals


def match_declarative(control_id, text, split_triggers):
    """
    P6: desired-state statement (NIST AI-RMF), no "shall", no leading verb:
        "<subject> are/is/has been <verb_ed>(, <verb_ed>)* ..."
    --> one leaf goal per participle, the subject becoming the object.

    Ex:
        "Roles and responsibilities for AI risk management are clearly
         defined, documented, and understood."
        --> "Define roles and responsibilities for AI risk management"
            "Document roles and responsibilities for AI risk management"
            "Understand roles and responsibilities for AI risk management"

    Implemented by reusing the P3 machinery (split_verb_segments /
    parse_segment): a declarative is grammatically a passive whose auxiliary
    is "are/is" instead of "shall be". A leading adverb ("clearly",
    "regularly") is stripped first, since it is not a participle.

    :param control_id: Identifier of the control/requirement.
    :param text: Requirement text to analyze.
    :param split_triggers: Whether disjunctive triggers should be expanded.
    """
    m = re.match(r"^(?P<subject>.+?)\s+(?:are|is|were|was|"
                 r"has\s+been|have\s+been)\s+(?P<verbs>.+)$", text)
    if not m:
        return None

    subject = lower_subject(m.group("subject"))
    verb_part = m.group("verbs")

    # Drop an adverb sitting between the auxiliary and the participle.
    m_adv = LEADING_ADVERB_RE.match(verb_part)
    if m_adv:
        verb_part = verb_part[m_adv.end():]

    goals = []
    for segment, connector in split_verb_segments(verb_part):
        participle, agent, rest = parse_segment(segment)

        # Guard: the first token must really be a participle, otherwise the
        # "are/is" we matched was a copula before an adjective
        # ("... are representative of the population") -> not a P6 statement.
        if not looks_like_participle(participle):
            return None

        base = to_base_form(participle)
        rests = split_disjunctive_triggers(rest) if split_triggers else [rest]
        for k, r in enumerate(rests):
            statement = f"{capitalize_first(base)} {subject}" + (f" {r}" if r else "")
            conn = "or" if k > 0 else connector
            goals.append(make_goal(control_id, "P6_declarative",
                                   base, statement, agent, conn))
    return goals or None



def make_goal(control_id, pattern, verb, statement, agent, connector=None):
    """
    Assemble a structured goal record (leaf goal) produced by a
    requirement transformation rule.
    Each goal represents a normalized requirement element extracted
    from a control statement.

    :param control_id: Identifier of the source requirement/control.
    :param pattern: Name of the transformation rule used (e.g., P2_state_complement, P3_passive_multi_verb, P4_active).
    :param verb: Canonical verb or base form of the action (e.g., "classify", "review", "establish").
    :param statement: Natural-language goal statement generated from the rule.
    :param agent: Responsible entity (subject of active voice or agent of passive construction). Can be None if not applicable.
    :param connector : str | None
        Logical connector linking this goal to the previous one:
        - "and"
        - "or"
        - ","
        - None (for the first goal in a sequence)
    """
    return {
        "control_id": control_id, # Source requirement identifier
        "pattern": pattern, # Transformation pattern used to generate this goal
        "verb": verb, # Normalized action verb (base form)
        "agent": agent, # Responsible agent (if identified)
        "connector": connector,   # Connector linking this goal to previous one in sequence
                                  # 'and' / 'or' / ',' linking it to the
                                  # previous leaf goal (None for the first)
        "statement": statement, # Final natural-language goal statement
    }




# ----- Entry point -----
# Subordinating conjunctions. If one of them sits in the SUBJECT of a
# candidate declarative, the "are/is" it introduces belongs to a subordinate
# clause, not to the main sentence -- see prefers_declarative().
SUBORDINATORS = {
    "until", "if", "when", "while", "unless", "because", "although",
    "whether", "before", "after", "once", "since", "provided", "where",
}


def split_independent_clauses(text):
    """
    Split a control that chains TWO independent declaratives with ", and".

        "The AI model is explained, validated, and documented, and output is
         interpreted within context."
        -> ["The AI model is explained, validated, and documented",
            "output is interpreted within context"]

    Without this, the second clause is swallowed as the "rest" of the last
    participle and produces
        "Document the AI model and output is interpreted within context".

    TWO conditions keep a coordinated NOUN LIST from being cut in half, which
    is the easy mistake here:
      * the LEFT part must already be a complete clause, i.e. contain its own
        "is"/"are". In "Test sets, metrics, and details ... are documented" the
        left part would be "Test sets, metrics" -- no auxiliary, so the items
        share one auxiliary and the sentence stays whole.
      * the RIGHT part must not start with a PARTICIPLE. In "... explained,
        validated, and documented, and output is ..." the first candidate
        opens on "documented", which is another participle of the same list,
        not a new subject.
    """
    text = text.strip()
    for m in re.finditer(r",\s+and\s+(?=([A-Za-z][\w'-]*)[\w\s,'-]{0,60}?\s+(?:is|are|was|were)\s)",
                         text):
        left, right = text[:m.start()], text[m.end():]
        if not re.search(r"\b(is|are|was|were)\b", left):
            continue                       # left is not a clause: a noun list
        if looks_like_participle(m.group(1).lower()):
            continue                       # right opens on another participle
        return [left.strip().rstrip("."), right.strip().rstrip(".")]
    return [text.rstrip(".")]


def prefers_declarative(text):
    """
    True when a sentence that COULD be read as an imperative is really a
    declarative statement of a desired state, so P6 must be tried before P5.

    The conflict is real: "Test" and "Ensure" are both in IMPERATIVE_VERBS, so
        "Test sets, metrics, and details ... are documented."   (declarative)
        "Ensure Authorized Software is Currently Supported"     (imperative)
    both start with a "verb". Three conditions separate them:
      1. the sentence must match  <subject> are/is <participle...>
      2. the subject must contain NO subordinator -- otherwise the auxiliary
         belongs to a subordinate clause:
            "Protect ... media until the media are destroyed ..."   (NIST MP-4)
            "Assess the controls ... if they are implemented ..."   (NIST CA-2)
         which are genuine imperatives.
      3. a SHORT subject opening with an imperative verb is an imperative
         ("Ensure Authorized Software"); a long noun phrase is a subject
         ("Test sets, metrics, and details about the tools used during TEVV").
    """
    m = re.match(r"^(?P<subject>.+?)\s+(?:are|is|were|was)\s+(?P<verbs>.+)$", text)
    if not m:
        return False
    subject, verbs = m.group("subject"), m.group("verbs")

    subject_words = subject.split()
    if any(w.lower().strip(",;:") in SUBORDINATORS for w in subject_words):
        return False                                    # condition 2
    if len(subject_words) <= 4 and \
            subject_words[0].lower().strip(",;:") in IMPERATIVE_VERBS:
        return False                                    # condition 3

    # condition 1: the auxiliary must really be followed by a participle
    m_adv = LEADING_ADVERB_RE.match(verbs)
    head = verbs[m_adv.end():] if m_adv else verbs
    first = head.split(",")[0].split(" and ")[0].split()
    return bool(first) and looks_like_participle(first[0])



# Auxiliaries and copulas: never the main verb of an active present clause.
NON_ACTION_PRESENT = {
    "is", "are", "was", "were", "has", "have", "does", "do", "seems",
    "remains", "becomes", "means", "includes", "consists", "comprises",
}


def deinflect_present(word):
    """
    Third-person singular present -> base form.
        establishes -> establish     identifies -> identify
        applies     -> apply         monitors   -> monitor
    Returns None when the word is not a plausible -s present form.
    """
    w = word.lower()
    if not w.endswith("s") or len(w) < 4:
        return None
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ses", "shes", "ches", "xes", "zes")):
        return w[:-2]
    return w[:-1]


def match_active_present(control_id, text):
    """
    P4b: ACTIVE PRESENT INDICATIVE, third person singular, no "shall".
        "<subject> <verb>s <rest>"  -->  "<Verb> <rest>", agent = <subject>

        "The organization establishes mechanisms for accountability and
         transparency related to AI systems."
        --> "Establish mechanisms for accountability and transparency related
             to AI systems"   [agent: the organization]

    This is the present-tense twin of P4 ("<subj> shall <verb> ..."): the same
    sentence with the same meaning, only without the modal. Leaving it to P7
    would produce "Ensure the organization establishes mechanisms ...", which
    keeps the subject inside the goal and hides the AGENT the control names --
    exactly the information a goal model wants to record separately.

    Tried LAST, just before P7, so that any sentence P5 or P6 can handle never
    reaches it. Three guards keep it from firing on a plural noun:
      * the candidate verb must de-inflect to a KNOWN verb, so "AI risks are
        ..." cannot read "risks" as a verb;
      * auxiliaries and copulas are excluded (NON_ACTION_PRESENT);
      * a subject and a remainder must both be present;
      * the sentence must carry NO modal -- a "shall" control belongs to
        P1-P4, and a plural noun before it would otherwise be read as a verb.
    Plural presents with no -s ("Evaluations ... meet applicable requirements")
    carry no marker to key on and are left to P7.

    :param control_id: Identifier of the control/requirement.
    :param text: Requirement text to analyze.
    """
    # A sentence carrying a MODAL is a "shall" control: P1-P4 own it, and if
    # they all declined, P4b must not guess. Without this guard, ISO A.13.2.1
    # ("Formal transfer policies, procedures and CONTROLS shall be in place
    # ...") had its plural noun "controls" read as the verb "control", giving
    # the nonsense "Control shall be in place to protect ...".
    if re.search(r"\b(shall|must|should|will|may|can)\b", text, re.IGNORECASE):
        return None

    known = set(PARTICIPLE_LEMMAS.values()) | IMPERATIVE_VERBS
    words = text.rstrip(".").split()

    for i in range(1, len(words) - 1):          # never the first or last word
        token = words[i].lower().strip(",;:")
        if token in NON_ACTION_PRESENT:
            break                               # a copula: not an active clause
        base = deinflect_present(token)
        if not base or base not in known:
            continue
        subject = " ".join(words[:i]).strip().rstrip(",")
        rest = " ".join(words[i + 1:]).strip().rstrip(".")
        if not subject or not rest:
            continue
        return [make_goal(control_id, "P4b_active_present", base,
                          capitalize_first(f"{base} {rest}"),
                          lower_subject(subject))]
    return None


def transform_shall_statement(control_id, text, split_triggers=True):
    """
    Transform one governance control into a list of leaf-goal records.

    This function acts as the main dispatcher of the requirement
    transformation pipeline. It tries multiple grammar-based patterns
    in order of specificity and returns the first successful match.

    :param control_id: Identifier of the control (e.g., "A.5.1.1").
    :param text: Raw governance control sentence.
    :param split_triggers: If True, disjunctive trigger clauses (e.g., "or if ...") are expanded into separate goals.
    """
    # Normalize the input text (fixes casing, removes formatting noise, standardizes whitespace)
    text = normalize(text)

    # A control may chain TWO independent declaratives with ", and" -- each is
    # parsed on its own, and their goals concatenated (FIX: the second clause
    # used to be swallowed as the "rest" of the last participle).
    clauses = split_independent_clauses(text)
    if len(clauses) > 1:
        goals = []
        for clause in clauses:
            goals.extend(transform_shall_statement(control_id, clause,
                                                   split_triggers))
        return assign_goal_ids(goals) if False else goals

    return _transform_single_clause(control_id, text, split_triggers)




def _transform_single_clause(control_id, text, split_triggers=True):
    """
    Transform ONE clause (no ", and <subject> is/are ..." coordination left).

    Order matters:
      1. "There shall be ..."                                    (P1)
      2. Passive "shall be + participle"                         (P2/P3)
      3. Active "shall + verb"                                   (P4)
      4. Imperative directive, no "shall"                        (P5)
      5. Declarative desired state, no "shall"                   (P6)
      6. Directive wrapper for anything left                     (P7)

    P5 normally comes BEFORE P6, because a directive may contain "are" in a
    subordinate clause ("... to determine if they are implemented correctly",
    NIST CA-2) and must not be read as a declarative. But some declaratives
    OPEN with a word that is also a verb ("Test sets, metrics, ... are
    documented"), and P5 would then swallow the whole sentence unchanged.
    prefers_declarative() detects exactly that case and flips the two.
    """
    matchers = [
        lambda: match_there_shall_be(control_id, text),            # P1
        lambda: match_passive(control_id, text, split_triggers),   # P2/P3
        lambda: match_active(control_id, text),                    # P4
    ]
    if prefers_declarative(text):
        matchers += [lambda: match_declarative(control_id, text, split_triggers),
                     lambda: match_imperative(control_id, text)]
    else:
        matchers += [lambda: match_imperative(control_id, text),
                     lambda: match_declarative(control_id, text, split_triggers)]

    matchers.append(lambda: match_active_present(control_id, text))  # P4b

    for matcher in matchers:
        goals = matcher()
        if goals:
            return goals

    # P7 -- last resort. The sentence is a statement of a desired state that no
    # pattern can turn into a verb-led goal (an active present indicative, a
    # copula: "Evaluations ... meet applicable requirements and are
    # representative of the relevant population"). Returning it verbatim leaves
    # a NON-GOAL in the model, so it is wrapped in a neutral directive verb:
    # nothing is invented, and the result reads as an objective to satisfy.
    body = text.rstrip(".").strip()
    if body:
        body = body[0].lower() + body[1:]
        return [make_goal(control_id, "P7_directive_wrapper", "ensure",
                          f"Ensure {body}", None)]
    return [make_goal(control_id, "UNMATCHED", None, text, None)]




def assign_goal_ids(goals, parent_id="G1.1"):
    """
    Assign hierarchical identifiers to a list of goal objects.
    Each goal receives a unique ID derived from its parent goal ID,
    following a structured numbering scheme: G1.1 → G1.1.1, G1.1.2, G1.1.3, ...
    This is useful for maintaining traceability in goal models,
    requirement decompositions, or ISO control transformations.

    :param goals: List of goal dictionaries produced by transformation rules.
    :param parent_id: Identifier of the parent goal under which these goals are grouped.
    """
    for i, g in enumerate(goals, start=1):
        g["goal_id"] = f"{parent_id}.{i}"
    return goals




def print_goals(goals):
    """
    Pretty-print the leaf goals of one control.

    :param goals: List of goal dictionaries.
    """
    for g in goals:
        # Safely retrieve the goal ID if it exists.
        gid = g.get("goal_id", "-")
        # Build a string showing the agent (if available).
        # Ex: "  [agent: Management]"
        # If agent is None or empty, use an empty string.
        agent = f"  [agent: {g['agent']}]" if g["agent"] else ""
        print(f"{gid:<8} ({g['control_id']}, {g['pattern']})") # Ex: G1.1.1   (A.5.1.1, P3_passive_multi_verb)
        print(f"         {g['statement']}{agent}") # Example: Review access control   [agent: Management]