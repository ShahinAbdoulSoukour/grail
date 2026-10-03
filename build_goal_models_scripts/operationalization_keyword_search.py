"""
Symbolic (keyword-search) operationalization of leaf goals into actions from TWO knowledge bases:
* MITRE D3FEND (d3fend.ttl)
* OntoSecAI (ontosecai.rdf)
"""

import glob
import json
import os
import re
from collections import Counter, deque

from rdflib import Graph


# ---- config ----
D3FEND_FILE = "d3fend.ttl"
ONTOSECAI_FILE = "ontosecai.rdf"

GOAL_MODEL_DIRS = {
    "ISO 27001": "goal_models_ISO27001",
    "ISO 42001": "goal_models_ISO42001",
    "NIST":      "goal_models_NIST",
    "CIS":       "goal_models_CIS",
    "AI-RMF":    "goal_models_AIRMF",
}

# How many distinct goal nouns must match an entry before we commit to it.
MIN_NOUN_MATCHES = 2

DP = """
PREFIX d3f:  <http://d3fend.mitre.org/ontologies/d3fend.owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
"""
HP = """
PREFIX hes:  <http://www.semanticweb.org/ubaid/ontologies/2023/5/HES#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "for", "in", "on", "with", "by",
    "all", "any", "their", "its", "shall", "be", "at", "upon", "when", "if",
    "information", "security", "system", "systems", "organization", "employees",
    "users", "user", "external", "party", "processing", "facilities", "rights",
    "use", "used", "relevant", "appropriate", "against", "during", "provided",
}

LEADING_ADVERBS = {
    "uniquely", "securely", "actively", "regularly", "periodically",
    "continuously", "automatically", "centrally", "properly", "effectively",
    "appropriately", "clearly", "timely", "fully",
}


# Curated concept --> artifact map, bridging vocabulary the ontology does not
# share with the standards. Checked before the keyword fallback.
ARTIFACT_HINTS = {
    "privileged access": "PrivilegedUserAccount",
    "access rights": "UserAccount",
    "authentication information": "Credential",
    "source code": "File",
    "account": "UserAccount",
    "credential": "Credential",
    "password": "Password",
    "media": "DigitalMedia",
    "network": "NetworkTraffic",
}

# leaf-goal verb --> D3FEND relation verbs with the same intent.
# An UNMAPPED verb yields NO action: without the filter every technique on the artifact would match whatever the goal asks.
INTENT_VERBS = {
    "remove": {"deletes", "disables", "terminates", "suspends", "erases", "restricts"},
    "revoke": {"deletes", "disables", "terminates", "restricts"},
    "delete": {"deletes", "erases"}, "erase": {"deletes", "erases"},
    "disable": {"disables", "terminates", "suspends", "blocks"},
    "terminate": {"terminates", "suspends", "disables"},
    "restrict": {"restricts", "filters", "isolates", "blocks", "limits", "hardens", "enforces"},
    "limit": {"restricts", "limits", "use-limits", "filters"},
    "control": {"restricts", "filters", "enforces", "hardens"},
    "prevent": {"blocks", "filters", "restricts", "isolates"},
    "block": {"blocks", "filters", "restricts"},
    "segregate": {"isolates", "filters", "restricts"},
    "separate": {"isolates", "filters", "restricts"},
    "isolate": {"isolates", "filters", "blocks", "quarantines"},
    "quarantine": {"quarantines", "isolates", "blocks"},
    "allowlist": {"filters", "restricts", "blocks"},
    "enforce": {"enforces", "restricts", "filters", "hardens"},
    "employ": {"enforces", "restricts", "hardens"},
    "protect": {"encrypts", "hardens", "strengthens", "obfuscates", "filters"},
    "secure": {"hardens", "strengthens", "encrypts", "restricts"},
    "harden": {"hardens", "strengthens"}, "strengthen": {"strengthens", "hardens"},
    "encrypt": {"encrypts", "obfuscates"}, "obfuscate": {"obfuscates", "encrypts"},
    "safeguard": {"hardens", "encrypts", "restricts"},
    "review": {"analyzes", "monitors", "evaluates", "verifies"},
    "monitor": {"monitors", "analyzes", "detects"},
    "detect": {"detects", "analyzes", "monitors"},
    "analyze": {"analyzes", "evaluates"}, "analyse": {"analyzes", "evaluates"},
    "assess": {"analyzes", "evaluates", "verifies"},
    "evaluate": {"evaluates", "analyzes", "verifies"},
    "examine": {"analyzes", "evaluates"},
    "test": {"validates", "verifies", "evaluates", "analyzes"},
    "verify": {"verifies", "validates", "authenticates"},
    "validate": {"validates", "verifies"},
    "audit": {"analyzes", "monitors", "evaluates"},
    "log": {"monitors", "analyzes"}, "measure": {"evaluates", "analyzes"},
    "track": {"monitors", "analyzes", "inventories"},
    "identify": {"maps", "inventories"}, "classify": {"maps", "inventories"},
    "categorize": {"maps", "inventories"}, "inventory": {"inventories", "maps"},
    "map": {"maps"}, "collect": {"inventories", "monitors", "queries"},
    "determine": {"analyzes", "evaluates", "maps"}, "select": {"maps", "evaluates"},
    "authenticate": {"authenticates", "verifies", "validates"},
    "authorize": {"restricts", "enforces", "authenticates"},
    "authorise": {"restricts", "enforces", "authenticates"},
    "configure": {"configures", "hardens", "modifies"},
    "adjust": {"modifies", "restricts", "filters", "hardens"},
    "modify": {"modifies", "updates", "configures"},
    "update": {"updates", "modifies", "regenerates"},
    "regenerate": {"regenerates", "updates"}, "rotate": {"regenerates", "updates"},
    "apply": {"hardens", "enforces", "restricts", "configures"},
    "implement": {"restricts", "filters", "hardens", "authenticates", "strengthens"},
    "manage": {"manages", "restricts", "modifies", "inventories"},
    "maintain": {"updates", "manages", "monitors"},
    "use": {"uses"}, "utilize": {"uses"},
    "deploy": {"uses", "hardens", "configures"},
    "restore": {"restores"}, "recover": {"restores"},
    "respond": {"restores", "quarantines", "blocks", "terminates"},
    "correct": {"restores", "updates", "modifies"},
    "remediate": {"restores", "updates", "quarantines"},
    "eradicate": {"deletes", "quarantines", "neutralizes"},
    "neutralize": {"neutralizes", "blocks"},
    "spoof": {"spoofs"}, "decoy": {"spoofs"},
}


# ---- OntoSecAI: hints ----
# A defensive goal almost never contains the NAME of the
# attack it defends against ("Monitor inputs and outputs of AI systems" shares
# no word with "Prompt Injection"), so this maps goal vocabulary to attacks.
# Keys are phrases looked up in the goal, values are attack names.
ATTACK_HINTS = {
    "input": "Prompt Injection",
    "prompt": "Prompt Injection",
    "output": "Output Integrity Attack",
    "training data": "Chatbot Poisoning",
    "poison": "Chatbot Poisoning",
    "supply chain": "Supply Chain Poisoning",
    "third-party": "Supply Chain Poisoning",
    "supplier": "Supply Chain Poisoning",
    "authentication": "Bypassing Authentication",
    "identity": "Identity Spoofing",
    "impersonation": "Identity Spoofing",
    "privacy": "Privacy Leak",
    "personal data": "Privacy Leak",
    "confidentiality": "Privacy Leak",
    "model": "Model Theft",
    "availability": "Denial of Services",
    # not the bare word "resource": once plurals match, it fired on
    # "Allocate resources for AI risk management activities" (budgeting,
    # not an attack). Only the phrases that name the attack itself:
    "resource exhaustion": "Excessive Allocation",
    "resource consumption": "Excessive Allocation",
    "code": "Arbitrary Code Execution",
    "malicious": "Malicious Logic Insertion",
    "hardware": "Hardware Trojan Attacks",
    "supply": "Supply Chain Poisoning",
}

# leaf-goal verb -> AI lifecycle phases with the same intent.
# This is the counterpart of D3FEND's verb filter: it keeps a mitigation only
# if it applies at the point of the lifecycle the goal is about. An unmapped
# verb means NO phase filter here (unlike D3FEND) because the mitigation was
# already selected by a thresholded text match, so the filter only refines.
INTENT_PHASES = {
    "monitor": {"Monitoring&Maintenance", "Deployment"},
    "detect": {"Monitoring&Maintenance", "ModelEvaluation"},
    "review": {"Monitoring&Maintenance", "ModelEvaluation"},
    "audit": {"Monitoring&Maintenance"},
    "log": {"Monitoring&Maintenance"},
    "maintain": {"Monitoring&Maintenance"},
    "test": {"ModelEvaluation"}, "validate": {"ModelEvaluation"},
    "verify": {"ModelEvaluation"}, "evaluate": {"ModelEvaluation"},
    "assess": {"ModelEvaluation", "Business&DataUnderstanding"},
    "measure": {"ModelEvaluation"},
    "restrict": {"AccessManagement"}, "control": {"AccessManagement"},
    "limit": {"AccessManagement"}, "authenticate": {"AccessManagement"},
    "authorize": {"AccessManagement"}, "remove": {"AccessManagement"},
    "revoke": {"AccessManagement"},
    "sanitize": {"DataPreparation"}, "clean": {"DataPreparation"},
    "prepare": {"DataPreparation"}, "label": {"DataPreparation"},
    "acquire": {"DataPreparation", "Business&DataUnderstanding"},
    "deploy": {"Deployment"}, "operate": {"Deployment", "Monitoring&Maintenance"},
    "design": {"ModelEngineering", "SoftwareSecurityEngineering"},
    "develop": {"ModelEngineering", "SoftwareSecurityEngineering"},
    "train": {"ModelEngineering"}, "harden": {"ModelEngineering"},
    "understand": {"Business&DataUnderstanding"},
    "identify": {"Business&DataUnderstanding"},
    "document": {"Business&DataUnderstanding"},
}



# ---- goal-model helpers ----
def leaf_goals_of(model):
    """
    Return the list of leaf-goal dictionaries of an already classified
    goal model.

    Input files are *_leaf_goals_classified.json, so this function does NOT
    classify goals. It only traverses the model and returns leaf nodes.

    A goal node is considered a leaf when it has a goal_name and no non-empty
    subgoals list.
    """
    def walk(node):
        if isinstance(node, dict):
            subgoals = node.get("subgoals")

            # A goal node with no children is a leaf goal.
            if node.get("goal_name") and not subgoals:
                yield node
                return

            if isinstance(subgoals, list):
                for child in subgoals:
                    yield from walk(child)

        elif isinstance(node, list):
            for child in node:
                yield from walk(child)

    # A LIST, not a generator. Callers iterate over the leaves several times
    # (one sum() per column in run_standard) and call len() on them. A
    # generator is exhausted by the first pass: every later count silently
    # came out 0, and len() raised "object of type 'generator' has no len()".
    return list(walk(model))


# --- text helpers ---
def tokens(text_low):
    """
    Whole-word tokens of a lower-cased text (punctuation stripped).
    """
    return set(re.findall(r"[a-z][a-z\-]+", text_low))


def phrase_in_text(phrase, text_low, allow_plural=False):
    """
    True when a phrase occurs as a whole-word phrase ("code" never matches
    inside "encode").

    allow_plural=True also accepts a plural ending ("s" / "es"). It is meant
    for the CURATED hints only (ARTIFACT_HINTS, ATTACK_HINTS), whose meaning
    is controlled: without it they never fired on the plural forms the
    standards actually use --
        "account"    missed "Manage information system ACCOUNTS"   (NIST AC-2)
        "credential" missed "Manage CREDENTIALS ..."
        "input"      missed "Monitor INPUTS and outputs of AI systems"

    It stays OFF (the default) when matching ONTOLOGY LABELS, because a plural
    in a goal can carry a different sense than the singular label: D3FEND's
    "Process" is an executing program, while "Assess PROCESSES for human
    oversight" means organizational procedures. Accepting the plural there
    attached D3-PSA "Process Spawn Analysis" to a human-oversight goal.
    """
    phrase = phrase.strip().lower()
    text_low = text_low.lower()
    if not phrase:
        return False
    plural = r"(?:e?s)?" if allow_plural else ""
    return re.search(
        rf"(?<![a-z0-9]){re.escape(phrase)}{plural}(?![a-z0-9])",
        text_low,
    ) is not None


def leading_verb(goal_name):
    """The goal's action verb, with a leading adverb removed."""
    words = [w.lower().strip(",;:.") for w in goal_name.split() if w.strip()]
    if not words:
        return ""
    if words[0] in LEADING_ADVERBS and len(words) > 1:
        return words[1]
    return words[0]


def extract_nouns(goal_name):
    """
    Content words of the goal's object (leading verb dropped, stopwords out).
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", goal_name.lower())
    return [w for w in words[1:] if w not in STOPWORDS and len(w) > 3]


def best_text_match(goal_name, names, descriptions=None,
                    min_nouns=MIN_NOUN_MATCHES):
    """
    Rank entries against the goal and return the best key, or None when
    nothing clears the bar.

    `names` is {key -> short name/label} and `descriptions` the optional
    {key -> long definition}. The two are scored DIFFERENTLY, which matters:
      * the NAME is what can occur verbatim in a goal, and it is where the
        discriminating words are ("Validate ML Model", "Sanitize Training
        Data"). Its words are matched against ALL the goal's content words,
        the leading VERB INCLUDED -- an OntoSecAI mitigation name is itself
        verb-led, so dropping the goal's verb throws away the best signal.
        (D3FEND is different: there the verb is handled by INTENT_VERBS, and
        an artifact label is a noun phrase.)
      * the DESCRIPTION is long and generic, so it only corroborates, never
        decides: its words are matched against the goal's nouns alone.
    Scoring name+description as one blob made the phrase test useless (a
    50-word definition never occurs in a goal) and diluted the overlap ratio.

    Accepted if the name occurs as a phrase in the goal, or >=2 name words
    match, or >=1 name word plus >=`min_nouns` description words match.
    """
    low = goal_name.lower()
    goal_words = tokens(low)
    content = {w for w in goal_words if w not in STOPWORDS and len(w) > 2}
    nouns = set(extract_nouns(goal_name))
    descriptions = descriptions or {}
    best, best_key = None, None

    for key, name in names.items():
        name_words = tokens(name.lower())
        if not name_words:
            continue
        phrase = " ".join(name.lower().split())
        phrase_hit = len(phrase) > 4 and phrase_in_text(phrase, low)
        name_hit = content & name_words
        desc_hit = nouns & tokens(descriptions.get(key, "").lower())

        if not (phrase_hit or len(name_hit) >= 2
                or (len(name_hit) >= 1 and len(desc_hit) >= min_nouns)):
            continue
        # tie-breaker only: how much of the entry's own name the goal
        # covers. An internal ranking score, not a reported metric.
        overlap = len(name_words & goal_words) / len(name_words)
        rank = (phrase_hit, len(name_hit), len(desc_hit), overlap)
        if best_key is None or rank > best_key:
            best, best_key = key, rank
    return best


# --- D3FEND index ---
def build_d3fend_index(graph):
    """
    Index D3FEND once:
      art_text     {artifact -> "label + synonyms"}    corpus for STEP 1
      parents      {artifact -> {direct superclasses}} for inheritance
      restrictions [(id, label, verb, artifact)]       technique--verb->artifact
      definitions  {id -> definition}
    """
    art_text, parents, restrictions, definitions = {}, {}, [], {}

    for r in graph.query(DP + """
        SELECT ?a ?label ?syn WHERE {
          ?a rdfs:subClassOf* d3f:DigitalArtifact ; rdfs:label ?label .
          OPTIONAL { ?a d3f:synonym ?syn } }"""):
        name = str(r.a).split("#")[-1]
        txt = str(r.label).lower() + " " + (str(r.syn).lower() if r.syn else "")
        art_text[name] = art_text.get(name, "") + " " + txt

    for r in graph.query(DP + """
        SELECT ?a ?p WHERE {
          ?a rdfs:subClassOf* d3f:DigitalArtifact ; rdfs:subClassOf ?p .
          ?p rdfs:subClassOf* d3f:DigitalArtifact . }"""):
        parents.setdefault(str(r.a).split("#")[-1], set()).add(
            str(r.p).split("#")[-1])

    for r in graph.query(DP + """
        SELECT ?id ?label ?verb ?art WHERE {
          ?t rdfs:subClassOf* d3f:DefensiveTechnique ;
             d3f:d3fend-id ?id ; rdfs:label ?label .
          ?t rdfs:subClassOf [ owl:onProperty ?verb ; owl:someValuesFrom ?art ] .
          ?art rdfs:subClassOf* d3f:DigitalArtifact . }"""):
        restrictions.append((str(r.id), str(r.label),
                             str(r.verb).split("#")[-1],
                             str(r.art).split("#")[-1]))

    for r in graph.query(DP + """
        SELECT ?id ?def WHERE { ?t d3f:d3fend-id ?id ; d3f:definition ?def . }"""):
        definitions[str(r.id)] = str(r["def"])

    # Keep ONLY the artifacts some technique actually acts on. Of the ~897
    # artifacts in the taxonomy, only ~141 are the target of a restriction:
    # the other 756 are dead ends -- a goal can match one and then necessarily
    # return no action, so indexing them only creates false positives at
    # STEP 1 without ever adding a possible action.
    usable = {art for _, _, _, art in restrictions} | set(ARTIFACT_HINTS.values())
    art_text = {a: t for a, t in art_text.items() if a in usable}

    return {"art_text": art_text, "parents": parents,
            "restrictions": restrictions, "definitions": definitions}


def ancestors_of(name, parents):
    """
    The artifact itself + all its superclasses (BFS over parent edges).
    """
    seen, queue = {name}, deque([name])
    while queue:
        for p in parents.get(queue.popleft(), ()):
            if p not in seen:
                seen.add(p)
                queue.append(p)
    return seen


def find_artifact(goal_name, idx):
    """
    D3FEND STEP 1 - curated hint first, else a thresholded keyword match.
    """
    low = goal_name.lower()
    for phrase in sorted(ARTIFACT_HINTS, key=len, reverse=True):
        if phrase_in_text(phrase, low, allow_plural=True):
            return ARTIFACT_HINTS[phrase]
    return best_text_match(goal_name, idx["art_text"])   # labels are short


def operationalize_d3fend(goal_name, idx):
    """
    D3FEND chain -> (artifact, actions, verb, verb_mapped).
    STEP 1 artifact, STEP 2 techniques on it and its ancestors, STEP 3 verb.
    """
    verb = leading_verb(goal_name)
    artifact = find_artifact(goal_name, idx)
    if not artifact:
        return None, [], verb, verb in INTENT_VERBS

    wanted = INTENT_VERBS.get(verb)
    if wanted is None:                     # unmapped verb -> no action, no noise
        return artifact, [], verb, False

    anc = ancestors_of(artifact, idx["parents"])
    seen, actions = set(), []
    for tid, label, v, art in idx["restrictions"]:
        if art in anc and v in wanted and tid not in seen:
            seen.add(tid)
            actions.append({"id": tid, "label": label, "verb": v,
                            "definition": idx["definitions"].get(tid, ""),
                            "source": "D3FEND"})
    return artifact, actions, verb, True


# ---- OntoSecAI index ----

def local_name(value):
    """
    Extract the local name from an RDF URI.

    Handles both fragment URIs (...#Name) and slash URIs (.../Name).
    """
    value = str(value)
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    return value.rstrip("/").rsplit("/", 1)[-1]


def build_ontosecai_index(graph):
    """
    Index OntoSecAI once:
      mitigations {id -> {id, name, definition, phases}}
      attacks     {id -> {id, name, definition, mitigations}}
      mit_text    {id -> "name + description"}   corpus for chain (A)
      att_text    {id -> "name + description"}   corpus for chain (B)
    """
    mitigations, attacks = {}, {}

    for r in graph.query(HP + """
        SELECT ?id ?n ?d WHERE {
          ?m a hes:Mitigations ; hes:Name ?n .
          OPTIONAL { ?m hes:Description ?d }
          BIND(REPLACE(STR(?m), "^.*[#/]", "") AS ?id) }"""):
        mitigations.setdefault(str(r[0]), {
            "id": str(r[0]), "name": str(r[1]),
            "definition": str(r[2]) if r[2] else "", "phases": set()})

    for r in graph.query(HP + """
        SELECT ?id ?p WHERE {
          ?m a hes:Mitigations ; hes:IsIncludedIn ?p .
          BIND(REPLACE(STR(?m), "^.*[#/]", "") AS ?id) }"""):
        if str(r[0]) in mitigations:
            mitigations[str(r[0])]["phases"].add(local_name(r[1]))

    for r in graph.query(HP + """
        SELECT ?aid ?an ?ad ?mid WHERE {
          ?a hes:Name ?an ; hes:IsMitigatedBy ?m .
          OPTIONAL { ?a hes:Description ?ad }
          BIND(REPLACE(STR(?a), "^.*[#/]", "") AS ?aid)
          BIND(REPLACE(STR(?m), "^.*[#/]", "") AS ?mid) }"""):
        entry = attacks.setdefault(str(r[0]), {
            "id": str(r[0]), "name": str(r[1]),
            "definition": str(r[2]) if r[2] else "", "mitigations": set()})
        entry["mitigations"].add(str(r[3]))

    # names and descriptions kept APART: best_text_match scores them
    # differently (see its docstring).
    return {
        "mitigations": mitigations, "attacks": attacks,
        "mit_name": {k: v["name"] for k, v in mitigations.items()},
        "mit_desc": {k: v["definition"] for k, v in mitigations.items()},
        "att_name": {k: v["name"] for k, v in attacks.items()},
        "att_desc": {k: v["definition"] for k, v in attacks.items()},
    }


def find_attack(goal_name, idx):
    """
    OntoSecAI chain (B) STEP 1 - the attack the goal defends against.
    ATTACK_HINTS first: a defensive goal shares no vocabulary with an
    offensive attack name, so the keyword fallback alone almost never fires.
    """
    low = goal_name.lower()
    by_name = {v["name"].lower(): k for k, v in idx["attacks"].items()}
    for phrase in sorted(ATTACK_HINTS, key=len, reverse=True):
        if phrase_in_text(phrase, low, allow_plural=True):
            target = ATTACK_HINTS[phrase].lower()
            if target in by_name:
                return by_name[target]
    return best_text_match(goal_name, idx["att_name"], idx["att_desc"])


def operationalize_ontosecai(goal_name, idx):
    """
    OntoSecAI chains -> (attack, actions).

    (A) direct: the mitigation whose own text matches the goal. Mitigation
        names are defensive, so this shares the goal's vocabulary.
    (B) via attack: the attack the goal defends against, then IsMitigatedBy,
        then keep the mitigations whose lifecycle phase matches the goal verb.
    The two results are merged, de-duplicated by mitigation id.
    """
    verb = leading_verb(goal_name)
    wanted_phases = INTENT_PHASES.get(verb)
    actions, seen = [], set()

    # --- chain (A): goal -> mitigation, directly
    direct = best_text_match(goal_name, idx["mit_name"], idx["mit_desc"])
    if direct:
        m = idx["mitigations"][direct]
        seen.add(direct)
        actions.append({"id": m["id"], "label": m["name"],
                        "phases": sorted(m["phases"]),
                        "definition": m["definition"], "via": "mitigation",
                        "source": "OntoSecAI"})

    # --- chain (B): goal -> attack -> mitigations -> phase filter
    attack = find_attack(goal_name, idx)
    if attack:
        for mid in sorted(idx["attacks"][attack]["mitigations"]):
            if mid in seen or mid not in idx["mitigations"]:
                continue
            m = idx["mitigations"][mid]
            # keep it only if its phase matches the goal's intent (when known)
            if wanted_phases and m["phases"] and not (m["phases"] & wanted_phases):
                continue
            seen.add(mid)
            actions.append({"id": m["id"], "label": m["name"],
                            "phases": sorted(m["phases"]),
                            "definition": m["definition"], "via": "attack",
                            "source": "OntoSecAI"})
    return attack, actions


# ---- orchestration ----
def operationalize_model(model, d3f_idx, onto_idx, unmapped_verbs=None):
    """
    Enrich every leaf in place with its class, the D3FEND artifact, the
    OntoSecAI attack, and the MERGED action list of both catalogues.

    EVERY leaf is operationalized -- ORG and PHYS included. The class is
    recorded but no longer short-circuits the search: it is a description of
    the goal, not a permission to look. Two reasons:
      * the classifier is a heuristic; skipping ORG outright means a
        genuinely technical goal it mislabels can NEVER be recovered;
      * an ORG goal that does match is worth seeing -- either the class is
        wrong, or the match is spurious, and both are findings.
    A goal that matches nothing simply keeps an empty action list, which is a
    coverage result and not a failure.
    """
    for leaf in leaf_goals_of(model):
        name = leaf["goal_name"]

        # Input files are already classified. Do not re-classify here.
        cls = str(leaf.get("class", "")).strip().upper()
        if cls not in {"ORG", "PHYS", "TECH"}:
            raise ValueError(
                f"Leaf goal '{name}' does not contain a valid class "
                f"(expected ORG, PHYS, or TECH)."
            )

        artifact, d3f_actions, verb, mapped = operationalize_d3fend(name, d3f_idx)
        attack, onto_actions = operationalize_ontosecai(name, onto_idx)

        # Anchors are recorded ONLY when they exist. A null "artifact" or
        # "attack" carries no information -- the empty action list already says
        # nothing was found -- and an unused key on every second leaf makes the
        # model harder to read.
        if artifact:
            leaf["artifact"] = artifact           # D3FEND anchor
        if attack:
            leaf["attack"] = attack               # OntoSecAI anchor
        leaf["actions"] = d3f_actions + onto_actions
        if not mapped and unmapped_verbs is not None:
            unmapped_verbs[verb] += 1             # counted, not written out
    return model


def source_models(directory):
    """
    Return ONLY classified leaf-goal JSON files.

    Required input pattern:
        *_leaf_goals_classified.json

    This deliberately excludes:
        *.operationalized_*.json
        other intermediate JSON files
        unclassified goal-model JSON files
    """
    pattern = os.path.join(directory, "*_leaf_goals_classified.json")
    return sorted(glob.glob(pattern))


def run_standard(name, directory, d3f_idx, onto_idx, unmapped_verbs):
    """
    Operationalize every goal model of one standard; return its totals.
    """
    files = source_models(directory)
    if not files:
        print(f"\n{name}: no goal model found in {directory}/")
        return None

    print(f"\n=== {name}  ({len(files)} models, {directory}/) ===")
    print("  Input files: *_leaf_goals_classified.json")
    print(f"  {'model':10} {'leaf':>4} | {'ORG':>3} {'PHYS':>4} {'TECH':>4} | "
          f"{'D3F':>4} {'ONTO':>5} {'both':>5} {'none':>5} {'acts':>5}")
    tot = Counter()
    for path in files:
        with open(path, encoding="utf-8") as f:
            model = json.load(f)
        operationalize_model(model, d3f_idx, onto_idx, unmapped_verbs)

        out = path[:-len(".json")] + ".operationalized_keyword_search.json"

        if os.path.abspath(out) == os.path.abspath(path):
            raise RuntimeError(f"Refusing to overwrite input file: {path}")

        with open(out, "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2, ensure_ascii=False)

        leaves = leaf_goals_of(model)
        org = sum(lf["class"] == "ORG" for lf in leaves)
        phys = sum(lf["class"] == "PHYS" for lf in leaves)
        tech = sum(lf["class"] == "TECH" for lf in leaves)
        # Counted over EVERY leaf now, since every leaf is operationalized.
        d3f = sum(1 for lf in leaves
                  if any(a["source"] == "D3FEND" for a in lf["actions"]))
        onto = sum(1 for lf in leaves
                   if any(a["source"] == "OntoSecAI" for a in lf["actions"]))
        both = sum(1 for lf in leaves
                   if {a["source"] for a in lf["actions"]} == {"D3FEND", "OntoSecAI"})
        none = sum(1 for lf in leaves if not lf["actions"])
        # "acts" counts ACTIONS, the columns above count GOALS: one goal may
        # carry several actions, so the two figures differ on purpose.
        acts = sum(len(lf["actions"]) for lf in leaves)
        tot.update({"leaf": len(leaves), "ORG": org, "PHYS": phys,
                    "TECH": tech, "D3F": d3f, "ONTO": onto,
                    "both": both, "none": none, "acts": acts})
        print(f"  {model['clause_id']:10} {len(leaves):>4} | {org:>3} {phys:>4} "
              f"{tech:>4} | {d3f:>4} {onto:>5} {both:>5} {none:>5} "
              f"{acts:>5}")
    return tot


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description=(
            "Keyword-search operationalization of *_leaf_goals_classified.json "
            "files using D3FEND and OntoSecAI."
        )
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=None,
        help=(
            "Directories to scan. If omitted, scan the configured standard "
            "directories that exist, plus the current directory when it "
            "contains classified files."
        ),
    )
    parser.add_argument(
        "--d3fend",
        default=D3FEND_FILE,
        help="Path to d3fend.ttl",
    )
    parser.add_argument(
        "--ontosecai",
        default=ONTOSECAI_FILE,
        help="Path to ontosecai.rdf",
    )
    args = parser.parse_args()

    print("loading ontologies ...")

    d3f_path = Path(args.d3fend)
    onto_path = Path(args.ontosecai)

    if not d3f_path.exists():
        raise FileNotFoundError(f"D3FEND ontology not found: {d3f_path}")
    if not onto_path.exists():
        raise FileNotFoundError(f"OntoSecAI ontology not found: {onto_path}")

    d3f_idx = build_d3fend_index(
        Graph().parse(str(d3f_path), format="turtle")
    )
    onto_idx = build_ontosecai_index(
        Graph().parse(str(onto_path), format="xml")
    )

    print(
        f"  D3FEND    : {len(d3f_idx['art_text'])} artifacts, "
        f"{len(d3f_idx['restrictions'])} restrictions"
    )
    print(
        f"  OntoSecAI : {len(onto_idx['mitigations'])} mitigations, "
        f"{len(onto_idx['attacks'])} attacks"
    )

    if args.dirs:
        directories = []
        for directory in args.dirs:
            p = Path(directory)
            if not p.exists():
                print(f"\nWARNING: directory not found: {p}")
                continue
            directories.append((p.name, str(p)))
    else:
        directories = []

        # Process the configured standard directories that actually exist.
        for standard, directory in GOAL_MODEL_DIRS.items():
            if Path(directory).exists():
                directories.append((standard, directory))

        # Also support a flat project directory containing classified files.
        current_files = list(Path(".").glob("*_leaf_goals_classified.json"))
        if current_files:
            directories.append(("Current directory", "."))

    if not directories:
        print(
            "\nNo input directory contains files matching "
            "'*_leaf_goals_classified.json'."
        )
        raise SystemExit(0)

    unmapped_verbs = Counter()
    totals = {}

    for std, directory in directories:
        t = run_standard(
            std,
            directory,
            d3f_idx,
            onto_idx,
            unmapped_verbs,
        )
        if t:
            totals[std] = t

    print(f"\n{'='*90}\nSUMMARY")
    print(
        "  Input pattern : *_leaf_goals_classified.json\n"
        "  Output suffix : .operationalized_keyword_search.json\n"
        "  Classification is read from the input files; a leaf without a "
        "valid class stops the run.\n"
    )
    print(
        f"  {'standard':18} {'leaf':>5} | {'ORG':>4} {'PHYS':>5} {'TECH':>5} | "
        f"{'D3FEND':>7} {'OntoSecAI':>10} {'both':>5} {'none':>5} {'acts':>5}"
    )

    grand = Counter()

    for std, t in totals.items():
        print(
            f"  {std:18} {t['leaf']:>5} | "
            f"{t['ORG']:>4} {t['PHYS']:>5} {t['TECH']:>5} | "
            f"{t['D3F']:>7} {t['ONTO']:>10} {t['both']:>5} "
            f"{t['none']:>5} {t['acts']:>5}"
        )
        grand.update(t)

    print(
        f"  {'TOTAL':18} {grand['leaf']:>5} | "
        f"{grand['ORG']:>4} {grand['PHYS']:>5} {grand['TECH']:>5} | "
        f"{grand['D3F']:>7} {grand['ONTO']:>10} {grand['both']:>5} "
        f"{grand['none']:>5} {grand['acts']:>5}"
    )

    if unmapped_verbs:
        print("\nUnmapped goal verbs:")
        for verb, count in sorted(unmapped_verbs.items()):
            print(f"  {verb or '<empty>'}: {count}")