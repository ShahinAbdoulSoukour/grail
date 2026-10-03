"""
Prompt-engineering operationalization of leaf goals with Qwen via llama.cpp.

Pipeline:
    1. Load D3FEND and OntoSecAI.
    2. Build candidate indices with SPARQL.
    3. Use SentenceTransformer to retrieve a small candidate set from each KB.
    4. Ask a local Qwen model (served by llama.cpp) to select zero or more
       operationalizations from ONLY those candidates.
    5. Validate the returned IDs against the candidate set.
    6. Save one JSON output per input goal-model file.

Output:
    - qwen_candidates: top-k candidates retrieved by embedding similarity.
    - actions: final operations selected by Qwen, with its reasons.

The LLM is a semantic selector.

Recommended local server (Qwen3 + llama.cpp):
    llama serve -hf Qwen/Qwen3-8B-GGUF:Q4_K_M

llama.cpp exposes an OpenAI-compatible API at http://localhost:8080/v1/.
Qwen3 non-thinking mode can be requested with chat_template_kwargs when the
server supports it.
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from rdflib import Graph, Namespace, OWL, RDF, RDFS, URIRef


# ---- Configuration ----
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Candidate retrieval is deliberately broader than the final number of operations.
# Qwen decides which candidates are truly relevant.
# Number of D3FEND techniques and OntoSecAI mitigations retrieved per leaf by cosine similarity before handing them to Qwen.
TOP_K_D3FEND = 10
TOP_K_ONTOSECAI = 10

# Final number of operations Qwen is allowed to select per leaf goal.
MAX_ACTIONS = 5

# Batch size for SentenceTransformer.encode. Larger batches are faster on GPU but consume more memory.
EMBEDDING_BATCH_SIZE = 64

LLAMA_BASE_URL = "http://127.0.0.1:11434/v1" # OpenAI-compatible base URL exposed by llama.cpp's server.
LLAMA_MODEL = "qwen3:8b" # Model identifier sent in the request body (llama.cpp mostly ignores it, but OpenAI clients require it).
LLAMA_API_KEY = "..." ### <--- API key here
LLAMA_TIMEOUT = 300 # HTTP timeout in seconds --> 5 minutes, since Qwen3-8B Q4_K_M can be slow on CPU.

# Qwen3 non-thinking parameters recommended by the Qwen documentation.
TEMPERATURE = 0.7 # Sampling temperature. Qwen3 docs recommend 0.7 for non-thinking tasks.
TOP_P = 0.8 # Nucleus sampling: only consider tokens whose cumulative probability ≤ 0.8.
TOP_K = 20 # Keep only the 20 highest-probability tokens at each step.
MAX_TOKENS = 2000 # Upper bound on generated tokens; enough for a JSON object with a handful of operations.
PRESENCE_PENALTY = 1.5 # Strongly penalises repeated tokens. Prevents the model from looping on the same phrase.

# Input directory containing the goal-model JSON files.
GOAL_MODEL_DIRS = {
    "ISO 27001": "goal_models_ISO27001",
    "ISO 42001": "goal_models_ISO42001",
    "NIST": "goal_models_NIST",
    "CIS": "goal_models_CIS",
    "AI-RMF": "goal_models_AIRMF",
}

INPUT_PATTERN = "*_leaf_goals_classified.json"

OUTPUT_SUFFIX = ".operationalized_qwen.json"

D3FEND_CANDIDATE_FILES = "d3fend.ttl"
ONTOSECAI_CANDIDATE_FILES = "ontosecai.rdf"

D3F = Namespace("http://d3fend.mitre.org/ontologies/d3fend.owl#")
HES = Namespace("http://www.semanticweb.org/ubaid/ontologies/2023/5/HES#")


# ---- RDF helpers ----
def local_name(uri):
    """
    Extracts the fragment or last path segment of a URI.
    """
    text = str(uri) # Normalise to string --> inputs may be URIRef objects.
    # If the URI has a fragment separator, the local name is what follows #.
    if "#" in text:
        # Split on the last # and take everything after it. rsplit(..., 1) ensures a URI containing multiple # still behaves predictably.
        return text.rsplit("#", 1)[1]
    # Otherwise strip any trailing slash and take the last / segment.
    # This becomes the canonical ID used in the candidate lists (e.g. M-12).
    return text.rstrip("/").rsplit("/", 1)[-1]


def load_rdf_graph(filename, label):
    """
    Loads an RDF file into an rdflib.Graph, choosing Turtle vs. RDF/XML by extension.
    """
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    graph = Graph() # Create an empty RDF graph.
    # Pick a parser format. .rdf falls through to "xml", which is the correct default for RDF/XML.
    fmt = "turtle" if path.suffix.lower() in {".ttl", ".turtle"} else "xml"
    # Parse the file into the graph. Any syntax error propagates as an exception.
    graph.parse(path, format=fmt)
    return graph # Return the populated graph to the caller.


# ---- D3FEND indexing ----
def build_d3fend_index(graph):
    """
    Index D3FEND defensive techniques and enrich them with ontology semantics.

    We keep defensive techniques as the candidate universe. Digital artifacts,
    actions, definitions, synonyms, and parent techniques are context used for
    retrieval and shown to Qwen.
    """

    # Finds every class that is a DefensiveTechnique (direct or transitive via subClassOf*) and has both an ID and a label.
    # This is the candidate universe.
    QUERY_DEFENSIVE_TECHNIQUES = """
    PREFIX d3f:  <http://d3fend.mitre.org/ontologies/d3fend.owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?technique ?id ?label WHERE {
        ?technique rdfs:subClassOf* d3f:DefensiveTechnique ;
                   d3f:d3fend-id ?id ;
                   rdfs:label ?label .
    }
    """

    # Builds a lookup table of artifacts → labels/definitions.
    # OPTIONAL means artifacts without a definition still appear.
    QUERY_DIGITAL_ARTIFACTS = """
    PREFIX d3f:  <http://d3fend.mitre.org/ontologies/d3fend.owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?artifact ?label ?definition WHERE {
        ?artifact rdfs:subClassOf* d3f:DigitalArtifact ;
                  rdfs:label ?label .
        OPTIONAL { ?artifact d3f:definition ?definition . }
    }
    """

    # Extracts OWL restrictions, the someValuesFrom pattern that says "this technique acts on this artifact via this property".
    # Used to populate the actions and artifacts fields of a technique.
    QUERY_TECHNIQUE_RESTRICTIONS = """
    PREFIX d3f: <http://d3fend.mitre.org/ontologies/d3fend.owl#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?technique ?action ?artifact WHERE {
        ?technique rdfs:subClassOf* d3f:DefensiveTechnique ;
                   d3f:d3fend-id ?id ;
                   rdfs:subClassOf ?restriction .
        ?restriction owl:onProperty ?action ;
                     owl:someValuesFrom ?artifact .
        ?artifact rdfs:subClassOf* d3f:DigitalArtifact .
    }
    """

    # Captures direct (non-OWL-restriction) relations between a technique and an artifact — e.g. d3f:produces.
    # FILTER removes structural predicates that would otherwise pollute the actions set.
    QUERY_TECHNIQUE_ARTIFACT_LINKS = """
    PREFIX d3f: <http://d3fend.mitre.org/ontologies/d3fend.owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT DISTINCT ?technique ?property ?artifact WHERE {
        ?technique rdfs:subClassOf* d3f:DefensiveTechnique ;
                   d3f:d3fend-id ?id .
        ?technique ?property ?artifact .
        ?artifact rdfs:subClassOf* d3f:DigitalArtifact .
        FILTER(
            ?property != rdfs:subClassOf &&
            ?property != rdf:type &&
            ?property != d3f:d3fend-id
        )
    }
    """

    # Fetches definitions and synonyms. Both OPTIONAL — a technique may have either, both, or neither.
    QUERY_TECHNIQUE_METADATA = """
    PREFIX d3f: <http://d3fend.mitre.org/ontologies/d3fend.owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?technique ?definition ?synonym WHERE {
        ?technique rdfs:subClassOf* d3f:DefensiveTechnique ;
                   d3f:d3fend-id ?id ;
                   rdfs:label ?label .
        OPTIONAL { ?technique d3f:definition ?definition . }
        OPTIONAL { ?technique d3f:synonym ?synonym . }
    }
    """

    # Collects ancestor labels (the + means at least one step) to give Qwen taxonomic context.
    QUERY_PARENT_TECHNIQUES = """
    PREFIX d3f: <http://d3fend.mitre.org/ontologies/d3fend.owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?technique ?parent ?parentLabel WHERE {
        ?technique rdfs:subClassOf+ d3f:DefensiveTechnique ;
                   d3f:d3fend-id ?id ;
                   rdfs:subClassOf ?parent .
        ?parent rdfs:subClassOf* d3f:DefensiveTechnique ;
               rdfs:label ?parentLabel .
    }
    """

    # --- Execute the queries
    # Each row is a ResultRow supporting attribute access (row.technique, row.id, row.label).
    # Converting to list materializes the results since graph.query returns a generator.`
    # All six queries run eagerly and their results are held in memory.
    technique_rows = list(graph.query(QUERY_DEFENSIVE_TECHNIQUES))
    artifact_rows = list(graph.query(QUERY_DIGITAL_ARTIFACTS))
    restriction_rows = list(graph.query(QUERY_TECHNIQUE_RESTRICTIONS))
    direct_link_rows = list(graph.query(QUERY_TECHNIQUE_ARTIFACT_LINKS))
    metadata_rows = list(graph.query(QUERY_TECHNIQUE_METADATA))
    parent_rows = list(graph.query(QUERY_PARENT_TECHNIQUES))

    # Maps artifact URI → {label, definition}.
    artifact_info: dict[str, dict[str, str]] = {}

    # Converts URIRef keys to plain strings, strips whitespace, and uses "" when row.definition is None.
    for row in artifact_rows:
        artifact_info[str(row.artifact)] = {
            "label": str(row.label).strip(),
            "definition": str(row.definition).strip() if row.definition else "",
        }

    # --- Building the relation map
    # Maps technique URI → {actions: set, artifacts: set}. Sets are used to deduplicate.
    relation_map: dict[str, dict[str, set[str]]] = {}
    # Normalise the technique URI to a string key.
    for row in restriction_rows:
        technique_uri = str(row.technique)
        # Ensure the entry exists before mutating.
        relation_map.setdefault(
            technique_uri,
            {"actions": set(), "artifacts": set()},
        )

        # The OWL property URI becomes an action name (e.g. d3f:Deceive --> "Deceive").
        relation_map[technique_uri]["actions"].add(local_name(row.action))

        # Look up the artifact in the info map; fall back to the URI's local name if not found.
        artifact_uri = str(row.artifact)
        info = artifact_info.get(
            artifact_uri,
            {"label": local_name(row.artifact), "definition": ""},
        )
        # Add the artifact's human-readable label.
        relation_map[technique_uri]["artifacts"].add(info["label"])

    # Same as above, but for direct links.
    # The 'if' guard is a defensive second filter, the SPARQL FILTER already removed these, but belt-and-braces is fine.
    for row in direct_link_rows:
        technique_uri = str(row.technique)
        relation_map.setdefault(
            technique_uri,
            {"actions": set(), "artifacts": set()},
        )
        property_name = local_name(row.property)
        if property_name not in {"type", "subClassOf", "d3fend-id"}:
            relation_map[technique_uri]["actions"].add(property_name)

        # Resolve the artifact label identically to the restriction loop.
        artifact_uri = str(row.artifact)
        info = artifact_info.get(
            artifact_uri,
            {"label": local_name(row.artifact), "definition": ""},
        )
        relation_map[technique_uri]["artifacts"].add(info["label"])

    # --- Building metadata
    # Use setdefault so the same technique visited multiple times (once per synonym) shares one entry.
    metadata_map: dict[str, dict[str, Any]] = {}
    for row in metadata_rows:
        uri = str(row.technique)
        entry = metadata_map.setdefault(uri, {"definition": "", "synonyms": set()})

        # Overwrite with the (only) definition; the last write wins, which is fine because there's at most one.
        if row.definition:
            entry["definition"] = str(row.definition).strip()

        # Add each synonym to the set; the inner if handles the case where the literal is only whitespace.
        if row.synonym:
            synonym = str(row.synonym).strip()
            if synonym:
                entry["synonyms"].add(synonym)

    # --- Building parents
    # Simple grouping by technique URI; duplicates removed by the set.
    parent_map: dict[str, set[str]] = {}
    for row in parent_rows:
        parent_map.setdefault(str(row.technique), set()).add(
            str(row.parentLabel).strip()
        )

    # --- Assembling the technique records
    techniques: list[dict[str, Any]] = [] # The final output list.

    # Sort by (id, label) so the candidate ordering is deterministic, important for reproducible output.
    for row in sorted(technique_rows, key=lambda r: (str(r.id), str(r.label))):
        # Extract the three primary fields.
        uri = str(row.technique)
        d3fend_id = str(row.id).strip()
        name = str(row.label).strip()
        # Fetch by URI, defaulting to empty structures so downstream code doesn't need to check for None.
        metadata = metadata_map.get(uri, {"definition": "", "synonyms": set()})
        relations = relation_map.get(uri, {"actions": set(), "artifacts": set()})
        # Convert all sets to sorted lists so the JSON output is stable across runs.
        definition = metadata["definition"]
        synonyms = sorted(metadata["synonyms"])
        actions = sorted(relations["actions"])
        artifacts = sorted(relations["artifacts"])
        parents = sorted(parent_map.get(uri, set()))
        # Start the embedding text with the two most informative fields.
        parts = [
            f"Defensive technique: {name}",
            f"D3FEND ID: {d3fend_id}",
        ]
        # Each conditional block appends a labelled sentence.
        # Fields are only appended when non-empty to avoid clutter like Action: with nothing after it.
        if actions:
            parts.append("Action: " + ", ".join(actions))
        if artifacts:
            parts.append("Digital artifact: " + "; ".join(artifacts))
        if definition:
            parts.append("Definition: " + definition)
        if synonyms:
            parts.append("Synonyms: " + "; ".join(synonyms))
        if parents:
            parts.append("Parent techniques: " + "; ".join(parents))

        # The final dict: this is the record shape consumed everywhere downstream.
        # embedding_text is the only string actually fed to the encoder.
        techniques.append({
            "source": "D3FEND",
            "uri": uri,
            "id": d3fend_id,
            "name": name,
            "description": definition,
            "actions": actions,
            "artifacts": artifacts,
            "parent_techniques": parents,
            "embedding_text": " ".join(parts),
        })

    return techniques # The full list of D3FEND technique records.



# ---- OntoSecAI indexing ----
def build_ontosecai_index(graph):

    # The only query needed for OntoSecAI.
    # Mitigations is a flat class with a Name and optional Description.
    QUERY_MITIGATIONS = """
    PREFIX hes: <http://www.semanticweb.org/ubaid/ontologies/2023/5/HES#>

    SELECT DISTINCT ?mitigation ?name ?description WHERE {
        ?mitigation a hes:Mitigations ;
                    hes:Name ?name .
        OPTIONAL { ?mitigation hes:Description ?description . }
    }
    """

    mitigations: list[dict[str, Any]] = [] # Output accumulator.

    # Extract fields; the ID is derived from the URI's local name (e.g. .../HES#M-12 --> "M-12").
    for row in graph.query(QUERY_MITIGATIONS):
        uri = str(row.mitigation)
        mitigation_id = local_name(row.mitigation)
        name = str(row.name).strip()
        description = str(row.description).strip() if row.description else ""

        # Build the embedding text --> same pattern as D3FEND but with fewer fields.
        parts = [
            f"AI-security mitigation: {name}",
            f"OntoSecAI ID: {mitigation_id}",
        ]
        if description:
            parts.append("Description: " + description)

        # The record shape mirrors the D3FEND one (minus actions/artifacts/parents) so downstream code can handle both uniformly.
        mitigations.append({
            "source": "OntoSecAI",
            "uri": uri,
            "id": mitigation_id,
            "name": name,
            "description": description,
            "embedding_text": " ".join(parts),
        })

    # Deterministic ordering, matching the D3FEND index.
    mitigations.sort(key=lambda item: (item["id"], item["name"]))
    return mitigations


# ---- Goal-model traversal ----
def iter_leaf_nodes(node, context: list[dict[str, Any]] | None = None):
    """
    Yield (leaf_goal, ancestor_goal_nodes). The hierarchy is retained only for traversal.
    Generator that walks a nested goal tree and yields (leaf, ancestors).
    """
    # Default argument is None rather than [] because mutable defaults are shared across calls, a common Python gotcha.
    if context is None:
        context = []

    # Handle dict nodes (the goal structure).
    if isinstance(node, dict):
        # A node is a goal iff it has a non-empty goal_name. subgoals may be None, [], or a list.
        subgoals = node.get("subgoals")
        is_goal = bool(node.get("goal_name"))
        # Extend the ancestor chain only if this node is a goal, otherwise it's a structural container.
        next_context = context + [node] if is_goal else context

        # Base case: a goal with no subgoals is a leaf.
        # Note it yields context (the parent chain), not next_context, the leaf is not an ancestor of itself.
        if is_goal and not subgoals:
            yield node, context
            return

        # Recurse into children, passing the updated context.
        # return prevents falling through to the list branch.
        if isinstance(subgoals, list):
            for child in subgoals:
                yield from iter_leaf_nodes(child, next_context)
        return

    # Top-level list support, some JSON files may have a root array.
    if isinstance(node, list):
        for child in node:
            yield from iter_leaf_nodes(child, context)


def source_models(directory):
    """
    Return an empty list rather than raising, the caller prints a friendly "no files" message.
    """
    if not directory.exists():
        return []
    # Glob, filter out previous outputs (so re-runs don't process their own results), and sort for determinism.
    return sorted(
        p for p in directory.glob(INPUT_PATTERN)
        if not p.name.endswith(OUTPUT_SUFFIX)
    )


# ---- Embedding retrieval ----
def load_embedding_model(model_name):
    """
    The import is inside the function so that importing this module doesn't require sentence-transformers to be installed.
    Downloads the model on first call and caches it.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def encode_texts(model, texts, batch_size):
    """
    Guard against empty input, model.encode([]) is not always well-behaved.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True, # L2-normalised vectors, so dot product = cosine similarity.
        show_progress_bar=True, # tqdm progress for large KBs.
        convert_to_numpy=True, # avoid torch tensors downstream.
    )

    # Ensure a 2-D array: a single input can come back as 1-D from some versions.
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    return vectors


def cosine_top_k(query_vector, candidate_vectors, k):
    if candidate_vectors.size == 0:
        return []
    scores = candidate_vectors @ query_vector
    k = min(k, len(scores))
    if k <= 0:
        return []

    indices = np.argpartition(-scores, k - 1)[:k]
    indices = indices[np.argsort(-scores[indices])]
    return [(int(i), float(scores[i])) for i in indices]


def build_goal_context(leaf, ancestors):
    """
    Return ONLY the leaf goal text used for retrieval and prompting.
    """
    return str(leaf.get("goal_name", "")).strip()



# ---- Prompt engineering ----
SYSTEM_PROMPT = """
You are an expert in cybersecurity requirements engineering and KAOS goal
operationalization.

Your task is to operationalize ONE KAOS leaf goal using ONLY the candidate
D3FEND defensive techniques and OntoSecAI mitigations supplied in the user
message.

Rules:
1. Select zero or more operations. It is valid to return no operation.
2. DO NOT invent, rename, paraphrase, or modify a candidate operation.
3. Every selected operation MUST use an ID that appears in the candidate list.
4. Select an operation only when it directly contributes to satisfying the
   leaf goal and its concrete defensive purpose.
5. Do not select a candidate merely because it shares a keyword with the goal.
6. Prefer concrete defensive actions over broad or weakly related concepts.
7. It is acceptable for D3FEND and/or OntoSecAI to have no suitable candidate.
8. Select at most the requested maximum number of operations.
9. Return JSON only. Do not return Markdown, explanations outside JSON, or
   code fences.

The output must have exactly this structure:
{
  "selected_operations": [
    {
      "source": "D3FEND" or "OntoSecAI",
      "id": "EXACT_CANDIDATE_ID",
      "reason": "Brief explanation of why this operation directly helps satisfy the goal."
    }
  ]
}

An empty list is valid:
{"selected_operations": []}
""".strip()


def build_user_prompt(goal_context, d3_candidates, onto_candidates, max_actions):
    """
    Build the prompt for the LLM
    """
    # Top-level prompt structure.
    # The two candidate lists are separated so the LLM sees which source each belongs to.
    payload: dict[str, Any] = {
        "goal": goal_context,
        "maximum_operations": max_actions,
        "candidate_d3fend_techniques": [],
        "candidate_ontosecai_mitigations": [],
    }

    # Only the fields Qwen needs are serialized, the raw URI and embedding_text are stripped.
    # "artifacts" is renamed to "digital_artifacts" for clarity.
    for candidate in d3_candidates:
        payload["candidate_d3fend_techniques"].append({
            "id": candidate["id"],
            "name": candidate["name"],
            "description": candidate["description"],
            "actions": candidate["actions"],
            "digital_artifacts": candidate["artifacts"],
            "parent_techniques": candidate["parent_techniques"],
        })

    # Same idea, but OntoSecAI records have fewer fields.
    for candidate in onto_candidates:
        payload["candidate_ontosecai_mitigations"].append({
            "id": candidate["id"],
            "name": candidate["name"],
            "description": candidate["description"],
        })

    # ensure_ascii=False preserves non-ASCII goal names verbatim.
    # indent=2 makes the payload easier for the LLM to parse (and easier to eyeball during debugging).
    return (
        "Operationalize the following leaf goal.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


# ---- llama.cpp OpenAI-compatible API ----
def strip_code_fences(text):
    """
    Removes optional ``` or ```json openers (case-insensitive) and any closing fence.
    Even with response_format: json_object, some local models occasionally emit fences; this is cheap insurance.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text):
    cleaned = strip_code_fences(text) # Normalise away fences first.

    # Fast path: the model obeyed and returned clean JSON.
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # Conservative fallback: locate the first balanced JSON object.
    start = cleaned.find("{") # Fallback: look for the first {.
    if start >= 0: # State for the brace matcher.
        depth = 0
        in_string = False
        escaped = False
        # Iterate character by character from the first {.
        for i in range(start, len(cleaned)):
            char = cleaned[i]
            # Inside a string: track escapes and close-quote, and ignore braces.
            # Order matters, the escaped check must come first so a backslash-escaped quote doesn't terminate the string.
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            # Outside a string: track brace depth, and the moment depth returns to zero we've found a balanced object.
            # Attempt to parse it; if it's a dict, return it.
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    value = json.loads(candidate)
                    if isinstance(value, dict):
                        return value
                    break

    # If nothing worked, raise with the raw output included for debugging.
    raise ValueError(f"Could not parse JSON object from model output: {text!r}")


def llama_chat_completion(base_url, model, system_prompt, user_prompt):
    # Construct the OpenAI-compatible endpoint.
    # rstrip("/") prevents a double slash if the caller passed a trailing slash.
    endpoint = base_url.rstrip("/") + "/chat/completions"

    # Standard chat-messages layout.
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Sampling hyperparameters from the module-level constants.
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens": MAX_TOKENS,
        "presence_penalty": PRESENCE_PENALTY,
        "response_format": {"type": "json_object"}, # Asks llama.cpp to constrain decoding to valid JSON.
        "chat_template_kwargs": {"enable_thinking": False}, # Disables Qwen3's chain-of-thought mode, otherwise the model can emit ... blocks before the JSON.
    }

    # Standard urllib.request.Request with JSON body and auth header.
    request = Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLAMA_API_KEY}",
        },
        method="POST",
    )

    # Send the request, read the body, decode and parse JSON. The with closes the connection.
    try:
        with urlopen(request, timeout=LLAMA_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"llama.cpp HTTP {exc.code} from {endpoint}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to llama.cpp at {endpoint}: {exc.reason}"
        ) from exc

    # A successful HTTP response can still have no choices (e.g. streamed and dropped).
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"llama.cpp returned no choices: {payload}")

    # Extract the text content defensively.
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"llama.cpp response has no text content: {payload}")

    # Return both the full response (useful for token accounting/logging) and the extracted text.
    return payload, content


def ping_llama(base_url):
    # Hit the standard OpenAI /v1/models endpoint.
    endpoint = base_url.rstrip("/") + "/models"
    request = Request(
        endpoint,
        headers={"Authorization": f"Bearer {LLAMA_API_KEY}"},
        method="GET",
    )
    # Any failure --> actionable error message telling the user to start the server.
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach llama.cpp at {base_url}. "
            f"Start llama-server/llama serve first. Details: {exc}"
        ) from exc

    # Diagnostic printout so the user can see whether the expected model is loaded.
    model_ids = [item.get("id") for item in payload.get("data", [])]
    print("llama.cpp models:", ", ".join(str(x) for x in model_ids) or "(none reported)")


# ---- Validation and output ----
def candidate_by_id(candidates):
    # Build an ID --> record dict for O(1) lookup during validation.
    return {candidate["id"]: candidate for candidate in candidates}


def validate_selected_operations(model_result, d3_candidates, onto_candidates, max_actions):
    # Defensive: if the model returned a dict or string here, abort.
    selected = model_result.get("selected_operations", [])
    if not isinstance(selected, list):
        raise ValueError("Model output field 'selected_operations' is not a list")

    # Lookups are built from the candidate lists, not the full KBs, so an ID that exists in D3FEND but wasn't in this prompt's top-k is still rejected.
    d3_by_id = candidate_by_id(d3_candidates)
    onto_by_id = candidate_by_id(onto_candidates)

    # Output accumulators: validated operations, warnings, and a dedup set keyed by (source, id).
    operations: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()

    # Skip non-dict entries.
    for item in selected:
        if not isinstance(item, dict):
            warnings.append(f"Ignoring malformed operation: {item!r}")
            continue

        # Normalize all three fields to strings.
        source = str(item.get("source", "")).strip()
        op_id = str(item.get("id", "")).strip()
        reason = str(item.get("reason", "")).strip()

        # Resolve the ID against the correct KB. Anything else is invalid.
        if source == "D3FEND":
            candidate = d3_by_id.get(op_id)
        elif source == "OntoSecAI":
            candidate = onto_by_id.get(op_id)
        else:
            candidate = None

        # Drop hallucinations.
        if candidate is None:
            warnings.append(
                f"Ignoring hallucinated/invalid operation: {source}:{op_id}"
            )
            continue

        # Skip duplicates.
        key = (source, op_id)
        if key in seen:
            continue
        seen.add(key)

        # Critical design point: only reason comes from the LLM.
        # id, name, and description are taken from the canonical KB record, a misbehaving model cannot corrupt them.
        operations.append({
            "source": source,
            "id": candidate["id"],
            "name": candidate["name"],
            "description": candidate["description"],
            "reason": reason,
        })

        # Enforce the cap.
        if len(operations) >= max_actions:
            break

    # Return both the validated list and any diagnostics.
    return operations, warnings


def count_goal_nodes(node):
    # Dict case.
    if isinstance(node, dict):
        # Initialise accumulators.
        has_goal = bool(node.get("goal_name"))
        subgoals = node.get("subgoals")
        child_goals = 0
        child_leaves = 0
        # Recurse into children, summing their counts.
        if isinstance(subgoals, list):
            for child in subgoals:
                goals, leaves = count_goal_nodes(child)
                child_goals += goals
                child_leaves += leaves
        # Total goals = this node (if it is one) + all descendants.
        # A leaf is a goal with no subgoals — not subgoals covers both None and [].
        total_goals = child_goals + (1 if has_goal else 0)
        total_leaves = child_leaves + (1 if has_goal and not subgoals else 0)
        return total_goals, total_leaves

    if isinstance(node, list):
        goals = leaves = 0
        for child in node:
            g, l = count_goal_nodes(child)
            goals += g
            leaves += l
        return goals, leaves

    return 0, 0


def build_summary(source_name, root, leaves):
    goals, leaf_count = count_goal_nodes(root)

    summary = {
        "json": source_name,
        "goals": goals,
        "leaf_goals": leaf_count,
        "ORG_goals": sum(leaf.get("class") == "ORG" for leaf in leaves),
        "PHYS_goals": sum(leaf.get("class") == "PHYS" for leaf in leaves),
        "TECH_goals": sum(leaf.get("class") == "TECH" for leaf in leaves),
        "ORG_operationalized_D3FEND": 0,
        "PHYS_operationalized_D3FEND": 0,
        "TECH_operationalized_D3FEND": 0,
        "ORG_operationalized_OntoSecAI": 0,
        "PHYS_operationalized_OntoSecAI": 0,
        "TECH_operationalized_OntoSecAI": 0,
    }

    for leaf in leaves:
        classes = leaf.get("class")
        actions = leaf.get("actions", [])
        sources = {action.get("source") for action in actions}

        if classes in {"ORG", "PHYS", "TECH"}:
            if "D3FEND" in sources:
                summary[f"{classes}_operationalized_D3FEND"] += 1
            if "OntoSecAI" in sources:
                summary[f"{classes}_operationalized_OntoSecAI"] += 1

    return summary


def print_summary(summaries):
    print("\n" + "=" * 146)
    print("SUMMARY — PER JSON")
    print("=" * 146)

    fields = [
        "goals",
        "leaf_goals",
        "ORG_goals",
        "PHYS_goals",
        "TECH_goals",
        "ORG_operationalized_D3FEND",
        "PHYS_operationalized_D3FEND",
        "TECH_operationalized_D3FEND",
        "ORG_operationalized_OntoSecAI",
        "PHYS_operationalized_OntoSecAI",
        "TECH_operationalized_OntoSecAI",
    ]

    totals = {field: 0 for field in fields}

    for summary in summaries:
        print(f"\n[{summary['json']}]")
        labels = {
            "goals": "Goals",
            "leaf_goals": "Leaf goals",
            "ORG_goals": "ORG goals",
            "PHYS_goals": "PHYS goals",
            "TECH_goals": "TECH goals",
            "ORG_operationalized_D3FEND": "ORG operationalized D3FEND",
            "PHYS_operationalized_D3FEND": "PHYS operationalized D3FEND",
            "TECH_operationalized_D3FEND": "TECH operationalized D3FEND",
            "ORG_operationalized_OntoSecAI": "ORG operationalized OntoSecAI",
            "PHYS_operationalized_OntoSecAI": "PHYS operationalized OntoSecAI",
            "TECH_operationalized_OntoSecAI": "TECH operationalized OntoSecAI",
        }

        for field in fields:
            value = int(summary[field])
            totals[field] += value
            print(f"  {labels[field]:34} : {value}")

    print("\n" + "-" * 146)
    print("[TOTAL]")
    labels = {
        "goals": "Goals",
        "leaf_goals": "Leaf goals",
        "ORG_goals": "ORG goals",
        "PHYS_goals": "PHYS goals",
        "TECH_goals": "TECH goals",
        "ORG_operationalized_D3FEND": "ORG operationalized D3FEND",
        "PHYS_operationalized_D3FEND": "PHYS operationalized D3FEND",
        "TECH_operationalized_D3FEND": "TECH operationalized D3FEND",
        "ORG_operationalized_OntoSecAI": "ORG operationalized OntoSecAI",
        "PHYS_operationalized_OntoSecAI": "PHYS operationalized OntoSecAI",
        "TECH_operationalized_OntoSecAI": "TECH operationalized OntoSecAI",
    }
    for field in fields:
        print(f"  {labels[field]:34} : {totals[field]}")


def output_path_for(source_path):
    return source_path.with_name(source_path.stem + OUTPUT_SUFFIX)


# ---- Per-file processing ----
def process_goal_model(
    source_path,
    model,
    d3fend_candidates,
    d3fend_vectors,
    onto_candidates,
    onto_vectors,
    base_url,
    llama_model,
    top_k_d3fend,
    top_k_ontosecai,
    max_actions,
    sleep_seconds):

    print(f"\nProcessing: {source_path}")

    with source_path.open("r", encoding="utf-8") as handle:
        root = json.load(handle)

    leaves = list(iter_leaf_nodes(root))
    leaf_nodes_for_output: list[dict[str, Any]] = []
    summary_leaves: list[dict[str, Any]] = []

    for leaf_index, (leaf, ancestors) in enumerate(leaves, start=1):
        # Use ONLY the leaf goal name for retrieval and for the LLM prompt.
        # The hierarchy is not included in the semantic representation.
        goal_context = str(leaf.get("goal_name", "")).strip()
        if not goal_context:
            raise ValueError(f"Empty leaf goal name for leaf: {leaf!r}")

        encoded_goal = encode_texts(
            model,
            [goal_context],
            EMBEDDING_BATCH_SIZE,
        )
        if encoded_goal.shape[0] != 1:
            raise ValueError(
                f"Expected one goal embedding, got shape {encoded_goal.shape} "
                f"for leaf {leaf.get('goal_id', '?')}"
            )
        query_vector = encoded_goal[0]

        d3_ranked = cosine_top_k(query_vector, d3fend_vectors, top_k_d3fend)
        onto_ranked = cosine_top_k(query_vector, onto_vectors, top_k_ontosecai)

        d3_candidates_for_goal = [
            d3fend_candidates[i] for i, _ in d3_ranked
        ]
        onto_candidates_for_goal = [
            onto_candidates[i] for i, _ in onto_ranked
        ]

        prompt = build_user_prompt(
            goal_context,
            d3_candidates_for_goal,
            onto_candidates_for_goal,
            max_actions,
        )

        try:
            _, raw_response = llama_chat_completion(
                base_url=base_url,
                model=llama_model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
            )
            parsed = extract_json_object(raw_response)
            actions, validation_warnings = validate_selected_operations(
                parsed,
                d3_candidates_for_goal,
                onto_candidates_for_goal,
                max_actions,
            )
            # The selected operations are stored directly in "actions".
        except Exception as exc:
            raw_response = ""
            parsed = {}
            actions = []
            validation_warnings = [f"LLM error: {exc}"]
            # Keep actions empty when Qwen fails; the warning records the error.

        # Preserve source leaf metadata and add Qwen fields.
        result_leaf = dict(leaf)
        result_leaf["embedding_context"] = goal_context
        # Candidates are the top-k results retrieved by cosine similarity
        # using SentenceTransformer embeddings. Qwen selects from this set.
        result_leaf["qwen_candidates"] = {
            "D3FEND": [
                {
                    "id": candidate["id"],
                    "name": candidate["name"],
                    "similarity_score": round(score, 6),
                }
                for candidate, (_, score) in zip(
                    d3_candidates_for_goal,
                    d3_ranked,
                    strict=False,
                )
            ],
            "OntoSecAI": [
                {
                    "id": candidate["id"],
                    "name": candidate["name"],
                    "similarity_score": round(score, 6),
                }
                for candidate, (_, score) in zip(
                    onto_candidates_for_goal,
                    onto_ranked,
                    strict=False,
                )
            ],
        }
        result_leaf["actions"] = actions

        if validation_warnings:
            result_leaf["qwen_warnings"] = validation_warnings

        # The selected operations and Qwen reasons are stored directly in
        # "actions"; no duplicate "qwen_decision" field is written.
        leaf_nodes_for_output.append(result_leaf)
        summary_leaves.append(result_leaf)

        d3_label = "✓" if any(a["source"] == "D3FEND" for a in actions) else "·"
        onto_label = "✓" if any(a["source"] == "OntoSecAI" for a in actions) else "·"
        goal_name = re.sub(r"\s+", " ", str(leaf.get("goal_name", ""))).strip()
        print(
            f"  {goal_name[:75]:75} | D3FEND={d3_label} | "
            f"OntoSecAI={onto_label} | actions={len(actions)}"
        )

        if validation_warnings:
            for warning in validation_warnings:
                print(f"      warning: {warning}")

        if sleep_seconds > 0 and leaf_index < len(leaves):
            time.sleep(sleep_seconds)

    # Reinsert transformed leaves into their original structure by recursively
    # consuming the generated leaf sequence.
    transformed_iter = iter(leaf_nodes_for_output)

    def replace_leaves(node: Any) -> Any:
        if isinstance(node, dict):
            subgoals = node.get("subgoals")
            is_goal = bool(node.get("goal_name"))
            if is_goal and not subgoals:
                return next(transformed_iter)
            result = dict(node)
            if isinstance(subgoals, list):
                result["subgoals"] = [replace_leaves(child) for child in subgoals]
            return result

        if isinstance(node, list):
            return [replace_leaves(child) for child in node]

        return node

    transformed_root = replace_leaves(root)
    output_path = output_path_for(source_path)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(transformed_root, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Saved: {output_path}")

    return build_summary(source_path.name, transformed_root, summary_leaves)


# ---- Main ----
def main():
    project_root = Path(__file__).resolve().parent

    print("Input directories:")
    for source_name, directory_name in GOAL_MODEL_DIRS.items():
        directory = project_root / directory_name
        print(f"  {source_name:12} -> {directory}")
    print(f"Input pattern    : {INPUT_PATTERN}")

    print("Loading D3FEND...")
    d3fend_graph = load_rdf_graph(D3FEND_CANDIDATE_FILES, "D3FEND ontology")

    print("Loading OntoSecAI...")
    ontosecai_graph = load_rdf_graph(ONTOSECAI_CANDIDATE_FILES, "OntoSecAI ontology")

    d3fend_candidates = build_d3fend_index(d3fend_graph)
    onto_candidates = build_ontosecai_index(ontosecai_graph)

    print("\nKnowledge-base index")
    print("-" * 60)
    print(f"D3FEND defensive techniques : {len(d3fend_candidates)}")
    print(f"OntoSecAI mitigations        : {len(onto_candidates)}")

    print(f"\nChecking llama.cpp at {LLAMA_BASE_URL}...")
    ping_llama(LLAMA_BASE_URL)

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedding_model = load_embedding_model(EMBEDDING_MODEL)

    print("\nEncoding D3FEND techniques...")
    d3fend_vectors = encode_texts(
        embedding_model,
        [item["embedding_text"] for item in d3fend_candidates],
        EMBEDDING_BATCH_SIZE,
    )

    print("\nEncoding OntoSecAI mitigations...")
    onto_vectors = encode_texts(
        embedding_model,
        [item["embedding_text"] for item in onto_candidates],
        EMBEDDING_BATCH_SIZE,
    )

    summaries: list[dict[str, Any]] = []

    for source_name, directory_name in GOAL_MODEL_DIRS.items():
        source_dir = project_root / directory_name
        source_files = source_models(source_dir)

        if not source_files:
            print(f"\n{source_name}: no files matching {INPUT_PATTERN} in {source_dir}")
            continue

        print("\n" + "=" * 90)
        print(f"{source_name} — {len(source_files)} goal-model files")
        print("=" * 90)

        for source_path in source_files:
            try:
                summary = process_goal_model(
                    source_path=source_path,
                    model=embedding_model,
                    d3fend_candidates=d3fend_candidates,
                    d3fend_vectors=d3fend_vectors,
                    onto_candidates=onto_candidates,
                    onto_vectors=onto_vectors,
                    base_url=LLAMA_BASE_URL,
                    llama_model=LLAMA_MODEL,
                    top_k_d3fend=TOP_K_D3FEND,
                    top_k_ontosecai=TOP_K_ONTOSECAI,
                    max_actions=MAX_ACTIONS,
                    sleep_seconds=0.0,
                )
                summaries.append(summary)
            except Exception as exc:
                print(f"\nERROR while processing {source_path.name}: {exc}", file=sys.stderr)
                traceback.print_exc()
                raise

    if not summaries:
        raise FileNotFoundError(
            f"No input goal-model JSON files found matching {INPUT_PATTERN!r} "
            f"in configured goal-model directories."
        )

    print_summary(summaries)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)