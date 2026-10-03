"""
Embedding-based operationalization of leaf goals with:

  1. MITRE D3FEND
  2. OntoSecAI

- Every leaf goal is processed, regardless of its PHYS / ORG / TECH class.
- PHYS / ORG / TECH is preserved in the output.
- Matching is semantic (SentenceTransformer + cosine similarity).
- Goal embeddings use only the leaf goal text.
- The complete knowledge graphs are NOT embedded indiscriminately:
      D3FEND --> only descendants of DefensiveTechnique
      OntoSecAI --> only individuals typed as Mitigations
- Ontology information is included in the text representation used for
  embeddings, so matching is not based on the class identifier alone.

Thresholds are deliberately configurable.
They are starting values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from rdflib import Graph, Namespace, OWL, RDF, RDFS, URIRef


# ---- Configuration ----
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" # embedding model

# Starting values only.
D3FEND_THRESHOLD = 0.45 # starting threshold
ONTOSECAI_THRESHOLD = 0.45

# We retrieve a small semantic neighborhood, then retain candidates above threshold.
# This makes the result auditable and avoids blindly keeping every moderately similar item.
TOP_K_D3FEND = 8 # retrieve 8 nearest D3FEND techniques
TOP_K_ONTOSECAI = 8 # retrieve 8 nearest OntoSecAI mitigations
MAX_D3FEND_ACTIONS = 5 # keep at most 5 accepted D3FEND actions
MAX_ONTOSECAI_ACTIONS = 5 # keep at most 5 accepted OntoSecAI actions

BATCH_SIZE = 64 # encoding batch size

# Goal-model directories produced by the updated goal-model builders.
GOAL_MODEL_DIRS = {
    "ISO 27001": "goal_models_ISO27001",
    "ISO 42001": "goal_models_ISO42001",
    "NIST": "goal_models_NIST",
    "CIS": "goal_models_CIS",
    "AI-RMF": "goal_models_AIRMF",
}

# the output file suffix
OUTPUT_SUFFIX = ".operationalized_embeddings.json"

# input file pattern
INPUT_PATTERN = "*_leaf_goals_classified.json"

# The script first tries the canonical names, then the names of the currently
# attached files. This makes it easy to run in the repository without renaming
# the ontology files.
#D3FEND_CANDIDATE_FILES = (
#    "d3fend.ttl",
#)

#ONTOSECAI_CANDIDATE_FILES = (
#    "ontosecai.rdf",
#)

# Single file names (strings) instead of tuples/lists
D3FEND_CANDIDATE_FILES = "d3fend.ttl"
ONTOSECAI_CANDIDATE_FILES = "ontosecai.rdf"

D3F = Namespace("http://d3fend.mitre.org/ontologies/d3fend.owl#")
HES = Namespace("http://www.semanticweb.org/ubaid/ontologies/2023/5/HES#")


def local_name(uri):
    """
    Return the local fragment/local name of an RDF URI.
    Convert URI to string.

    :param uri: RDF URI (URIRef or string)
    :return: local fragment/local name as a string
    """
    text = str(uri)
    if "#" in text:
        return text.rsplit("#", 1)[1]
    return text.rstrip("/").rsplit("/", 1)[-1]


# ---- D3FEND indexing ----
def build_d3fend_index(graph):
    """
    Build the D3FEND operationalization index.

    We deliberately do not use every OWL class in the D3FEND knowledge graph.
    Operationalization candidates are defensive techniques. However, we use
    several SPARQL queries to capture their useful semantics:

      1. Defensive techniques, including all descendants.
      2. Defensive technique labels/definitions/synonyms.
      3. Technique restrictions: action verb --> digital artifact.
      4. Direct "associated-with" / related artifact links where present.
      5. Parent defensive-technique relationships.

    This follows the structure exposed by D3FEND: defensive techniques are
    related to digital artifacts and are organized hierarchically. The artifact
    ontology itself contains hundreds of useful artifacts, but artifacts are
    used here as semantic context, not as independent operational actions.

    :param graph: rdflib.Graph containing the D3FEND ontology
    :return: list of dicts, each representing a defensive technique with
             uri, d3fend_id, name, definition, synonyms, actions, artifacts,
             parent_techniques, and embedding_text
    """

    # ------------------------------------------------------------------
    # QUERY 1: ALL DEFENSIVE TECHNIQUES
    #
    # rdfs:subClassOf* captures direct and transitive descendants.
    # ------------------------------------------------------------------
    QUERY_DEFENSIVE_TECHNIQUES = """
    PREFIX d3f:  <http://d3fend.mitre.org/ontologies/d3fend.owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?technique ?id ?label WHERE {
        ?technique rdfs:subClassOf* d3f:DefensiveTechnique ;
                   d3f:d3fend-id ?id ;
                   rdfs:label ?label .
    }
    """

    # ------------------------------------------------------------------
    # QUERY 2: DIGITAL ARTIFACTS
    #
    # These are semantic context for techniques. The D3FEND Artifact Ontology
    # contains high-level and specialized artifacts (User Account, Credential,
    # File, Network Traffic, etc.).
    # ------------------------------------------------------------------
    QUERY_DIGITAL_ARTIFACTS = """
    PREFIX d3f:  <http://d3fend.mitre.org/ontologies/d3fend.owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?artifact ?label ?definition WHERE {
        ?artifact rdfs:subClassOf* d3f:DigitalArtifact ;
                  rdfs:label ?label .

        OPTIONAL {
            ?artifact d3f:definition ?definition .
        }
    }
    """

    # ------------------------------------------------------------------
    # QUERY 3: TECHNIQUE RESTRICTIONS
    #
    # This captures the core D3FEND semantics:
    #
    #   defensive technique
    #       --action predicate--> digital artifact
    #
    # Example shape:
    #   UserAccountPermissions
    #       --restricts--> UserAccount
    #
    # We allow materialized OWL restrictions at any level below
    # DefensiveTechnique.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # QUERY 4: DIRECT ARTIFACT ASSOCIATIONS
    #
    # D3FEND has evolved and materialized more artifact/event/agent
    # restrictions in newer ontology releases. We therefore also inspect
    # object properties whose target is a DigitalArtifact, rather than relying
    # exclusively on blank-node restrictions.
    # ------------------------------------------------------------------
    QUERY_TECHNIQUE_ARTIFACT_LINKS = """
    PREFIX d3f: <http://d3fend.mitre.org/ontologies/d3fend.owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
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

    # ------------------------------------------------------------------
    # QUERY 5: TECHNIQUE METADATA
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # QUERY 6: DIRECT PARENT TECHNIQUES
    # ------------------------------------------------------------------
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

    technique_rows = list(graph.query(QUERY_DEFENSIVE_TECHNIQUES))
    artifact_rows = list(graph.query(QUERY_DIGITAL_ARTIFACTS))
    restriction_rows = list(graph.query(QUERY_TECHNIQUE_RESTRICTIONS))
    direct_link_rows = list(graph.query(QUERY_TECHNIQUE_ARTIFACT_LINKS))
    metadata_rows = list(graph.query(QUERY_TECHNIQUE_METADATA))
    parent_rows = list(graph.query(QUERY_PARENT_TECHNIQUES))

    # Artifact labels + definitions.
    artifact_info: dict[str, dict[str, str]] = {}
    for row in artifact_rows:
        uri = str(row.artifact)
        artifact_info[uri] = {
            "label": str(row.label).strip(),
            "definition": str(row.definition).strip() if row.definition else "",
        }

    # Technique --> actions/artifacts from OWL restrictions.
    relation_map: dict[str, dict[str, set[str]]] = {}
    for row in restriction_rows:
        technique_uri = str(row.technique)
        relation_map.setdefault(technique_uri, {"actions": set(), "artifacts": set()},)

        relation_map[technique_uri]["actions"].add(local_name(row.action))

        artifact_uri = str(row.artifact)
        info = artifact_info.get(artifact_uri, {"label": local_name(row.artifact), "definition": ""})
        relation_map[technique_uri]["artifacts"].add(info["label"])

    # Technique --> direct artifact links.  We keep the property name as an
    # additional semantic action signal when it is meaningful.
    for row in direct_link_rows:
        technique_uri = str(row.technique)
        relation_map.setdefault(technique_uri, {"actions": set(), "artifacts": set()},)

        property_name = local_name(row.property)
        artifact_uri = str(row.artifact)

        if property_name not in {"type", "subClassOf", "d3fend-id"}:
            relation_map[technique_uri]["actions"].add(property_name)

        info = artifact_info.get(artifact_uri, {"label": local_name(row.artifact), "definition": ""})
        relation_map[technique_uri]["artifacts"].add(info["label"])

    # Metadata.
    metadata_map: dict[str, dict[str, Any]] = {}
    for row in metadata_rows:
        uri = str(row.technique)
        entry = metadata_map.setdefault(uri, {"definition": "", "synonyms": set()})
        if row.definition:
            entry["definition"] = str(row.definition).strip()
        if row.synonym:
            synonym = str(row.synonym).strip()
            if synonym:
                entry["synonyms"].add(synonym)

    # Parent labels
    parent_map: dict[str, set[str]] = {}
    for row in parent_rows:
        parent_map.setdefault(str(row.technique), set()).add(str(row.parentLabel).strip())

    techniques: list[dict[str, Any]] = []

    for row in sorted(technique_rows, key=lambda r: (str(r.id), str(r.label))):
        uri = str(row.technique)
        d3fend_id = str(row.id).strip()
        label = str(row.label).strip()

        metadata = metadata_map.get(uri, {"definition": "", "synonyms": set()})
        relations = relation_map.get(uri, {"actions": set(), "artifacts": set()})

        definition = metadata["definition"]
        synonyms = sorted(metadata["synonyms"])
        actions = sorted(relations["actions"])
        artifacts = sorted(relations["artifacts"])
        parents = sorted(parent_map.get(uri, set()))

        parts = [
            f"Defensive technique: {label}",
            f"D3FEND ID: {d3fend_id}",
        ]

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

        techniques.append({
            "uri": uri,
            "d3fend_id": d3fend_id,
            "name": label,
            "definition": definition,
            "synonyms": synonyms,
            "actions": actions,
            "artifacts": artifacts,
            "parent_techniques": parents,
            "embedding_text": " ".join(parts),
        })

    return techniques


# ---- OntoSecAI indexing ----
def build_ontosecai_index(graph):
    """
    Build the OntoSecAI operationalization index.

    Only individuals explicitly typed as hes:Mitigations are candidates.
    Attack and lifecycle information is deliberately NOT included in the
    final output. It is contextual metadata, not the operationalization
    itself.

    :param graph: rdflib.Graph containing the OntoSecAI ontology
    :return: list of dicts, each representing a mitigation with
             uri, mitigation_id, name, description, embedding_text
    """

    # ------------------------------------------------------------------
    # SPARQL QUERY — OntoSecAI mitigations
    #
    # Only mitigation individuals are operationalization candidates.
    # The graph contains attack concepts and other ontology entities too,
    # so we explicitly restrict the candidate universe here.
    # ------------------------------------------------------------------
    QUERY_MITIGATIONS = """
    PREFIX hes: <http://www.semanticweb.org/ubaid/ontologies/2023/5/HES#>

    SELECT DISTINCT ?mitigation ?name ?description WHERE {
        ?mitigation a hes:Mitigations ;
                    hes:Name ?name .

        OPTIONAL {
            ?mitigation hes:Description ?description .
        }
    }
    """

    mitigations: list[dict[str, Any]] = []

    for row in graph.query(QUERY_MITIGATIONS):
        mitigation_uri = str(row.mitigation)
        mitigation_id = local_name(row.mitigation)
        name = str(row.name).strip()
        description = str(row.description).strip() if row.description else ""

        # The embedding representation intentionally focuses on the
        # mitigation itself. Lifecycle phases and related attacks are not
        # required to identify the mitigation that operationalizes a goal.
        embedding_parts = [
            f"AI-security mitigation: {name}",
            f"OntoSecAI ID: {mitigation_id}",
        ]

        if description:
            embedding_parts.append("Description: " + description)

        mitigations.append({
            "uri": mitigation_uri,
            "mitigation_id": mitigation_id,
            "name": name,
            "description": description,
            "embedding_text": " ".join(embedding_parts),
        })

    mitigations.sort(key=lambda item: (item["mitigation_id"], item["name"]))
    return mitigations


# ---- Goal-model traversal ----
def iter_leaf_nodes(node, context: list[dict[str, Any]] | None = None):
    """
    Recursively collect leaf goals together with their goal hierarchy.

    The returned context contains ancestor goal nodes, from the highest
    available goal level down to the parent of the leaf. The hierarchy is
    retained for traversal and analysis, but is not included in the goal
    embedding text. Structural nodes that do not carry a goal_name are ignored.

    :param node: current node (dict or list)
    :param context: list of ancestor goal nodes
    :return: generator yielding (leaf_node, context) tuples
    """
    if context is None:
        context = []

    if isinstance(node, dict):
        subgoals = node.get("subgoals")
        is_goal = bool(node.get("goal_name"))
        next_context = context + [node] if is_goal else context

        if is_goal and not subgoals:
            # Context contains ancestors only; the leaf itself is added
            # explicitly when building the embedding text.
            yield node, context
            return

        if isinstance(subgoals, list):
            for child in subgoals:
                yield from iter_leaf_nodes(child, next_context)
        return

    if isinstance(node, list):
        for child in node:
            yield from iter_leaf_nodes(child, context)




def source_models(directory):
    """
    Return only source goal-model JSON files matching the required pattern.

    :param directory: pathlib.Path to the directory
    :return: sorted list of pathlib.Path objects
    """
    if not directory.exists():
        return []

    return sorted(
        p
        for p in directory.glob(INPUT_PATTERN)
        if p.is_file() and not p.name.endswith(OUTPUT_SUFFIX)
    )


# ---- Embedding model ----
def load_embedding_model(model_name):
    """
    Load the SentenceTransformer model once.

    :param model_name: name of the SentenceTransformer model
    :return: loaded SentenceTransformer instance
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def encode_texts(model, texts, batch_size = BATCH_SIZE):
    """
    Encode texts in batches and normalize them.

    With normalized vectors:
        cosine(a, b) == dot(a, b)

    :param model: SentenceTransformer model
    :param texts: list of strings to encode
    :param batch_size: batch size for encoding
    :return: numpy array of shape (len(texts), embedding_dim)
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


# ---- Semantic retrieval ----
def cosine_top_k(goal_vector, candidate_matrix, top_k,):
    """
    Return (candidate_index, similarity) sorted descending.

    Since both matrices are normalized, the matrix product is cosine similarity.

    :param goal_vector: 1D numpy array of the goal embedding
    :param candidate_matrix: 2D numpy array of candidate embeddings
    :param top_k: number of top candidates to return
    :return: list of (index, similarity_score) tuples
    """
    if len(candidate_matrix) == 0:
        return []

    scores = candidate_matrix @ goal_vector
    k = min(top_k, len(scores))

    # Partial selection first, then exact sorting among the selected indices.
    if k < len(scores):
        idx = np.argpartition(scores, -k)[-k:]
    else:
        idx = np.arange(len(scores))

    idx = idx[np.argsort(scores[idx])[::-1]]

    return [(int(i), float(scores[i])) for i in idx]


def accepted_candidates(ranked, threshold, max_actions,):
    """
    Keep candidates above threshold.

    This is intentionally not an argmax-only policy: one goal may genuinely
    have several valid operationalizations.

    :param ranked: list of (index, score) tuples
    :param threshold: minimum similarity score
    :param max_actions: maximum number of candidates to keep
    :return: filtered list of (index, score) tuples
    """
    return [
        (idx, score)
        for idx, score in ranked
        if score >= threshold
    ][:max_actions]


# ---- Output construction ----
def d3fend_action_dict(item, score):
    """
    Return the compact D3FEND action representation.

    :param item: D3FEND technique dict
    :param score: similarity score
    :return: compact dict with source, id, name, description, similarity_score
    """
    return {
        "source": "D3FEND",
        "id": item["d3fend_id"],
        "name": item["name"],
        "description": item["definition"],
        "similarity_score": round(score, 6),
    }


def ontosecai_action_dict(item, score):
    """
    Return the compact OntoSecAI action representation.

    :param item: OntoSecAI mitigation dict
    :param score: similarity score
    :return: compact dict with source, id, name, description, similarity_score
    """
    return {
        "source": "OntoSecAI",
        "id": item["mitigation_id"],
        "name": item["name"],
        "description": item["description"],
        "similarity_score": round(score, 6),
    }


def candidate_preview(item, score):
    """
    Compact candidate information for optional audit output.

    :param item: candidate dict (D3FEND or OntoSecAI)
    :param score: similarity score
    :return: compact dict with source, id, name, similarity_score
    """
    if "d3fend_id" in item:
        return {
            "source": "D3FEND",
            "id": item["d3fend_id"],
            "name": item["name"],
            "similarity_score": round(score, 6),
        }

    return {
        "source": "OntoSecAI",
        "id": item["mitigation_id"],
        "name": item["name"],
        "similarity_score": round(score, 6),
    }


def operationalize_model(model, embedding_model, d3fend_items,
                         d3fend_vectors, onto_items,
                         onto_vectors, d3fend_threshold, onto_threshold):
    """
    Operationalize ALL leaf goals.

    PHYS / ORG / TECH is not used as a gate. It remains available in each
    leaf for analysis, but all leaves are independently compared with both KBs.
    Goal embeddings use the leaf goal name.

    :param model: goal model as a dict/list
    :param embedding_model: SentenceTransformer model
    :param d3fend_items: list of D3FEND technique dicts
    :param d3fend_vectors: numpy array of D3FEND embeddings
    :param onto_items: list of OntoSecAI mitigation dicts
    :param onto_vectors: numpy array of OntoSecAI embeddings
    :param d3fend_threshold: threshold for D3FEND matches
    :param onto_threshold: threshold for OntoSecAI matches
    :return: tuple (model, stats) where model is updated with operationalization
             results and stats is a dictionary of coverage statistics
    """
    leaf_records = list(iter_leaf_nodes(model))
    leaves = [leaf for leaf, _context in leaf_records]

    # Embed the leaf goal text.
    # The cosine similarity is therefore computed directly between the
    # semantic representation of the leaf goal and the semantic
    # representations of D3FEND / OntoSecAI actions.

    # --- Example of embedding context used for cosine similarity ---
    # If the leaf goal is:
    # {
    #   'goal_id': 'G1.1.1',
    #   'goal_name': 'Establish organizational policies, processes, and procedures for mapping, measuring, and managing AI risks',
    #   'class': 'ORG'
    # }

    # then the exact string encoded by SentenceTransformer and later compared
    # against D3FEND and OntoSecAI embedding vectors is:
    # "Establish organizational policies, processes, and procedures for mapping, measuring, and managing AI risks"

    # This string becomes leaf["embedding_context"] and is the sole input to
    # encode_texts() for goal-side embeddings. The cosine similarity is
    # therefore computed between this leaf-goal sentence and, for example:

    # --> D3fend side:
    # Defensive technique: System Vulnerability Assessment
    # D3FEND ID: D3-SYSVA
    # Action: evaluates
    # Digital artifact: Digital System
    # Definition: System vulnerability assessment relates all the vulnerabilities of a system's components in the context of their configuration and internal dependencies and can also include assessing risk emerging from the system's design as a whole, not just the sum of individual component vulnerabilities.
    # Parent techniques: System Mapping

    # --> OntoSecAI side:
    # AI-security mitigation: Limit Release of Public Information
    # OntoSecAI ID: AML.M0001
    # Description: Limit the public release of technical information about the machine learning stack used in an organization's products or services. Technical knowledge of how machine learning is used can be leveraged by adversaries to perform targeting and tailor attacks to the target system. Additionally, consider limiting the release of organizational information - including physical locations, researcher names, and department structures - from which technical details such as machine learning techniques, model architectures, or datasets may be inferred.

    goal_texts: list[str] = []
    for leaf, _context in leaf_records:
        leaf_name = str(leaf["goal_name"]).strip()
        embedding_text = leaf_name
        goal_texts.append(embedding_text)

        # Store the exact text used for retrieval so results are auditable.
        leaf["embedding_context"] = embedding_text

    goal_vectors = encode_texts(embedding_model, goal_texts)

    stats = {
        "leaf_goals": len(leaves),
        "d3fend_match": 0,
        "ontosecai_match": 0,
        "both": 0,
        "none": 0,
        "by_class": {},
    }

    for leaf, goal_vector in zip(leaves, goal_vectors):
        goal_name = str(leaf["goal_name"]).strip()

        # --- D3FEND ---
        d3_ranked = cosine_top_k(
            goal_vector,
            d3fend_vectors,
            TOP_K_D3FEND,
        )
        d3_accepted = accepted_candidates(
            d3_ranked,
            d3fend_threshold,
            MAX_D3FEND_ACTIONS,
        )

        # --- OntoSecAI ---
        onto_ranked = cosine_top_k(
            goal_vector,
            onto_vectors,
            TOP_K_ONTOSECAI,
        )
        onto_accepted = accepted_candidates(
            onto_ranked,
            onto_threshold,
            MAX_ONTOSECAI_ACTIONS,
        )

        d3_actions = [
            d3fend_action_dict(d3fend_items[idx], score)
            for idx, score in d3_accepted
        ]
        onto_actions = [
            ontosecai_action_dict(onto_items[idx], score)
            for idx, score in onto_accepted
        ]

        all_actions = d3_actions + onto_actions

        # Keep the best candidate from each KB even when it is below the
        # threshold. This lets us distinguish a poor semantic match from a
        # plausible match that was rejected only by the threshold.
        leaf["best_D3FEND_candidate"] = (
            candidate_preview(d3fend_items[d3_ranked[0][0]], d3_ranked[0][1])
            if d3_ranked
            else None
        )
        leaf["best_OntoSecAI_candidate"] = (
            candidate_preview(onto_items[onto_ranked[0][0]], onto_ranked[0][1])
            if onto_ranked
            else None
        )

        # Single compact action list. Each item is either a D3FEND
        # technique or an OntoSecAI mitigation
        leaf["actions"] = all_actions

        leaf["operationalization_status"] = (
            "D3FEND+OntoSecAI"
            if d3_actions and onto_actions
            else "D3FEND"
            if d3_actions
            else "OntoSecAI"
            if onto_actions
            else "none_above_threshold"
        )

        # Coverage statistics
        has_d3 = bool(d3_actions)
        has_onto = bool(onto_actions)

        if has_d3:
            stats["d3fend_match"] += 1
        if has_onto:
            stats["ontosecai_match"] += 1

        if has_d3 and has_onto:
            stats["both"] += 1
        if not has_d3 and not has_onto:
            stats["none"] += 1

        cls = leaf.get("class", "UNCLASSIFIED")
        stats["by_class"].setdefault(
            cls,
            {
                "leaf_goals": 0,
                "d3fend_match": 0,
                "ontosecai_match": 0,
                "both": 0,
                "none": 0,
            },
        )
        cls_stats = stats["by_class"][cls]
        cls_stats["leaf_goals"] += 1
        cls_stats["d3fend_match"] += int(has_d3)
        cls_stats["ontosecai_match"] += int(has_onto)
        cls_stats["both"] += int(has_d3 and has_onto)
        cls_stats["none"] += int(not has_d3 and not has_onto)

        # Lightweight progress line for long runs. Always show the best
        # score, including when it is below the operationalization threshold.
        best_d3_score = d3_ranked[0][1] if d3_ranked else float("nan")
        best_onto_score = onto_ranked[0][1] if onto_ranked else float("nan")

        print(
            f"  {goal_name[:72]:72s} | "
            f"D3FEND={best_d3_score:.3f} "
            f"{'✓' if has_d3 else '·'} | "
            f"OntoSecAI={best_onto_score:.3f} "
            f"{'✓' if has_onto else '·'}"
        )

    return model, stats

# ----------

def find_first_existing(base_dir, filename):
    """
    Return the path to the file if it exists, otherwise raise FileNotFoundError.

    :param base_dir: pathlib.Path to the base directory
    :param filename: name of the file to look for
    :return: pathlib.Path to the existing file
    :raises FileNotFoundError: if the file does not exist
    """
    #for name in candidates:
    #    path = base_dir / name
    #    if path.exists():
    #        return path
    path = base_dir / filename
    if path.exists():
        return path
    #raise FileNotFoundError(
    #    "None of these files exists:\n"
    #    + "\n".join(f"  - {base_dir / name}" for name in candidates)
    #)
    raise FileNotFoundError(f"File not found: {path}")


def print_kb_summary(d3fend_items, onto_items):
    """
    Print the summary of the Knowledge Base.

    :param d3fend_items: list of D3FEND technique dicts
    :param onto_items: list of OntoSecAI mitigation dicts
    :return: None
    """
    print("\nKnowledge-base index")
    print("-" * 60)
    print(f"D3FEND defensive techniques : {len(d3fend_items)}")
    print(f"OntoSecAI mitigations        : {len(onto_items)}")


def count_goal_nodes(model):
    """
    Return (leaf_goals, total_goal_nodes).

    Goal nodes are dictionaries carrying goal_id + goal_name. This excludes
    control/metadata nodes that do not themselves represent goals.

    :param model: goal model as a dict/list
    :return: tuple (leaves, total_goal_nodes)
    """
    leaf_records = list(iter_leaf_nodes(model))
    leaves = [leaf for leaf, _context in leaf_records]

    total_goal_nodes = 0

    def walk(node: Any) -> None:
        nonlocal total_goal_nodes

        if isinstance(node, dict):
            if "goal_id" in node and "goal_name" in node:
                total_goal_nodes += 1

            for child in node.get("subgoals", []) or []:
                walk(child)

        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(model)
    return leaves, total_goal_nodes


def build_json_summary(model, stats, source_path):
    """
    Summary requested by the user, per JSON file.

    Counts:
      - total goals
      - leaf goals
      - ORG / PHYS / TECH leaves
      - D3FEND operationalized by each class
      - OntoSecAI operationalized by each class

    :param model: goal model as a dict/list
    :param stats: statistics dict from operationalize_model
    :param source_path: pathlib.Path to the source JSON file
    :return: summary dict
    """
    leaves, total_goals = count_goal_nodes(model)

    classes = ("ORG", "PHYS", "TECH")

    summary = {
        "json": source_path.name,
        "goals": total_goals,
        "leaf_goals": len(leaves),
    }

    for cls in classes:
        summary[f"{cls}_goals"] = sum(
            leaf.get("class") == cls for leaf in leaves
        )

        summary[f"{cls}_operationalized_D3FEND"] = sum(
            leaf.get("class") == cls
            and any(
                action.get("source") == "D3FEND"
                for action in leaf.get("actions", [])
            )
            for leaf in leaves
        )

        summary[f"{cls}_operationalized_OntoSecAI"] = sum(
            leaf.get("class") == cls
            and any(
                action.get("source") == "OntoSecAI"
                for action in leaf.get("actions", [])
            )
            for leaf in leaves
        )

    return summary


def print_summary(json_summaries):
    """
    Print the requested summary once per JSON and then overall totals.

    :param json_summaries: list of summary dicts from build_json_summary
    :return: None
    """
    columns = [
        "json",
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

    print("\n" + "=" * 190)
    print("SUMMARY — PER JSON")
    print("=" * 190)

    # Compact one-line summary with every requested field.
    for summary in json_summaries:
        print(f"\n[{summary['json']}]")
        print(f"Goals      : {summary['goals']}")
        print(f"Leaf goals : {summary['leaf_goals']}")
        print(f"ORG goals  : {summary['ORG_goals']}")
        print(f"PHYS goals : {summary['PHYS_goals']}")
        print(f"TECH goals : {summary['TECH_goals']}")
        print(
            f"  ORG operationalized D3FEND     : "
            f"{summary['ORG_operationalized_D3FEND']}"
        )
        print(
            f"  PHYS operationalized D3FEND    : "
            f"{summary['PHYS_operationalized_D3FEND']}"
        )
        print(
            f"  TECH operationalized D3FEND    : "
            f"{summary['TECH_operationalized_D3FEND']}"
        )
        print(
            f"  ORG operationalized OntoSecAI  : "
            f"{summary['ORG_operationalized_OntoSecAI']}"
        )
        print(
            f"  PHYS operationalized OntoSecAI : "
            f"{summary['PHYS_operationalized_OntoSecAI']}"
        )
        print(
            f"  TECH operationalized OntoSecAI : "
            f"{summary['TECH_operationalized_OntoSecAI']}"
        )

    if not json_summaries:
        return

    # Overall totals are useful for comparing standards.
    total = {"json": "TOTAL"}
    numeric_columns = columns[1:]

    for column in numeric_columns:
        total[column] = sum(item[column] for item in json_summaries)

    print("\n" + "-" * 190)
    print("[TOTAL]")
    print(f"Goals      : {total['goals']}")
    print(f"Leaf goals : {total['leaf_goals']}")
    print(f"ORG goals  : {total['ORG_goals']}")
    print(f"PHYS goals : {total['PHYS_goals']}")
    print(f"TECH goals : {total['TECH_goals']}")
    print(
        f"  ORG operationalized D3FEND     : "
        f"{total['ORG_operationalized_D3FEND']}"
    )
    print(
        f"  PHYS operationalized D3FEND    : "
        f"{total['PHYS_operationalized_D3FEND']}"
    )
    print(
        f"  TECH operationalized D3FEND    : "
        f"{total['TECH_operationalized_D3FEND']}"
    )
    print(
        f"  ORG operationalized OntoSecAI  : "
        f"{total['ORG_operationalized_OntoSecAI']}"
    )
    print(
        f"  PHYS operationalized OntoSecAI : "
        f"{total['PHYS_operationalized_OntoSecAI']}"
    )
    print(
        f"  TECH operationalized OntoSecAI : "
        f"{total['TECH_operationalized_OntoSecAI']}"
    )


def parse_args():
    """
    Parse command line arguments.

    :return: argparse.Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Embedding-based operationalization of leaf goals."
    )

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"SentenceTransformer model (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--d3fend-threshold",
        type=float,
        default=D3FEND_THRESHOLD,
        help=f"D3FEND cosine threshold (default: {D3FEND_THRESHOLD})",
    )
    parser.add_argument(
        "--ontosecai-threshold",
        type=float,
        default=ONTOSECAI_THRESHOLD,
        help=f"OntoSecAI cosine threshold (default: {ONTOSECAI_THRESHOLD})",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Repository/project root (default: current directory).",
    )

    return parser.parse_args()


def main():
    """
    Main entry point for the script.

    # Resolve base_dir.
    # Find D3FEND and OntoSecAI files.
    # Load RDF graphs.
    # Build indexes.
    # Print KB summary.
    # Load embedding model.
    # Encode D3FEND and OntoSecAI texts.
    # For each standard in GOAL_MODEL_DIRS:
    #   Find source JSON files.
    #   For each file:
    #     Load JSON.
    #     operationalize_model(...)
    #     Save output with OUTPUT_SUFFIX.
    #     Build JSON summary.
    #     Accumulate standard stats.
    #   Store standard stats.
    # Print final summary.

    :return: None
    """
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()

    d3fend_file = find_first_existing(base_dir, D3FEND_CANDIDATE_FILES)
    ontosecai_file = find_first_existing(base_dir, ONTOSECAI_CANDIDATE_FILES)

    print("Loading D3FEND...")
    d3fend_graph = Graph().parse(d3fend_file, format="turtle")

    print("Loading OntoSecAI...")
    ontosecai_graph = Graph().parse(ontosecai_file)

    d3fend_items = build_d3fend_index(d3fend_graph)
    onto_items = build_ontosecai_index(ontosecai_graph)

    print_kb_summary(d3fend_items, onto_items)

    # Semantic index is built ONCE for the complete run.
    print(f"\nLoading embedding model: {args.model}")
    embedding_model = load_embedding_model(args.model)

    print("\nEncoding D3FEND techniques...")
    d3fend_vectors = encode_texts(embedding_model,
        [item["embedding_text"] for item in d3fend_items]
    )

    print("\nEncoding OntoSecAI mitigations...")
    onto_vectors = encode_texts(embedding_model,
        [item["embedding_text"] for item in onto_items]
    )

    all_stats: dict[str, dict[str, Any]] = {}
    json_summaries: list[dict[str, Any]] = []

    for standard, directory_name in GOAL_MODEL_DIRS.items():
        directory = base_dir / directory_name
        files = source_models(directory)

        if not files:
            print(
                f"\n{standard}: no source goal models found in "
                f"{directory}"
            )
            continue

        print(
            f"\n{'=' * 90}\n"
            f"{standard} — {len(files)} goal-model files\n"
            f"{'=' * 90}"
        )

        standard_stats = {
            "leaf_goals": 0,
            "d3fend_match": 0,
            "ontosecai_match": 0,
            "both": 0,
            "none": 0,
            "by_class": {},
        }

        for path in files:
            print(f"\nProcessing: {path}")

            with path.open("r", encoding="utf-8") as handle:
                model = json.load(handle)

            model, stats = operationalize_model(
                model=model,
                embedding_model=embedding_model,
                d3fend_items=d3fend_items,
                d3fend_vectors=d3fend_vectors,
                onto_items=onto_items,
                onto_vectors=onto_vectors,
                d3fend_threshold=args.d3fend_threshold,
                onto_threshold=args.ontosecai_threshold,
            )

            output_path = path.with_name(
                path.stem + OUTPUT_SUFFIX
            )

            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    model,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

            json_summaries.append(
                build_json_summary(
                    model=model,
                    stats=stats,
                    source_path=path,
                )
            )

            for key in (
                "leaf_goals",
                "d3fend_match",
                "ontosecai_match",
                "both",
                "none",
            ):
                standard_stats[key] += stats[key]

            for cls, cls_stats in stats["by_class"].items():
                standard_stats["by_class"].setdefault(
                    cls,
                    {
                        "leaf_goals": 0,
                        "d3fend_match": 0,
                        "ontosecai_match": 0,
                        "both": 0,
                        "none": 0,
                    },
                )
                for key in standard_stats["by_class"][cls]:
                    standard_stats["by_class"][cls][key] += cls_stats[key]

            print(f"Saved: {output_path}")

        all_stats[standard] = standard_stats

    print_summary(json_summaries)


if __name__ == "__main__":
    main()
