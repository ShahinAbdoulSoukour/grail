"""
Merge operationalized goal-model JSON files into one RDF-star graph.

The script is procedural and uses dictionaries/functions only.
It preserves:
    - goal hierarchy
    - goal IDs and names
    - control/objective/function metadata when present
    - ORG / PHYS / TECH classification
    - embedding context
    - selected operationalizations (actions) and similarity scores
    - similarity scores
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import RDFS, XSD

# ---- Configuration ----
DEFAULT_INPUT_PATTERN = "*_leaf_goals_classified.operationalized_embeddings.json"
DEFAULT_OUTPUT = "merged_goal_models.ttl" # default Turtle output name

BASE = Namespace("https://example.org/goal-model/")
GOAL = Namespace("https://example.org/goal-model/ontology/")

# ---- URI / literal helpers ----
def slug(text):
    """
    Convert arbitrary text into a URI-safe fragment.

    :param text: arbitrary string
    :return: URI-safe fragment
    """
    # Strips whitespace, replaces invalid URI characters with _, collapses consecutive _,
    # trims leading/trailing _, and falls back to "unknown" when empty.
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "unknown"


def safe_goal_local_name(standard, function_name, goal_id):
    """
    Create a globally unique goal URI fragment.

    :param standard: standard name
    :param function_name: function name
    :param goal_id: goal identifier
    :return: unique goal URI fragment
    """
    return f"goal_{slug(standard)}_{slug(function_name)}_{slug(goal_id)}"


def safe_operation_local_name(source, operation_id):
    """
    Create a globally unique operation URI fragment.

    :param source: operation source (D3FEND or OntoSecAI)
    :param operation_id: operation identifier
    :return: unique operation URI fragment
    """
    return f"operation_{slug(source)}_{slug(operation_id)}"


def add_literal(graph, subject, predicate, value):
    """
    Add a string literal only when the value is non-empty.

    :param graph: rdflib.Graph
    :param subject: RDF subject
    :param predicate: RDF predicate
    :param value: value to add
    :return: None
    """
    if value is None:
        return
    text = str(value).strip()
    if text:
        graph.add((subject, predicate, Literal(text)))


def add_decimal(graph, subject, predicate, value):
    """
    Add a decimal literal when possible.

    :param graph: rdflib.Graph
    :param subject: RDF subject
    :param predicate: RDF predicate
    :param value: numeric value
    :return: None
    """
    if value is None or value == "":
        return
    try:
        graph.add((subject, predicate, Literal(float(value), datatype=XSD.decimal)))
    except (TypeError, ValueError):
        pass

# ---- RDF schema ----
def bind_namespaces(graph):
    """
    Bind short prefixes for readable Turtle output.

    :param graph: rdflib.Graph
    :return: None
    """
    graph.bind("gm", GOAL)
    graph.bind("gmb", BASE)
    graph.bind("rdfs", RDFS)
    graph.bind("xsd", XSD)


def declare_schema(graph):
    """
    Declare a small vocabulary for the merged goal-model graph.

    :param graph: rdflib.Graph
    :return: None
    """
    classes = [
        GOAL.Standard,
        GOAL.Function,
        GOAL.Goal,
        GOAL.RootGoal,
        GOAL.ObjectiveGoal,
        GOAL.LeafGoal,
        GOAL.GoalClass,
        GOAL.Operation,
        GOAL.D3FENDOperation,
        GOAL.OntoSecAIOperation,
    ]

    for cls in classes:
        graph.add((cls, RDF.type, RDFS.Class))

    class_relations = [
        (GOAL.RootGoal, RDFS.subClassOf, GOAL.Goal),
        (GOAL.ObjectiveGoal, RDFS.subClassOf, GOAL.Goal),
        (GOAL.LeafGoal, RDFS.subClassOf, GOAL.Goal),
        (GOAL.D3FENDOperation, RDFS.subClassOf, GOAL.Operation),
        (GOAL.OntoSecAIOperation, RDFS.subClassOf, GOAL.Operation),
    ]

    for triple in class_relations:
        graph.add(triple)

    properties = [
        (GOAL.goalId, "Goal identifier"),
        (GOAL.goalName, "Goal name"),
        (GOAL.standardName, "Standard name"),
        (GOAL.functionName, "Function name"),
        (GOAL.controlId, "Control identifier"),
        (GOAL.controlName, "Control name"),
        (GOAL.objectiveName, "Objective name"),
        (GOAL.outcomeName, "Outcome name"),
        (GOAL.classification, "Goal classification"),
        (GOAL.embeddingContext, "Embedding context"),
        (GOAL.parentGoal, "Parent goal"),
        (GOAL.operationalizedBy, "Operationalized by"),
        (GOAL.source, "Operation source"),
        (GOAL.operationId, "Operation identifier"),
        (GOAL.operationName, "Operation name"),
        (GOAL.description, "Operation description"),
        (GOAL.similarityScore, "Embedding similarity score"),
        (GOAL.selectionReason, "Reason for selecting operation"),
    ]

    for prop, label in properties:
        graph.add((prop, RDF.type, RDF.Property))
        graph.add((prop, RDFS.label, Literal(label)))

    for cls_name, label in (
        (GOAL.ORG, "ORG"),
        (GOAL.PHYS, "PHYS"),
        (GOAL.TECH, "TECH"),
    ):
        graph.add((cls_name, RDF.type, GOAL.GoalClass))
        graph.add((cls_name, RDFS.label, Literal(label)))


# ---- JSON hierarchy handling ----
def goal_depth(goal, ancestors):
    """
    Return the depth used to classify root/objective/control/leaf goals.

    :param goal: current goal dict
    :param ancestors: list of ancestor goal dicts
    :return: depth as int
    """
    return len(ancestors)


def classify_goal_level(goal, ancestors, has_children):
    """
    Infer the RDF goal class from the JSON structure.

    :param goal: current goal dict
    :param ancestors: list of ancestor goal dicts
    :param has_children: whether the goal has subgoals
    :return: RDF class URI for the goal
    """
    if not has_children:
        return GOAL.LeafGoal

    depth = goal_depth(goal, ancestors)
    if depth == 0:
        return GOAL.RootGoal

    if depth == 1:
        return GOAL.ObjectiveGoal

    return GOAL.Goal


def iter_goal_nodes(node, ancestors: list[dict[str, Any]] | None = None):
    """
    Return every goal node with its ancestors and leaf status.

    :param node: current node (dict or list)
    :param ancestors: list of ancestor goal dicts
    :return: list of (goal, ancestors, is_leaf) tuples
    """
    if ancestors is None:
        ancestors = []

    results: list[tuple[dict[str, Any], list[dict[str, Any]], bool]] = []

    if isinstance(node, dict):
        has_goal = bool(node.get("goal_id") and node.get("goal_name"))
        subgoals = node.get("subgoals")
        has_children = isinstance(subgoals, list) and len(subgoals) > 0

        next_ancestors = ancestors + [node] if has_goal else ancestors

        if has_goal:
            results.append((node, ancestors, not has_children))

        if has_children:
            for child in subgoals:
                results.extend(iter_goal_nodes(child, next_ancestors))

    elif isinstance(node, list):
        for child in node:
            results.extend(iter_goal_nodes(child, ancestors))

    return results


# ---- RDF conversion ----
def determine_standard(file_path, model):
    """
    Determine the source standard and function name for a JSON goal model.

    Priority order for the standard:
      1. Explicit metadata in the JSON, if available.
      2. Parent directory / file path.
      3. AIRMF function names (GOVERN, MANAGE, MAP, MEASURE).
      4. Function/clause naming conventions when unambiguous.
      5. ``Unknown`` as a safe fallback.

    :param file_path: path to the input JSON file
    :param model: loaded JSON dict
    :return: tuple (standard_name, function_name)
    """

    file_path = Path(file_path)

    # --- 1. Explicit standard metadata in the JSON -------------------------
    standard = (
        model.get("standard")
        or model.get("standard_name")
        or model.get("standardName")
    )
    if standard:
        standard = str(standard).strip()

    # Function name can also be explicit in AI-RMF models.
    function_name = str(
        model.get("airmf_function")
        or model.get("clause_id")
        or model.get("function")
        or file_path.stem
    ).strip()

    # --- 2. Infer from directory / file path --------------------------------
    path_text = "/".join(str(part).lower() for part in file_path.parts)

    path_rules = (
        ("goal_models_iso27001", "ISO 27001"),
        ("goal_models_iso42001", "ISO 42001"),
        ("goal_models_nist", "NIST"),
        ("goal_models_cis", "CIS Controls"),
        ("goal_models_airmf", "AI-RMF"),
        ("iso27001", "ISO 27001"),
        ("iso42001", "ISO 42001"),
        ("nist", "NIST"),
        ("cis", "CIS Controls"),
        ("airmf", "AI-RMF"),
    )

    if not standard:
        for marker, standard_name in path_rules:
            if marker in path_text:
                standard = standard_name
                break

    # --- 3. AI-RMF function names -------------------------------------------
    if not standard and function_name.upper() in {
        "GOVERN", "MANAGE", "MAP", "MEASURE"
    }:
        standard = "AI-RMF"

    # --- 4. Unambiguous function/clause conventions ------------------------
    if not standard:
        function_upper = function_name.upper()
        if function_upper.startswith("CIS."):
            standard = "CIS Controls"
        elif function_upper.startswith("NIST"):
            standard = "NIST"

    # Do not guess between ISO 27001 and ISO 42001 from an A.x identifier.
    if not standard:
        standard = "Unknown"

    return standard, function_name


def add_standard_and_function(graph, standard_name, function_name):
    """
    Create or reuse a Standard and Function resource and link them.

    :param graph: rdflib.Graph
    :param standard_name: standard name
    :param function_name: function name
    :return: tuple (standard_uri, function_uri)
    """
    standard_uri = BASE[f"standard_{slug(standard_name)}"]
    function_uri = BASE[f"function_{slug(standard_name)}_{slug(function_name)}"]

    graph.add((standard_uri, RDF.type, GOAL.Standard))
    add_literal(graph, standard_uri, GOAL.standardName, standard_name)

    graph.add((function_uri, RDF.type, GOAL.Function))
    add_literal(graph, function_uri, GOAL.functionName, function_name)

    graph.add((GOAL.partOf, RDF.type, RDF.Property))
    graph.add((GOAL.partOf, RDFS.label, Literal("Part of")))
    graph.add((function_uri, GOAL.partOf, standard_uri))

    return standard_uri, function_uri


def operation_uri(source, operation_id):
    """
    Return the URI for an operation.

    :param source: operation source
    :param operation_id: operation identifier
    :return: URIRef
    """
    return BASE[safe_operation_local_name(source, operation_id)]


def add_action(graph, goal_uri, action, score_annotations):
    """
    Represent one selected operationalization.

    The similarity score belongs to the specific goal -> operation relation,
    not to the operation resource itself. Therefore it is emitted as an
    RDF-star annotation and collected for serialization after the base graph.

    :param graph: rdflib.Graph
    :param goal_uri: URI of the goal being operationalized
    :param action: action dict from the JSON
    :param score_annotations: list collecting (goal_uri, predicate, op_uri, score)
    :return: None
    """
    source = str(action.get("source", "")).strip()
    action_id = str(action.get("id", "")).strip()
    if not source or not action_id:
        return

    op_uri = operation_uri(source, action_id)
    graph.add((op_uri, RDF.type, GOAL.Operation))
    if source == "D3FEND":
        graph.add((op_uri, RDF.type, GOAL.D3FENDOperation))
    elif source == "OntoSecAI":
        graph.add((op_uri, RDF.type, GOAL.OntoSecAIOperation))

    add_literal(graph, op_uri, GOAL.source, source)
    add_literal(graph, op_uri, GOAL.operationId, action_id)
    add_literal(graph, op_uri, GOAL.operationName, action.get("name"))
    add_literal(graph, op_uri, GOAL.description, action.get("description"))

    # Direct traceability: leaf goal -> selected security operation
    graph.add((goal_uri, GOAL.operationalizedBy, op_uri))

    # The score is specific to this goal-operation association.
    score = action.get("similarity_score")
    if score is not None and score != "":
        try:
            score_annotations.append((
                goal_uri,
                GOAL.operationalizedBy,
                op_uri,
                float(score),
            ))
        except (TypeError, ValueError):
            pass


def convert_goal_model(graph, model, source_path):
    """
    Convert one JSON goal model into the shared RDF graph.

    :param graph: rdflib.Graph
    :param model: JSON model
    :param source_path: path to the JSON file
    :return: dict with counts of goals, leaf goals, operations
    """
    standard_name, function_name = determine_standard(source_path, model)
    _, function_uri = add_standard_and_function(
        graph,
        standard_name,
        function_name,
    )

    counters = {
        "goals": 0,
        "leaf_goals": 0,
        "operations": 0,
    }
    score_annotations = graph._goal_model_score_annotations

    node_records = iter_goal_nodes(model)
    uri_by_goal_key: dict[tuple[str, str], URIRef] = {}

    # First pass: create all goal resources.
    for goal, ancestors, is_leaf in node_records:
        goal_id = str(goal.get("goal_id", "")).strip()
        goal_name = str(goal.get("goal_name", "")).strip()
        if not goal_id or not goal_name:
            continue

        uri = BASE[safe_goal_local_name(standard_name, function_name, goal_id)]
        uri_by_goal_key[(function_name, goal_id)] = uri

        rdf_class = classify_goal_level(goal, ancestors, not is_leaf)
        graph.add((uri, RDF.type, rdf_class))
        add_literal(graph, uri, GOAL.goalId, goal_id)
        add_literal(graph, uri, GOAL.goalName, goal_name)
        add_literal(graph, uri, GOAL.standardName, standard_name)
        add_literal(graph, uri, GOAL.functionName, function_name)

        graph.add((GOAL.partOfFunction, RDF.type, RDF.Property))
        graph.add((GOAL.partOfFunction, RDFS.label, Literal("Part of function")))
        graph.add((uri, GOAL.partOfFunction, function_uri))

        if "control_id" in goal:
            add_literal(graph, uri, GOAL.controlId, goal.get("control_id"))
        if "control_name" in goal:
            add_literal(graph, uri, GOAL.controlName, goal.get("control_name"))
        if "objective_name" in goal:
            add_literal(graph, uri, GOAL.objectiveName, goal.get("objective_name"))
        if "outcome_name" in goal:
            add_literal(graph, uri, GOAL.outcomeName, goal.get("outcome_name"))
        if "class" in goal:
            cls = str(goal.get("class", "")).strip()
            if cls in {"ORG", "PHYS", "TECH"}:
                graph.add((uri, GOAL.classification, GOAL[cls]))

        add_literal(graph, uri, GOAL.embeddingContext, goal.get("embedding_context"))

        counters["goals"] += 1
        counters["leaf_goals"] += int(is_leaf)

    # Second pass: hierarchy and operationalizations.
    for goal, ancestors, is_leaf in node_records:
        goal_id = str(goal.get("goal_id", "")).strip()
        if not goal_id:
            continue

        uri = uri_by_goal_key.get((function_name, goal_id))
        if uri is None:
            continue

        if ancestors:
            parent = ancestors[-1]
            parent_id = str(parent.get("goal_id", "")).strip()
            if parent_id:
                parent_uri = uri_by_goal_key.get((function_name, parent_id))
                if parent_uri is not None:
                    graph.add((uri, GOAL.parentGoal, parent_uri))

        actions = goal.get("actions", [])
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict):
                    before = len(graph)
                    add_action(graph, uri, action, score_annotations)
                    counters["operations"] += int(len(graph) > before)

    return counters


def serialize_turtle_star(graph, output_path, score_annotations):
    """
    Serialize the base RDF graph and append RDF-star relation annotations.

    RDFLib's standard Graph API does not currently provide the quoted-triple
    construction required for RDF-star, so the base graph is serialized with
    RDFLib and the RDF-star statements are emitted as Turtle-star text.
    """
    turtle = graph.serialize(format="turtle")
    if not isinstance(turtle, str):
        turtle = turtle.decode("utf-8")

    lines = [turtle.rstrip(), ""]
    if score_annotations:
        lines.append("# RDF-star relation annotations")
        for subject, predicate, obj, score in score_annotations:
            s = subject.n3(graph.namespace_manager)
            p = predicate.n3(graph.namespace_manager)
            o = obj.n3(graph.namespace_manager)
            lines.append(
                f"<< {s} {p} {o} >> "
                f"{GOAL.similarityScore.n3(graph.namespace_manager)} {score:.6f} ."
            )
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


# ---- Main ----
def parse_args():
    """
    Parse command line arguments.

    :return: argparse.Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Merge *_leaf_goals_classified.operationalized_embeddings.json into RDF/Turtle."
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Project directory containing the goal-model directories.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output Turtle file (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--input-pattern",
        default=DEFAULT_INPUT_PATTERN,
        help=f"Glob pattern for input JSON files (default: {DEFAULT_INPUT_PATTERN}).",
    )
    return parser.parse_args()


def main():
    """
    Main entry point for the script.

    :return: None
    """
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    output_path = base_dir / args.output
    input_pattern = args.input_pattern

    graph = Graph()
    bind_namespaces(graph)
    declare_schema(graph)
    # RDFLib does not expose RDF-star quoted triples in the Graph API used by
    # this generator, so keep relation annotations separately and emit them
    # in Turtle-star after serializing the base RDF graph.
    graph._goal_model_score_annotations = []

    input_files = sorted(
        p
        for p in base_dir.rglob(input_pattern)
        if p.is_file() and p.resolve() != output_path.resolve()
    )

    print(f"Base directory : {base_dir}")
    print(f"Input pattern  : {input_pattern}")
    print(f"Input files    : {len(input_files)}")

    if not input_files:
        raise FileNotFoundError(
            f"No files matching '{input_pattern}' found under {base_dir}"
        )

    total = {"goals": 0, "leaf_goals": 0, "operations": 0}

    for path in input_files:
        print(f"\nProcessing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            model = json.load(handle)

        counters = convert_goal_model(graph, model, path)
        for key in total:
            total[key] += counters[key]

        print(f"Goals      : {counters['goals']}")
        print(f"Leaf goals : {counters['leaf_goals']}")
        print(f"Operations : {counters['operations']}")

    serialize_turtle_star(graph, output_path, graph._goal_model_score_annotations)

    print("\n" + "=" * 70)
    print("Merged RDF graph")
    print("=" * 70)
    print(f"Triples    : {len(graph)}")
    print(f"Goals      : {total['goals']}")
    print(f"Leaf goals : {total['leaf_goals']}")
    print(f"Operations : {total['operations']}")
    print(f"Saved      : {output_path}")


if __name__ == "__main__":
    main()