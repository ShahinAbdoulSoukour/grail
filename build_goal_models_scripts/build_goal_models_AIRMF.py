"""
Build one independent goal model per NIST AI-RMF FUNCTION found in AIRMF.json
(GOVERN, MAP, MEASURE, MANAGE), as a NESTED tree:

G0 (root, from the function title)
  └─ G1 (from the function objective)
       └─ G1.i (one per outcome statement)
            └─ G1.i.j (leaf goals)

Same shape as the ISO 27001 / ISO 42001 / NIST / CIS models, so the downstream
stages (leaves goals classification, and operationalization) work unchanged.
AI-RMF describes risk-MANAGEMENT outcomes, not technical operations.

AI-RMF is not a control catalogue: its statements describe DESIRED STATES
("outcomes"), in the present passive, with no "shall" and no leading verb:
    "Roles and responsibilities for AI risk management are clearly defined,
     documented, and understood."
Pattern P6 in gov_controls_to_goals script turns such a statement into directives,
one per participle, the subject becoming the object:
    -> "Define roles and responsibilities for AI risk management"
       "Document roles and responsibilities for AI risk management"
       "Understand roles and responsibilities for AI risk management"
An adverb between the auxiliary and the participle ("are CLEARLY defined") is
dropped, and "has been / have been / was / were" are accepted as auxiliaries.
"""

import json
import os

from clause_to_root_goal import clause_to_root_goal
from objectives_to_goals import convert_objective_to_goal
from gov_controls_to_goals import transform_shall_statement, assign_goal_ids

OUTDIR = "goal_models_AIRMF"
SOURCE = "data/format_standards/AIRMF.json"


def leaf_node(leaf):
    """
    Turn a raw leaf-goal record into a clean node (drop pattern/verb/connector).

    :param leaf: raw leaf-goal record
    """
    node = {"goal_id": leaf["goal_id"], "goal_name": leaf["statement"]}
    if leaf.get("agent"):
        node["agent"] = leaf["agent"]
    return node


def objective_to_goal_statement(function_id, objective, title):
    """
    Turn an AI-RMF function objective into the G1 goal statement.

    The functions carry an ISO-style objective ("To establish and maintain
    effective AI risk management."), so convert_objective_to_goal applies
    directly. If it is missing, G1 is derived from the TITLE so the
    three-level tree still exists and leaf_goals_of() keeps working.

    :param function_id: e.g. "GOVERN"
    :param objective: the function's raw objective text
    :param title: the function title, used as a fallback
    """
    text = (objective or "").strip()
    if not text:
        return clause_to_root_goal(f"{function_id} {title}", verbalize=True)["statement"]
    return convert_objective_to_goal(text)


def build_goal_model(function_id, function):
    """
    Build one AI-RMF function's nested goal model from its JSON entry.

    :param function_id: e.g. "GOVERN" (AI-RMF function identifier)
    :param function: dict with keys "title", "objective", "controls" (outcomes)
    """
    g0 = clause_to_root_goal(f"{function_id} {function['title']}", verbalize=True)
    g1_name = objective_to_goal_statement(function_id, function.get("objective", ""),
                                          function["title"])

    outcome_nodes = []
    for i, (outcome_id, text) in enumerate(function["controls"].items(), start=1):
        # P6 (declarative) fires here: AI-RMF outcomes are desired states.
        leaves = assign_goal_ids(transform_shall_statement(outcome_id, text),
                                 parent_id=f"G1.{i}")
        outcome_nodes.append({
            "goal_id": f"G1.{i}",
            "control_id": outcome_id,       # e.g. "GOVERN 2"
            "control_name": text,           # the source outcome (verbatim)
            "subgoals": [leaf_node(lf) for lf in leaves],
        })

    return {
        "goal_id": "G0",
        "clause_id": function_id,           # same key as ISO (downstream)
        "clause_name": function["title"],
        "goal_name": g0["statement"],
        "airmf_function": function_id,      # GOVERN | MAP | MEASURE | MANAGE
        "subgoals": [
            {
                "goal_id": "G1",
                "objective_name": function.get("objective", ""),
                "goal_name": g1_name,
                "subgoals": outcome_nodes,
            }
        ],
    }


def print_model(model):
    """
    Pretty-print the nested tree (indented).

    :param model: the nested dictionary returned by build_goal_model
    """
    print(f"\n{'='*72}\n{model['goal_id']}  {model['goal_name']}  ({model['clause_id']})")
    g1 = model["subgoals"][0]
    print(f"  {g1['goal_id']}  {g1['goal_name']}")
    for ctrl in g1["subgoals"]:
        print(f"    {ctrl['goal_id']}  [{ctrl['control_id']}] {ctrl['control_name'][:58]}...")
        for leaf in ctrl["subgoals"]:
            agent = f"   [agent: {leaf['agent']}]" if leaf.get("agent") else ""
            print(f"      {leaf['goal_id']:<10} {leaf['goal_name']}{agent}")


def save_model(model, outdir=OUTDIR):
    """
    Write one AI-RMF function's goal model to its own JSON file.

    :param model: one function's goal model
    :param outdir: output directory
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{model['clause_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
    return path


if __name__ == "__main__":
    with open(SOURCE, encoding="utf-8") as f:
        airmf = json.load(f)

    n_leaves = 0
    n_unmatched = 0
    for function_id, function in airmf.items():
        model = build_goal_model(function_id, function)
        print_model(model)
        save_model(model)
        n_leaves += sum(len(c["subgoals"])
                        for c in model["subgoals"][0]["subgoals"])
        # count the statements no pattern could rewrite (kept verbatim)
        for outcome_id, text in function["controls"].items():
            n_unmatched += sum(1 for lf in transform_shall_statement(outcome_id, text)
                               if lf["pattern"] == "UNMATCHED")

    print(f"\n{len(airmf)} AI-RMF functions --> one JSON each in {OUTDIR}/")
    print(f"{n_leaves} leaf goals generated")
    if n_unmatched:
        print(f"note: {n_unmatched} statements use grammar outside P1-P6 "
              f"(active present indicative, copulas) and were kept verbatim")
