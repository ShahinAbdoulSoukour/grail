"""
Build one independent goal model per NIST SP 800-53 control FAMILY found in
NIST.json, as a NESTED tree:

G0 (root, from the family title)
  └─ G1 (from the family objective)
       └─ G1.i (one per control)
            └─ G1.i.j (leaf goals)

Same shape as the ISO 27001 / ISO 42001 models, so the downstream stages
(leaves goals classification, and operationalization) work unchanged.
"""

import json
import os

from clause_to_root_goal import clause_to_root_goal
from objectives_to_goals import convert_objective_to_goal
from gov_controls_to_goals import transform_shall_statement, assign_goal_ids

OUTDIR = "goal_models_NIST"
SOURCE = "data/format_standards/NIST.json"


def leaf_node(leaf):
    """
    Turn a raw leaf-goal record into a clean node (drop pattern/verb/connector).

    :param leaf: raw leaf-goal record
    """
    node = {"goal_id": leaf["goal_id"], "goal_name": leaf["statement"]}
    if leaf.get("agent"):
        node["agent"] = leaf["agent"]
    return node


def objective_to_goal_statement(family_id, objective, title):
    """
    Turn a NIST family objective into the G1 goal statement.

    NIST families now carry an ISO-style objective ("To ensure authorized and
    controlled access ..."), so convert_objective_to_goal applies directly.
    If the objective is missing, G1 is derived from the family TITLE instead,
    so the three-level tree still exists and leaf_goals_of() keeps working.

    :param family_id: e.g. "AC"
    :param objective: the family's raw objective text
    :param title: the family title, used as a fallback
    """
    text = (objective or "").strip()
    if not text:
        return clause_to_root_goal(f"{family_id} {title}", verbalize=True)["statement"]
    return convert_objective_to_goal(text)


def build_goal_model(family_id, family):
    """
    Build one family's nested goal model from its JSON entry.

    :param family_id: e.g. "AC" (control family identifier)
    :param family: dict with keys "title", "objective", "controls"
    """
    g0 = clause_to_root_goal(f"{family_id} {family['title']}", verbalize=True)
    g1_name = objective_to_goal_statement(family_id, family.get("objective", ""),
                                          family["title"])

    control_nodes = []
    for i, (ctrl_id, text) in enumerate(family["controls"].items(), start=1):
        # P5 (imperative) fires here; the same entry point is used for every
        # standard, the pattern is chosen from the sentence's grammar.
        leaves = assign_goal_ids(transform_shall_statement(ctrl_id, text),
                                 parent_id=f"G1.{i}")
        control_nodes.append({
            "goal_id": f"G1.{i}",
            "control_id": ctrl_id,          # e.g. "AC-2"
            "control_name": text,           # the source control (verbatim)
            "subgoals": [leaf_node(lf) for lf in leaves],
        })

    return {
        "goal_id": "G0",
        "clause_id": family_id,             # same key as ISO (downstream)
        "clause_name": family["title"],
        "goal_name": g0["statement"],
        "subgoals": [
            {
                "goal_id": "G1",
                "objective_name": family.get("objective", ""),
                "goal_name": g1_name,
                "subgoals": control_nodes,
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
        print(f"    {ctrl['goal_id']}  [{ctrl['control_id']}] {ctrl['control_name'][:60]}...")
        for leaf in ctrl["subgoals"]:
            agent = f"   [agent: {leaf['agent']}]" if leaf.get("agent") else ""
            print(f"      {leaf['goal_id']:<10} {leaf['goal_name']}{agent}")


def save_model(model, outdir=OUTDIR):
    """
    Write one family's goal model to its own JSON file.

    :param model: one family's goal model
    :param outdir: output directory
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{model['clause_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
    return path


if __name__ == "__main__":
    with open(SOURCE, encoding="utf-8") as f:
        nist = json.load(f)

    n_leaves = 0
    for family_id, family in nist.items():
        model = build_goal_model(family_id, family)
        print_model(model)
        save_model(model)
        n_leaves += sum(len(c["subgoals"])
                        for c in model["subgoals"][0]["subgoals"])

    print(f"\n{len(nist)} families --> one JSON each in {OUTDIR}/")
    print(f"{n_leaves} leaf goals generated")
