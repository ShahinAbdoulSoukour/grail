"""
Build one independent goal model per ISO/IEC 42001 Annex A group found in
ISO42001.json, as a NESTED tree:

G0 (root, from the group title)
  └─ G1 (from the group objective)
       └─ G1.i (one per control)
            └─ G1.i.j (leaf goals)

Each node has: goal_id, goal_name, and (except leaves) a "subgoals" list.
Leaf goals may carry "agent"; control-level nodes carry "control_id".
The tree has the SAME shape as the ISO 27001 models, so the downstream
stages (leaves goals classification, and operationalization) work unchanged.
"""

import json
import os

from clause_to_root_goal import clause_to_root_goal
from objectives_to_goals import convert_objective_to_goal
from gov_controls_to_goals import transform_shall_statement, assign_goal_ids

OUTDIR = "goal_models_ISO42001"
SOURCE = "data/format_standards/ISO42001.json"


def leaf_node(leaf):
    """
    Turn a raw leaf-goal record into a clean node (drop pattern/verb/connector).

    :param leaf: raw leaf-goal record
    """
    node = {"goal_id": leaf["goal_id"], "goal_name": leaf["statement"]}
    if leaf.get("agent"):
        node["agent"] = leaf["agent"]
    return node


def objective_to_goal_statement(group_id, objective, title):
    """
    Turn an ISO 42001 group objective into the G1 goal statement.

    Three cases, dispatched on the FORM of the objective:
      * empty            -> fall back to the title ("Ensure <title>")
      * contains "shall" -> it is a control sentence: reuse
                            transform_shall_statement (ISO 42001 case)
      * "To <verb> ..."  -> the ISO 27001 form: convert_objective_to_goal

    :param group_id: e.g. "A.2"
    :param objective: the group's raw objective text
    :param title: the group title, used as a fallback
    """
    text = (objective or "").strip()

    # No objective -> derive one from the title so G1 still exists.
    if not text:
        return clause_to_root_goal(f"{group_id} {title}", verbalize=True)["statement"]

    # ISO 42001: the objective is a "shall" sentence -> use the control parser.
    if " shall " in text.lower():
        leaves = transform_shall_statement(group_id, text)
        if leaves:
            # An objective may carry several verbs ("... shall be defined and
            # assigned"); join them so G1 stays ONE node above the controls.
            return " and ".join(lf["statement"] for lf in leaves)

    # ISO 27001 style ("To <verb> ...") or anything else.
    return convert_objective_to_goal(text)


def build_goal_model(group_id, group):
    """
    Build one group's nested goal model from its JSON entry.

    :param group_id: e.g. "A.2" (group identifier)
    :param group: dict with keys "title", "objective", "controls"
    """
    g0 = clause_to_root_goal(f"{group_id} {group['title']}", verbalize=True)
    g1_name = objective_to_goal_statement(group_id, group.get("objective", ""),
                                          group["title"])

    # Process each control into subgoals
    control_nodes = []
    for i, (ctrl_id, text) in enumerate(group["controls"].items(), start=1):
        leaves = assign_goal_ids(transform_shall_statement(ctrl_id, text),
                                 parent_id=f"G1.{i}")
        control_nodes.append({
            "goal_id": f"G1.{i}",
            "control_id": ctrl_id,          # e.g. "A.2.2"
            "control_name": text,           # the source control (verbatim)
            "subgoals": [leaf_node(lf) for lf in leaves],
        })

    return {
        "goal_id": "G0",
        "clause_id": group_id,              # same key as ISO 27001 (downstream)
        "clause_name": group["title"],
        "goal_name": g0["statement"],
        "subgoals": [
            {
                "goal_id": "G1",
                "objective_name": group.get("objective", ""),  # raw, for traceability
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
    Write one group's goal model to its own JSON file.

    :param model: one group's goal model
    :param outdir: output directory
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{model['clause_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
    return path


if __name__ == "__main__":
    with open(SOURCE, encoding="utf-8") as f:
        iso = json.load(f)

    n_leaves = 0
    n_dup_objective = 0
    for group_id, group in iso.items():
        model = build_goal_model(group_id, group)
        print_model(model)
        save_model(model)

        g1 = model["subgoals"][0]
        n_leaves += sum(len(c["subgoals"]) for c in g1["subgoals"])
        # Report the known redundancy of the source data (see module docstring).
        controls = list(group["controls"].values())
        if controls and group.get("objective", "").strip().rstrip(".") == \
                controls[0].strip().rstrip("."):
            n_dup_objective += 1

    print(f"\n{len(iso)} groups --> one JSON each in {OUTDIR}/")
    print(f"{n_leaves} leaf goals generated")
    if n_dup_objective:
        print(f"note: in {n_dup_objective}/{len(iso)} groups the objective is "
              f"identical to the first control, so G1 repeats the first leaf "
              f"goal (property of the source data, not of the transformation)")
