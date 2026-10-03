"""
Build one independent goal model per ISO/IEC 27001 Annex A clause found in iso27001.json, as a NESTED tree:
G0 (root, from title)
  └─ G1 (from objective)
       └─ G1.i (one per control)
            └─ G1.i.j (leaf goals)
Each node has: goal_id, goal_name, and (except leaves) a "subgoals" list.
Leaf goals may carry "agent"; control-level nodes carry "control_id".
"""

import json
import os
from clause_to_root_goal import clause_to_root_goal
from objectives_to_goals import convert_objective_to_goal
from gov_controls_to_goals import transform_shall_statement, assign_goal_ids

OUTDIR = "goal_models_ISO27001"

def leaf_node(leaf):
    """
    Turn a raw leaf-goal record into a clean node (drop pattern/verb/connector).

    :param leaf: raw leaf-goal record
    """
    # Create the core node.
    # Extracts the goal_id (e.g., "G1.2.3") from the raw record.
    # Extracts the "statement" (the final, fully‑formed goal text) and renames it to "goal_name".
    node = {"goal_id": leaf["goal_id"], "goal_name": leaf["statement"]}
    if leaf.get("agent"):
        node["agent"] = leaf["agent"]
    return node


def build_goal_model(clause_id, clause):
    """
    Build one clause's nested goal model from its JSON entry.

    :param clause_id: e.g. "A.5.1" (clause identifier)
    :param clause: dict with keys "title" (the clause heading), "objective" (the clause's objective), "controls" (a dict mapping control IDs (e.g., "A.5.1.1")).
    """
    g0 = clause_to_root_goal(f"{clause_id} {clause['title']}", verbalize=True)
    g1_name = convert_objective_to_goal(clause["objective"])

    # Process each control into subgoals
    control_nodes = []
    for i, (ctrl_id, text) in enumerate(clause["controls"].items(), start=1):
        # Parse the control text with transform_shall_statement(ctrl_id, text)
        # Assign permanent goal IDs
        leaves = assign_goal_ids(transform_shall_statement(ctrl_id, text), parent_id=f"G1.{i}")
        # Build the control node
        control_nodes.append({
            "goal_id": f"G1.{i}", # e.g., "G1.1"
            "control_id": ctrl_id, # the original ISO control ID (e.g., "A.5.1.1")
            "control_name": text, # the source control (verbatim)
            "subgoals": [leaf_node(lf) for lf in leaves], # the list of cleaned leaf nodes
        })
    # The returned dictionary is a three‑level hierarchy
    return {
        "goal_id": "G0",
        "clause_id": clause_id,
        "clause_name": clause["title"], # raw ISO clause title
        "goal_name": g0["statement"],
        "subgoals": [
            {
                "goal_id": "G1",
                "objective_name": clause["objective"], # raw ISO objective
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
    # Print the root goal (G0)
    print(f"\n{'='*72}\n{model['goal_id']}  {model['goal_name']}  ({model['clause_id']})")
    # Print the first subgoal (G1)
    g1 = model["subgoals"][0]
    print(f"  {g1['goal_id']}  {g1['goal_name']}")
    # Print each control node
    for ctrl in g1["subgoals"]: # Loops over the list of controls under g1["subgoals"]
        print(f"    {ctrl['goal_id']}  [{ctrl['control_id']}] {ctrl['control_name'][:60]}...")
        for leaf in ctrl["subgoals"]:
            agent = f"   [agent: {leaf['agent']}]" if leaf.get("agent") else ""
            print(f"      {leaf['goal_id']:<10} {leaf['goal_name']}{agent}")



def save_model(model, outdir=OUTDIR):
    """
    Write one clause's goal model to its own JSON file.

    :param model: one clause's goal model
    :param outdir: output directory
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{model['clause_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
    return path



if __name__ == "__main__":
    with open("iso27001.json", encoding="utf-8") as f:
        iso = json.load(f)

    for clause_id, clause in iso.items():
        model = build_goal_model(clause_id, clause)
        print_model(model)
        save_model(model)

    print(f"\n{len(iso)} clauses --> one JSON each in {OUTDIR}/")