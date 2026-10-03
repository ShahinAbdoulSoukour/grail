"""
Build one independent goal model per CIS Control found in CIS.json, as a
NESTED tree:

G0 (root, from the control title)
  └─ G1 (from the control's descriptive objective)
       └─ G1.i (one per safeguard)
            └─ G1.i.j (leaf goals)

Same shape as the ISO 27001 / ISO 42001 / NIST models, so the downstream
stages (leaves goals classification, and operationalization) work unchanged.

1. The SAFEGUARDS (controls) are short IMPERATIVE titles, not "shall"
   sentences: "Establish and Maintain Detailed Enterprise Asset Inventory".
   They are already goal statements, handled by pattern P5 in
   gov_controls_to_goals, which only splits the leading verb chain:
       "Establish and Maintain Detailed Enterprise Asset Inventory"
         -> "Establish Detailed Enterprise Asset Inventory"
            "Maintain Detailed Enterprise Asset Inventory"
   A single-verb safeguard stays ONE leaf goal ("Address Unauthorized Assets").

2. The OBJECTIVE is a long descriptive PARAGRAPH, not an ISO "To <verb> ..."
   sentence, and it is already imperative:
       "Actively manage (inventory, track, and correct) all enterprise assets
        (end-user devices, ...) connected to the infrastructure ... . This
        will also support identifying unauthorized and unmanaged assets ..."
   Only the FIRST sentence is a directive; the rest is rationale. The
   parenthetical asides are enumerations that make an unreadable goal name.
"""

import json
import os
import re

from clause_to_root_goal import clause_to_root_goal
from objectives_to_goals import convert_objective_to_goal
from gov_controls_to_goals import transform_shall_statement, assign_goal_ids

OUTDIR = "goal_models_CIS"
SOURCE = "data/format_standards/CIS.json"


def leaf_node(leaf):
    """
    Turn a raw leaf-goal record into a clean node (drop pattern/verb/connector).

    :param leaf: raw leaf-goal record
    """
    node = {"goal_id": leaf["goal_id"], "goal_name": leaf["statement"]}
    if leaf.get("agent"):
        node["agent"] = leaf["agent"]
    return node


def first_sentence(text):
    """
    Keep only the first sentence of a descriptive paragraph.

    The split point is a period followed by whitespace and a capital letter,
    so decimals and abbreviations inside the sentence are not split on.
    """
    parts = re.split(r"(?<=[a-z\)])\.\s+(?=[A-Z])", text.strip(), maxsplit=1)
    return parts[0].rstrip(".")


def drop_parentheticals(text):
    """
    Remove parenthetical enumerations that bloat a goal statement.
        "manage (inventory, track, and correct) all assets (end-user devices,
         ...) connected"  ->  "manage all assets connected"
    """
    # Applied REPEATEDLY, innermost group first: CIS objectives contain
    # NESTED parentheses ("... non-computing/Internet of Things (IoT)
    # devices; and servers)"), which a single pass cannot match.
    out, prev = text, None
    while out != prev:
        prev = out
        out = re.sub(r"\s*\([^()]*\)", "", out)
    out = re.sub(r"\s{2,}", " ", out)
    # tidy the punctuation left behind by a removed aside
    out = re.sub(r"\s+([,;.])", r"\1", out)
    return out.strip()


def objective_to_goal_statement(control_id, objective, title):
    """
    Turn a CIS control objective into the G1 goal statement.

    Dispatch on the FORM of the objective:
      * empty            -> fall back to the title
      * "To <verb> ..."  -> the ISO form: convert_objective_to_goal
      * otherwise        -> a descriptive CIS paragraph: keep the first
                            sentence, drop the parenthetical enumerations

    :param control_id: e.g. "CIS.1"
    :param objective: the control's raw objective paragraph
    :param title: the control title, used as a fallback
    """
    text = (objective or "").strip()
    if not text:
        return clause_to_root_goal(f"{control_id} {title}", verbalize=True)["statement"]

    if re.match(r"^[Tt]o\s+", text):
        return convert_objective_to_goal(text)

    directive = drop_parentheticals(first_sentence(text))
    return directive[0].upper() + directive[1:] if directive else text


def build_goal_model(control_id, control):
    """
    Build one CIS control's nested goal model from its JSON entry.

    :param control_id: e.g. "CIS.1" (control identifier)
    :param control: dict with keys "title", "objective", "controls" (safeguards)
    """
    g0 = clause_to_root_goal(f"{control_id} {control['title']}", verbalize=True)
    g1_name = objective_to_goal_statement(control_id, control.get("objective", ""),
                                          control["title"])

    safeguard_nodes = []
    for i, (sg_id, text) in enumerate(control["controls"].items(), start=1):
        # P5 (imperative) fires here: CIS safeguards are directive titles.
        leaves = assign_goal_ids(transform_shall_statement(sg_id, text),
                                 parent_id=f"G1.{i}")
        safeguard_nodes.append({
            "goal_id": f"G1.{i}",
            "control_id": sg_id,            # e.g. "CIS.1.1"
            "control_name": text,           # the source safeguard (verbatim)
            "subgoals": [leaf_node(lf) for lf in leaves],
        })

    return {
        "goal_id": "G0",
        "clause_id": control_id,            # same key as ISO (downstream)
        "clause_name": control["title"],
        "goal_name": g0["statement"],
        "subgoals": [
            {
                "goal_id": "G1",
                "objective_name": control.get("objective", ""),  # raw paragraph
                "goal_name": g1_name,
                "subgoals": safeguard_nodes,
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
    print(f"  {g1['goal_id']}  {g1['goal_name'][:100]}")
    for ctrl in g1["subgoals"]:
        print(f"    {ctrl['goal_id']}  [{ctrl['control_id']}] {ctrl['control_name'][:60]}...")
        for leaf in ctrl["subgoals"]:
            agent = f"   [agent: {leaf['agent']}]" if leaf.get("agent") else ""
            print(f"      {leaf['goal_id']:<10} {leaf['goal_name']}{agent}")


def save_model(model, outdir=OUTDIR):
    """
    Write one CIS control's goal model to its own JSON file.

    :param model: one control's goal model
    :param outdir: output directory
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{model['clause_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
    return path


if __name__ == "__main__":
    with open(SOURCE, encoding="utf-8") as f:
        cis = json.load(f)

    n_leaves = 0
    for control_id, control in cis.items():
        model = build_goal_model(control_id, control)
        print_model(model)
        save_model(model)
        n_leaves += sum(len(c["subgoals"])
                        for c in model["subgoals"][0]["subgoals"])

    print(f"\n{len(cis)} CIS controls --> one JSON each in {OUTDIR}/")
    print(f"{n_leaves} leaf goals generated")
