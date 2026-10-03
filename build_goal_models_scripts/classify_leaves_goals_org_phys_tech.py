"""
Classify each leaf goal into one of THREE classes, with a PRE-TRAINED
sentence-transformer.

ORG : organizational / governance control --> realized by PEOPLE
(define/approve/publish a policy, assign responsibilities, awareness
training, agreements, risk-management outcomes). No D3FEND.

PHYS : physical / environmental control --> protects buildings, offices,
rooms, secure areas, perimeters, entry/loading points, or guards
against fire and natural disasters. Outside D3FEND's digital scope.

TECH : technical control --> enforced by software/systems and therefore
operationalizable into D3FEND actions (accounts, credentials, media,
network, source code, permissions, encryption...).

Only TECH goals are sent to the D3FEND operationalizer. ORG and PHYS are
recorded as "no D3FEND coverage" --> that absence is a coverage finding, not
a failure.

Each class is described by SEVERAL short prototype sentences. They are encoded
once, averaged into one vector per class, and re-normalized; a goal is then
labeled by the nearest class vector (cosine). Averaging several prototypes
covers a class's vocabulary far better than a single sentence: a class is a
region of the embedding space, not a point.
"""

import numpy as np


MODEL_NAME = "all-mpnet-base-v2"


# --- CLASS PROTOTYPES ---
# Several short, concrete sentences per class, in the style of the goals they
# must match (imperative, verb + object). Edit these, not a word list.
CLASS_PROTOTYPES = {
    "ORG": [
        "Define, approve, publish and communicate an information security "
        "policy to employees and external parties.",
        "Assign roles, responsibilities and accountability for information "
        "security across the organization.",
        "Provide security awareness education and training to personnel, and "
        "operate a disciplinary process.",
        "Agree contractual terms, non-disclosure agreements and supplier "
        "obligations, and review supplier service delivery.",
        "Determine organizational risk tolerances, accept residual risks and "
        "document risk treatment decisions.",
        "Understand the intended purposes, business value, benefits and costs "
        "of a system, and document the findings.",
        "Assess and improve organizational processes, procedures and human "
        "oversight, and evaluate environmental impact and sustainability.",
    ],
    "PHYS": [
        "Define and use physical security perimeters to protect areas that "
        "contain sensitive information processing facilities.",
        "Protect secure areas with entry controls so that only authorized "
        "personnel are allowed physical access.",
        "Design and apply physical security for offices, rooms, buildings and "
        "the premises.",
        "Protect equipment against fire, flooding, natural disasters and "
        "environmental threats, including power supply and cabling.",
        "Control delivery and loading areas and other access points where "
        "unauthorized persons could enter the site.",
        "Secure media, documents and equipment in locked cabinets, safes or "
        "vaults when unattended.",
    ],
    "TECH": [
        "Remove, restrict or revoke the access rights and privileges of user "
        "accounts on systems and services.",
        "Manage credentials, passwords, cryptographic keys and certificates, "
        "and enforce multi-factor authentication.",
        "Restrict access to program source code, files and application "
        "functions using permissions.",
        "Encrypt data at rest and in transit, and protect storage media "
        "against unauthorized access.",
        "Filter, monitor and segment network traffic at boundaries using "
        "firewalls and intrusion detection.",
        "Collect, store and analyse audit logs and events to detect attacks "
        "on the system.",
        "Harden system configuration, apply patches, remediate "
        "vulnerabilities and deploy malware protection on endpoints.",
        "Maintain an inventory of software and hardware assets and scan them "
        "automatically with tooling.",
    ],
}


# --- MODEL + PROTOTYPE VECTORS (loaded once, cached) ---
_CACHE = {}  # {model_name: (model, labels, class_vectors)}


def build_prototype_vectors(model):
    """
    Encode every prototype, average them PER CLASS, and re-normalize so that a
    dot product with a normalized goal vector is exactly the cosine.

    Returns (labels, matrix) with one row per class, in the order of `labels`.
    """
    labels = list(CLASS_PROTOTYPES)
    rows = []
    for label in labels:
        vecs = model.encode(CLASS_PROTOTYPES[label], normalize_embeddings=True)
        mean = np.asarray(vecs).mean(axis=0)          # centroid of the class
        norm = np.linalg.norm(mean)
        rows.append(mean / norm if norm else mean)    # re-normalize
    return labels, np.vstack(rows)


def _get_model(model_name=MODEL_NAME):
    """
    Load the sentence-transformer and its class vectors once, then reuse them.
    The import is lazy so that merely importing this module does not pull in
    torch (useful for tools that only need leaf_goals_of()).
    """
    if model_name not in _CACHE:
        from sentence_transformers import SentenceTransformer  # lazy import
        model = SentenceTransformer(model_name)
        labels, vectors = build_prototype_vectors(model)
        _CACHE[model_name] = (model, labels, vectors)
    return _CACHE[model_name]


# --- CLASSIFICATION ---
def classify_goals(goal_texts, model_name=MODEL_NAME):
    """
    Classify a LIST of goals in one batch -> list of "ORG" | "PHYS" | "TECH".

    Batching matters: encoding 620 goals one by one is dominated by per-call
    overhead, while a single encode() call runs them through the model
    together. Prefer this over calling classify_goal() in a loop.
    """
    if not goal_texts:
        return []
    model, labels, vectors = _get_model(model_name)
    goal_vectors = model.encode(list(goal_texts), normalize_embeddings=True)
    sims = np.asarray(goal_vectors) @ vectors.T   # (n_goals, n_classes) cosines
    return [labels[i] for i in sims.argmax(axis=1)]


def classify_goal(goal_text, model_name=MODEL_NAME, **_ignored):
    """
    Return "ORG" | "PHYS" | "TECH" for ONE goal.
    """
    return classify_goals([goal_text], model_name)[0]


def classify_scores(goal_text, model_name=MODEL_NAME):
    """
    The cosine of a goal against each class vector, e.g.
        {"ORG": 0.41, "PHYS": 0.12, "TECH": 0.55}

    Use it to audit a label, to spot near-ties (a small gap between the top two
    means the goal sits between two classes), or to calibrate the prototypes:
    a systematically wrong goal is usually one whose vocabulary is missing from
    the prototype sentences of its true class.
    """
    model, labels, vectors = _get_model(model_name)
    v = model.encode([goal_text], normalize_embeddings=True)[0]
    sims = vectors @ np.asarray(v)
    return {label: round(float(score), 3) for label, score in zip(labels, sims)}


# --- Goal-model traversal + in-place tagging --------------------------------
def leaf_goals_of(model):
    """
    Collect every leaf-level goal node (G0 -> G1 -> Controls -> Leaves).
    """
    leaves = []
    for g1 in model.get("subgoals", []):
        for control in g1.get("subgoals", []):
            leaves.extend(control.get("subgoals", []))
    return leaves


def classify_leaf_goal(model, model_name=MODEL_NAME, **_ignored):
    """
    Tag every leaf goal in place with "class": "ORG" | "PHYS" | "TECH".
    All leaves of the model are encoded in a single batch.
    """
    leaves = leaf_goals_of(model)
    for leaf, label in zip(leaves,
                           classify_goals([lf["goal_name"] for lf in leaves],
                                          model_name)):
        leaf["class"] = label
    return model
