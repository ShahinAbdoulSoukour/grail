"""
Run the RQ3 without-goal-model (-GM) baselines for GRAIL.

The workflow is:
    source description
        --> direct mitigation mapping
        --> same SWE-Agent trajectory
        --> deployment adapter
        --> protected .traj
        --> before/after formal requirement evidence

The script produces the runtime evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from rdflib import Graph, Namespace, RDFS
except ImportError:  # pragma: no cover - only needed for raw KB mode
    Graph = None
    Namespace = None
    RDFS = None

GM = "https://example.org/goal-model/ontology/"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL = "qwen3:8b"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "by",
    "with", "from", "that", "this", "all", "be", "is", "are", "as", "into",
    "at", "through", "within", "system", "use", "using", "ensure", "establish",
    "must", "shall", "should", "can", "may", "their", "its", "via", "such",
}

BLOCKED_HTTP_OBSERVATION = (
    "HTTP/1.1 403 Forbidden\n"
    "X-Security-Agent: blocked\n"
    "Content-Type: application/json\n"
    '{"error":"blocked by GRAIL security agent"}'
)

BLOCKED_COMMAND_OBSERVATION = (
    "[GRAIL] command refused\n"
    "reason: command does not satisfy the active authorization policy"
)

BLOCKED_INSTALL_OBSERVATION = (
    "[GRAIL] installation refused\n"
    "reason: dependency does not satisfy the active authorization policy"
)


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def text(value) :
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def parse_json_object(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def tokens(value):
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_-]+", (value or "").lower())
        if token not in STOPWORDS and len(token) > 2
    }


def keyword_score(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------- Direct source descriptions ----------------

def load_source_controls(scenario_path, controls_json: Path | None):
    """
    Load direct source descriptions without any goal model.
    """
    if controls_json is not None:
        data = load_json(controls_json)
        if isinstance(data, dict) and isinstance(data.get("controls"), list):
            data = data["controls"]
        if not isinstance(data, list):
            raise ValueError("--controls-json must contain a JSON list or {'controls': [...]}")

        controls: list[dict[str, Any]] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            if not row.get("id") or not row.get("description"):
                continue
            controls.append({
                "id": str(row["id"]),
                "description": str(row["description"]),
                "event_types": list(row.get("event_types", [])),
                "source": row.get("source", "external_control_export"),
            })
        if not controls:
            raise ValueError("No usable controls were found in --controls-json")
        return controls

    scenario = load_json(scenario_path)
    requirements = scenario.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError(f"No requirements found in scenario: {scenario_path}")

    # Temporary/fallback source descriptions.
    return [
        {
            "id": str(req["id"]),
            "description": str(req.get("description", "")),
            "event_types": list(req.get("event_types", [])),
            "source": "scenario_requirement_description",
        }
        for req in requirements
        if req.get("id") and req.get("description")
    ]


# ---------------- Raw D3FEND / OntoSecAI mitigation extraction ----------------

D3F = Namespace("http://d3fend.mitre.org/ontologies/d3fend.owl#") if Namespace else None
HES = Namespace("http://www.semanticweb.org/ubaid/ontologies/2023/5/HES#") if Namespace else None

D3FEND_QUERY = """
PREFIX d3f:  <http://d3fend.mitre.org/ontologies/d3fend.owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?technique ?id ?label ?definition
WHERE {
    ?technique rdfs:subClassOf* d3f:DefensiveTechnique ;
               d3f:d3fend-id ?id ;
               rdfs:label ?label .
    OPTIONAL { ?technique d3f:definition ?definition . }
}
ORDER BY STR(?id)
"""

ONTOSECAI_QUERY = """
PREFIX hes: <http://www.semanticweb.org/ubaid/ontologies/2023/5/HES#>

SELECT DISTINCT ?mitigation ?name ?description
WHERE {
    ?mitigation a hes:Mitigations ;
                hes:Name ?name .
    OPTIONAL { ?mitigation hes:Description ?description . }
}
ORDER BY STR(?mitigation)
"""


def _require_rdflib():
    if Graph is None:
        raise RuntimeError(
            "rdflib is required for raw D3FEND/OntoSecAI KB mode. "
            "Install it with: pip install rdflib"
        )


def _local_name(uri):
    value = str(uri)
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    return value.rstrip("/").rsplit("/", 1)[-1]


def _load_rdf_graph(path, label: str):
    _require_rdflib()
    if not path.exists():
        raise FileNotFoundError(path)
    graph = Graph()
    suffix = path.suffix.lower()
    fmt = "turtle" if suffix in {".ttl", ".turtle"} else "xml"
    graph.parse(path, format=fmt)
    return graph


def extract_operations_from_raw_kbs(d3fend_path: Path | None, ontosecai_path: Path | None):
    """
    Extract the independent mitigation universe directly from raw KBs.

    D3FEND candidates are descendants of d3f:DefensiveTechnique and use
    d3f:d3fend-id + rdfs:label + optional d3f:definition.

    OntoSecAI candidates are individuals of hes:Mitigations and use the
    hes:Name + optional hes:Description vocabulary used by the project's
    existing operationalization scripts.
    """
    merged: dict[str, dict[str, Any]] = {}
    counts = {"D3FEND": 0, "OntoSecAI": 0}

    if d3fend_path is not None:
        graph = _load_rdf_graph(d3fend_path, "D3FEND KB")
        for row in graph.query(D3FEND_QUERY):
            oid = str(row.id)
            name = str(row.label)
            description = str(row.definition) if row.definition is not None else ""
            if not oid or not name:
                continue
            op = {
                "operation_id": oid,
                "operation_name": name,
                "source": "D3FEND",
                "description": description,
                "uri": str(row.technique),
            }
            if oid not in merged:
                merged[oid] = op
                counts["D3FEND"] += 1

    if ontosecai_path is not None:
        graph = _load_rdf_graph(ontosecai_path, "OntoSecAI KB")
        for row in graph.query(ONTOSECAI_QUERY):
            oid = _local_name(row.mitigation)
            name = str(row.name)
            description = str(row.description) if row.description is not None else ""
            if not oid or not name:
                continue
            op = {
                "operation_id": oid,
                "operation_name": name,
                "source": "OntoSecAI",
                "description": description,
                "uri": str(row.mitigation),
            }
            # IDs are expected to be distinct across these two catalogs
            # (e.g. D3-* vs AML.M*/OWASP.M*). Preserve the first definition if
            # a source publishes a duplicate ID.
            if oid not in merged:
                merged[oid] = op
                counts["OntoSecAI"] += 1

    if not merged:
        raise ValueError("No mitigation operations were extracted from the raw KBs")

    return sorted(merged.values(), key=lambda row: row["operation_id"]), counts, "raw_sparql"


# ---------------- Direct mappings: control/description --> operation ----------------

def build_keyword_mapping(controls, operations, threshold, top_k):
    rows = []
    for control in controls:
        ranked = []
        for op in operations:
            score = keyword_score(
                control["description"],
                f"{op['operation_name']}. {op['description']}",
            )
            ranked.append((score, op))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["operation_id"]))
        selected = [
            {**op, "mapping_score": float(score)}
            for score, op in ranked[:top_k]
            if score >= threshold
        ]
        rows.append({**control, "operations": selected})

    return {"method": "keyword", "goal_model_used": False, "controls": rows}


def load_embedder(model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for --method embedding"
        ) from exc
    return SentenceTransformer(model_name)


def build_embedding_mapping(controls, operations, threshold, top_k, model_name):
    model = load_embedder(model_name)
    control_texts = [control["description"] for control in controls]
    operation_texts = [
        f"{op['operation_name']}. {op['description']}" for op in operations
    ]

    control_vectors = model.encode(
        control_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    operation_vectors = model.encode(
        operation_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    rows = []
    for control, vector in zip(controls, control_vectors):
        scores = operation_vectors @ vector
        indices = np.argsort(-scores)[: min(top_k, len(operations))]
        selected = []
        for index in indices:
            score = float(scores[int(index)])
            if score >= threshold:
                selected.append({
                    **operations[int(index)],
                    "mapping_score": score,
                })
        rows.append({**control, "operations": selected})

    return {"method": "embedding", "goal_model_used": False, "controls": rows}


def ollama_chat(base_url, model, system_prompt, user_prompt, timeout):
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "num_predict": 2000,
            "presence_penalty": 1.5,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot contact Ollama at {base_url}: {exc}") from exc
    return str(body.get("message", {}).get("content", ""))


def extract_json_object(raw_text):
    text_value = (raw_text or "").strip()
    try:
        value = json.loads(text_value)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text_value, re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def build_llm_mapping(controls, operations, candidate_k, base_url, model, timeout):
    system_prompt = (
        "You are performing a direct security control-to-mitigation mapping. "
        "There is NO goal model and NO goal hierarchy in this baseline. "
        "Map the supplied source description directly to zero or more technical "
        "mitigations. Select ONLY exact operation IDs from candidate_mitigations. "
        "Do not invent IDs. Match defensive intent rather than superficial wording. "
        "Return JSON only as {\"selected_operations\":[{\"id\":\"...\",\"reason\":\"...\"}]}. "
        "Use an empty list when no mitigation is justified."
    )

    valid = {operation["operation_id"]: operation for operation in operations}
    rows = []

    for i, control in enumerate(controls, start=1):
        candidates = sorted(
            operations,
            key=lambda operation: keyword_score(
                control["description"],
                f"{operation['operation_name']} {operation['description']}",
            ),
            reverse=True,
        )[:candidate_k]

        user_payload = {
            "control_id": control["id"],
            "source_description": control["description"],
            "event_types": control.get("event_types", []),
            "candidate_mitigations": [
                {
                    "id": operation["operation_id"],
                    "name": operation["operation_name"],
                    "source": operation["source"],
                    "description": operation["description"],
                }
                for operation in candidates
            ],
        }

        raw_response = ollama_chat(
            base_url,
            model,
            system_prompt,
            json.dumps(user_payload, ensure_ascii=False),
            timeout,
        )
        parsed = extract_json_object(raw_response)
        selected = []
        for item in parsed.get("selected_operations", []) or []:
            operation_id = item.get("id") if isinstance(item, dict) else item
            if operation_id not in valid:
                continue
            operation = dict(valid[operation_id])
            if isinstance(item, dict) and item.get("reason"):
                operation["reason"] = item["reason"]
            selected.append(operation)

        rows.append({
            **control,
            "operations": selected,
            "raw_response": raw_response,
        })
        print(f"LLM mapping: {i}/{len(controls)}")

    return {
        "method": "llm",
        "goal_model_used": False,
        "llm_model": model,
        "controls": rows,
    }


# ---------------- Self-contained SWE-Agent runtime semantics ----------------

def infer_event_type(action, observation, state) -> str | None:
    explicit = (
        state.get("event_type")
        or state.get("eventType")
        or state.get("security_event")
        or state.get("securityEvent")
        or state.get("event")
    )
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    action_l = action.lower()
    haystack = f"{action}\n{observation}\n{text(state)}".lower()

    if re.search(r"\b(curl|wget)\b", action_l) and any(token in haystack for token in (
        "../", "..\\", "%2e%2e", "path traversal", "parent-directory"
    )):
        return "path_traversal"
    if re.search(r"\b(pip|pip3)\s+install\b|\bnpm\s+install\b|\bpoetry\s+add\b", haystack):
        return "install_dependency"
    if any(token in haystack for token in ("chmod ", "chown ", "setfacl ")):
        return "permission_change"
    if any(token in haystack for token in ("apply_patch", "sed -i", "modify source", "patch source")):
        return "modify_source"
    if re.search(r"\b(python|python3|bash|sh)\b", action_l):
        return "execute_command"
    if re.search(r"\b(curl|wget)\b", action_l) and "http" in haystack:
        return "external_network_access"
    if re.search(r"(^|\s)(cat|head|tail|less|more|sed|grep|rg)\s+", action_l):
        return "read_file"
    if any(token in haystack for token in ("../", "..\\", "%2e%2e", "path traversal", "parent-directory")):
        return "path_traversal"
    return None


def infer_facts(action, observation, state, event_type: str | None):
    facts: dict[str, bool] = {}
    aliases = {
        "path_traversal_detected": "pathTraversalDetected",
        "file_access_granted": "fileAccessGranted",
        "file_access_attempt": "fileAccessAttempt",
        "sensitive_data_exposed": "sensitiveDataExposed",
        "threat_detected": "threatDetected",
        "mitigation_active": "mitigationActive",
        "action_blocked": "actionBlocked",
        "command_executed": "commandExecuted",
        "command_authorized": "commandAuthorized",
        "remote_access": "remoteAccess",
        "remote_request": "remoteRequest",
        "unauthorized_access_attempt": "unauthorizedAccessAttempt",
    }

    for key, value in state.items():
        if isinstance(value, bool):
            facts[aliases.get(str(key), str(key))] = value

    haystack = f"{action}\n{observation}\n{text(state)}".lower()

    if event_type == "path_traversal":
        facts.setdefault("pathTraversalDetected", True)
        facts.setdefault("fileAccessAttempt", True)
        facts.setdefault("threatDetected", True)
        if re.search(r"\b200\s+ok\b", haystack):
            facts.setdefault("fileAccessGranted", True)

    elif event_type == "execute_command":
        facts.setdefault("commandExecuted", True)
        authorized_markers = (
            "pytest", "python -m pytest", "python -m unittest", "python -c ",
            "python3 -c ", "python - <<", "python3 - <<", "tox", "make ",
            "npm test", "npm run", "git ", "ls ", "cat ", "grep ",
        )
        dangerous_markers = (
            "| bash", "| sh", "curl |bash", "curl | sh", "wget |bash",
            "wget | sh", "rm -rf /", "sudo ",
        )
        if any(marker in haystack for marker in authorized_markers) and not any(
            marker in haystack for marker in dangerous_markers
        ):
            facts.setdefault("commandAuthorized", True)
        else:
            facts.setdefault("threatDetected", True)

    elif event_type == "install_dependency":
        facts.setdefault("installDependency", True)

    elif event_type == "read_file":
        facts.setdefault("fileAccessAttempt", True)

    if re.search(r"\b403\b|\bforbidden\b|\bblocked\b|\bdenied\b", haystack):
        facts.setdefault("actionBlocked", True)
    if "x-security-agent: blocked" in haystack or "mitigation active" in haystack:
        facts.setdefault("mitigationActive", True)
    if "sensitive data exposed" in haystack or "secret exposed" in haystack:
        facts.setdefault("sensitiveDataExposed", True)

    if event_type == "path_traversal" and state.get("remote_access") is True:
        facts.setdefault("remoteRequest", True)
    if facts.get("actionBlocked") and facts.get("fileAccessGranted") is not True:
        facts.setdefault("fileAccessGranted", False)
    if facts.get("pathTraversalDetected") and facts.get("fileAccessGranted"):
        facts.setdefault("sensitiveDataExposed", True)
    if facts.get("pathTraversalDetected") and state.get("authentication") is False:
        facts.setdefault("unauthorizedAccessAttempt", True)

    return facts


def normalize_trace(traj, scenario):
    assumptions = {
        "remote_http_requests_unauthenticated": False,
        **(scenario.get("runtime_assumptions", {}) or {}),
    }
    normalized = []
    for index, raw in enumerate(traj.get("trajectory", [])):
        if not isinstance(raw, dict):
            continue
        action = text(raw.get("action")).strip()
        observation = text(raw.get("observation")).strip()
        state = parse_json_object(raw.get("state"))
        event_type = infer_event_type(action, observation, state)

        if event_type in {"path_traversal", "external_network_access"}:
            is_http_request = bool(re.search(r"\b(curl|wget)\b.*https?://", action.lower()))
            if is_http_request:
                state.setdefault("remote_access", True)
                if assumptions.get("remote_http_requests_unauthenticated", False):
                    state.setdefault("authentication", False)

        facts = infer_facts(action, observation, state, event_type)
        normalized.append({
            "step": index,
            "time": index,
            "action": action,
            "observation": observation,
            "state": state,
            "event_type": event_type,
            "facts": facts,
        })
    return normalized


def _labels(steps):
    return [
        {key for key, value in (step.get("facts") or {}).items() if value is True}
        for step in steps
    ]


def _formula_kind(formula):
    f = re.sub(r"\s+", " ", formula.strip())
    patterns = [
        ("G_NOT", re.compile(r"^G\(\s*(\w+)\s*->\s*!\s*(\w+)\s*\)$")),
        ("G_POS", re.compile(r"^G\(\s*(\w+)\s*->\s*(\w+)\s*\)$")),
        ("G_F", re.compile(r"^G\(\s*(\w+)\s*->\s*F\(\s*(\w+)\s*\)\s*\)$")),
    ]
    for kind, regex in patterns:
        match = regex.match(f)
        if match:
            return kind, match.groups()
    raise ValueError(f"Unsupported LTL formula for the self-contained -GM runtime: {formula}")


def direct_ltl_status(steps, formula):
    """
    Evaluate the finite observed trace with final-state stuttering.
    """
    labels = _labels(steps)
    n = len(labels)
    if not n:
        return "SATISFIED", [], []

    kind, args = _formula_kind(formula)

    def suffix_holds(start: int) -> bool:
        for k in range(start, n):
            if kind == "G_NOT":
                p, q = args
                if p in labels[k] and q in labels[k]:
                    return False
            elif kind == "G_POS":
                p, q = args
                if p in labels[k] and q not in labels[k]:
                    return False
            elif kind == "G_F":
                p, q = args
                if p in labels[k] and not any(q in labels[j] for j in range(k, n)):
                    return False
        return True

    satisfying = [i for i in range(n) if suffix_holds(i)]
    violating = [i for i in range(n) if i not in satisfying]
    return (
        "SATISFIED" if 0 in satisfying else "VIOLATED",
        satisfying,
        violating,
    )


def requirement_trigger_states(steps, requirement, violating_states):
    event_types = set(requirement.get("event_types", []))
    if not event_types:
        return list(violating_states)
    return [
        i for i in violating_states
        if 0 <= i < len(steps) and steps[i].get("event_type") in event_types
    ]


# Shared deployment adapter, copied deliberately into the independent -GM
# baseline so the experiment does not depend on the GRAIL reasoning script.

def infer_enforcement_adapter(operation):
    name = f"{operation.get('operation_name', '')} {operation.get('description', '')}".lower()

    if any(keyword in name for keyword in (
        "path validation", "canonical", "file access mediation",
        "remote file access mediation", "network resource access mediation",
        "network access mediation", "local file permissions",
        "user account permissions", "user group permissions",
        "permissions control", "access control", "access mediation",
        "data access control", "downstream authorization",
    )):
        return "path_or_access_control"
    if "authentication" in name or "multi-factor" in name:
        return "authentication_gate"
    if any(keyword in name for keyword in (
        "trusted library", "software allowlist", "package allowlist", "dependency",
    )):
        return "dependency_allowlist"
    if any(keyword in name for keyword in (
        "allowlisted", "script allowlist", "command allowlist", "authorized script",
    )):
        return "command_allowlist"
    return "generic_block"


def operation_compatible_with_event(operation, event_type: str | None):
    if not event_type:
        return True
    adapter = infer_enforcement_adapter(operation)
    allowed = {
        "path_traversal": {"path_or_access_control", "authentication_gate"},
        "install_dependency": {"dependency_allowlist"},
        "execute_command": {"command_allowlist"},
    }.get(event_type)
    return adapter in allowed if allowed is not None else True


def apply_enforcement_adapter(raw_step, operation: dict[str, Any] | None):
    patched = json.loads(json.dumps(raw_step))
    state = parse_json_object(patched.get("state"))

    if operation is None:
        state["action_blocked"] = True
        state["mitigation_active"] = False
        patched["state"] = json.dumps(state, ensure_ascii=False)
        patched["security_action"] = "escalate"
        return patched, {
            "adapter": None,
            "applied": False,
            "requires_human": True,
            "reason": "No compatible mitigation was selected.",
        }

    adapter = infer_enforcement_adapter(operation)
    requires_human = False

    if adapter in {"path_or_access_control", "authentication_gate"}:
        if adapter == "path_or_access_control":
            state["path_validation"] = True
        else:
            state["authentication"] = True
        state["file_access_granted"] = False
        state["sensitive_data_exposed"] = False
        state["mitigation_active"] = True
        state["action_blocked"] = True
        patched["original_observation"] = patched.get("observation")
        patched["observation"] = BLOCKED_HTTP_OBSERVATION

    elif adapter == "dependency_allowlist":
        state["dependency_allowlist"] = True
        state["dependency_vulnerable"] = False
        state["untrusted_source"] = False
        state["mitigation_active"] = True
        state["action_blocked"] = True
        patched["original_observation"] = patched.get("observation")
        patched["observation"] = BLOCKED_INSTALL_OBSERVATION
        requires_human = True

    elif adapter == "command_allowlist":
        state["command_allowlist"] = True
        state["command_executed"] = False
        state["mitigation_active"] = True
        state["action_blocked"] = True
        patched["original_observation"] = patched.get("observation")
        patched["observation"] = BLOCKED_COMMAND_OBSERVATION
        requires_human = True

    else:
        state["action_blocked"] = True
        patched["original_observation"] = patched.get("observation")
        patched["observation"] = "[GRAIL] action blocked; mitigation adapter unavailable"
        requires_human = True

    patched["state"] = json.dumps(state, ensure_ascii=False)
    patched["security_action"] = {
        "adapter": adapter,
        "operation_id": operation.get("operation_id"),
        "operation_name": operation.get("operation_name"),
    }
    return patched, {
        "adapter": adapter,
        "applied": adapter != "generic_block",
        "requires_human": requires_human,
        "approval_basis": "deployment_adapter_policy" if requires_human else "automatic_case_study_adapter",
    }


# ---------------- Runtime replay ----------------

def choose_control_for_requirement(requirement, controls_by_id, controls):
    """
    Resolve an applicable direct source description for a requirement.
    """
    exact = controls_by_id.get(str(requirement.get("id")))
    if exact is not None:
        return exact

    req_events = set(requirement.get("event_types", []))
    candidates = [
        control for control in controls
        if not req_events or not control.get("event_types")
        or req_events.intersection(control.get("event_types", []))
    ]
    if not candidates:
        candidates = controls
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda control: keyword_score(
            requirement.get("description", ""), control.get("description", "")
        ),
    )


def select_operation(control: dict[str, Any] | None, event_type: str | None, method):
    if control is None:
        return None
    candidates = [
        operation
        for operation in control.get("operations", [])
        if operation_compatible_with_event(operation, event_type)
    ]
    if not candidates:
        return None
    if method in {"keyword", "embedding"}:
        return max(candidates, key=lambda op: float(op.get("mapping_score", 0.0)))
    # LLM mapping has no ranking semantics; preserve the first selected item.
    return candidates[0]


def run_direct_runtime(before, scenario, mapping, method):
    controls = mapping.get("controls", [])
    controls_by_id = {str(control["id"]): control for control in controls}
    persistent: dict[str, Any] = {}
    protected_steps: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    requirements = list(scenario.get("requirements", []))

    for index, original in enumerate(before.get("trajectory", [])):
        raw = json.loads(json.dumps(original))
        state = parse_json_object(raw.get("state"))
        state.update(persistent)
        raw["state"] = json.dumps(state, ensure_ascii=False)

        normalized = normalize_trace({"trajectory": protected_steps + [raw]}, scenario)
        violated_now: list[str] = []
        requirement_evidence: list[dict[str, Any]] = []

        for requirement in requirements:
            applicable = any(
                step.get("event_type") in set(requirement.get("event_types", []))
                for step in normalized
            )
            if not applicable:
                requirement_evidence.append({
                    "id": requirement["id"],
                    "status": "NOT_TRIGGERED",
                    "violation_trigger_states": [],
                })
                continue

            status, satisfying, violating = direct_ltl_status(
                normalized,
                requirement["formula"],
            )
            trigger_states = requirement_trigger_states(
                normalized,
                requirement,
                violating,
            )
            requirement_evidence.append({
                "id": requirement["id"],
                "status": status,
                "violation_trigger_states": trigger_states,
            })
            if status == "VIOLATED" and index in trigger_states:
                violated_now.append(requirement["id"])

        decision: dict[str, Any] = {
            "decision": "ALLOW",
            "violated_requirements": violated_now,
            "baseline": "without_goal_model",
            "trace": [],
        }

        if violated_now:
            event_type = infer_event_type(
                raw.get("action", ""),
                raw.get("observation", ""),
                parse_json_object(raw.get("state")),
            )
            applied_any = False

            for req_id in violated_now:
                requirement = next(req for req in requirements if req["id"] == req_id)
                control = choose_control_for_requirement(
                    requirement,
                    controls_by_id,
                    controls,
                )
                operation = select_operation(control, event_type, method)
                trace_row = {
                    "requirement": req_id,
                    "source_description": control.get("description", "") if control else requirement.get("description", ""),
                    "source_control_id": control.get("id") if control else req_id,
                    "event_type": event_type,
                    "operation": operation,
                }

                if operation is not None:
                    raw, enforcement = apply_enforcement_adapter(raw, operation)
                    trace_row["enforcement"] = enforcement
                    applied_any = applied_any or bool(enforcement.get("applied"))
                else:
                    trace_row["enforcement"] = {
                        "applied": False,
                        "adapter": None,
                        "requires_human": True,
                        "reason": "No event-compatible mitigation selected.",
                    }

                decision["trace"].append(trace_row)

            if applied_any:
                decision["decision"] = "BLOCK"
                decision["reason"] = "direct source-description-to-mitigation match applied"
                decision["escalate_to_human"] = False
            else:
                decision["decision"] = "ESCALATE"
                decision["reason"] = "no directly applicable mitigation selected"
                decision["escalate_to_human"] = True

        # Persist the same case-study security state across subsequent steps.
        updated_state = parse_json_object(raw.get("state"))
        for key in (
            "path_validation", "authentication", "dependency_allowlist",
            "command_allowlist", "file_access_granted", "sensitive_data_exposed",
            "mitigation_active",
        ):
            if key in updated_state:
                persistent[key] = updated_state[key]

        raw["security_decision"] = decision
        protected_steps.append(raw)
        decisions.append({
            "step": index,
            **decision,
            "requirements": requirement_evidence,
        })

    result = json.loads(json.dumps(before))
    result["trajectory"] = protected_steps
    result["grail_security_agent"] = {
        "baseline": "without_goal_model",
        "goal_model_used": False,
        "decision_counts": dict(Counter(item["decision"] for item in decisions)),
        "final_constraints": persistent,
    }
    return result, decisions


def requirement_verdicts(traj, scenario):
    steps = normalize_trace(traj, scenario)
    result: dict[str, str] = {}
    for requirement in scenario.get("requirements", []):
        status, _, _ = direct_ltl_status(steps, requirement["formula"])
        result[requirement["id"]] = status
    return result


# ---------------- CLI ----------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a self-contained direct-control/direct-mitigation RQ3 -GM baseline"
    )
    parser.add_argument("--method", choices=["keyword", "embedding", "llm"], required=True)
    parser.add_argument("--traj-before", type=Path, required=True)
    parser.add_argument("--traj-after", type=Path, required=True,
                        help="Output path for the protected trajectory")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument(
        "--d3fend-kb",
        type=Path,
        default=None,
        help="Raw MITRE D3FEND ontology (d3fend.ttl). Candidates are extracted with SPARQL.",
    )
    parser.add_argument(
        "--ontosecai-kb",
        type=Path,
        default=None,
        help="Raw OntoSecAI ontology (ontosecai.rdf). Candidates are extracted with SPARQL.",
    )
    parser.add_argument(
        "--controls-json",
        type=Path,
        default=None,
        help="Optional direct source-control export. Without it, scenario requirement descriptions are used.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-base-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-candidates", type=int, default=30)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=Path("evaluation_results/rq3_nogm_mapping.json"),
    )
    parser.add_argument(
        "--evidence-output",
        type=Path,
        default=Path("evaluation_results/rq3_nogm_evidence.json"),
    )
    args = parser.parse_args()

    paths_to_check = [args.traj_before, args.scenario, args.d3fend_kb, args.ontosecai_kb]
    for path in paths_to_check:
        if path is not None and not path.exists():
            raise FileNotFoundError(path)

    if args.d3fend_kb is None and args.ontosecai_kb is None:
        parser.error("Provide at least one raw KB: --d3fend-kb and/or --ontosecai-kb")

    scenario = load_json(args.scenario)
    before = load_json(args.traj_before)
    if not isinstance(before, dict) or not isinstance(before.get("trajectory"), list):
        raise ValueError(f"Invalid .traj file: {args.traj_before}")

    args.threshold = (
        0.05 if args.method == "keyword" else 0.45
    ) if args.threshold is None else args.threshold

    controls = load_source_controls(args.scenario, args.controls_json)

    operations, raw_kb_counts, source_mode = extract_operations_from_raw_kbs(
        args.d3fend_kb, args.ontosecai_kb
    )

    print(f"Method: {args.method} -GM")
    print("Runtime implementation: self-contained (no GRAIL runtime script imported)")
    print(f"Source descriptions: {len(controls)}")
    print(f"Mitigation operations: {len(operations)}")
    print(f"Mitigation source: {source_mode}")
    if raw_kb_counts:
        print(f"Raw KB extraction counts: {json.dumps(raw_kb_counts)}")
    print("Goal hierarchy used: NO")

    if args.method == "keyword":
        mapping = build_keyword_mapping(
            controls, operations, args.threshold, args.top_k
        )
    elif args.method == "embedding":
        mapping = build_embedding_mapping(
            controls, operations, args.threshold, args.top_k, args.embedding_model
        )
    else:
        mapping = build_llm_mapping(
            controls,
            operations,
            args.llm_candidates,
            args.llm_base_url,
            args.llm_model,
            args.llm_timeout,
        )

    save_json(args.mapping_output, mapping)

    after, decisions = run_direct_runtime(
        before,
        scenario,
        mapping,
        args.method,
    )
    args.traj_after.parent.mkdir(parents=True, exist_ok=True)
    args.traj_after.write_text(
        json.dumps(after, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    before_verdicts = requirement_verdicts(before, scenario)
    after_verdicts = requirement_verdicts(after, scenario)

    selected = []
    for decision in decisions:
        for trace in decision.get("trace", []):
            if trace.get("operation"):
                selected.append(trace["operation"])

    evidence = {
        "method": f"{args.method} -GM",
        "goal_model_used": False,
        "source_description_basis": (
            "external_controls" if args.controls_json else "scenario_requirement_descriptions"
        ),
        "traj_before": str(args.traj_before),
        "traj_after": str(args.traj_after),
        "mitigation_source_mode": source_mode,
        "raw_kbs": {
            "d3fend": str(args.d3fend_kb) if args.d3fend_kb else None,
            "ontosecai": str(args.ontosecai_kb) if args.ontosecai_kb else None,
        },
        "raw_kb_extraction_counts": raw_kb_counts,
        "mapping_output": str(args.mapping_output),
        "before_requirement_verdicts": before_verdicts,
        "after_requirement_verdicts": after_verdicts,
        "decisions": decisions,
        "selected_operations": selected,
        "decision_counts": dict(Counter(d["decision"] for d in decisions)),
        "traceability": {
            "description": "direct source description -> selected mitigation -> enforcement adapter; no goal hierarchy",
            "selected_operation_count": len(selected),
        },
    }
    save_json(args.evidence_output, evidence)

    print("\nProtected trajectory:", args.traj_after)
    print("Mapping:", args.mapping_output)
    print("Evidence:", args.evidence_output)
    print("\nDecision counts:")
    print(json.dumps(evidence["decision_counts"], indent=2))
    print("\nBEFORE verdicts:")
    print(json.dumps(before_verdicts, indent=2))
    print("\nAFTER verdicts:")
    print(json.dumps(after_verdicts, indent=2))


if __name__ == "__main__":
    main()
