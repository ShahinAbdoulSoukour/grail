"""
Generate the RQ1 figures of the GRAIL paper from the three goal-model KGs.

Inputs (same directory, or pass --kg-dir):
    merged_goal_models_kws.ttl
    merged_goal_models_embedding.ttl
    merged_goal_models_llm.ttl

Optional inputs (for catalog utilization):
    d3fend.ttl, ontosecai.rdf   (--d3fend / --ontosecai)

Outputs:
    fig_operationalization_coverage.pdf   (coverage per governance framework)
    fig_operationalization_profile.pdf    (coverage per goal class + operations per leaf)
    rq1_metrics.json                      (all values behind Table 1 and the figures)

Requirements: pip install pyoxigraph matplotlib numpy
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyoxigraph import RdfFormat, Store

GM = "PREFIX gm: <https://example.org/goal-model/ontology/> "

STRATEGIES = [  # (file suffix, label, colour)
    ("kws", "Keyword +GM", "#f0b45a"),
    ("embedding", "Embedding +GM", "#6c8ebf"),
    ("llm", "LLM +GM", "#82b366"),
]
STANDARDS = [  # (standardName in the KG, label on the x axis)
    ("ISO 27001", "ISO/IEC\n27001"),
    ("NIST", "NIST SP\n800-53"),
    ("CIS Controls", "CIS\nControls"),
    ("AI-RMF", "NIST\nAI RMF"),
    ("ISO 42001", "ISO/IEC\n42001"),
]
CLASSES = [("TECH", "Technical"), ("PHYS", "Physical"), ("ORG", "Organizational")]
# The leaf class is taken from one reference KG, so that all strategies are
# compared on the same partition (the LLM KG contains a few doubly-classified leaves).
CLASS_REFERENCE = "embedding"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42})


def load(path):
    store = Store()
    store.load(path=str(path), format=RdfFormat.TURTLE)
    return store


def leaf_info(store):
    """{leaf IRI: (standard, class)}"""
    out = {}
    q = GM + """SELECT ?g ?std ?c WHERE {
        ?g a gm:LeafGoal ; gm:standardName ?std .
        OPTIONAL { ?g gm:classification ?c } }"""
    for r in store.query(q):
        cls = r["c"].value.rsplit("/", 1)[-1] if r["c"] is not None else None
        out.setdefault(r["g"].value, (r["std"].value, cls))
    return out


def leaf_operations(store):
    """{leaf IRI: set of operation ids}"""
    ops = defaultdict(set)
    q = GM + """SELECT ?g ?id WHERE {
        ?g a gm:LeafGoal ; gm:operationalizedBy ?o . ?o gm:operationId ?id }"""
    for r in store.query(q):
        ops[r["g"].value].add(r["id"].value)
    return ops


def pct(num, den):
    return round(100 * num / den, 1) if den else None


def count(store, query):
    return int(next(iter(store.query(GM + query)))["n"].value)


def operation_sources(store):
    """{operation id: 'D3FEND' | 'OntoSecAI'}"""
    q = GM + "SELECT ?id ?src WHERE { ?o a gm:Operation ; gm:operationId ?id ; gm:source ?src }"
    return {r["id"].value: r["src"].value for r in store.query(q)}


def leaves_with_complete_trace(store):
    """Leaves with a path leaf -> objective -> root -> control family -> standard."""
    q = GM + """SELECT (COUNT(DISTINCT ?g) AS ?n) WHERE {
        ?g a gm:LeafGoal ; gm:parentGoal ?p .
        ?p a gm:ObjectiveGoal ; gm:parentGoal ?r .
        ?r a gm:RootGoal ; gm:partOfFunction ?f .
        ?f gm:partOf ?s . }"""
    return int(next(iter(store.query(q)))["n"].value)


def similarity_scores(store):
    q = GM + "SELECT ?s WHERE { << ?g gm:operationalizedBy ?o >> gm:similarityScore ?s }"
    return [float(r["s"].value) for r in store.query(q)]


def catalog_sizes(d3fend_path, ontosecai_path):
    """Number of D3FEND defensive techniques and OntoSecAI mitigations (None if file missing)."""
    sizes = {"D3FEND": None, "OntoSecAI": None}
    if d3fend_path and Path(d3fend_path).exists():
        st = Store()
        st.load(path=str(d3fend_path), format=RdfFormat.TURTLE)
        q = """PREFIX d3f: <http://d3fend.mitre.org/ontologies/d3fend.owl#>
               PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
               SELECT (COUNT(DISTINCT ?id) AS ?n) WHERE {
                 ?t rdfs:subClassOf* d3f:DefensiveTechnique ; d3f:d3fend-id ?id ; rdfs:label ?l }"""
        sizes["D3FEND"] = int(next(iter(st.query(q)))["n"].value)
    if ontosecai_path and Path(ontosecai_path).exists():
        st = Store()
        st.load(path=str(ontosecai_path), format=RdfFormat.RDF_XML)
        q = """PREFIX hes: <http://www.semanticweb.org/ubaid/ontologies/2023/5/HES#>
               SELECT (COUNT(DISTINCT ?m) AS ?n) WHERE { ?m a hes:Mitigations ; hes:Name ?x }"""
        sizes["OntoSecAI"] = int(next(iter(st.query(q)))["n"].value)
    return sizes


def compute_metrics(stores, leaves, ops, catalogs, kg_files):
    """All values reported in Table 1 and in the RQ1 figures."""
    ref = stores[CLASS_REFERENCE]
    n_leaves = len(leaves)
    controls = set()
    q = GM + "SELECT ?std ?f ?id WHERE { ?g a gm:LeafGoal ; gm:standardName ?std ; gm:functionName ?f ; gm:goalId ?id }"
    for r in ref.query(q):
        controls.add((r["std"].value, r["f"].value, ".".join(r["id"].value.split(".")[:2])))
    n_roots = count(ref, "SELECT (COUNT(DISTINCT ?g) AS ?n) WHERE { ?g a gm:RootGoal }")
    n_obj = count(ref, "SELECT (COUNT(DISTINCT ?g) AS ?n) WHERE { ?g a gm:ObjectiveGoal }")
    n_fam = count(ref, "SELECT (COUNT(DISTINCT ?f) AS ?n) WHERE { ?f a gm:Function }")
    n_std = len({s for s, _ in leaves.values()})
    orphans = count(ref, """SELECT (COUNT(DISTINCT ?g) AS ?n) WHERE {
        ?g a ?t . FILTER(?t IN (gm:ObjectiveGoal, gm:LeafGoal))
        FILTER NOT EXISTS { ?g gm:parentGoal ?p } }""")
    traced = leaves_with_complete_trace(ref)
    class_counts = Counter(c for _, c in leaves.values())

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {k: str(v) for k, v in kg_files.items()},
        "class_reference_kg": CLASS_REFERENCE,
        "structure": {
            "standards": n_std,
            "control_families": n_fam,
            "controls": len(controls),
            "goals_total": n_roots + n_obj + n_leaves,
            "root_goals": n_roots,
            "objective_goals": n_obj,
            "leaf_goals": n_leaves,
            "leaf_goals_per_control_mean": round(n_leaves / len(controls), 2),
            "refinement_depth": 3,
            "leaf_classes": {c: class_counts.get(c, 0) for c, _ in CLASSES},
            "orphan_non_root_goals": orphans,
            "leaves_traceable_to_standard": traced,
            "leaves_traceable_to_standard_pct": pct(traced, n_leaves),
        },
        "catalog_sizes": catalogs,
        "strategies": {},
        "agreement": {},
    }

    for k, label, _ in STRATEGIES:
        st = stores[k]
        o = {g: v for g, v in ops[k].items() if g in leaves and v}
        src = operation_sources(st)
        per_leaf = [len(v) for v in o.values()]
        used = set().union(*o.values()) if o else set()
        used_by_src = Counter(src.get(x, "unknown") for x in used)
        links_by_src = Counter(src.get(x, "unknown") for v in o.values() for x in v)
        scores = similarity_scores(st)
        entry = {
            "label": label,
            "operationalized_leaves": len(o),
            "operationalized_leaves_pct": pct(len(o), n_leaves),
            "per_class": {
                c: {"operationalized": sum(1 for g in o if leaves[g][1] == c),
                    "total": class_counts.get(c, 0),
                    "pct": pct(sum(1 for g in o if leaves[g][1] == c), class_counts.get(c, 0))}
                for c, _ in CLASSES},
            "per_standard": {
                std: {"operationalized": sum(1 for g in o if leaves[g][0] == std),
                      "total": sum(1 for s, _ in leaves.values() if s == std),
                      "pct": pct(sum(1 for g in o if leaves[g][0] == std),
                                 sum(1 for s, _ in leaves.values() if s == std))}
                for std, _ in STANDARDS},
            "goal_operation_links": sum(per_leaf),
            "links_by_catalog": dict(links_by_src),
            "operations_per_operationalized_leaf": {
                "mean": round(statistics.mean(per_leaf), 2) if per_leaf else None,
                "median": statistics.median(per_leaf) if per_leaf else None,
                "max": max(per_leaf) if per_leaf else None,
                "distribution": {str(n): c for n, c in sorted(Counter(per_leaf).items())},
            },
            "distinct_operations_used": len(used),
            "distinct_operations_by_catalog": dict(used_by_src),
            "catalog_utilization_pct": {
                cat: pct(used_by_src.get(cat, 0), size) for cat, size in catalogs.items()},
            "scored_links": bool(scores),
            "similarity_scores": ({"min": round(min(scores), 3), "mean": round(statistics.mean(scores), 3),
                                   "max": round(max(scores), 3), "count": len(scores)} if scores else None),
        }
        metrics["strategies"][k] = entry

    for a, b in combinations([k for k, _, _ in STRATEGIES], 2):
        A = {g: v for g, v in ops[a].items() if g in leaves and v}
        B = {g: v for g, v in ops[b].items() if g in leaves and v}
        common = sorted(set(A) & set(B))
        jac = [len(A[g] & B[g]) / len(A[g] | B[g]) for g in common]
        share = sum(1 for g in common if A[g] & B[g])
        metrics["agreement"][f"{a}__{b}"] = {
            "leaves_operationalized_by_both": len(common),
            "leaves_sharing_at_least_one_operation": share,
            "leaves_sharing_at_least_one_operation_pct": pct(share, len(common)),
            "mean_jaccard": round(statistics.mean(jac), 3) if jac else None,
        }
    return metrics


def bar_labels(ax, xs, vals):
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-dir", default=".")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--d3fend", default="d3fend.ttl", help="D3FEND ontology (for catalog utilization)")
    ap.add_argument("--ontosecai", default="ontosecai.rdf", help="OntoSecAI ontology (for catalog utilization)")
    args = ap.parse_args()
    kg_dir, out_dir = Path(args.kg_dir), Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    kg_files = {k: kg_dir / f"merged_goal_models_{k}.ttl" for k, _, _ in STRATEGIES}
    stores = {k: load(p) for k, p in kg_files.items()}
    leaves = leaf_info(stores[CLASS_REFERENCE])
    ops = {k: leaf_operations(s) for k, s in stores.items()}
    n_total = len(leaves)

    # ---------------- Metrics (JSON) ----------------
    d3f = Path(args.d3fend) if Path(args.d3fend).is_absolute() else kg_dir / args.d3fend
    osa = Path(args.ontosecai) if Path(args.ontosecai).is_absolute() else kg_dir / args.ontosecai
    metrics = compute_metrics(stores, leaves, ops, catalog_sizes(d3f, osa), kg_files)
    with open(out_dir / "rq1_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---------------- Figure A: coverage per framework ----------------
    labels = [f"{lab}\n(n={sum(1 for s, _ in leaves.values() if s == std)})"
              for std, lab in STANDARDS] + [f"All\n\n(n={n_total})"]
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    for i, (k, lab, col) in enumerate(STRATEGIES):
        vals = []
        for std, _ in STANDARDS:
            sel = [g for g, (s, _) in leaves.items() if s == std]
            vals.append(100 * sum(1 for g in sel if ops[k].get(g)) / len(sel))
        vals.append(100 * sum(1 for g in leaves if ops[k].get(g)) / n_total)
        xs = x + (i - 1) * w
        ax.bar(xs, vals, w, label=lab, color=col, edgecolor="black", linewidth=0.4)
        bar_labels(ax, xs, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Operationalized leaves (%)")
    ax.set_ylim(0, 110)
    ax.axvline(len(labels) - 1.5, color="grey", lw=0.6, ls=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_operationalization_coverage.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------------- Figure B: profile per class + operations per leaf ----------------
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(5.4, 2.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) coverage per goal class
    x = np.arange(len(CLASSES))
    for i, (k, lab, col) in enumerate(STRATEGIES):
        vals = []
        for c, _ in CLASSES:
            sel = [g for g, (_, cc) in leaves.items() if cc == c]
            vals.append(100 * sum(1 for g in sel if ops[k].get(g)) / len(sel))
        xs = x + (i - 1) * w
        axa.bar(xs, vals, w, label=lab, color=col, edgecolor="black", linewidth=0.4)
        bar_labels(axa, xs, vals)
    axa.set_xticks(x)
    axa.set_xticklabels([f"{lab}\n(n={sum(1 for _, cc in leaves.values() if cc == c)})"
                         for c, lab in CLASSES], fontsize=7)
    axa.set_ylabel("Operationalized leaves (%)")
    axa.set_ylim(0, 110)
    axa.set_title("(a) Coverage per goal class", fontsize=8)
    axa.spines[["top", "right"]].set_visible(False)

    # (b) number of operations per operationalized leaf
    data = [[len(v) for g, v in ops[k].items() if g in leaves and v] for k, _, _ in STRATEGIES]
    bp = axb.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                     medianprops={"color": "black", "linewidth": 1},
                     flierprops={"marker": "o", "markersize": 2, "alpha": 0.5})
    for patch, (_, _, col) in zip(bp["boxes"], STRATEGIES):
        patch.set_facecolor(col)
        patch.set_linewidth(0.6)
    for i, d in enumerate(data, start=1):
        axb.plot(i, np.mean(d), marker="D", color="white", markeredgecolor="black",
                 markersize=3.5, zorder=3)
    axb.set_xticks([1, 2, 3])
    axb.set_xticklabels(["Keyword", "Embedding", "LLM"], fontsize=7)
    axb.set_ylabel("Operations per leaf")
    axb.set_title("(b) Operations per operationalized leaf", fontsize=8)
    axb.spines[["top", "right"]].set_visible(False)

    handles, lbls = axa.get_legend_handles_labels()
    fig.legend(handles, lbls, ncol=3, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.11))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_operationalization_profile.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Written:", out_dir / "fig_operationalization_coverage.pdf",
          out_dir / "fig_operationalization_profile.pdf",
          out_dir / "rq1_metrics.json")


if __name__ == "__main__":
    main()