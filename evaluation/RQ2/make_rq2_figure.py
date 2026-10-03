"""
Generate the RQ2 figure: outcome of the malicious actions per configuration and
adapter variant (blocked automatically / blocked with human approval / escalated).

Inputs: the rq2_metrics.json files written by compute_rq2_metrics.py, e.g.
    rq2_results_prototype/rq2_metrics.json        (--adapters prototype --top-k-goals 5)
    rq2_results_extended_k5/rq2_metrics.json      (--adapters extended  --top-k-goals 5)
    rq2_results_extended_k10/rq2_metrics.json     (--adapters extended  --top-k-goals 10)

Output: fig_rq2_enforcement.pdf  (+ rq2_figure_data.json with the plotted values)

Usage:
    python make_rq2_figure.py --runs rq2_results_prototype rq2_results_extended_k5 \
           rq2_results_extended_k10 --out-dir figures/
Requirements: pip install matplotlib numpy
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42})

CONFIGS = [("kws", "Keyword +GM"), ("embedding", "Embedding +GM"), ("llm", "LLM +GM")]
COLORS = {"auto": "#4f8a3c", "human": "#a9d18e", "esc": "#f0b45a"}


def run_label(m):
    adapters = "Proto." if m.get("adapters") == "prototype" else "Ext."
    return f"{adapters}\nk={m.get('top_k_goals', 5)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Result directories (one per run)")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = [json.loads((Path(r) / "rq2_metrics.json").read_text()) for r in args.runs]
    data = []
    for m in runs:
        row = {"run": run_label(m).replace("\n", ", "), "configurations": {}}
        for k, _ in CONFIGS:
            c = m["configurations"][k]
            blocked = c["blocked_violations"]
            human = c.get("blocked_with_human_approval", 0)
            row["configurations"][k] = {"blocked_automatic": blocked - human,
                                        "blocked_human_approval": human,
                                        "escalated": c["escalated_violations"],
                                        "total": c["violation_decisions"],
                                        "traced_to_standard_pct": c["traced_to_standard_pct"]}
        data.append(row)
    (out / "rq2_figure_data.json").write_text(json.dumps(data, indent=2))

    n_runs = len(runs)
    width = 0.8 / n_runs
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    x = np.arange(len(CONFIGS))
    for j, (m, row) in enumerate(zip(runs, data)):
        xs = x + (j - (n_runs - 1) / 2) * width
        auto = np.array([row["configurations"][k]["blocked_automatic"] for k, _ in CONFIGS])
        hum = np.array([row["configurations"][k]["blocked_human_approval"] for k, _ in CONFIGS])
        esc = np.array([row["configurations"][k]["escalated"] for k, _ in CONFIGS])
        kw = dict(width=width * 0.92, edgecolor="black", linewidth=0.4)
        ax.bar(xs, auto, color=COLORS["auto"], label="Blocked (automatic)" if j == 0 else None, **kw)
        ax.bar(xs, hum, bottom=auto, color=COLORS["human"], hatch="////",
               label="Blocked (human approval)" if j == 0 else None, **kw)
        ax.bar(xs, esc, bottom=auto + hum, color=COLORS["esc"],
               label="Escalated (explained, not blocked)" if j == 0 else None, **kw)
        for xi, a, h in zip(xs, auto, hum):
            ax.text(xi, 10.25, f"{a + h}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
            ax.text(xi, -0.35, run_label(m), ha="center", va="top", fontsize=6, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in CONFIGS], fontsize=8)
    ax.tick_params(axis="x", pad=19, length=0)
    ax.set_ylabel("Malicious actions (of 10)")
    ax.set_ylim(0, 11.3)
    ax.set_yticks(range(0, 11, 2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_rq2_enforcement.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_rq2_enforcement.png", dpi=200, bbox_inches="tight")
    print("Written:", out / "fig_rq2_enforcement.pdf", out / "rq2_figure_data.json")


if __name__ == "__main__":
    main()
