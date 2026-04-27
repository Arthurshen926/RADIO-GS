#!/usr/bin/env python3
"""
build_paper_main_table.py
--------------------------
Reads all available lerf_eval_best/summary.json files for the 4-scene LERF-OVS
benchmark and generates a LaTeX + Markdown main results table for the paper.

Usage:
    python radio_gs/scripts/build_paper_main_table.py
    python radio_gs/scripts/build_paper_main_table.py --output output/radio_gs/reports/paper_main_table.md
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output" / "radio_gs"

SCENES = ["figurines", "ramen", "teatime", "waldo_kitchen"]
SCENE_LABELS = {
    "figurines": "Figurines",
    "ramen": "Ramen",
    "teatime": "Teatime",
    "waldo_kitchen": "Waldo Kitchen",
}

# Map (scene, method) -> {seed: exp_dir}
EXP_DIRS = {
    ("figurines", "nofdh"): {
        42: OUTPUT_ROOT / "lerf_figurines_v14_nofdh_240ep",
        7:  OUTPUT_ROOT / "lerf_figurines_v14_nofdh_240ep_seed7",
        123: OUTPUT_ROOT / "lerf_figurines_v14_nofdh_240ep_seed123",
    },
    ("figurines", "fdh_ws240"): {
        42: OUTPUT_ROOT / "lerf_figurines_v14_fdh_ws240_240ep",
        7:  OUTPUT_ROOT / "lerf_figurines_v14_fdh_ws240_240ep_seed7",
        123: OUTPUT_ROOT / "lerf_figurines_v14_fdh_ws240_240ep_seed123",
    },
    ("ramen", "nofdh"): {
        42: OUTPUT_ROOT / "lerf_ramen_v14_nofdh_240ep",
        7:  OUTPUT_ROOT / "lerf_ramen_v14_nofdh_240ep_seed7",
        123: OUTPUT_ROOT / "lerf_ramen_v14_nofdh_240ep_seed123",
    },
    ("ramen", "fdh_ws240"): {
        42: OUTPUT_ROOT / "lerf_ramen_v14_fdh_ws240_240ep",
        7:  OUTPUT_ROOT / "lerf_ramen_v14_fdh_ws240_240ep_seed7",
        123: OUTPUT_ROOT / "lerf_ramen_v14_fdh_ws240_240ep_seed123",
    },
    ("teatime", "nofdh"): {
        42: OUTPUT_ROOT / "lerf_teatime_v14_nofdh_240ep",
        7:  OUTPUT_ROOT / "lerf_teatime_v14_nofdh_240ep_seed7",
        123: OUTPUT_ROOT / "lerf_teatime_v14_nofdh_240ep_seed123",
    },
    ("teatime", "fdh_ws240"): {
        42: OUTPUT_ROOT / "lerf_teatime_v14_fdh_ws240_240ep",
        7:  OUTPUT_ROOT / "lerf_teatime_v14_fdh_ws240_240ep_seed7",
        123: OUTPUT_ROOT / "lerf_teatime_v14_fdh_ws240_240ep_seed123",
    },
    ("waldo_kitchen", "nofdh"): {
        42: OUTPUT_ROOT / "lerf_waldo_kitchen_v14_nofdh_240ep",
        7:  OUTPUT_ROOT / "lerf_waldo_kitchen_v14_nofdh_240ep_seed7",
        123: OUTPUT_ROOT / "lerf_waldo_kitchen_v14_nofdh_240ep_seed123",
    },
    ("waldo_kitchen", "fdh_ws240"): {
        42: OUTPUT_ROOT / "lerf_waldo_kitchen_v14_fdh_ws240_240ep",
        7:  OUTPUT_ROOT / "lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7",
        123: OUTPUT_ROOT / "lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed123",
    },
}

def mean(vals): return sum(vals) / len(vals)
def std(vals):
    if len(vals) < 2: return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m)**2 for v in vals) / (len(vals) - 1))

def load_result(exp_dir: Path):
    """Load loc_acc and miou from lerf_eval_best/summary.json"""
    p = exp_dir / "lerf_eval_best" / "summary.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        best = d.get("best", {})
        return {"loc_acc": float(best["loc_acc"]), "miou": float(best["miou"])}
    except Exception:
        return None

def collect(scene, method):
    """Returns dict: seed -> result (or None if missing)"""
    dirs = EXP_DIRS.get((scene, method), {})
    return {seed: load_result(d) for seed, d in dirs.items()}

def agg(results_by_seed):
    """Aggregate available results -> (mean, std, n, values_by_seed)"""
    vals = [(s, r) for s, r in results_by_seed.items() if r is not None]
    if not vals:
        return None, None, 0, {}
    loc_accs = [r["loc_acc"] for _, r in vals]
    return mean(loc_accs), std(loc_accs), len(loc_accs), {s: r["loc_acc"] for s, r in vals}

def fmt_cell(m, s, n):
    if m is None:
        return "—"
    if n == 3:
        return f"{m:.4f} ± {s:.4f}"
    return f"{m:.4f} ({n}/3)"

def build_report():
    lines = []
    lines.append("# RADIO-GS: Main Results Table (LERF-OVS)")
    lines.append(f"*Auto-generated. Last updated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    # Collect all data
    data = {}
    for scene in SCENES:
        for method in ["nofdh", "fdh_ws240"]:
            by_seed = collect(scene, method)
            m, s, n, vals = agg(by_seed)
            data[(scene, method)] = {"mean": m, "std": s, "n": n, "by_seed": by_seed, "vals": vals}

    # --- Markdown main table ---
    lines.append("## Main Table: LocAcc (n=3 seeds, mean ± std)")
    lines.append("")
    lines.append("| Scene | noFDH | FDH-WS240 (Ours) | Δ |")
    lines.append("|---|---|---|---|")

    macro_nofdh, macro_fdh, macro_n = [], [], 0
    for scene in SCENES:
        nd = data[(scene, "nofdh")]
        fd = data[(scene, "fdh_ws240")]
        nofdh_cell = fmt_cell(nd["mean"], nd["std"], nd["n"])
        fdh_cell = fmt_cell(fd["mean"], fd["std"], fd["n"])
        if nd["mean"] is not None and fd["mean"] is not None:
            delta = f"+{fd['mean'] - nd['mean']:.4f}" if fd["mean"] >= nd["mean"] else f"{fd['mean'] - nd['mean']:.4f}"
            # Bold the winner
            if fd["mean"] > nd["mean"]:
                fdh_cell = f"**{fdh_cell}**"
            elif nd["mean"] > fd["mean"]:
                nofdh_cell = f"**{nofdh_cell}**"
        else:
            delta = "—"
        lines.append(f"| {SCENE_LABELS[scene]} | {nofdh_cell} | {fdh_cell} | {delta} |")
        if nd["mean"] is not None: macro_nofdh.append(nd["mean"])
        if fd["mean"] is not None: macro_fdh.append(fd["mean"])

    lines.append("")
    if macro_nofdh and macro_fdh and len(macro_nofdh) == len(macro_fdh) == 4:
        mn = mean(macro_nofdh)
        mf = mean(macro_fdh)
        delta = f"+{mf - mn:.4f}" if mf >= mn else f"{mf - mn:.4f}"
        lines.append(f"**Macro Average (4 scenes):** noFDH = {mn:.4f} | FDH-WS240 = **{mf:.4f}** | Δ = {delta}")
    else:
        lines.append(f"**Macro Average:** Partial ({len(macro_fdh)}/4 scenes complete for FDH-WS240)")
    lines.append("")

    # --- Per-seed breakdown ---
    lines.append("## Per-Seed Breakdown")
    lines.append("")
    for scene in SCENES:
        lines.append(f"### {SCENE_LABELS[scene]}")
        lines.append("")
        lines.append("| Method | Seed 42 | Seed 7 | Seed 123 | Mean | Std | N |")
        lines.append("|---|---|---|---|---|---|---|")
        for method, label in [("nofdh", "noFDH"), ("fdh_ws240", "FDH-WS240")]:
            d = data[(scene, method)]
            s42 = f"{d['by_seed'].get(42, {}).get('loc_acc', None):.4f}" if d['by_seed'].get(42) else "—"
            s7  = f"{d['by_seed'].get(7, {}).get('loc_acc', None):.4f}" if d['by_seed'].get(7) else "—"
            s123= f"{d['by_seed'].get(123, {}).get('loc_acc', None):.4f}" if d['by_seed'].get(123) else "—"
            m_str = f"{d['mean']:.4f}" if d['mean'] is not None else "—"
            s_str = f"{d['std']:.4f}" if d['std'] is not None else "—"
            lines.append(f"| {label} | {s42} | {s7} | {s123} | {m_str} | {s_str} | {d['n']}/3 |")
        lines.append("")

    # --- LaTeX table ---
    lines.append("## LaTeX Table (for paper)")
    lines.append("")
    lines.append("```latex")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{LERF-OVS Localization Accuracy (LocAcc). Results are mean $\pm$ std over $n=3$ random seeds.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Scene & noFDH & FDH-WS240 (Ours) \\")
    lines.append(r"\midrule")
    for scene in SCENES:
        nd = data[(scene, "nofdh")]
        fd = data[(scene, "fdh_ws240")]
        def latex_cell(d):
            if d["mean"] is None: return r"\textemdash"
            if d["n"] == 3:
                return f"{d['mean']:.3f} $\\pm$ {d['std']:.3f}"
            return f"{d['mean']:.3f} ({d['n']}/3)"
        nc = latex_cell(nd)
        fc = latex_cell(fd)
        if fd["mean"] is not None and nd["mean"] is not None and fd["mean"] > nd["mean"]:
            fc = r"\textbf{" + fc + "}"
        elif nd["mean"] is not None and fd["mean"] is not None and nd["mean"] > fd["mean"]:
            nc = r"\textbf{" + nc + "}"
        lines.append(f"{SCENE_LABELS[scene]} & {nc} & {fc} \\\\")
    lines.append(r"\midrule")
    if len(macro_nofdh) == len(macro_fdh) == 4:
        mn, mf = mean(macro_nofdh), mean(macro_fdh)
        nc = f"{mn:.3f}"
        fc = f"{mf:.3f}"
        if mf > mn: fc = r"\textbf{" + fc + "}"
        else: nc = r"\textbf{" + nc + "}"
        lines.append(f"Macro Avg & {nc} & {fc} \\\\")
    else:
        lines.append(f"Macro Avg & \\multicolumn{{2}}{{c}}{{partial ({len(macro_fdh)}/4 scenes)}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("```")
    lines.append("")

    # --- Completeness status ---
    total = 4 * 2 * 3  # scenes * methods * seeds
    done = sum(1 for scene in SCENES for method in ["nofdh", "fdh_ws240"]
               for seed in [42, 7, 123] if data[(scene, method)]["by_seed"].get(seed) is not None)
    lines.append(f"## Completeness: {done}/{total} runs have eval results")
    lines.append("")
    for scene in SCENES:
        for method in ["nofdh", "fdh_ws240"]:
            d = data[(scene, method)]
            seeds_done = [s for s in [42, 7, 123] if d["by_seed"].get(s) is not None]
            seeds_pend = [s for s in [42, 7, 123] if d["by_seed"].get(s) is None]
            status = "✅" if len(seeds_done) == 3 else "⏳"
            lines.append(f"- {status} {scene}/{method}: seeds done={seeds_done}, pending={seeds_pend}")

    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "reports" / "paper_main_table.md"))
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build_report()
    out.write_text(text)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
