#!/usr/bin/env python3
"""Build a markdown report for the conservative LERF seed sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output" / "radio_gs"

SCENE_SPECS = [
    {
        "scene": "figurines",
        "label": "Figurines",
        "variants": [
            {
                "name": "nofdh",
                "label": "noFDH",
                "summary_paths": {
                    42: OUTPUT_ROOT / "lerf_figurines_v14_nofdh_240ep" / "lerf_eval_best" / "summary.json",
                    7: OUTPUT_ROOT / "lerf_figurines_v14_nofdh_240ep_seed7" / "lerf_eval_best" / "summary.json",
                    123: OUTPUT_ROOT / "lerf_figurines_v14_nofdh_240ep_seed123" / "lerf_eval_best" / "summary.json",
                },
            },
            {
                "name": "fdh",
                "label": "FDH",
                "summary_paths": {
                    42: OUTPUT_ROOT / "lerf_figurines_v14_fdh_ws240_240ep" / "lerf_eval_best" / "summary.json",
                    7: OUTPUT_ROOT / "lerf_figurines_v14_fdh_ws240_240ep_seed7" / "lerf_eval_best" / "summary.json",
                    123: OUTPUT_ROOT / "lerf_figurines_v14_fdh_ws240_240ep_seed123" / "lerf_eval_best" / "summary.json",
                },
            },
        ],
    },
    {
        "scene": "ramen",
        "label": "Ramen",
        "variants": [
            {
                "name": "nofdh",
                "label": "noFDH",
                "summary_paths": {
                    42: OUTPUT_ROOT / "lerf_ramen_v14_nofdh_240ep" / "lerf_eval_best" / "summary.json",
                    7: OUTPUT_ROOT / "lerf_ramen_v14_nofdh_240ep_seed7" / "lerf_eval_best" / "summary.json",
                    123: OUTPUT_ROOT / "lerf_ramen_v14_nofdh_240ep_seed123" / "lerf_eval_best" / "summary.json",
                },
            },
            {
                "name": "fdh",
                "label": "FDH",
                "summary_paths": {
                    42: OUTPUT_ROOT / "lerf_ramen_v14_fdh_ws240_240ep" / "lerf_eval_best" / "summary.json",
                    7: OUTPUT_ROOT / "lerf_ramen_v14_fdh_ws240_240ep_seed7" / "lerf_eval_best" / "summary.json",
                    123: OUTPUT_ROOT / "lerf_ramen_v14_fdh_ws240_240ep_seed123" / "lerf_eval_best" / "summary.json",
                },
            },
        ],
    },
    {
        "scene": "teatime",
        "label": "Teatime",
        "variants": [
            {
                "name": "nofdh",
                "label": "noFDH",
                "summary_paths": {
                    42: OUTPUT_ROOT / "lerf_teatime_v14_nofdh_240ep" / "lerf_eval_best" / "summary.json",
                    7: OUTPUT_ROOT / "lerf_teatime_v14_nofdh_240ep_seed7" / "lerf_eval_best" / "summary.json",
                    123: OUTPUT_ROOT / "lerf_teatime_v14_nofdh_240ep_seed123" / "lerf_eval_best" / "summary.json",
                },
            },
            {
                "name": "fdh",
                "label": "FDH",
                "summary_paths": {
                    42: OUTPUT_ROOT / "lerf_teatime_v14_fdh_ws240_240ep" / "lerf_eval_best" / "summary.json",
                    7: OUTPUT_ROOT / "lerf_teatime_v14_fdh_ws240_240ep_seed7" / "lerf_eval_best" / "summary.json",
                    123: OUTPUT_ROOT / "lerf_teatime_v14_fdh_ws240_240ep_seed123" / "lerf_eval_best" / "summary.json",
                },
            },
        ],
    },
    {
        "scene": "waldo_kitchen",
        "label": "Waldo Kitchen",
        "variants": [
            {
                "name": "nofdh",
                "label": "noFDH",
                "summary_paths": {
                    42: OUTPUT_ROOT / "lerf_waldo_kitchen_v14_nofdh_240ep" / "lerf_eval_best" / "summary.json",
                    7: OUTPUT_ROOT / "lerf_waldo_kitchen_v14_nofdh_240ep_seed7" / "lerf_eval_best" / "summary.json",
                    123: OUTPUT_ROOT / "lerf_waldo_kitchen_v14_nofdh_240ep_seed123" / "lerf_eval_best" / "summary.json",
                },
            },
            {
                "name": "fdh",
                "label": "FDH",
                "summary_paths": {
                    42: OUTPUT_ROOT / "lerf_waldo_kitchen_v14_fdh_ws240_240ep" / "lerf_eval_best" / "summary.json",
                    7: OUTPUT_ROOT / "lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7" / "lerf_eval_best" / "summary.json",
                    123: OUTPUT_ROOT / "lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed123" / "lerf_eval_best" / "summary.json",
                },
            },
        ],
    },
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def fmt(value: float) -> str:
    return f"{value:.4f}"


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_variant_rows(summary_paths: dict[int, Path]) -> tuple[list[dict], list[int]]:
    rows: list[dict] = []
    missing: list[int] = []
    for seed in sorted(summary_paths):
        path = summary_paths[seed]
        if not path.exists():
            missing.append(seed)
            continue
        payload = load_summary(path)
        best = payload["best"]
        rows.append(
            {
                "seed": seed,
                "path": path,
                "loc_acc": float(best["loc_acc"]),
                "miou": float(best["miou"]),
                "temp": float(best["temp"]),
                "loc_total": int(best["loc_total"]),
            }
        )
    return rows, missing


def build_scene_section(scene_spec: dict) -> tuple[list[str], bool]:
    lines: list[str] = []
    complete = True

    lines.append(f"## {scene_spec['label']}")
    lines.append("")
    lines.append("| Variant | Seed | Best LocAcc | Best mIoU | Best Temp | Loc Total | Source |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    summary_rows: list[tuple[str, list[dict]]] = []
    pending_lines: list[str] = []

    for variant in scene_spec["variants"]:
        rows, missing = collect_variant_rows(variant["summary_paths"])
        summary_rows.append((variant["label"], rows))
        for row in rows:
            lines.append(
                "| {variant} | {seed} | {loc_acc} | {miou} | {temp:.1f} | {loc_total} | `{source}` |".format(
                    variant=variant["label"],
                    seed=row["seed"],
                    loc_acc=fmt(row["loc_acc"]),
                    miou=fmt(row["miou"]),
                    temp=row["temp"],
                    loc_total=row["loc_total"],
                    source=relpath(row["path"]),
                )
            )
        if missing:
            complete = False
            pending_lines.append(
                "- `{variant}` pending seeds: {seeds}".format(
                    variant=variant["label"],
                    seeds=", ".join(str(seed) for seed in missing),
                )
            )

    lines.append("")
    lines.append("### Mean ± Std")
    lines.append("")
    lines.append("| Variant | Seeds Present | LocAcc Mean | LocAcc Std | mIoU Mean | mIoU Std |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for variant_label, rows in summary_rows:
        if not rows:
            lines.append(f"| {variant_label} | 0 | - | - | - | - |")
            continue
        loc_values = [row["loc_acc"] for row in rows]
        miou_values = [row["miou"] for row in rows]
        lines.append(
            f"| {variant_label} | {len(rows)} | {fmt(mean(loc_values))} | {fmt(std(loc_values))} | {fmt(mean(miou_values))} | {fmt(std(miou_values))} |"
        )

    if pending_lines:
        lines.append("")
        lines.append("### Pending")
        lines.append("")
        lines.extend(pending_lines)

    lines.append("")
    return lines, complete


def build_report() -> tuple[str, bool]:
    lines = [
        "# LERF Seed Robustness Summary",
        "",
        "This report summarizes the conservative `n=3` seed sweep for the key LERF scenes and compares the best rendered-feature LERF-OVS scores selected by each run's own temperature sweep.",
        "",
        "- Target scenes: `figurines`, `ramen`, `teatime`, `waldo_kitchen`",
        "- Variants: `nofdh`, `fdh`",
        "- Seeds: `42`, `7`, `123`",
        "- Per-run score: `lerf_eval_best/summary.json -> best.loc_acc` with `best.miou` as the tie-aware supporting metric",
        "",
    ]

    all_complete = True
    for scene_spec in SCENE_SPECS:
        scene_lines, scene_complete = build_scene_section(scene_spec)
        lines.extend(scene_lines)
        all_complete = all_complete and scene_complete

    lines.append("## Status")
    lines.append("")
    if all_complete:
        lines.append("All targeted seed runs and eval sweeps are present.")
    else:
        lines.append("This report is partial because one or more targeted seed runs or eval sweeps are still missing.")
    lines.append("")
    return "\n".join(lines) + "\n", all_complete


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LERF seed robustness markdown report")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_ROOT / "reports" / "seed_robustness_summary.md"),
        help="Output markdown path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text, complete = build_report()
    output_path.write_text(text, encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"complete={str(complete).lower()}")


if __name__ == "__main__":
    main()
