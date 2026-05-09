#!/usr/bin/env python3
"""Build a paper-facing LERF failure-analysis table."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv"


@dataclass(frozen=True)
class CategoryRow:
    scene: str
    category: str
    loc_acc: float
    miou: float
    n_samples: int


def load_category_rows(scene_label: str, result_path: Path) -> list[CategoryRow]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    rows: list[CategoryRow] = []
    for scene_payload in payload.get("scenes", {}).values():
        mode_payload = scene_payload.get("rendered") or scene_payload.get("gt") or {}
        per_category = mode_payload.get("per_category", {})
        for category, metrics in per_category.items():
            rows.append(
                CategoryRow(
                    scene=scene_label,
                    category=str(category),
                    loc_acc=float(metrics.get("loc_acc", 0.0)),
                    miou=float(metrics.get("miou", 0.0)),
                    n_samples=int(metrics.get("n_samples", 0)),
                )
            )
    return rows


def select_fragile_rows(rows: Iterable[CategoryRow], limit: int = 12) -> list[CategoryRow]:
    candidates = [row for row in rows if row.loc_acc < 1.0 or row.miou < 0.25]
    return sorted(candidates, key=lambda row: (row.loc_acc, row.miou, -row.n_samples, row.scene, row.category))[
        :limit
    ]


def _scene_label(scene: str) -> str:
    return {
        "figurines": "Figurines",
        "ramen": "Ramen",
        "teatime": "Teatime",
        "waldo_kitchen": "Waldo Kitchen",
    }.get(scene, scene)


def load_current_best_rows(csv_path: Path = DEFAULT_CSV) -> list[CategoryRow]:
    rows: list[CategoryRow] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            if item["scene"] == "macro":
                continue
            output_dir = Path(item["output_dir"])
            result_path = output_dir / "lerf_ovs_results.json"
            if not result_path.is_absolute():
                result_path = REPO_ROOT / result_path
            rows.extend(load_category_rows(_scene_label(item["scene"]), result_path))
    return rows


def write_markdown(rows: list[CategoryRow], fragile: list[CategoryRow], path: Path) -> None:
    by_scene: dict[str, list[CategoryRow]] = {}
    for row in rows:
        by_scene.setdefault(row.scene, []).append(row)

    lines = [
        "# LERF Failure Analysis",
        "",
        "This report is generated from the current best rendered LERF-OVS JSON files. "
        "Rows are ranked by localization failure first and mIoU second, exposing "
        "the small-object and peak-placement failure modes discussed in the paper.",
        "",
        "## Worst / Fragile Categories",
        "",
        "| Scene | Category | LocAcc | mIoU | Samples |",
        "|---|---|---:|---:|---:|",
    ]
    for row in fragile:
        lines.append(
            f"| {row.scene} | {row.category} | {row.loc_acc:.3f} | {row.miou:.3f} | {row.n_samples} |"
        )

    lines.extend(["", "## Per-Scene Summary", "", "| Scene | Categories | Mean LocAcc | Mean mIoU |", "|---|---:|---:|---:|"])
    for scene, scene_rows in sorted(by_scene.items()):
        mean_loc = sum(row.loc_acc for row in scene_rows) / len(scene_rows)
        mean_miou = sum(row.miou for row in scene_rows) / len(scene_rows)
        lines.append(f"| {scene} | {len(scene_rows)} | {mean_loc:.3f} | {mean_miou:.3f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Most hard cases are categories where one feature-cell peak shift is enough "
            "to fail the LERF LocAcc metric.",
            "- Figurines contributes many fragile small-object categories, supporting a "
            "targeted small-object/feature-resolution analysis.",
            "- mIoU and LocAcc should be interpreted together: broader object regions can "
            "raise overlap while moving the argmax outside a small annotation mask.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(fragile: list[CategoryRow], path: Path, limit: int = 8) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Representative fragile LERF-OVS categories from the frozen rendered-feature evaluation. Low LocAcc with nonzero mIoU highlights the peak-vs-region trade-off.}",
        r"\label{tab:failure_analysis}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Scene & Category & LocAcc & mIoU \\",
        r"\midrule",
    ]
    for row in fragile[:limit]:
        category = row.category.replace("_", r"\_")
        scene = row.scene.replace("_", r"\_")
        lines.append(f"{scene} & {category} & {row.loc_acc:.2f} & {row.miou:.2f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = load_current_best_rows()
    fragile = select_fragile_rows(rows)
    write_markdown(rows, fragile, REPO_ROOT / "output/radio_gs/reports/lerf_failure_analysis.md")
    write_latex(fragile, REPO_ROOT / "paper/lerf_failure_analysis_table.tex")
    print("Wrote output/radio_gs/reports/lerf_failure_analysis.md")
    print("Wrote paper/lerf_failure_analysis_table.tex")


if __name__ == "__main__":
    main()
