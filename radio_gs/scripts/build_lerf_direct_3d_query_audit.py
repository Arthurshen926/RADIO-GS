#!/usr/bin/env python3
"""Build a query-level audit for LERF direct 3D object selection."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from radio_gs.scripts.eval_lerf_direct_3d_selection import bootstrap_mean_ci


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENES = ["figurines", "ramen", "teatime", "waldo_kitchen"]
SCENE_LABELS = {
    "figurines": "Figurines",
    "ramen": "Ramen",
    "teatime": "Teatime",
    "waldo_kitchen": "Waldo Kitchen",
}


@dataclass(frozen=True)
class QueryRow:
    scene: str
    frame: str
    category: str
    iou: float
    pred_pixels: int
    gt_pixels: int
    overselect_ratio: float

    @property
    def zero_pred(self) -> bool:
        return self.pred_pixels == 0 and self.gt_pixels > 0


def load_query_rows(root: Path, scenes: Sequence[str], tag: str) -> list[QueryRow]:
    rows: list[QueryRow] = []
    for scene in scenes:
        path = root / scene / "lerf_direct_3d_selection_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing direct 3D result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("scene", {}).get("results", {})
        if tag not in results:
            raise KeyError(f"Selection tag {tag!r} not found in {path}")
        for item in results[tag].get("query_details", []):
            rows.append(
                QueryRow(
                    scene=SCENE_LABELS.get(scene, scene),
                    frame=str(item.get("frame", "")),
                    category=str(item.get("category", "")),
                    iou=float(item.get("iou", 0.0)),
                    pred_pixels=int(item.get("pred_pixels", 0)),
                    gt_pixels=int(item.get("gt_pixels", 0)),
                    overselect_ratio=float(item.get("overselect_ratio", 0.0)),
                )
            )
    return rows


def summarize_rows(rows: Sequence[QueryRow]) -> dict[str, float | int]:
    if not rows:
        return {
            "n": 0,
            "miou": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "acc025": 0.0,
            "zero_pred_rate": 0.0,
            "mean_overselect_ratio": 0.0,
        }
    ious = [row.iou for row in rows]
    ci = bootstrap_mean_ci(ious, num_samples=2000, seed=17)
    return {
        "n": len(rows),
        "miou": float(ci["mean"]),
        "ci_low": float(ci["ci_low"]),
        "ci_high": float(ci["ci_high"]),
        "acc025": sum(row.iou > 0.25 for row in rows) / len(rows),
        "zero_pred_rate": sum(row.zero_pred for row in rows) / len(rows),
        "mean_overselect_ratio": sum(row.overselect_ratio for row in rows) / len(rows),
    }


def group_by_scene(rows: Iterable[QueryRow]) -> dict[str, list[QueryRow]]:
    grouped: dict[str, list[QueryRow]] = {}
    for row in rows:
        grouped.setdefault(row.scene, []).append(row)
    return grouped


def worst_rows(rows: Sequence[QueryRow], limit: int) -> list[QueryRow]:
    return sorted(
        rows,
        key=lambda row: (row.iou, row.zero_pred is False, -row.gt_pixels, row.scene, row.category),
    )[:limit]


def write_markdown(rows: Sequence[QueryRow], tag: str, root: Path, path: Path) -> None:
    grouped = group_by_scene(rows)
    lines = [
        "# LERF Direct 3D Query-Level Audit",
        "",
        f"- Result root: `{root}`",
        f"- Selection tag: `{tag}`",
        "- Purpose: expose query-level uncertainty, zero-prediction failures, and over-selection failures for the VPR direct 3D readout.",
        "",
        "## Per-Scene Summary",
        "",
        "| Scene | Queries | mIoU | 95% bootstrap CI | Acc@0.25 | Zero-pred rate | Mean overselect |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scene in sorted(grouped):
        summary = summarize_rows(grouped[scene])
        lines.append(
            f"| {scene} | {summary['n']} | {summary['miou']:.4f} | "
            f"[{summary['ci_low']:.4f}, {summary['ci_high']:.4f}] | "
            f"{summary['acc025']:.4f} | {summary['zero_pred_rate']:.4f} | "
            f"{summary['mean_overselect_ratio']:.2f} |"
        )
    macro = summarize_rows(rows)
    lines.append(
        f"| Macro/query pool | {macro['n']} | {macro['miou']:.4f} | "
        f"[{macro['ci_low']:.4f}, {macro['ci_high']:.4f}] | "
        f"{macro['acc025']:.4f} | {macro['zero_pred_rate']:.4f} | "
        f"{macro['mean_overselect_ratio']:.2f} |"
    )

    lines.extend(
        [
            "",
            "## Worst Queries",
            "",
            "| Scene | Frame | Category | IoU | Pred px | GT px | Overselect |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in worst_rows(rows, limit=16):
        lines.append(
            f"| {row.scene} | {row.frame} | {row.category} | {row.iou:.4f} | "
            f"{row.pred_pixels} | {row.gt_pixels} | {row.overselect_ratio:.2f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: zero-prediction rows usually indicate that selected primitives are not visible in the annotated view after rendering, while high overselect ratios indicate clutter/background leakage. These are the two dominant failure modes to discuss for Waldo Kitchen.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(rows: Sequence[QueryRow], tag: str, path: Path) -> None:
    grouped = group_by_scene(rows)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Query-level audit for LERF direct 3D object selection. Confidence intervals are bootstrap intervals over query masks for the fixed " + tag.replace("_", r"\_") + r" selector.}",
        r"  \label{tab:direct3d_query_audit}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Scene & Queries & mIoU & Acc@0.25 & Zero pred. \\",
        r"    \midrule",
    ]
    for scene in sorted(grouped):
        summary = summarize_rows(grouped[scene])
        lines.append(
            f"    {scene} & {summary['n']} & {summary['miou']:.3f} & "
            f"{summary['acc025']:.3f} & {summary['zero_pred_rate']:.3f} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="output/radio_gs/lerf_direct_3d_selection_meanstd_20260513")
    parser.add_argument("--tag", default="meanstd2p5")
    parser.add_argument("--output_md", default="output/radio_gs/reports/lerf_direct_3d_query_audit.md")
    parser.add_argument("--output_tex", default="paper/lerf_direct_3d_query_audit_table.tex")
    args = parser.parse_args()

    root = Path(args.results_root)
    rows = load_query_rows(root, SCENES, args.tag)
    write_markdown(rows, args.tag, root, Path(args.output_md))
    write_latex(rows, args.tag, Path(args.output_tex))
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_tex}")


if __name__ == "__main__":
    main()
