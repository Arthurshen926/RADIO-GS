#!/usr/bin/env python3
"""Build LERF per-query breakdown diagnostics from direct-3D result JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_INPUTS = (
    Path("paper/artifacts/lerf_direct_3d_selection_figurines_results.json"),
    Path("paper/artifacts/lerf_direct_3d_selection_ramen_results.json"),
    Path("paper/artifacts/lerf_direct_3d_selection_teatime_results.json"),
    Path("paper/artifacts/lerf_direct_3d_selection_waldo_kitchen_results.json"),
)
DEFAULT_OUTPUT_JSON = Path("paper/artifacts/lerf_query_breakdown.json")
DEFAULT_OUTPUT_MD = Path("paper/artifacts/lerf_query_breakdown.md")

FOOTPRINT_BINS = (
    ("tiny", 0.0, 0.01),
    ("small", 0.01, 0.05),
    ("medium", 0.05, 0.20),
    ("large", 0.20, float("inf")),
)

LABEL_GROUP_KEYWORDS = {
    "reflective_or_transparent": ("glass", "stainless steel", "refrigerator", "sink"),
    "texture_like": ("noodle", "nori", "napkin", "dall-e brand", "paper", "cookie"),
    "container_or_part": (
        "bag",
        "bowl",
        "cabinet",
        "cup",
        "door handle",
        "hand",
        "hooves",
        "mug",
        "nose",
        "plate",
        "pot",
        "sink",
        "vessel",
    ),
    "multi_instance_likely": ("cookies", "onion segments", "chopsticks", "pots"),
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate_above(values: list[float], threshold: float) -> float:
    return sum(1 for value in values if value > threshold) / len(values) if values else 0.0


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ious = [float(row["iou"]) for row in rows]
    return {
        "count": len(rows),
        "miou": _mean(ious),
        "acc025": _rate_above(ious, 0.25),
        "acc05": _rate_above(ious, 0.5),
        "mean_gt_pixels": _mean([float(row.get("gt_pixels", 0.0)) for row in rows]),
        "mean_selected_gaussians": _mean([float(row.get("selected_gaussians", 0.0)) for row in rows]),
    }


def _footprint_bin(gt_pixels: float, image_area: float) -> str:
    ratio = gt_pixels / image_area if image_area > 0 else 0.0
    for name, low, high in FOOTPRINT_BINS:
        if ratio >= low and ratio <= high:
            return name
    return "large"


def _label_groups(label: str) -> list[str]:
    normalized = label.lower()
    groups = [
        group
        for group, keywords in LABEL_GROUP_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    ]
    return groups or ["other"]


def _load_query_rows(path: Path, selection: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scene_payload = payload["scene"]
    scene = scene_payload["scene"]
    image_area = float(scene_payload["image_height"]) * float(scene_payload["image_width"])
    details = scene_payload["results"][selection]["query_details"]
    rows: list[dict[str, Any]] = []
    for item in details:
        gt_pixels = float(item.get("gt_pixels", 0.0))
        category = str(item["category"])
        enriched = dict(item)
        enriched["scene"] = scene
        enriched["footprint_ratio"] = gt_pixels / image_area if image_area > 0 else 0.0
        enriched["footprint_bin"] = _footprint_bin(gt_pixels, image_area)
        enriched["label_groups"] = _label_groups(category)
        rows.append(enriched)
    return rows


def _bucket(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(row)
    return {name: _stats(items) for name, items in sorted(buckets.items())}


def _label_group_buckets(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for group in row["label_groups"]:
            buckets.setdefault(group, []).append(row)
    return {name: _stats(items) for name, items in sorted(buckets.items())}


def _scene_mean_stats(scene_breakdown: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scene_stats = list(scene_breakdown.values())
    return {
        "scene_count": len(scene_stats),
        "query_count": sum(int(stats["count"]) for stats in scene_stats),
        "miou": _mean([float(stats["miou"]) for stats in scene_stats]),
        "acc025": _mean([float(stats["acc025"]) for stats in scene_stats]),
        "acc05": _mean([float(stats["acc05"]) for stats in scene_stats]),
        "mean_gt_pixels": _mean([float(stats["mean_gt_pixels"]) for stats in scene_stats]),
        "mean_selected_gaussians": _mean([float(stats["mean_selected_gaussians"]) for stats in scene_stats]),
    }


def build_summary(inputs: Sequence[str | Path], *, selection: str = "thr0p25") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in inputs:
        rows.extend(_load_query_rows(Path(path), selection))
    scene_breakdown = _bucket(rows, "scene")
    return {
        "selection": selection,
        "inputs": [str(path) for path in inputs],
        "object_weighted": _stats(rows),
        "scene_mean": _scene_mean_stats(scene_breakdown),
        "scene_breakdown": scene_breakdown,
        "footprint_bins": {name: _stats([row for row in rows if row["footprint_bin"] == name]) for name, _, _ in FOOTPRINT_BINS},
        "label_groups": _label_group_buckets(rows),
        "queries": rows,
        "caveat": (
            "Object-weighted metrics are diagnostic query-level aggregates. "
            "Scene-mean metrics match the paper-facing four-scene aggregate. "
            "Label groups are deterministic keyword diagnostics for appendix analysis; "
            "only footprint bins are purely geometric."
        ),
    }


def _fmt(value: object) -> str:
    return f"{float(value):.4f}"


def _append_table(lines: list[str], title: str, rows: dict[str, dict[str, Any]]) -> None:
    lines.extend([f"## {title}", "", "| Group | Count | mIoU | Acc@0.25 | Acc@0.5 | Mean GT px | Mean selected Gaussians |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, stats in rows.items():
        lines.append(
            "| {name} | {count} | {miou} | {acc025} | {acc05} | {gt} | {selected} |".format(
                name=name,
                count=stats["count"],
                miou=_fmt(stats["miou"]),
                acc025=_fmt(stats["acc025"]),
                acc05=_fmt(stats["acc05"]),
                gt=_fmt(stats["mean_gt_pixels"]),
                selected=_fmt(stats["mean_selected_gaussians"]),
            )
        )
    lines.append("")


def render_markdown(summary: dict[str, Any]) -> str:
    object_weighted = summary["object_weighted"]
    scene_mean = summary["scene_mean"]
    lines = [
        "# LERF Query Breakdown",
        "",
        f"- selection: `{summary['selection']}`",
        f"- queries: {object_weighted['count']}",
        f"- object-weighted mIoU: {_fmt(object_weighted['miou'])}",
        f"- object-weighted Acc@0.25: {_fmt(object_weighted['acc025'])}",
        f"- scene-mean mIoU: {_fmt(scene_mean['miou'])}",
        f"- scene-mean Acc@0.25: {_fmt(scene_mean['acc025'])}",
        f"- caveat: {summary['caveat']}",
        "",
    ]
    _append_table(lines, "Scene Breakdown", summary["scene_breakdown"])
    _append_table(lines, "Footprint Bins", summary["footprint_bins"])
    _append_table(lines, "Label Groups", summary["label_groups"])
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--selection", default="thr0p25")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_summary(args.inputs, selection=args.selection)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
