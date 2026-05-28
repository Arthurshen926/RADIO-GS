#!/usr/bin/env python3
"""Summarize direct-3D refinement deltas by initial coarse-mask IoU buckets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("lt_025", 0.0, 0.25),
    ("025_050", 0.25, 0.50),
    ("050_075", 0.50, 0.75),
    ("gte_075", 0.75, float("inf")),
)


def _mean(items: Iterable[float]) -> float:
    values = list(items)
    return float(mean(values)) if values else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _scene_name(payload: dict[str, Any], path: Path) -> str:
    scene = payload.get("scene")
    if isinstance(scene, dict):
        value = scene.get("scene")
        if value:
            return str(value)
    if isinstance(scene, str):
        return scene
    return path.parent.name


def _iter_result_blocks(
    payload: dict[str, Any],
    *,
    selection: str,
) -> Iterable[tuple[str, dict[str, Any]]]:
    scene = payload.get("scene")
    if isinstance(scene, dict):
        results = scene.get("results", {})
        best = scene.get("best_by_miou")
    else:
        results = payload.get("results", {})
        best = payload.get("best_by_miou")
    if not isinstance(results, dict):
        return
    if selection == "all":
        for tag, block in results.items():
            if isinstance(block, dict):
                yield str(tag), block
        return
    tag = str(best) if selection == "best_by_miou" and best else selection
    block = results.get(tag)
    if isinstance(block, dict):
        yield tag, block


def _bucket_for(initial_iou: float) -> str:
    for name, low, high in BUCKETS:
        if low <= initial_iou < high:
            return name
    return BUCKETS[-1][0]


def _summarize_details(details: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in BUCKETS}
    skipped = 0
    for item in details:
        if "initial_iou" not in item or "iou" not in item:
            skipped += 1
            continue
        initial_iou = float(item.get("initial_iou", 0.0) or 0.0)
        buckets[_bucket_for(initial_iou)].append(item)

    out: dict[str, Any] = {}
    for name, items in buckets.items():
        accepted = [bool(item.get("sam3_accepted", False)) for item in items]
        out[name] = {
            "n": len(items),
            "initial_miou": _mean(float(item.get("initial_iou", 0.0) or 0.0) for item in items),
            "final_miou": _mean(float(item.get("iou", 0.0) or 0.0) for item in items),
            "delta_miou": _mean(
                float(item.get("delta_iou", float(item.get("iou", 0.0) or 0.0) - float(item.get("initial_iou", 0.0) or 0.0)) or 0.0)
                for item in items
            ),
            "initial_boundary_f": _mean(float(item.get("initial_boundary_f", 0.0) or 0.0) for item in items),
            "final_boundary_f": _mean(float(item.get("boundary_f", 0.0) or 0.0) for item in items),
            "delta_boundary_f": _mean(
                float(item.get("delta_boundary_f", float(item.get("boundary_f", 0.0) or 0.0) - float(item.get("initial_boundary_f", 0.0) or 0.0)) or 0.0)
                for item in items
            ),
            "initial_trimap_iou": _mean(float(item.get("initial_trimap_iou", 0.0) or 0.0) for item in items),
            "final_trimap_iou": _mean(float(item.get("trimap_iou", 0.0) or 0.0) for item in items),
            "delta_trimap_iou": _mean(
                float(item.get("delta_trimap_iou", float(item.get("trimap_iou", 0.0) or 0.0) - float(item.get("initial_trimap_iou", 0.0) or 0.0)) or 0.0)
                for item in items
            ),
            "sam_accept_rate": _mean(1.0 if value else 0.0 for value in accepted),
        }
    out["skipped_without_initial_iou"] = skipped
    out["n"] = sum(len(items) for items in buckets.values())
    return out


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Direct-3D Initial-IoU Bucket Diagnostic",
        "",
        "| Source | Scene | Selection | Bucket | n | Initial mIoU | Final mIoU | Delta | Initial BF | Final BF | Delta BF | SAM accept |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in summary["entries"]:
        for bucket, stats in entry["buckets"].items():
            if bucket in {"n", "skipped_without_initial_iou"}:
                continue
            lines.append(
                "| {source} | {scene} | {selection} | {bucket} | {n} | "
                "{initial_miou:.4f} | {final_miou:.4f} | {delta_miou:+.4f} | "
                "{initial_boundary_f:.4f} | {final_boundary_f:.4f} | "
                "{delta_boundary_f:+.4f} | {sam_accept_rate:.3f} |".format(
                    source=entry["source"],
                    scene=entry["scene"],
                    selection=entry["selection"],
                    bucket=bucket,
                    **stats,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="direct-3D result JSON files")
    parser.add_argument(
        "--selection",
        default="best_by_miou",
        help="'best_by_miou', 'all', or a concrete selection tag",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    args = parser.parse_args()

    entries: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []
    for raw_path in args.results:
        path = Path(raw_path)
        payload = _load_json(path)
        scene = _scene_name(payload, path)
        for selection, block in _iter_result_blocks(payload, selection=args.selection):
            details = list(block.get("query_details", []))
            all_details.extend(details)
            entries.append(
                {
                    "source": str(path),
                    "scene": scene,
                    "selection": selection,
                    "buckets": _summarize_details(details),
                }
            )

    summary = {
        "selection": args.selection,
        "entries": entries,
        "overall": _summarize_details(all_details),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(summary, output_md)
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
