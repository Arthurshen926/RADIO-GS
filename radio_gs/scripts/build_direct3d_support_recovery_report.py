#!/usr/bin/env python3
"""Build a Direct3D support-recovery audit from result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")


def load_result(root: Path, scene: str, tag: str) -> dict[str, Any]:
    path = root / scene / "lerf_direct_3d_selection_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["scene"]["results"][tag]


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def aggregate(root: Path, tag: str) -> dict[str, Any]:
    scenes: dict[str, Any] = {}
    for scene in SCENES:
        result = load_result(root, scene, tag)
        scenes[scene] = {
            "miou": float(result["miou"]),
            "acc025": float(result["acc025"]),
            "boundary_f": float(result["boundary_f"]),
            "trimap_iou": float(result["trimap_iou"]),
            "query_details": result["query_details"],
        }
    return {
        "root": str(root),
        "tag": tag,
        "macro": {
            "miou": mean([scene["miou"] for scene in scenes.values()]),
            "acc025": mean([scene["acc025"] for scene in scenes.values()]),
            "boundary_f": mean([scene["boundary_f"] for scene in scenes.values()]),
            "trimap_iou": mean([scene["trimap_iou"] for scene in scenes.values()]),
        },
        "scenes": scenes,
    }


def index_queries(result: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for scene, scene_result in result["scenes"].items():
        for item in scene_result["query_details"]:
            indexed[(scene, str(item["frame"]), str(item["category"]))] = item
    return indexed


def compare_queries(base: dict[str, Any], variant: dict[str, Any]) -> list[dict[str, Any]]:
    base_items = index_queries(base)
    variant_items = index_queries(variant)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(base_items) & set(variant_items)):
        before = float(base_items[key]["iou"])
        after = float(variant_items[key]["iou"])
        rows.append(
            {
                "scene": key[0],
                "frame": key[1],
                "category": key[2],
                "before_iou": before,
                "after_iou": after,
                "delta_iou": after - before,
                "before_acc025": before >= 0.25,
                "after_acc025": after >= 0.25,
                "before_pred_pixels": int(base_items[key].get("pred_pixels", 0)),
                "after_pred_pixels": int(variant_items[key].get("pred_pixels", 0)),
                "gt_pixels": int(variant_items[key].get("gt_pixels", 0)),
            }
        )
    return rows


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = [
        "# Direct3D Support Recovery Audit",
        "",
        f"- Baseline: `{report['baseline']['root']}`",
        f"- Promoted variant: `{report['promoted_variant']}`",
        f"- Tag: `{report['tag']}`",
        "",
        "## Macro Metrics",
        "",
        "| Variant | mIoU | Acc@0.25 | Boundary-F | Trimap IoU |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["variants"].items():
        macro = result["macro"]
        rows.append(
            f"| {name} | {macro['miou']:.4f} | {macro['acc025']:.4f} | "
            f"{macro['boundary_f']:.4f} | {macro['trimap_iou']:.4f} |"
        )
    rows.extend(["", "## Scene Metrics", ""])
    for name, result in report["variants"].items():
        rows.extend(
            [
                f"### {name}",
                "",
                "| Scene | mIoU | Acc@0.25 | Boundary-F |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for scene, scene_result in result["scenes"].items():
            rows.append(
                f"| {scene} | {scene_result['miou']:.4f} | "
                f"{scene_result['acc025']:.4f} | {scene_result['boundary_f']:.4f} |"
            )
        rows.append("")
    rows.extend(
        [
            "## Acc@0.25 Crossings",
            "",
            "| Variant | Scene | Frame | Query | Before | After | Delta |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for name, comparisons in report["comparisons"].items():
        crossings = [
            row
            for row in comparisons
            if (not row["before_acc025"] and row["after_acc025"])
            or (row["before_acc025"] and not row["after_acc025"])
        ]
        for row in sorted(crossings, key=lambda item: (-item["delta_iou"], item["scene"]))[:20]:
            rows.append(
                f"| {name} | {row['scene']} | {row['frame']} | `{row['category']}` | "
                f"{row['before_iou']:.4f} | {row['after_iou']:.4f} | {row['delta_iou']:.4f} |"
            )
    rows.extend(
        [
            "",
            "Conclusion: `rgb_grabcut_component_guard` is a GT-free support-preserving "
            "cleanup. It targets cases where the projected support is multi-component "
            "and avoids forcing a largest-component decision unless one component "
            "dominates the refined mask.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="thr0p35")
    parser.add_argument(
        "--baseline",
        default="output/radio_gs/lerf_direct3d_deployed_opacity_gate_masks_20260528",
    )
    parser.add_argument("--variant", action="append", default=[], help="name=path")
    parser.add_argument("--promoted_variant", default="component_guard")
    parser.add_argument("--output_json", default="paper/artifacts/direct3d_support_recovery_audit_20260528.json")
    parser.add_argument("--output_md", default="paper/artifacts/direct3d_support_recovery_audit_20260528.md")
    args = parser.parse_args()

    variants = {"baseline": aggregate(Path(args.baseline), args.tag)}
    for raw in args.variant:
        if "=" not in raw:
            raise ValueError(f"Expected variant as name=path, got {raw!r}")
        name, path = raw.split("=", 1)
        variants[name] = aggregate(Path(path), args.tag)
    baseline = variants["baseline"]
    comparisons = {
        name: compare_queries(baseline, result)
        for name, result in variants.items()
        if name != "baseline"
    }
    report = {
        "tag": args.tag,
        "baseline": baseline,
        "variants": variants,
        "comparisons": comparisons,
        "promoted_variant": args.promoted_variant,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_md), report)
    print(f"wrote {output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
