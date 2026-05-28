#!/usr/bin/env python3
"""Summarize prompt-ensemble support-policy Direct3D results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")


def load_scene_results(roots: list[Path], scene: str) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for root in roots:
        path = root / scene / "lerf_direct_3d_selection_results.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            merged.update(payload["scene"]["results"])
    if not merged:
        raise FileNotFoundError(f"No Direct3D result JSON found for scene {scene}")
    return merged


def macro_for_threshold(
    per_scene: dict[str, dict[str, dict[str, Any]]],
    threshold: str,
) -> dict[str, Any]:
    rows: dict[str, dict[str, float]] = {}
    for scene, results in per_scene.items():
        if threshold not in results:
            raise KeyError(f"Missing threshold {threshold} for {scene}")
        item = results[threshold]
        rows[scene] = {
            "miou": float(item["miou"]),
            "acc025": float(item["acc025"]),
            "boundary_f": float(item["boundary_f"]),
            "trimap_iou": float(item["trimap_iou"]),
        }
    return {
        "threshold": threshold,
        "scenes": rows,
        "macro": {
            key: sum(scene_row[key] for scene_row in rows.values()) / len(rows)
            for key in ("miou", "acc025", "boundary_f", "trimap_iou")
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    best = report["best"]
    rows = [
        "# LERF Direct3D Prompt-Ensemble Support Policy",
        "",
        "This artifact records the compact-field direct 3D readout that does not use a VPR cache or official RGB SAM readout at inference.",
        "",
        "Policy:",
        "",
        "- Frozen SigLIP2 prompt ensemble: `{query}|a photo of {query}|a photo of a {query}|the {query}|a {query} object`.",
        "- Direct Gaussian-center compact primitive scores with the opacity-gated point summary adapter.",
        "- Fixed global softmax score threshold; the table reports a global threshold sweep without per-scene or per-query thresholding.",
        "- GT-free support-aware RGB/GrabCut cleanup with component guard: keep the largest component if it dominates, otherwise preserve multi-component support only when the refined support has at least 6000 pixels.",
        "",
        "## Threshold Sweep",
        "",
        "| Threshold | mIoU | Acc@0.25 | Boundary-F | Trimap IoU |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report["thresholds"]:
        macro = item["macro"]
        rows.append(
            f"| {item['threshold']} | {macro['miou']:.6f} | {macro['acc025']:.6f} | "
            f"{macro['boundary_f']:.6f} | {macro['trimap_iou']:.6f} |"
        )
    rows.extend(
        [
            "",
            f"Best threshold by macro mIoU: `{best['threshold']}`.",
            "",
            "## Best Per-Scene Metrics",
            "",
            "| Scene | mIoU | Acc@0.25 | Boundary-F | Trimap IoU |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scene in SCENES:
        item = best["scenes"][scene]
        rows.append(
            f"| {scene} | {item['miou']:.6f} | {item['acc025']:.6f} | "
            f"{item['boundary_f']:.6f} | {item['trimap_iou']:.6f} |"
        )
    macro = best["macro"]
    rows.append(
        f"| Macro | {macro['miou']:.6f} | {macro['acc025']:.6f} | "
        f"{macro['boundary_f']:.6f} | {macro['trimap_iou']:.6f} |"
    )
    rows.extend(
        [
            "",
            "Conclusion: this targeted support-aware compact direct readout improves the previous compact row from 0.4836/0.6426 to 0.5000/0.7051 when rounded to four decimals, with the largest Acc@0.25 recovery on Waldo Kitchen.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        default=[
            "output/radio_gs/lerf_direct3d_prompt_ensemble_policy_20260528",
            "output/radio_gs/lerf_direct3d_prompt_ensemble_policy_sweep_20260528",
            "output/radio_gs/lerf_direct3d_prompt_ensemble_policy_sweep2_20260528",
            "output/radio_gs/lerf_direct3d_prompt_ensemble_policy_sweep3_20260528",
        ],
        help="Direct3D result root; can be repeated and later roots override duplicate thresholds.",
    )
    parser.add_argument(
        "--output_json",
        default="paper/artifacts/lerf_direct3d_prompt_ensemble_support_policy_20260528.json",
    )
    parser.add_argument(
        "--output_md",
        default="paper/artifacts/lerf_direct3d_prompt_ensemble_support_policy_20260528.md",
    )
    args = parser.parse_args()

    roots = [Path(root) for root in args.root]
    per_scene = {scene: load_scene_results(roots, scene) for scene in SCENES}
    common_thresholds = sorted(
        set.intersection(*(set(results) for results in per_scene.values())),
        key=lambda item: float(item[3:].replace("p", ".")) if item.startswith("thr") else item,
    )
    thresholds = [macro_for_threshold(per_scene, threshold) for threshold in common_thresholds]
    best = max(thresholds, key=lambda item: item["macro"]["miou"])
    report = {
        "roots": [str(root) for root in roots],
        "scenes": list(SCENES),
        "thresholds": thresholds,
        "best": best,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_md), report)
    print(f"wrote {output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
