"""Select a deployment coverage margin using source-dev masks only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = []
    for name in args.candidate:
        path = Path(name).resolve(strict=True)
        with path.open() as handle:
            report = json.load(handle)
        if (
            report.get("schema") != "radio_gs.sugm_v3.image_query_source_dev.v1"
            or not report.get("source_only")
            or report.get("target_rgb_opened")
            or report.get("benchmark_metrics_opened")
            or report.get("evaluation_residue") != 3
        ):
            raise ValueError("margin selection input is not source-dev authority")
        reports.append((path, report))
    scenes = {report["scene"] for _, report in reports}
    if len(scenes) != 1:
        raise ValueError("margin candidates cross scene boundaries")
    # IoU is the primary source capability. On effectively tied candidates,
    # retain the smaller intervention instead of enforcing a brittle gate.
    best_iou = max(float(report["metrics"]["mask_iou"]) for _, report in reports)
    eligible = [
        item for item in reports
        if float(item[1]["metrics"]["mask_iou"]) >= best_iou - 1e-4
    ]
    selected_path, selected = min(
        eligible, key=lambda item: abs(float(item[1].get("membership_margin", 0.0)))
    )
    payload = {
        "schema": "radio_gs.sugm_v3.source_membership_margin_selection.v1",
        "scene": selected["scene"],
        "selected_margin": float(selected.get("membership_margin", 0.0)),
        "selected_source_dev_iou": float(selected["metrics"]["mask_iou"]),
        "selection_rule": "maximum_source_dev_iou_with_minimum_intervention_tiebreak",
        "hard_gate": False,
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "candidates": [
            {
                "margin": float(report.get("membership_margin", 0.0)),
                "mask_iou": float(report["metrics"]["mask_iou"]),
                "unknown_fp_mass": float(report["metrics"]["unknown_fp_mass"]),
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path, report in reports
        ],
        "selected_candidate": {
            "path": str(selected_path), "sha256": sha256_file(selected_path)
        },
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()
