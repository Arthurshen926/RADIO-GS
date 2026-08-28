"""Select a SUGM-v3 private architecture from fixed source-dev reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import sha256_file


METRICS = ("mask_iou", "brier", "boundary_f", "unknown_fp_mass")


def summarize(reports: dict[str, dict[str, dict]]) -> dict[str, object]:
    arms = sorted(next(iter(reports.values())))
    scenes = sorted(reports)
    summaries = {}
    for arm in arms:
        values = [reports[scene][arm] for scene in scenes]
        for report in values:
            protected = report["protected_block_max_abs_delta"]
            if protected["shared"] != 0.0 or protected["semantic"] != 0.0:
                raise ValueError("private candidate rewrote a protected capability block")
        per_scene = {scene: reports[scene][arm]["delta"] for scene in scenes}
        summaries[arm] = {
            "per_scene_delta": per_scene,
            "macro_delta": {
                metric: sum(value["delta"][metric] for value in values) / len(values)
                for metric in METRICS
            },
            "all_scene_iou_gate_pass": all(
                value["delta"]["mask_iou"] >= 0.05 for value in values
            ),
            "all_scene_brier_improved": all(value["delta"]["brier"] < 0 for value in values),
            "all_scene_boundary_improved": all(
                value["delta"]["boundary_f"] > 0 for value in values
            ),
            "all_scene_unknown_mass_not_increased": all(
                value["delta"]["unknown_fp_mass"] <= 0 for value in values
            ),
        }
    eligible = [
        arm for arm, value in summaries.items()
        if all(value[name] for name in (
            "all_scene_iou_gate_pass",
            "all_scene_brier_improved",
            "all_scene_boundary_improved",
            "all_scene_unknown_mass_not_increased",
        ))
    ]
    if len(eligible) != 1:
        raise ValueError("source-dev private architecture gate is not uniquely resolved")
    return {"scenes": scenes, "arms": summaries, "selected": eligible[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", nargs=3, metavar=("SCENE", "ARM", "PATH"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports: dict[str, dict[str, dict]] = {}
    inputs = []
    for scene, arm, value in args.report:
        path = Path(value).resolve(strict=True)
        reports.setdefault(scene, {})[arm] = json.loads(path.read_text())
        inputs.append({"scene": scene, "arm": arm, "path": str(path), "sha256": sha256_file(path)})
    result = summarize(reports)
    payload = {
        "schema": "radio_gs.sugm_v3.private_architecture_selection.v1",
        **result,
        "gate": {
            "per_scene_mask_iou_delta_minimum": 0.05,
            "brier_must_improve": True,
            "boundary_f_must_improve": True,
            "unknown_fp_mass_must_not_increase": True,
            "shared_and_semantic_max_abs_delta": 0.0,
        },
        "inputs": inputs,
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "status": "development_evidence_only",
    }
    output = Path(args.output).resolve()
    write_frozen_json(output, payload)
    print(payload)


if __name__ == "__main__":
    main()
