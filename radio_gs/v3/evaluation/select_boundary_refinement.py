"""Select the boundary arm after corrected D16-head source-dev evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radio_gs.utils.immutable_artifacts import write_frozen_json
from radio_gs.v3.training.instance_upper_bound import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", nargs=3, metavar=("SCENE", "ARM", "PATH"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports: dict[str, dict[str, dict]] = {}
    inputs = []
    for scene, arm, value in args.report:
        path = Path(value).resolve(strict=True)
        report = json.loads(path.read_text())
        reports.setdefault(scene, {})[arm] = report
        inputs.append({"scene": scene, "arm": arm, "path": str(path), "sha256": sha256_file(path)})
    scenes = sorted(reports)
    arms = sorted(next(iter(reports.values())))
    summary = {}
    for arm in arms:
        values = [reports[scene][arm] for scene in scenes]
        eligible = all(
            value["nondegeneracy"]["pass"]
            and value["candidate_metrics"]["boundary_f"] > value["baseline_metrics"]["boundary_f"]
            and all(value["protected_block_max_abs_delta"][name] == 0.0 for name in ("shared", "semantic", "instance"))
            for value in values
        )
        summary[arm] = {
            "per_scene_boundary_f": {
                scene: reports[scene][arm]["candidate_metrics"]["boundary_f"]
                for scene in scenes
            },
            "macro_boundary_f": sum(value["candidate_metrics"]["boundary_f"] for value in values) / len(values),
            "eligible": eligible,
        }
    eligible = [arm for arm in arms if summary[arm]["eligible"]]
    if not eligible:
        raise ValueError("no boundary refinement arm passes source-dev gates")
    selected = max(eligible, key=lambda arm: summary[arm]["macro_boundary_f"])
    payload = {
        "schema": "radio_gs.sugm_v3.boundary_refinement_selection.v1",
        "scenes": scenes,
        "arms": summary,
        "selected": selected,
        "selection_metric": "scene_macro_corrected_boundary_f",
        "instance_architecture_fixed": "shared_core_low_rank_private_instance_branch",
        "source_only": True,
        "target_rgb_opened": False,
        "benchmark_metrics_opened": False,
        "inputs": inputs,
        "status": "development_evidence_only",
    }
    write_frozen_json(Path(args.output).resolve(), payload)
    print(payload)


if __name__ == "__main__":
    main()
