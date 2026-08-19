#!/usr/bin/env python3
"""Aggregate the fixed scene0000-selected ScanNet instance-topology transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import DEFAULT_SCENES


def _load_scene(root: Path, scene: str, subdir: str = "") -> dict[str, object]:
    path = root / scene / subdir / "scannet_vala_gaussian_protocol_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["scenes"][scene]


def _macro(rows: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    return {
        split: {
            metric: float(
                np.mean(
                    [
                        float(row["vala_pseudo_volume"][split][metric])
                        for row in rows.values()
                    ]
                )
            )
            for metric in ("miou", "macc")
        }
        for split in ("19", "15", "10")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-subdir", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = {scene: _load_scene(args.candidate_root, scene) for scene in DEFAULT_SCENES}
    baseline = {
        scene: _load_scene(args.baseline_root, scene, args.baseline_subdir)
        for scene in DEFAULT_SCENES
    }
    candidate_macro = _macro(candidate)
    baseline_macro = _macro(baseline)
    deltas = {
        split: {
            metric: candidate_macro[split][metric] - baseline_macro[split][metric]
            for metric in ("miou", "macc")
        }
        for split in ("19", "15", "10")
    }
    transfer_scenes = tuple(scene for scene in DEFAULT_SCENES if scene != "scene0000_00")
    transfer_delta = {
        split: {
            metric: float(
                np.mean(
                    [
                        float(candidate[s]["vala_pseudo_volume"][split][metric])
                        - float(baseline[s]["vala_pseudo_volume"][split][metric])
                        for s in transfer_scenes
                    ]
                )
            )
            for metric in ("miou", "macc")
        }
        for split in ("19", "15", "10")
    }
    report = {
        "artifact_type": "radio_gs_scannet_instance_topology_full8_summary",
        "selection_scene": "scene0000_00",
        "transfer_scenes": list(transfer_scenes),
        "candidate_root": str(args.candidate_root.resolve()),
        "baseline_root": str(args.baseline_root.resolve()),
        "candidate_macro": candidate_macro,
        "baseline_macro": baseline_macro,
        "delta": deltas,
        "transfer_seven_mean_delta": transfer_delta,
        "all_splits_positive": all(
            deltas[split][metric] > 0
            for split in ("19", "15", "10")
            for metric in ("miou", "macc")
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
