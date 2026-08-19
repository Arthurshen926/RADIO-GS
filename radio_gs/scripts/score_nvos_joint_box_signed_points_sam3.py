#!/usr/bin/env python3
"""Verify the full8 pre-GT barrier, then score joint box-point SAM3 masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from radio_gs.evaluation.promptable_segmentation import evaluate_manifest
from radio_gs.scripts.fuse_nvos_box_point_consensus import _resolve_prediction
from radio_gs.scripts.predict_nvos_joint_box_signed_points_sam3 import CANDIDATE_ID
from radio_gs.scripts.predict_nvos_method_v1_field_box_sam3 import _write_json
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import _sha256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--expected-prediction-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    path = Path(args.prediction_manifest).expanduser().resolve(strict=True)
    if _sha256(path) != args.expected_prediction_manifest_sha256:
        raise ValueError("joint box-point prediction manifest SHA-256 differs")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    scenes = [str(value) for value in manifest.get("scene_order", [])]
    if (
        manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("kind")
        != "promptable_nvs_method_v1_joint_box_signed_points_sam3_predictions"
        or len(scenes) != 8
        or manifest.get("all_eight_predictions_sealed") is not True
        or manifest.get("evaluation_performed") is not False
        or manifest.get("target_mask_opened") is not False
        or manifest.get("target_metric_opened") is not False
    ):
        raise ValueError("joint box-point pre-GT barrier differs")
    for scene in scenes:
        frames = list(manifest["predictions"][scene])
        if len(frames) != 1:
            raise ValueError(f"joint box-point target frame differs: {scene}")
        _resolve_prediction(manifest, path, scene, str(frames[0]))
    evaluation = evaluate_manifest(
        Path(args.dataset_manifest).expanduser().resolve(strict=True),
        prediction_manifest=path,
    )
    parent_macro = 0.8177621399957551
    parent_scenes = {
        "fern": 0.8304732612835244,
        "flower": 0.9680415254004953,
        "fortress": 0.9754431952966958,
        "horns_center": 0.6968090986196901,
        "horns_left": 0.6898833651238061,
        "leaves": 0.6397007365739295,
        "orchids": 0.877451185671343,
        "trex": 0.8642947519965567,
    }
    scene_scores = {
        str(row["scene_id"]): float(row["foreground_iou"])
        for row in evaluation["scenes"]
    }
    deltas = {
        scene: scene_scores[scene] - parent_scenes[scene] for scene in parent_scenes
    }
    macro = float(evaluation["dataset"]["foreground_iou"])
    promotion = macro > parent_macro and min(deltas.values()) >= -0.02
    result = {
        "schema_version": 1,
        "artifact_type": "radio_gs_nvos_joint_box_signed_points_sam3_result",
        "candidate_id": CANDIDATE_ID,
        "pre_gt_barrier": {
            "prediction_manifest": str(path),
            "prediction_manifest_sha256": args.expected_prediction_manifest_sha256,
            "all_eight_predictions_verified_before_first_target_mask_open": True,
        },
        "evaluation": evaluation,
        "comparison": {
            "parent_macro_foreground_iou": parent_macro,
            "candidate_macro_foreground_iou": macro,
            "delta_macro_foreground_iou": macro - parent_macro,
            "scene_iou_delta": deltas,
        },
        "promotion_gate": {
            "passed": promotion,
            "macro_strictly_improved": macro > parent_macro,
            "maximum_per_scene_regression": 0.02,
            "observed_worst_scene_delta": min(deltas.values()),
        },
        "eligibility": {
            "development_use_only": True,
            "target_rgb_assisted": True,
            "target_mask_used_for_prediction": False,
        },
    }
    output = Path(args.output).expanduser().resolve()
    _write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "foreground_iou": macro,
                "delta": macro - parent_macro,
                "promotion": promotion,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
