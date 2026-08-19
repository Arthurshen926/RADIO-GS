#!/usr/bin/env python3
"""Verify then score a sealed NVOS box-plus-point consensus batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from radio_gs.evaluation.promptable_segmentation import evaluate_manifest
from radio_gs.scripts.fuse_nvos_box_point_consensus import CANDIDATE_ID, _resolve_prediction
from radio_gs.scripts.predict_nvos_method_v1_field_box_sam3 import _write_json
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import _sha256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--expected-prediction-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest_path = Path(args.prediction_manifest).expanduser().resolve(strict=True)
    if _sha256(manifest_path) != args.expected_prediction_manifest_sha256:
        raise ValueError("consensus prediction manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = [str(value) for value in manifest.get("scene_order", [])]
    if (
        manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("kind")
        != "promptable_nvs_method_v1_box_point_consensus_predictions"
        or len(scenes) != 8
        or manifest.get("all_eight_predictions_sealed") is not True
        or manifest.get("evaluation_performed") is not False
        or manifest.get("target_mask_opened") is not False
        or manifest.get("target_metric_opened") is not False
    ):
        raise ValueError("consensus pre-GT barrier differs")
    for scene in scenes:
        frames = list(manifest["predictions"][scene])
        if len(frames) != 1:
            raise ValueError(f"consensus target frame differs: {scene}")
        _resolve_prediction(manifest, manifest_path, scene, str(frames[0]))
    evaluation = evaluate_manifest(
        Path(args.dataset_manifest).expanduser().resolve(strict=True),
        prediction_manifest=manifest_path,
    )
    result = {
        "schema_version": 1,
        "artifact_type": "radio_gs_nvos_box_point_consensus_result",
        "candidate_id": CANDIDATE_ID,
        "pre_gt_barrier": {
            "prediction_manifest": str(manifest_path),
            "prediction_manifest_sha256": args.expected_prediction_manifest_sha256,
            "all_eight_predictions_verified_before_first_target_mask_open": True,
        },
        "evaluation": evaluation,
        "eligibility": {
            "development_use_only": True,
            "target_rgb_assisted": True,
            "strict_unseen_protocol_eligible": False,
        },
    }
    output = Path(args.output).expanduser().resolve()
    _write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "foreground_iou": evaluation["dataset"]["foreground_iou"],
                "pixel_accuracy": evaluation["dataset"]["pixel_accuracy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
