#!/usr/bin/env python3
"""Score a fully sealed NVOS SAM3-video reliability-gate batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask, resize_mask_nearest
from radio_gs.scripts.materialize_nvos_ludvig_region_reliability_gate import _sha256, _write_json


def score(args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = Path(args.prediction_manifest).resolve(strict=True)
    dataset_path = Path(args.dataset_manifest).resolve(strict=True)
    if _sha256(prediction_path) != args.expected_prediction_manifest_sha256:
        raise ValueError("prediction manifest hash differs")
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    scenes = [str(value) for value in prediction["scene_order"]]
    if (
        len(scenes) != 8
        or prediction.get("all_eight_predictions_sealed") is not True
        or prediction.get("target_mask_opened") is not False
        or prediction.get("target_metric_opened") is not False
    ):
        raise ValueError("prediction batch is not sealed full8")
    # Verify and load every prediction before resolving any target mask.
    loaded: dict[str, tuple[str, np.ndarray]] = {}
    for scene in scenes:
        frame, raw_path = next(iter(prediction["predictions"][scene].items()))
        path = Path(raw_path).resolve(strict=True)
        if _sha256(path) != prediction["prediction_sha256"][scene][frame]:
            raise ValueError(f"{scene}: prediction hash differs")
        loaded[scene] = (frame, np.load(path, allow_pickle=False))
    dataset_rows = {str(row["scene_id"]): row for row in dataset["scenes"]}
    per_scene: dict[str, Any] = {}
    for scene in scenes:
        frame, margin = loaded[scene]
        row = dataset_rows[scene]
        frame_row = next(item for item in row["frames"] if str(item["frame_id"]) == frame)
        target_path = Path(frame_row["ground_truth"]).resolve(strict=True)
        if _sha256(target_path) != frame_row["ground_truth_sha256"]:
            raise ValueError(f"{scene}: target hash differs")
        target = load_ground_truth_mask(target_path).astype(bool)
        candidate = resize_mask_nearest(margin >= 0.0, target.shape).astype(bool)
        union = int(np.logical_or(candidate, target).sum())
        intersection = int(np.logical_and(candidate, target).sum())
        per_scene[scene] = {
            "foreground_iou": float(intersection / union) if union else 1.0,
            "pixel_accuracy": float((candidate == target).mean()),
        }
    result = {
        "schema_version": 1,
        "artifact_type": "nvos_sam3_prompt_proposal_video_reliability_gate_result_v1",
        "scene_order": scenes,
        "per_scene": per_scene,
        "macro_foreground_iou": float(np.mean([row["foreground_iou"] for row in per_scene.values()])),
        "macro_pixel_accuracy": float(np.mean([row["pixel_accuracy"] for row in per_scene.values()])),
        "development_only": True,
        "sota_claim": False,
    }
    _write_json(Path(args.output).resolve(), result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--expected-prediction-manifest-sha256", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(score(parser.parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
