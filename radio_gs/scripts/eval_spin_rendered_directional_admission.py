#!/usr/bin/env python3
"""Score one sealed rendered directional-admission receipt under frozen SPIn GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask
from radio_gs.scripts.build_spin_source_footprint_quantile_oof import (
    file_sha256,
    json_sha256,
)
from radio_gs.scripts.materialize_spin_rendered_directional_admission import (
    PREDICTION_RECEIPT_TYPE,
    _array,
    _load_json_authority,
    _require_file,
    build_candidate_frame,
)
from radio_gs.querying.source_oof_transport_admission import (
    DirectionalAdmissionCalibration,
)


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return float(intersection / union) if union else 1.0


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    receipt_path = _require_file(
        args.prediction_receipt,
        args.prediction_receipt_sha256,
        "sealed directional-admission receipt",
    )
    receipt = _load_json_authority(receipt_path)
    if receipt.get("artifact_type") != PREDICTION_RECEIPT_TYPE or receipt.get(
        "sealed_before_target_ground_truth_open"
    ) is not True:
        raise ValueError("directional-admission receipt is not pre-metric")
    if any(
        receipt.get(key) is not False
        for key in ("target_rgb_opened", "target_mask_opened", "target_metric_computed")
    ):
        raise ValueError("directional-admission receipt violates pre-metric safety")
    values = receipt["calibration"]
    calibration = DirectionalAdmissionCalibration(
        expansion=float(values["expansion"]),
        contraction=float(values["contraction"]),
        leave_one_fold_expansion=tuple(float(v) for v in values["leave_one_fold_expansion"]),
        leave_one_fold_contraction=tuple(float(v) for v in values["leave_one_fold_contraction"]),
        folds=tuple(int(v) for v in values["folds"]),
        eligible_rows=int(values["eligible_rows"]),
    )
    frames = receipt.get("frames")
    if not isinstance(frames, Mapping) or int(receipt.get("frame_count", -1)) != len(frames):
        raise ValueError("directional-admission receipt frame authority differs")

    # Exhaustively verify and recompute every candidate before opening GT.
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for frame_id in sorted(frames):
        record = frames[frame_id]
        _, coverage = _array(record["coverage"], f"{frame_id} coverage")
        _, unary = _array(record["unary_probability"], f"{frame_id} unary")
        _, proposal = _array(record["proposal_probability"], f"{frame_id} proposal")
        _, candidate = _array(record["candidate_probability"], f"{frame_id} candidate")
        _, threshold = _array(record["fixed_threshold"], f"{frame_id} threshold")
        _, margin = _array(record["continuous_margin"], f"{frame_id} margin")
        _, quantile_margin = _array(
            record["quantile_baseline_margin"], f"{frame_id} quantile baseline"
        )
        replay = build_candidate_frame(unary, proposal, coverage, calibration)
        if not np.array_equal(replay, candidate):
            raise ValueError(f"directional-admission replay differs: {frame_id}")
        if not np.array_equal(threshold, np.full(threshold.shape, 0.5, dtype=np.float32)):
            raise ValueError(f"fixed threshold differs: {frame_id}")
        if not np.array_equal(margin, candidate - threshold):
            raise ValueError(f"continuous margin identity differs: {frame_id}")
        loaded[frame_id] = {
            "unary": unary,
            "proposal": proposal,
            "candidate_margin": margin,
            "quantile_margin": quantile_margin,
        }

    manifest_record = receipt.get("manifest")
    if not isinstance(manifest_record, Mapping):
        raise ValueError("directional-admission receipt lacks manifest authority")
    manifest_path = _require_file(
        manifest_record["path"], manifest_record["sha256"], "frozen manifest"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_hash") != receipt.get("protocol_hash"):
        raise ValueError("frozen manifest protocol differs")
    scenes = [
        scene
        for scene in manifest.get("scenes", [])
        if scene.get("scene_id") == receipt.get("scene_id")
    ]
    if len(scenes) != 1:
        raise ValueError("frozen manifest scene identity is ambiguous")
    scene = scenes[0]
    frame_records = {str(frame["frame_id"]): frame for frame in scene["frames"]}
    evaluation_frames = [str(value) for value in scene["evaluation_frame_ids"]]
    if set(evaluation_frames) != set(loaded):
        raise ValueError("prediction and frozen evaluation frames differ")

    results: list[dict[str, object]] = []
    for frame_id in evaluation_frames:
        authority = frame_records[frame_id]
        gt_path = _require_file(
            authority["ground_truth"],
            authority["ground_truth_sha256"],
            f"target mask {frame_id}",
        )
        target = load_ground_truth_mask(gt_path).astype(bool)
        values = loaded[frame_id]
        resized = {
            key: cv2.resize(
                value,
                (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            for key, value in values.items()
        }
        predictions = {
            "unary_at_0_5": resized["unary"] >= 0.5,
            "proposal_at_0_5": resized["proposal"] >= 0.5,
            "quantile_baseline": resized["quantile_margin"] >= 0.0,
            "directional_admission": resized["candidate_margin"] >= 0.0,
        }
        results.append(
            {
                "frame_id": frame_id,
                "foreground_iou": {
                    key: _iou(prediction, target)
                    for key, prediction in predictions.items()
                },
                "positive_fraction": {
                    key: float(prediction.mean())
                    for key, prediction in predictions.items()
                },
                "target_positive_fraction": float(target.mean()),
            }
        )
    names = (
        "unary_at_0_5",
        "proposal_at_0_5",
        "quantile_baseline",
        "directional_admission",
    )
    macro = {
        name: float(np.mean([row["foreground_iou"][name] for row in results]))
        for name in names
    }
    report = {
        "schema": "radio_gs.spin_rendered_directional_admission_exact_result.v1",
        "schema_version": 1,
        "status": "single_exact_evaluation_complete_no_parameter_scan",
        "scene_id": receipt["scene_id"],
        "prediction_receipt": {
            "path": str(receipt_path),
            "sha256": args.prediction_receipt_sha256,
        },
        "manifest": {"path": str(manifest_path), "sha256": manifest_record["sha256"]},
        "foreground_iou": macro,
        "delta_candidate_vs_quantile_baseline": (
            macro["directional_admission"] - macro["quantile_baseline"]
        ),
        "delta_candidate_vs_proposal_at_0_5": (
            macro["directional_admission"] - macro["proposal_at_0_5"]
        ),
        "frames": results,
        "evaluation_contract": {
            "within_scene_aggregation": "unweighted_frame_mean",
            "margin_resize": "cv2.INTER_LINEAR",
            "comparison": "greater_equal_zero",
            "empty_union_value": 1.0,
            "parameter_scan": False,
        },
        "target_rgb_opened": False,
        "target_mask_opened": True,
        "target_metric_computed": True,
    }
    report["content_sha256"] = json_sha256(report)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite exact evaluation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "report_path": str(output), "report_sha256": file_sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-receipt", required=True)
    parser.add_argument("--prediction-receipt-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps({"foreground_iou": result["foreground_iou"], "sha256": result["report_sha256"]}))


if __name__ == "__main__":
    main()
