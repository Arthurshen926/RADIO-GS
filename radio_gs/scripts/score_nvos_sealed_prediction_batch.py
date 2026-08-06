"""Score an all-scene NVOS prediction batch only after its pre-GT barrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_no_clobber(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different batch score: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _score_frame(score: np.ndarray, ground_truth: np.ndarray, threshold: float) -> dict[str, float]:
    values = np.asarray(score)
    target = np.asarray(ground_truth, dtype=bool)
    if values.ndim != 2 or target.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("score and ground truth must be finite aligned 2D arrays")
    resized = cv2.resize(
        values.astype(np.float32, copy=False),
        (target.shape[1], target.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    prediction = resized >= float(threshold)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return {
        "foreground_iou": float(intersection / union) if union else 1.0,
        "pixel_accuracy": float((prediction == target).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--batch-authority", required=True)
    parser.add_argument("--batch-authority-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    authority_path = Path(args.batch_authority).expanduser().resolve()
    if _file_sha256(manifest_path) != args.manifest_sha256:
        raise ValueError("manifest SHA-256 differs")
    if _file_sha256(authority_path) != args.batch_authority_sha256:
        raise ValueError("batch authority SHA-256 differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    scene_order = tuple(authority["scene_order"])
    scene_records = {str(scene["scene_id"]): scene for scene in manifest["scenes"]}
    if tuple(scene_records) != scene_order or tuple(authority["records"]) != scene_order:
        raise ValueError("manifest and batch scene order differ")
    if (
        authority.get("artifact_type")
        != "nvos_hierarchical_trust_local_positive_full8_prediction_batch_authority_v2"
        or authority.get("all_eight_receipts_verified_before_any_target_ground_truth_open")
        is not True
        or authority.get("authorized_next_step")
        != "single_full8_cpu_scoring_pass_under_frozen_manifest"
    ):
        raise ValueError("batch authority does not authorize scoring")

    # Complete the prediction-side verification barrier before opening any GT.
    verified: dict[str, dict[str, object]] = {}
    for scene_id in scene_order:
        record = authority["records"][scene_id]
        receipt_path = Path(record["prediction_receipt"]).expanduser().resolve()
        if _file_sha256(receipt_path) != record["prediction_receipt_sha256"]:
            raise ValueError(f"{scene_id} prediction receipt SHA-256 differs")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("scene_id") != scene_id
            or receipt.get("sealed_before_target_ground_truth_open") is not True
            or receipt.get("target_mask_opened") is not False
            or receipt.get("target_metric_opened") is not False
        ):
            raise ValueError(f"{scene_id} prediction receipt safety differs")
        method = receipt["method_contract"]
        if (
            method["evaluator_sha256"] != authority["evaluator_sha256"]
            or method["candidate_args"]["prediction_only"] is not True
            or method["candidate_args"]["source_completion_calibration"]
            != "all_trial_loo_hierarchical_local_positive_v2"
        ):
            raise ValueError(f"{scene_id} candidate method differs")
        primitive = method["primitive_unary_artifact"]
        if _file_sha256(primitive["path"]) != primitive["file_sha256"]:
            raise ValueError(f"{scene_id} primitive unary SHA-256 differs")
        scores: dict[str, np.ndarray] = {}
        for frame_id, item in receipt["target_scores"].items():
            score_path = Path(item["path"]).expanduser().resolve()
            if _file_sha256(score_path) != item["sha256"]:
                raise ValueError(f"{scene_id} target score SHA-256 differs")
            scores[str(frame_id)] = np.load(score_path, allow_pickle=False)
        threshold = float(method["score_threshold"])
        if threshold != 0.5:
            raise ValueError(f"{scene_id} frozen threshold differs")
        verified[scene_id] = {
            "receipt_path": str(receipt_path),
            "receipt_sha256": record["prediction_receipt_sha256"],
            "hierarchical_branch": record["hierarchical_branch"],
            "scores": scores,
            "score_records": receipt["target_scores"],
            "threshold": threshold,
        }

    per_scene: dict[str, object] = {}
    for scene_id in scene_order:
        scene = scene_records[scene_id]
        expected_frames = tuple(str(value) for value in scene["evaluation_frame_ids"])
        if tuple(verified[scene_id]["scores"]) != expected_frames:
            raise ValueError(f"{scene_id} evaluation frames differ")
        frames = []
        for frame_id in expected_frames:
            frame = next(row for row in scene["frames"] if str(row["frame_id"]) == frame_id)
            ground_truth_path = Path(frame["ground_truth"]).expanduser().resolve()
            if _file_sha256(ground_truth_path) != frame["ground_truth_sha256"]:
                raise ValueError(f"{scene_id} ground-truth SHA-256 differs")
            metrics = _score_frame(
                verified[scene_id]["scores"][frame_id],
                load_ground_truth_mask(ground_truth_path),
                float(verified[scene_id]["threshold"]),
            )
            frames.append({"frame_id": frame_id, **metrics})
        per_scene[scene_id] = {
            "foreground_iou": float(np.mean([row["foreground_iou"] for row in frames])),
            "pixel_accuracy": float(np.mean([row["pixel_accuracy"] for row in frames])),
            "hierarchical_branch": verified[scene_id]["hierarchical_branch"],
            "prediction_receipt": verified[scene_id]["receipt_path"],
            "prediction_receipt_sha256": verified[scene_id]["receipt_sha256"],
            "score_records": verified[scene_id]["score_records"],
            "frames": frames,
        }
    payload = {
        "schema_version": 1,
        "artifact_type": "nvos_hierarchical_trust_local_positive_full8_exact_results_v2",
        "manifest": {"path": str(manifest_path), "sha256": args.manifest_sha256},
        "prediction_batch_authority": {
            "path": str(authority_path),
            "sha256": args.batch_authority_sha256,
        },
        "scene_order": list(scene_order),
        "per_scene": per_scene,
        "aggregate": {
            "scene_macro_foreground_iou": float(
                np.mean([row["foreground_iou"] for row in per_scene.values()])
            ),
            "scene_macro_pixel_accuracy": float(
                np.mean([row["pixel_accuracy"] for row in per_scene.values()])
            ),
        },
        "evaluation_protocol": {
            "score_resize": "cv2.INTER_LINEAR",
            "threshold": 0.5,
            "comparison": "greater_or_equal",
            "empty_union_value": 1.0,
            "aggregation": "per_frame_then_task_instance_scene_macro",
        },
        "safety": {
            "all_eight_receipts_verified_before_first_target_ground_truth_open": True,
            "prediction_changed_after_receipt": False,
            "target_metrics_used_for_method_selection": False,
        },
    }
    output = Path(args.output).expanduser().resolve()
    _atomic_write_no_clobber(output, payload)
    print(json.dumps({"output": str(output), "sha256": _file_sha256(output), **payload["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
