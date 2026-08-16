#!/usr/bin/env python3
"""Verify the complete SPIn Method-v1 batch before opening target masks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from radio_gs.data.promptable_nvs_manifest import (
    validate_manifest as validate_dataset_manifest,
)
from radio_gs.evaluation.promptable_segmentation import evaluate_manifest
from radio_gs.five_benchmark_method_v1 import METHOD_ID, validate_method_authority
from radio_gs.scripts.run_spin9_method_v1_scene import (
    DATASET_MANIFEST,
    METHOD_AUTHORITY,
)
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


HISTORICAL_TARGET = 0.9484146824995366


def _write_json_noclobber(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _nested(
    value: Mapping[str, Any], scene_id: str, frame_id: str, *, label: str
) -> str:
    scene = value.get(scene_id)
    if not isinstance(scene, Mapping) or frame_id not in scene:
        raise ValueError(f"{label} is absent for {scene_id}/{frame_id}")
    return str(scene[frame_id])


def verify_full9_before_gt(
    *,
    dataset_manifest_path: str | Path,
    prediction_manifest_path: str | Path,
    method_authority_path: str | Path,
) -> dict[str, Any]:
    dataset, dataset_sha, dataset_path = load_json_object(
        dataset_manifest_path, label="SPIn Available-Nine dataset manifest"
    )
    normalized = validate_dataset_manifest(dataset, check_files=False)
    authority, authority_sha, authority_path = load_json_object(
        method_authority_path, label="Method-v1 authority"
    )
    validate_method_authority(authority)
    scene_order = [str(value) for value in dataset["protocol"]["cohort"]]
    normalized_order = [str(row["scene_id"]) for row in normalized["scenes"]]
    method_scenes = [
        str(value) for value in authority["frozen_cohorts"]["spin_nerf_available9"]
    ]
    if normalized_order != scene_order or set(method_scenes) != set(scene_order):
        raise ValueError("SPIn Method-v1 and dataset Available-Nine cohorts differ")

    prediction, prediction_sha, prediction_path = load_json_object(
        prediction_manifest_path, label="SPIn Method-v1 prediction manifest"
    )
    predictions = prediction.get("predictions")
    prediction_hashes = prediction.get("prediction_sha256")
    if (
        prediction.get("kind")
        != "promptable_nvs_method_v1_spin9_transient_sam_predictions"
        or prediction.get("method_id") != METHOD_ID
        or prediction.get("protocol_hash") != normalized["protocol_hash"]
        or prediction.get("scene_order") != scene_order
        or prediction.get("evaluation_performed") is not False
        or prediction.get("target_mask_opened") is not False
        or prediction.get("target_metric_opened") is not False
        or prediction.get("all_nine_scene_predictions_sealed") is not True
        or not isinstance(predictions, Mapping)
        or not isinstance(prediction_hashes, Mapping)
        or list(predictions) != scene_order
        or list(prediction_hashes) != scene_order
    ):
        raise ValueError("SPIn Method-v1 prediction batch contract differs")
    receipt_rows = prediction.get("receipts")
    if not isinstance(receipt_rows, list):
        raise ValueError("SPIn Method-v1 prediction receipts are absent")
    receipt_index = {
        str(row.get("scene_id")): row
        for row in receipt_rows
        if isinstance(row, Mapping)
    }
    if list(receipt_index) != scene_order:
        raise ValueError("SPIn Method-v1 receipt batch is not ordered full9")

    prediction_root = Path(str(prediction.get("prediction_root", ".")))
    if not prediction_root.is_absolute():
        prediction_root = prediction_path.parent / prediction_root
    scene_index = {str(row["scene_id"]): row for row in normalized["scenes"]}
    verified: dict[str, Any] = {}
    for scene_id in scene_order:
        target_ids = [
            str(value) for value in scene_index[scene_id]["evaluation_frame_ids"]
        ]
        if list(predictions[scene_id]) != target_ids:
            raise ValueError(f"{scene_id} prediction frame order differs")
        receipt_row = receipt_index[scene_id]
        receipt_path = Path(str(receipt_row.get("path", ""))).resolve(strict=True)
        receipt_sha = sha256_file(receipt_path)
        if receipt_sha != receipt_row.get("sha256"):
            raise ValueError(f"{scene_id} transient receipt SHA-256 differs")
        receipt, _digest, _source = load_json_object(
            receipt_path, label=f"{scene_id} transient SAM receipt"
        )
        output_index = {
            str(row.get("frame_id")): row
            for row in receipt.get("outputs", [])
            if isinstance(row, Mapping)
        }
        safety = receipt.get("safety", {})
        reference = receipt.get("reference", {})
        candidate = receipt.get("candidate_policy", {})
        if (
            receipt.get("artifact_type")
            != "radio_gs_method_v1_spin9_transient_sam_receipt"
            or receipt.get("method_id") != METHOD_ID
            or receipt.get("scene_id") != scene_id
            or list(output_index) != target_ids
            or safety.get(
                "all_signed_margins_and_fields_verified_before_first_rgb_open"
            )
            is not True
            or safety.get("reference_mask_opened") is not True
            or safety.get("target_rgb_opened") is not True
            or safety.get("target_mask_opened") is not False
            or safety.get("target_metric_opened") is not False
            or safety.get("reference_mask_selection") is not True
            or safety.get("target_metric_used_for_selection") is not False
            or safety.get("graph_used") is not False
            or safety.get("connected_component_used") is not False
            or candidate.get("multimask_output") is not True
            or candidate.get("candidate_count") != 3
            or candidate.get("reference_only_calibration") is not True
            or candidate.get("canonical_branch_fallback") is not False
            or int(reference.get("selected_candidate", -1)) not in (0, 1, 2)
            or not 0.0 < float(reference.get("selected_threshold", -1.0)) < 1.0
        ):
            raise ValueError(f"{scene_id} transient receipt contract differs")
        field = Path(str(receipt.get("field", {}).get("path", ""))).resolve(strict=True)
        field_sha = str(receipt.get("field", {}).get("sha256", ""))
        if len(field_sha) != 64 or sha256_file(field) != field_sha:
            raise ValueError(f"{scene_id} final Method-v1 field SHA-256 differs")

        scene_verified: dict[str, Any] = {}
        for frame_id in target_ids:
            relative = Path(
                _nested(predictions, scene_id, frame_id, label="prediction")
            )
            path = relative if relative.is_absolute() else prediction_root / relative
            path = path.resolve(strict=True)
            expected = _nested(
                prediction_hashes, scene_id, frame_id, label="prediction hash"
            )
            row = output_index[frame_id]
            if (
                sha256_file(path) != expected
                or Path(str(row.get("path", ""))).resolve() != path
                or row.get("sha256") != expected
            ):
                raise ValueError(f"{scene_id}/{frame_id} sealed prediction differs")
            scene_verified[frame_id] = {
                "prediction": str(path),
                "prediction_sha256": expected,
            }
        verified[scene_id] = {
            "field": str(field),
            "field_sha256": field_sha,
            "receipt": str(receipt_path),
            "receipt_sha256": receipt_sha,
            "selected_candidate": int(reference["selected_candidate"]),
            "selected_threshold": float(reference["selected_threshold"]),
            "selected_reference_iou": float(reference["selected_reference_iou"]),
            "frames": scene_verified,
        }
    return {
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": dataset_sha,
        "prediction_manifest": str(prediction_path),
        "prediction_manifest_sha256": prediction_sha,
        "method_authority": str(authority_path),
        "method_authority_sha256": authority_sha,
        "protocol_hash": normalized["protocol_hash"],
        "scene_order": scene_order,
        "verified": verified,
        "all_nine_scene_receipts_and_predictions_verified_before_first_target_mask_open": True,
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    barrier = verify_full9_before_gt(
        dataset_manifest_path=args.manifest,
        prediction_manifest_path=args.prediction_manifest,
        method_authority_path=args.method_authority,
    )
    dataset = json.loads(Path(barrier["dataset_manifest"]).read_text(encoding="utf-8"))
    validate_dataset_manifest(dataset, check_files=True)
    evaluation = evaluate_manifest(
        barrier["dataset_manifest"],
        prediction_manifest=barrier["prediction_manifest"],
    )
    foreground_iou = float(evaluation["dataset"]["foreground_iou"])
    payload = {
        "schema_version": 1,
        "artifact_type": "radio_gs_method_v1_spin9_full9_results",
        "method_id": METHOD_ID,
        "pre_gt_barrier": barrier,
        "evaluation": evaluation,
        "comparison": {
            "historical_reference_selected_target": HISTORICAL_TARGET,
            "delta_method_v1_minus_historical_target": foreground_iou
            - HISTORICAL_TARGET,
            "meets_or_exceeds_historical_target": foreground_iou >= HISTORICAL_TARGET,
        },
        "eligibility": {
            "exact_available_nine_scene_cohort": True,
            "exact_frozen_metric_implementation": True,
            "reference_frame_not_scored": True,
            "query_transient_target_rgb_authorized_by_method_v1": True,
            "reusable_field_track_target_rgb_policy": "forbidden",
            "target_rgb_assisted_track": True,
            "full_ten_scene_eligible": False,
            "missing_scene": "fork",
            "prospectively_blind": False,
        },
        "safety": {
            "all_nine_scene_receipts_and_predictions_verified_before_first_target_mask_open": True,
            "target_metrics_used_for_method_selection": False,
            "reference_mask_only_calibration": True,
        },
    }
    output = Path(args.output).expanduser().resolve()
    _write_json_noclobber(output, payload)
    return {**payload, "output": str(output), "output_sha256": sha256_file(output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DATASET_MANIFEST))
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--method-authority", default=str(METHOD_AUTHORITY))
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = score(build_parser().parse_args(argv))
    dataset = report["evaluation"]["dataset"]
    print(
        json.dumps(
            {
                "output": report["output"],
                "output_sha256": report["output_sha256"],
                "foreground_iou": dataset["foreground_iou"],
                "pixel_accuracy": dataset["pixel_accuracy"],
                "meets_historical_target": report["comparison"][
                    "meets_or_exceeds_historical_target"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
