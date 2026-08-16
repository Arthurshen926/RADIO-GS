#!/usr/bin/env python3
"""Score the sealed NVOS field-box SAM3 candidate after a full8 barrier."""

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
from radio_gs.five_benchmark_method_v1 import validate_method_authority
from radio_gs.scripts.predict_nvos_method_v1_field_box_sam3 import (
    CANDIDATE_ID,
    SIGNED_POINT_CANDIDATE_ID,
    SIGNED_POINT_PREREGISTRATION,
)
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import (
    DEFAULT_METHOD_AUTHORITY,
)
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def verify_before_gt(
    *,
    dataset_manifest_path: str | Path,
    prediction_manifest_path: str | Path,
    method_authority_path: str | Path,
    expected_candidate_id: str = CANDIDATE_ID,
) -> dict[str, Any]:
    dataset, dataset_sha, dataset_path = load_json_object(
        dataset_manifest_path, label="NVOS dataset manifest"
    )
    normalized = validate_dataset_manifest(dataset, check_files=False)
    authority, authority_sha, authority_path = load_json_object(
        method_authority_path, label="Method-v1 authority"
    )
    validate_method_authority(authority)
    scene_order = [str(value) for value in authority["frozen_cohorts"]["nvos"]]
    if [str(row["scene_id"]) for row in normalized["scenes"]] != scene_order:
        raise ValueError("NVOS field-box candidate cohort differs")
    prediction, prediction_sha, prediction_path = load_json_object(
        prediction_manifest_path, label="NVOS field-box candidate manifest"
    )
    predictions = prediction.get("predictions")
    hashes = prediction.get("prediction_sha256")
    if (
        prediction.get("kind") != "promptable_nvs_method_v1_field_box_sam3_predictions"
        or prediction.get("candidate_id") != expected_candidate_id
        or prediction.get("protocol_hash") != normalized["protocol_hash"]
        or prediction.get("scene_order") != scene_order
        or prediction.get("all_eight_predictions_sealed") is not True
        or prediction.get("evaluation_performed") is not False
        or prediction.get("target_mask_opened") is not False
        or prediction.get("target_metric_opened") is not False
        or not isinstance(predictions, Mapping)
        or not isinstance(hashes, Mapping)
        or list(predictions) != scene_order
        or list(hashes) != scene_order
    ):
        raise ValueError("NVOS field-box prediction batch differs")
    receipts = {
        str(row.get("scene_id")): row
        for row in prediction.get("receipts", [])
        if isinstance(row, Mapping)
    }
    if list(receipts) != scene_order:
        raise ValueError("NVOS field-box receipts are not ordered full8")
    prediction_root = Path(str(prediction.get("prediction_root", ".")))
    if not prediction_root.is_absolute():
        prediction_root = prediction_path.parent / prediction_root
    normalized_index = {str(row["scene_id"]): row for row in normalized["scenes"]}
    verified = {}
    signed_point_selection = expected_candidate_id == SIGNED_POINT_CANDIDATE_ID
    expected_safety = {
        "candidate_selected_by_coarse_field_overlap": not signed_point_selection,
        "candidate_selected_by_signed_field_points": signed_point_selection,
        "coarse_field_overlap_used_as_tie_break": signed_point_selection,
    }
    for scene_id in scene_order:
        target_ids = [
            str(value) for value in normalized_index[scene_id]["evaluation_frame_ids"]
        ]
        if len(target_ids) != 1 or list(predictions[scene_id]) != target_ids:
            raise ValueError(f"{scene_id} field-box target identity differs")
        frame_id = target_ids[0]
        relative = Path(str(predictions[scene_id][frame_id]))
        score = relative if relative.is_absolute() else prediction_root / relative
        score = score.resolve(strict=True)
        expected = str(hashes[scene_id][frame_id])
        receipt_path = Path(str(receipts[scene_id]["path"])).resolve(strict=True)
        if sha256_file(receipt_path) != receipts[scene_id]["sha256"]:
            raise ValueError(f"{scene_id} field-box receipt SHA-256 differs")
        receipt, _digest, _source = load_json_object(
            receipt_path, label=f"{scene_id} field-box receipt"
        )
        safety = receipt.get("safety", {})
        field = receipt.get("field", {})
        if (
            receipt.get("artifact_type")
            != "radio_gs_nvos_method_v1_field_box_sam3_receipt"
            or receipt.get("candidate_id") != expected_candidate_id
            or receipt.get("scene_id") != scene_id
            or receipt.get("frame_id") != frame_id
            or receipt.get("output", {}).get("sha256") != expected
            or Path(str(receipt.get("output", {}).get("path", ""))).resolve() != score
            or sha256_file(score) != expected
            or safety.get("target_rgb_opened") is not True
            or safety.get("target_mask_opened") is not False
            or safety.get("target_metric_opened") is not False
            or any(
                safety.get(key) is not expected
                for key, expected in expected_safety.items()
            )
            or safety.get("target_metric_used_for_selection") is not False
            or safety.get("graph_used") is not False
            or safety.get("connected_component_used") is not False
            or receipt.get("authorities", {}).get("method_authority_sha256")
            != authority_sha
            or field.get("field_checkpoint_schema") != "factorized-v2"
        ):
            raise ValueError(f"{scene_id} field-box receipt contract differs")
        field_path = Path(str(field.get("field_checkpoint", ""))).resolve(strict=True)
        field_sha = str(field.get("field_checkpoint_sha256", ""))
        if len(field_sha) != 64 or sha256_file(field_path) != field_sha:
            raise ValueError(f"{scene_id} final Method-v1 field SHA-256 differs")
        verified[scene_id] = {
            "field": str(field_path),
            "field_sha256": field_sha,
            "prediction": str(score),
            "prediction_sha256": expected,
            "receipt": str(receipt_path),
        }
    return {
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": dataset_sha,
        "prediction_manifest": str(prediction_path),
        "prediction_manifest_sha256": prediction_sha,
        "method_authority": str(authority_path),
        "method_authority_sha256": authority_sha,
        "scene_order": scene_order,
        "verified": verified,
        "all_eight_predictions_verified_before_first_target_mask_open": True,
    }


def score(args: argparse.Namespace) -> dict[str, Any]:
    if args.candidate_id not in {CANDIDATE_ID, SIGNED_POINT_CANDIDATE_ID}:
        raise ValueError("unknown NVOS field-box candidate ID")
    barrier = verify_before_gt(
        dataset_manifest_path=args.manifest,
        prediction_manifest_path=args.prediction_manifest,
        method_authority_path=args.method_authority,
        expected_candidate_id=args.candidate_id,
    )
    dataset = json.loads(Path(barrier["dataset_manifest"]).read_text(encoding="utf-8"))
    validate_dataset_manifest(dataset, check_files=True)
    evaluation = evaluate_manifest(
        barrier["dataset_manifest"],
        prediction_manifest=barrier["prediction_manifest"],
    )
    macro = float(evaluation["dataset"]["foreground_iou"])
    if args.candidate_id == SIGNED_POINT_CANDIDATE_ID:
        preregistration = json.loads(
            SIGNED_POINT_PREREGISTRATION.read_text(encoding="utf-8")
        )
        parent = preregistration["parent_candidate"]
        parent_result_path = Path(parent["result"]["path"]).resolve(strict=True)
        if sha256_file(parent_result_path) != parent["result"]["sha256"]:
            raise ValueError("signed-point parent result SHA-256 differs")
        parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
        parent_macro = float(parent_result["evaluation"]["dataset"]["foreground_iou"])
        if parent_macro != float(parent["macro_foreground_iou"]):
            raise ValueError("signed-point parent macro differs from preregistration")
        parent_by_scene = {
            str(row["scene_id"]): float(row["foreground_iou"])
            for row in parent_result["evaluation"]["scenes"]
        }
        candidate_by_scene = {
            str(row["scene_id"]): float(row["foreground_iou"])
            for row in evaluation["scenes"]
        }
        minimum_macro = 0.768805
        minimum_delta = 0.01
        minimum_horns_left = 0.45
        maximum_scene_regression = 0.02
        scene_regressions = {
            scene_id: candidate_by_scene[scene_id] - parent_iou
            for scene_id, parent_iou in parent_by_scene.items()
        }
        gate = {
            "minimum_macro_foreground_iou": minimum_macro,
            "minimum_delta_vs_parent": minimum_delta,
            "parent_macro_foreground_iou": parent_macro,
            "minimum_horns_left_foreground_iou": minimum_horns_left,
            "maximum_allowed_scene_regression": maximum_scene_regression,
            "scene_delta_vs_parent": scene_regressions,
            "passed": (
                macro >= minimum_macro
                and macro - parent_macro >= minimum_delta
                and candidate_by_scene["horns_left"] >= minimum_horns_left
                and min(scene_regressions.values()) >= -maximum_scene_regression
            ),
        }
    else:
        gate = {
            "minimum_macro_foreground_iou": 0.768805,
            "minimum_delta_vs_majority_vote_baseline": 0.01,
            "baseline_macro_foreground_iou": 0.5268735259096345,
            "passed": (macro >= 0.768805 and macro - 0.5268735259096345 >= 0.01),
        }
    payload = {
        "schema_version": 1,
        "artifact_type": "radio_gs_nvos_method_v1_field_box_sam3_result",
        "candidate_id": args.candidate_id,
        "pre_gt_barrier": barrier,
        "evaluation": evaluation,
        "promotion_gate": gate,
        "eligibility": {
            "development_use_only": True,
            "prospectively_blind": False,
            "target_rgb_assisted": True,
            "strict_unseen_protocol_eligible": False,
        },
    }
    output = Path(args.output).expanduser().resolve()
    _write_json(output, payload)
    return {**payload, "output": str(output), "output_sha256": sha256_file(output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--method-authority", default=str(DEFAULT_METHOD_AUTHORITY))
    parser.add_argument("--candidate-id", default=CANDIDATE_ID)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = score(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "output": report["output"],
                "output_sha256": report["output_sha256"],
                "foreground_iou": report["evaluation"]["dataset"]["foreground_iou"],
                "promotion_gate_passed": report["promotion_gate"]["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
