#!/usr/bin/env python3
"""Verify the complete NVOS two-round batch before opening target masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from radio_gs.data.promptable_nvs_manifest import validate_manifest as validate_dataset_manifest
from radio_gs.evaluation.promptable_segmentation import evaluate_manifest
from radio_gs.scripts.build_nvos_two_round_exact_consensus import (
    CANDIDATE_ID,
    _prediction_path,
    _write_json,
)
from radio_gs.scripts.predict_nvos_method_v1_transient_sam import (
    DEFAULT_EVALUATION_CONTRACT,
    DEFAULT_METHOD_AUTHORITY,
    _sha256,
)
from radio_gs.scripts.validate_nvos_rgb_assisted_contract import (
    validate_contract as validate_evaluation_contract,
)


def verify_before_gt(args: argparse.Namespace) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path = Path(args.prediction_manifest).expanduser().resolve(strict=True)
    if _sha256(manifest_path) != str(args.expected_prediction_manifest_sha256):
        raise ValueError("two-round prediction manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_path = Path(args.dataset_manifest).expanduser().resolve(strict=True)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    normalized = validate_dataset_manifest(dataset, check_files=False)
    contract = validate_evaluation_contract(args.evaluation_contract)
    authority = json.loads(Path(contract["method_authority"]).read_text(encoding="utf-8"))
    frozen_order = [str(value) for value in authority["frozen_cohorts"]["nvos"]]
    scenes = [str(value) for value in manifest.get("scene_order", [])]
    if (
        manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("kind")
        != "promptable_nvs_method_v1_two_round_mask_prompt_sam3_predictions"
        or scenes != frozen_order
        or [str(row["scene_id"]) for row in normalized["scenes"]] != frozen_order
        or manifest.get("protocol_hash") != normalized["protocol_hash"]
        or manifest.get("all_requested_predictions_sealed") is not True
        or manifest.get("evaluation_performed") is not False
        or manifest.get("target_rgb_opened") is not True
        or manifest.get("target_mask_opened") is not False
        or manifest.get("target_metric_opened") is not False
    ):
        raise ValueError("two-round pre-GT barrier differs")
    receipt_rows = manifest.get("receipts")
    if not isinstance(receipt_rows, list) or [str(row.get("scene_id")) for row in receipt_rows] != scenes:
        raise ValueError("two-round receipt order differs")
    scene_index = {str(row["scene_id"]): row for row in normalized["scenes"]}
    verified: dict[str, Any] = {}
    for scene, receipt_row in zip(scenes, receipt_rows):
        frame_ids = list(map(str, scene_index[scene]["evaluation_frame_ids"]))
        if len(frame_ids) != 1 or list(manifest["predictions"][scene]) != frame_ids:
            raise ValueError(f"two-round target frame differs: {scene}")
        frame = frame_ids[0]
        prediction = _prediction_path(manifest, manifest_path, scene, frame)
        receipt_path = Path(str(receipt_row.get("path", ""))).resolve(strict=True)
        if _sha256(receipt_path) != receipt_row.get("sha256"):
            raise ValueError(f"two-round receipt SHA-256 differs: {scene}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        safety = receipt.get("safety", {})
        authorities = receipt.get("authorities", {})
        output = receipt.get("output", {}).get("continuous_margin", {})
        if (
            receipt.get("artifact_type")
            != "radio_gs_nvos_two_round_mask_prompt_sam3_receipt"
            or receipt.get("candidate_id") != CANDIDATE_ID
            or receipt.get("scene_id") != scene
            or receipt.get("frame_id") != frame
            or Path(str(output.get("path", ""))).resolve() != prediction
            or output.get("sha256") != manifest["prediction_sha256"][scene][frame]
            or authorities.get("protocol_hash") != normalized["protocol_hash"]
            or Path(str(authorities.get("evaluation_contract", ""))).resolve()
            != Path(contract["contract"])
            or authorities.get("evaluation_contract_sha256") != contract["contract_sha256"]
            or authorities.get("evaluation_contract_id") != contract["contract_id"]
            or safety.get("target_rgb_opened") is not True
            or safety.get("target_mask_opened") is not False
            or safety.get("target_metric_opened") is not False
            or safety.get("target_metric_used_for_selection") is not False
            or safety.get("scene_specific_parameter") is not False
        ):
            raise ValueError(f"two-round receipt contract differs: {scene}")
        verified[scene] = {
            "frame_id": frame,
            "prediction": str(prediction),
            "prediction_sha256": _sha256(prediction),
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
        }
    barrier = {
        "prediction_manifest": str(manifest_path),
        "prediction_manifest_sha256": args.expected_prediction_manifest_sha256,
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": _sha256(dataset_path),
        "evaluation_contract": contract,
        "scene_order": scenes,
        "verified": verified,
        "all_eight_predictions_and_receipts_verified_before_first_target_mask_open": True,
    }
    return manifest, manifest_path, barrier


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--expected-prediction-manifest-sha256", required=True)
    parser.add_argument("--evaluation-contract", default=str(DEFAULT_EVALUATION_CONTRACT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    _manifest, manifest_path, barrier = verify_before_gt(args)
    dataset = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    validate_dataset_manifest(dataset, check_files=True)
    evaluation = evaluate_manifest(
        Path(args.dataset_manifest).expanduser().resolve(strict=True),
        prediction_manifest=manifest_path,
    )
    result = {
        "schema_version": 1,
        "artifact_type": "radio_gs_nvos_two_round_exact_consensus_sam3_result",
        "candidate_id": CANDIDATE_ID,
        "pre_gt_barrier": barrier,
        "evaluation": evaluation,
        "eligibility": {
            "development_use_only": True,
            "target_rgb_assisted": True,
            "exact_rgb_assisted_evaluation_contract": True,
            "strict_unseen_protocol_eligible": False,
            "sota_claim": False,
        },
        "safety": {
            "target_metrics_used_for_prediction_or_selection": False,
            "all_eight_verified_before_first_target_mask_open": True,
        },
    }
    output = Path(args.output).expanduser().resolve()
    _write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": _sha256(output),
                "foreground_iou": evaluation["dataset"]["foreground_iou"],
                "pixel_accuracy": evaluation["dataset"]["pixel_accuracy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
