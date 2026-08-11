"""Prepare and score the frozen NVOS valid-support two-scene Stage-A probe.

The probe changes exactly one method factor from the sealed hierarchical-v2
full8 run: the frozen scaled-renderer scalar readout conditions its denominator
on canonical capability validity.  Coverage power is fixed to zero, so the
candidate is ``sum(w*v*p) / sum(w*v)`` without a coverage penalty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask
from radio_gs.scripts.nvos_prompt_native_stage_a import (
    PATH_ONLY_ARGS,
    SCENES,
    _candidate_arg_diff,
    _json_sha256,
    _load,
    _option,
    _replace_option,
    _score_frame,
    _sha256,
    _write_no_clobber,
)


ARTIFACT_TYPE = "nvos_valid_support_stage_a_fern_trex_manifest_v1"
RESULT_TYPE = "nvos_valid_support_stage_a_fern_trex_exact_results_v1"


def _enable_valid_support(command: list[str]) -> list[str]:
    if "--valid-support-normalization" in command:
        raise ValueError("base command already enables valid-support normalization")
    if "--valid-support-coverage-power" in command:
        raise ValueError("base command explicitly sets valid-support coverage power")
    device_index = command.index("--device")
    return [
        *command[:device_index],
        "--valid-support-normalization",
        "--valid-support-coverage-power",
        "0",
        *command[device_index:],
    ]


def _prepare(args: argparse.Namespace) -> None:
    base_manifest_path = Path(args.base_run_manifest).expanduser().resolve()
    base_results_path = Path(args.base_exact_results).expanduser().resolve()
    evaluator_path = Path(args.evaluator).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if str(output_root).startswith("/mnt/pool/"):
        raise ValueError("Stage-A output must be on local SSD, not /mnt/pool")
    base_manifest = _load(base_manifest_path)
    base_results = _load(base_results_path)
    if base_manifest.get("artifact_type") != (
        "nvos_hierarchical_trust_local_positive_full8_prediction_run_manifest_v2"
    ):
        raise ValueError("unexpected frozen full8-v2 run manifest")
    if _sha256(evaluator_path) != base_manifest["evaluator_sha256"]:
        raise ValueError("working evaluator differs from the frozen evaluator")

    records: dict[str, object] = {}
    for scene in SCENES:
        base_record = base_manifest["records"][scene]
        base_command = list(base_record["command"])
        if _option(base_command, "--score-render-resolution") != "scaled_renderer":
            raise ValueError(f"{scene}: base score-render mode is not scaled_renderer")
        scene_root = output_root / scene
        command = _replace_option(base_command, "--output-dir", str(scene_root))
        command = _replace_option(
            command,
            "--primitive-unary-output",
            str(scene_root / "primitive_unary.pt"),
        )
        command = _replace_option(
            command,
            "--prediction-receipt-output",
            str(scene_root / "pre_metric_prediction_receipt.json"),
        )
        command = _enable_valid_support(command)
        records[scene] = {
            "base_command_sha256": _json_sha256(base_command),
            "command": command,
            "command_sha256": _json_sha256(command),
            "environment": {"CUDA_VISIBLE_DEVICES": "0"},
            "execution_only_changes": {
                "physical_gpu": {"base": base_record["physical_gpu"], "candidate": 0},
                "output_dir": str(scene_root),
                "prediction_receipt_output": str(
                    scene_root / "pre_metric_prediction_receipt.json"
                ),
                "primitive_unary_output": str(scene_root / "primitive_unary.pt"),
            },
            "method_factor": {
                "name": "valid_support_normalization",
                "base": False,
                "candidate": True,
                "coverage_power": 0.0,
                "formula": "sum(w*v*p)/sum(w*v)",
            },
            "base_prediction_receipt": {
                "path": base_record["prediction_receipt"],
                "sha256": _sha256(base_record["prediction_receipt"]),
            },
            "candidate_prediction_receipt": str(
                scene_root / "pre_metric_prediction_receipt.json"
            ),
            "baseline": {
                "foreground_iou": base_results["per_scene"][scene][
                    "foreground_iou"
                ],
                "pixel_accuracy": base_results["per_scene"][scene][
                    "pixel_accuracy"
                ],
            },
        }

    payload = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "prepared_before_candidate_target_scoring",
        "scene_order": list(SCENES),
        "base_run_manifest": {
            "path": str(base_manifest_path),
            "sha256": _sha256(base_manifest_path),
        },
        "base_exact_results": {
            "path": str(base_results_path),
            "sha256": _sha256(base_results_path),
        },
        "evaluator": {"path": str(evaluator_path), "sha256": _sha256(evaluator_path)},
        "implementation_audit": {
            "primitive_unary_export_precedes_valid_support_construction": True,
            "valid_support_consumers": ["render_scalar_scores"],
            "changes_geometry_alpha_or_primitive_unary": False,
            "base_pixel_formula": "sum(w*p)/sum(w)",
            "candidate_pixel_formula": "sum(w*v*p)/sum(w*v)",
            "coverage_power": 0.0,
        },
        "single_factor_contract": {
            "method_change": "valid_support_normalization: false -> true",
            "allowed_execution_changes": sorted(PATH_ONLY_ARGS | {"physical_gpu"}),
            "all_other_candidate_arg_values_must_match": True,
            "primitive_unary_tensor_must_be_numerically_equivalent_to_base": True,
            "score_render_resolution": "scaled_renderer",
            "target_rgb_used": False,
            "target_mask_used_before_exact_scoring": False,
        },
        "promotion_gate_preregistered_before_candidate_scoring": {
            "pair_macro_iou_delta_minimum": 0.005,
            "per_scene_iou_delta_minimum": -0.002,
            "requires_both_conditions": True,
            "full8_may_not_start_before_parent_report": True,
        },
        "thermal_policy": {
            "physical_gpu": 0,
            "continue_at_or_below_c": 80,
            "pause_at_or_above_c": 82,
            "polling": "launch, scene boundary, and low-frequency while running",
        },
        "output_root": str(output_root),
        "records": records,
    }
    output = _write_no_clobber(output_root / "stage_a_manifest.json", payload)
    print(json.dumps({"manifest": str(output), "sha256": _sha256(output)}))
    for scene in SCENES:
        record = records[scene]
        print(
            "CUDA_VISIBLE_DEVICES=0 "
            + " ".join(json.dumps(item) for item in record["command"])
        )


def _score(args: argparse.Namespace) -> None:
    stage_manifest_path = Path(args.stage_manifest).expanduser().resolve()
    stage = _load(stage_manifest_path)
    if stage.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("unexpected Stage-A manifest")
    if tuple(stage.get("scene_order", ())) != SCENES:
        raise ValueError("Stage-A scene order differs")
    base_manifest_path = Path(stage["base_run_manifest"]["path"])
    base_results_path = Path(stage["base_exact_results"]["path"])
    if _sha256(base_manifest_path) != stage["base_run_manifest"]["sha256"]:
        raise ValueError("base run manifest changed")
    if _sha256(base_results_path) != stage["base_exact_results"]["sha256"]:
        raise ValueError("base exact results changed")
    evaluator_path = Path(stage["evaluator"]["path"])
    if _sha256(evaluator_path) != stage["evaluator"]["sha256"]:
        raise ValueError("frozen evaluator changed")
    base_manifest = _load(base_manifest_path)
    base_results = _load(base_results_path)
    benchmark_manifest_path = Path(base_manifest["manifest"])
    if _sha256(benchmark_manifest_path) != base_manifest["manifest_sha256"]:
        raise ValueError("benchmark manifest changed")
    benchmark = _load(benchmark_manifest_path)
    scene_rows = {row["scene_id"]: row for row in benchmark["scenes"]}

    # Verify both candidate predictions and primitive replay before opening GT.
    verified: dict[str, object] = {}
    for scene in SCENES:
        record = stage["records"][scene]
        receipt_path = Path(record["candidate_prediction_receipt"])
        receipt = _load(receipt_path)
        base_receipt_path = Path(record["base_prediction_receipt"]["path"])
        base_receipt = _load(base_receipt_path)
        if _sha256(base_receipt_path) != record["base_prediction_receipt"]["sha256"]:
            raise ValueError(f"{scene}: base receipt changed")
        if not (
            receipt.get("sealed_before_target_ground_truth_open") is True
            and receipt.get("target_rgb_opened") is False
            and receipt.get("target_mask_opened") is False
            and receipt.get("target_metric_opened") is False
        ):
            raise ValueError(f"{scene}: candidate prediction barrier differs")
        base_method = dict(base_receipt["method_contract"])
        candidate_method = dict(receipt["method_contract"])
        base_args = dict(base_method.pop("candidate_args"))
        candidate_args = dict(candidate_method.pop("candidate_args"))
        differences = _candidate_arg_diff(base_args, candidate_args)
        expected_differences = PATH_ONLY_ARGS | {"valid_support_normalization"}
        if set(differences) != expected_differences:
            raise ValueError(
                f"{scene}: candidate argument differences are not single-factor: "
                f"{sorted(differences)}"
            )
        if differences["valid_support_normalization"] != (False, True):
            raise ValueError(f"{scene}: valid-support factor differs")
        if base_args["score_render_resolution"] != "scaled_renderer" or (
            candidate_args["score_render_resolution"] != "scaled_renderer"
        ):
            raise ValueError(f"{scene}: score-render resolution drifted")
        if base_args["valid_support_coverage_power"] != 0.0 or (
            candidate_args["valid_support_coverage_power"] != 0.0
        ):
            raise ValueError(f"{scene}: valid-support coverage power drifted")

        base_primitive = dict(base_method.pop("primitive_unary_artifact"))
        candidate_primitive = dict(candidate_method.pop("primitive_unary_artifact"))
        if _sha256(base_primitive["path"]) != base_primitive["file_sha256"]:
            raise ValueError(f"{scene}: base primitive artifact changed")
        if _sha256(candidate_primitive["path"]) != candidate_primitive["file_sha256"]:
            raise ValueError(f"{scene}: candidate primitive artifact changed")
        base_payload = torch.load(
            base_primitive["path"], map_location="cpu", weights_only=True
        )
        candidate_payload = torch.load(
            candidate_primitive["path"], map_location="cpu", weights_only=True
        )
        base_unary = base_payload["primitive_unary_probability"].float()
        candidate_unary = candidate_payload["primitive_unary_probability"].float()
        if base_unary.shape != candidate_unary.shape:
            raise ValueError(f"{scene}: primitive unary shape changed")
        unary_difference = (base_unary - candidate_unary).abs()
        unary_max = float(unary_difference.max().item())
        unary_mean = float(unary_difference.mean().item())
        # This pre-GT tolerance admits only last-bit float32 CUDA replay drift.
        # It was finalized after sealed predictions but before any target mask
        # was opened; no element in either sentinel differs by more than 1e-6.
        if unary_max > 1e-6 or unary_mean > 1e-8:
            raise ValueError(f"{scene}: primitive unary numerical drift exceeds tolerance")

        base_valid = base_method.pop("valid_support_normalization")
        candidate_valid = candidate_method.pop("valid_support_normalization")
        if (base_valid, candidate_valid) != (False, True):
            raise ValueError(f"{scene}: receipt valid-support contract differs")
        if base_method != candidate_method:
            raise ValueError(f"{scene}: method contract changed beyond valid support")

        candidate_scores: dict[str, np.ndarray] = {}
        candidate_score_records = receipt["target_scores"]
        base_score_records = base_receipt["target_scores"]
        if set(candidate_score_records) != set(base_score_records):
            raise ValueError(f"{scene}: target score frame ids changed")
        score_shapes: dict[str, list[int]] = {}
        base_score_shapes: dict[str, list[int]] = {}
        for frame_id, item in candidate_score_records.items():
            if _sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"{scene}/{frame_id}: candidate score changed")
            base_item = base_score_records[frame_id]
            if _sha256(base_item["path"]) != base_item["sha256"]:
                raise ValueError(f"{scene}/{frame_id}: base score changed")
            candidate_score = np.load(item["path"], allow_pickle=False)
            base_score = np.load(base_item["path"], allow_pickle=False)
            if candidate_score.shape != base_score.shape:
                raise ValueError(f"{scene}/{frame_id}: scaled renderer shape changed")
            candidate_scores[frame_id] = candidate_score
            score_shapes[frame_id] = list(candidate_score.shape)
            base_score_shapes[frame_id] = list(base_score.shape)
        verified[scene] = {
            "receipt": str(receipt_path.resolve()),
            "receipt_sha256": _sha256(receipt_path),
            "score_records": candidate_score_records,
            "score_shapes": score_shapes,
            "base_score_shapes": base_score_shapes,
            "scores": candidate_scores,
            "primitive_unary": candidate_primitive,
            "primitive_unary_base": base_primitive,
            "primitive_unary_numerical_replay": {
                "bit_exact": bool(torch.equal(base_unary, candidate_unary)),
                "max_abs_difference": unary_max,
                "mean_abs_difference": unary_mean,
                "maximum_allowed": 1e-6,
                "mean_allowed": 1e-8,
                "accepted_before_target_ground_truth_open": True,
            },
            "candidate_arg_differences": {
                key: {"base": value[0], "candidate": value[1]}
                for key, value in differences.items()
            },
        }

    # Exact target scoring starts only after both sealed receipts pass above.
    per_scene: dict[str, object] = {}
    for scene in SCENES:
        scene_row = scene_rows[scene]
        frames = []
        for frame_id in scene_row["evaluation_frame_ids"]:
            frame = next(
                row for row in scene_row["frames"] if row["frame_id"] == frame_id
            )
            target_path = Path(frame["ground_truth"])
            if _sha256(target_path) != frame["ground_truth_sha256"]:
                raise ValueError(f"{scene}/{frame_id}: target mask changed")
            metrics = _score_frame(
                verified[scene]["scores"][frame_id],
                load_ground_truth_mask(target_path),
            )
            frames.append({"frame_id": frame_id, **metrics})
        foreground_iou = float(np.mean([row["foreground_iou"] for row in frames]))
        pixel_accuracy = float(np.mean([row["pixel_accuracy"] for row in frames]))
        baseline = base_results["per_scene"][scene]
        per_scene[scene] = {
            "foreground_iou": foreground_iou,
            "pixel_accuracy": pixel_accuracy,
            "delta_foreground_iou": foreground_iou - baseline["foreground_iou"],
            "delta_pixel_accuracy": pixel_accuracy - baseline["pixel_accuracy"],
            "baseline_foreground_iou": baseline["foreground_iou"],
            "baseline_pixel_accuracy": baseline["pixel_accuracy"],
            "frames": frames,
            **verified[scene],
        }
        per_scene[scene].pop("scores")
    pair_iou = float(np.mean([row["foreground_iou"] for row in per_scene.values()]))
    pair_baseline = float(
        np.mean([row["baseline_foreground_iou"] for row in per_scene.values()])
    )
    pair_delta = pair_iou - pair_baseline
    gate_contract = stage["promotion_gate_preregistered_before_candidate_scoring"]
    macro_pass = pair_delta >= gate_contract["pair_macro_iou_delta_minimum"]
    safety_pass = all(
        row["delta_foreground_iou"] >= gate_contract["per_scene_iou_delta_minimum"]
        for row in per_scene.values()
    )
    payload = {
        "schema_version": 1,
        "artifact_type": RESULT_TYPE,
        "status": "two_scene_prediction_barrier_and_exact_scoring_complete",
        "stage_manifest": {
            "path": str(stage_manifest_path),
            "sha256": _sha256(stage_manifest_path),
        },
        "scorer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "scene_order": list(SCENES),
        "per_scene": per_scene,
        "aggregate": {
            "pair_macro_foreground_iou": pair_iou,
            "baseline_pair_macro_foreground_iou": pair_baseline,
            "delta_foreground_iou": pair_delta,
            "pair_macro_pixel_accuracy": float(
                np.mean([row["pixel_accuracy"] for row in per_scene.values()])
            ),
        },
        "promotion_gate": {
            "contract": gate_contract,
            "pair_macro_pass": macro_pass,
            "per_scene_safety_pass": safety_pass,
            "passed": bool(macro_pass and safety_pass),
            "full8_started": False,
        },
        "evaluation_protocol": {
            "score_resize": "cv2.INTER_LINEAR",
            "threshold": 0.5,
            "comparison": "greater_or_equal",
            "aggregation": "per-frame then equal two-scene macro",
        },
        "safety": {
            "both_receipts_verified_before_first_target_ground_truth_open": True,
            "target_rgb_used": False,
            "target_metrics_used_to_change_candidate": False,
            "primitive_unary_bit_exact_to_base": all(
                row["primitive_unary_numerical_replay"]["bit_exact"]
                for row in per_scene.values()
            ),
            "primitive_unary_numerically_equivalent_to_base": True,
            "primitive_unary_tolerance_fixed_before_target_ground_truth_open": True,
        },
    }
    output = _write_no_clobber(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                **payload["aggregate"],
                "gate_passed": payload["promotion_gate"]["passed"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--base-run-manifest", required=True)
    prepare.add_argument("--base-exact-results", required=True)
    prepare.add_argument("--evaluator", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.set_defaults(func=_prepare)
    score = subparsers.add_parser("score")
    score.add_argument("--stage-manifest", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(func=_score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
