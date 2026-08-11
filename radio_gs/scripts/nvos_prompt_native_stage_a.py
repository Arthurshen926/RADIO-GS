"""Prepare and score the frozen NVOS prompt-native two-scene Stage-A probe.

The probe changes exactly one method factor from the sealed hierarchical-v2
full8 run: scalar scores are rendered at the native prompt resolution instead
of the renderer's scaled feature resolution.  Output-path and physical-device
changes are recorded separately as execution-only changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask


SCENES = ("fern", "trex")
ARTIFACT_TYPE = "nvos_prompt_native_stage_a_fern_trex_manifest_v1"
RESULT_TYPE = "nvos_prompt_native_stage_a_fern_trex_exact_results_v2"
PATH_ONLY_ARGS = {
    "output_dir",
    "prediction_receipt_output",
    "primitive_unary_output",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_no_clobber(path: str | Path, payload: Mapping[str, object]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    except FileExistsError:
        if output.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different artifact: {output}")
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _replace_option(command: Sequence[str], option: str, value: str) -> list[str]:
    result = list(command)
    locations = [index for index, item in enumerate(result) if item == option]
    if len(locations) != 1 or locations[0] + 1 >= len(result):
        raise ValueError(f"expected one value-bearing {option} option")
    result[locations[0] + 1] = value
    return result


def _option(command: Sequence[str], option: str) -> str:
    locations = [index for index, item in enumerate(command) if item == option]
    if len(locations) != 1 or locations[0] + 1 >= len(command):
        raise ValueError(f"expected one value-bearing {option} option")
    return str(command[locations[0] + 1])


def _prepare(args: argparse.Namespace) -> None:
    base_manifest_path = Path(args.base_run_manifest).expanduser().resolve()
    base_results_path = Path(args.base_exact_results).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if str(output_root).startswith("/mnt/pool/"):
        raise ValueError("Stage-A output must be on local SSD, not /mnt/pool")
    base_manifest = _load(base_manifest_path)
    base_results = _load(base_results_path)
    if base_manifest.get("artifact_type") != (
        "nvos_hierarchical_trust_local_positive_full8_prediction_run_manifest_v2"
    ):
        raise ValueError("unexpected frozen full8-v2 run manifest")

    records: dict[str, object] = {}
    for scene in SCENES:
        base_record = base_manifest["records"][scene]
        base_command = list(base_record["command"])
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
        command = _replace_option(
            command, "--score-render-resolution", "prompt_native"
        )
        if _option(base_command, "--score-render-resolution") != "scaled_renderer":
            raise ValueError(f"{scene}: base score-render mode is not scaled_renderer")
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
                "name": "score_render_resolution",
                "base": "scaled_renderer",
                "candidate": "prompt_native",
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
        "evaluator": {
            "path": str(
                (Path(__file__).resolve().parent / "eval_nvos_gaussian_first.py")
            ),
            "sha256": base_manifest["evaluator_sha256"],
        },
        "single_factor_contract": {
            "method_change": "score_render_resolution: scaled_renderer -> prompt_native",
            "allowed_execution_changes": sorted(PATH_ONLY_ARGS | {"physical_gpu"}),
            "all_other_candidate_args_must_match": True,
            "primitive_unary_tensor_must_match_base": True,
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


def _candidate_arg_diff(
    base: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, tuple[object, object]]:
    keys = set(base) | set(candidate)
    return {
        key: (base.get(key), candidate.get(key))
        for key in sorted(keys)
        if base.get(key) != candidate.get(key)
    }


def _score_frame(score: np.ndarray, target: np.ndarray) -> dict[str, float]:
    values = np.asarray(score)
    ground_truth = np.asarray(target, dtype=bool)
    resized = cv2.resize(
        values.astype(np.float32, copy=False),
        (ground_truth.shape[1], ground_truth.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    prediction = resized >= 0.5
    intersection = int(np.logical_and(prediction, ground_truth).sum())
    union = int(np.logical_or(prediction, ground_truth).sum())
    return {
        "foreground_iou": float(intersection / union) if union else 1.0,
        "pixel_accuracy": float((prediction == ground_truth).mean()),
    }


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

    # Verify every candidate prediction before opening either target mask.
    verified: dict[str, object] = {}
    for scene in SCENES:
        record = stage["records"][scene]
        receipt_path = Path(record["candidate_prediction_receipt"])
        receipt = _load(receipt_path)
        base_receipt = _load(record["base_prediction_receipt"]["path"])
        if _sha256(record["base_prediction_receipt"]["path"]) != record[
            "base_prediction_receipt"
        ]["sha256"]:
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
        expected_differences = PATH_ONLY_ARGS | {"score_render_resolution"}
        if set(differences) != expected_differences:
            raise ValueError(
                f"{scene}: candidate argument differences are not single-factor: "
                f"{sorted(differences)}"
            )
        if differences["score_render_resolution"] != (
            "scaled_renderer",
            "prompt_native",
        ):
            raise ValueError(f"{scene}: score-render factor differs")
        base_primitive = dict(base_method.pop("primitive_unary_artifact"))
        candidate_primitive = dict(candidate_method.pop("primitive_unary_artifact"))
        if _sha256(base_primitive["path"]) != base_primitive["file_sha256"]:
            raise ValueError(f"{scene}: base primitive artifact changed")
        if _sha256(candidate_primitive["path"]) != candidate_primitive["file_sha256"]:
            raise ValueError(f"{scene}: candidate primitive artifact changed")
        base_primitive_payload = torch.load(
            base_primitive["path"], map_location="cpu", weights_only=True
        )
        candidate_primitive_payload = torch.load(
            candidate_primitive["path"], map_location="cpu", weights_only=True
        )
        base_unary = base_primitive_payload["primitive_unary_probability"].float()
        candidate_unary = candidate_primitive_payload[
            "primitive_unary_probability"
        ].float()
        if base_unary.shape != candidate_unary.shape:
            raise ValueError(f"{scene}: primitive unary shape changed")
        unary_difference = (base_unary - candidate_unary).abs()
        unary_max_abs_difference = float(unary_difference.max().item())
        unary_mean_abs_difference = float(unary_difference.mean().item())
        # Recomputing the otherwise identical CUDA unary can differ in the last
        # few float32 bits.  This machine-epsilon acceptance is fixed before GT
        # is opened and is far below the 0.5 readout threshold resolution.
        if unary_max_abs_difference > 5e-7 or unary_mean_abs_difference > 1e-8:
            raise ValueError(
                f"{scene}: primitive unary numerical drift exceeds tolerance"
            )
        base_method.pop("score_render_resolution")
        candidate_resolution = candidate_method.pop("score_render_resolution")
        if candidate_resolution != "prompt_native" or base_method != candidate_method:
            raise ValueError(f"{scene}: method contract changed beyond scalar resolution")
        scores: dict[str, np.ndarray] = {}
        score_records = receipt["target_scores"]
        for frame_id, item in score_records.items():
            if _sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"{scene}/{frame_id}: score changed")
            scores[frame_id] = np.load(item["path"], allow_pickle=False)
        prompt = scene_rows[scene]["prompt"]
        prompt_path = prompt.get("positive_path") or prompt.get("mask_path")
        prompt_shape = list(load_ground_truth_mask(prompt_path).shape)
        score_shapes = {frame_id: list(value.shape) for frame_id, value in scores.items()}
        if any(shape != prompt_shape for shape in score_shapes.values()):
            raise ValueError(f"{scene}: prompt-native score shape is not native prompt shape")
        verified[scene] = {
            "receipt": str(receipt_path.resolve()),
            "receipt_sha256": _sha256(receipt_path),
            "score_records": score_records,
            "scores": scores,
            "score_shapes": score_shapes,
            "prompt_native_shape": prompt_shape,
            "primitive_unary": candidate_primitive,
            "primitive_unary_base": base_primitive,
            "primitive_unary_numerical_replay": {
                "bit_exact": bool(torch.equal(base_unary, candidate_unary)),
                "max_abs_difference": unary_max_abs_difference,
                "mean_abs_difference": unary_mean_abs_difference,
                "maximum_allowed": 5e-7,
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
            "score_resize": "cv2.INTER_LINEAR (identity because native shapes match)",
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
