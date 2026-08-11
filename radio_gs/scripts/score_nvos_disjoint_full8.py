#!/usr/bin/env python3
"""Verify a sealed NVOS full8 prediction batch, then score its six new scenes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask
from radio_gs.querying.disjoint_domain_composition import MODE
from radio_gs.querying.multiview_region_memory_runtime import RUNTIME_MODE
from radio_gs.scripts.run_nvos_disjoint_full8_prediction import (
    ARTIFACT_TYPE as PREREGISTRATION_ARTIFACT_TYPE,
    LIKELIHOOD_MODE,
    load_torch_record,
    normalized_arguments,
    safe_prediction_receipt,
    sha256_file,
    verify_partition_artifact,
)


BATCH_ARTIFACT_TYPE = "nvos_disjoint_domain_composition_full8_prediction_batch_authority_v1"
RESULT_ARTIFACT_TYPE = "nvos_disjoint_domain_composition_full8_exact_result_v1"
FULL8_SCENES = (
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
)
NEW_SCENES = (
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
)


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_noclobber(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    except FileExistsError:
        if output.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different result: {output}")
    finally:
        temporary.unlink(missing_ok=True)
    return output


def score_probability(score: np.ndarray, target: np.ndarray) -> dict[str, float]:
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


def verify_new_scene_before_gt(
    scene: str,
    *,
    batch: Mapping[str, Any],
    prereg: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = prereg["sealed_inputs"][scene]
    batch_row = batch["predictions"][scene]
    receipt_path = Path(batch_row["prediction_receipt"]["path"]).resolve(strict=True)
    if sha256_file(receipt_path) != batch_row["prediction_receipt"]["sha256"]:
        raise ValueError(f"{scene}: prediction receipt changed")
    receipt = load_json(receipt_path)
    if not safe_prediction_receipt(receipt, scene):
        raise ValueError(f"{scene}: prediction safety barrier differs")
    base_receipt_path = Path(inputs["base_prediction_receipt"]["path"])
    if sha256_file(base_receipt_path) != inputs["base_prediction_receipt"]["sha256"]:
        raise ValueError(f"{scene}: base receipt changed")
    base_receipt = load_json(base_receipt_path)
    method = receipt["method_contract"]
    candidate_args = dict(method["candidate_args"])
    base_args = dict(base_receipt["method_contract"]["candidate_args"])
    evaluator = prereg["frozen_implementation"]["evaluator"]
    if not (
        method.get("evaluator_sha256") == evaluator["sha256"]
        and candidate_args.get("registered_query_likelihood_calibration")
        == LIKELIHOOD_MODE
        and candidate_args.get("registered_disjoint_domain_composition") == MODE
        and candidate_args.get("object_multiview_region_memory") == RUNTIME_MODE
        and normalized_arguments(candidate_args) == normalized_arguments(base_args)
        and float(method.get("score_threshold", -1)) == 0.5
    ):
        raise ValueError(f"{scene}: candidate method differs")
    primitive_record = batch_row["primitive_unary"]
    method_primitive = method["primitive_unary_artifact"]
    if not (
        method_primitive["path"] == primitive_record["path"]
        and method_primitive["file_sha256"] == primitive_record["sha256"]
    ):
        raise ValueError(f"{scene}: primitive receipt binding differs")
    primitive = load_torch_record(primitive_record, label=f"{scene} candidate primitive")
    base_primitive = load_torch_record(inputs["base_primitive"], label=f"{scene} base primitive")
    partition = verify_partition_artifact(primitive, base_primitive)
    if partition != batch_row["partition"]:
        raise ValueError(f"{scene}: sealed partition summary differs")

    scores: dict[str, np.ndarray] = {}
    score_records = receipt.get("target_scores")
    if not isinstance(score_records, Mapping) or not score_records:
        raise ValueError(f"{scene}: no sealed target score maps")
    for frame_id, record in score_records.items():
        score_path = Path(record["path"]).resolve(strict=True)
        if sha256_file(score_path) != record["sha256"]:
            raise ValueError(f"{scene}/{frame_id}: score map changed")
        scores[str(frame_id)] = np.load(score_path, allow_pickle=False)
    return {
        "prediction_receipt": batch_row["prediction_receipt"],
        "primitive_unary": primitive_record,
        "partition": partition,
        "score_records": dict(score_records),
        "scores": scores,
    }


def run(args: argparse.Namespace) -> Path:
    batch_path = Path(args.batch_authority).expanduser().resolve(strict=True)
    if sha256_file(batch_path) != args.batch_authority_sha256:
        raise ValueError("full8 batch authority changed")
    batch = load_json(batch_path)
    scorer_path = Path(__file__).resolve()
    if not (
        batch.get("artifact_type") == BATCH_ARTIFACT_TYPE
        and batch.get("status")
        == "all_eight_composed_predictions_sealed_before_new_six_target_scoring"
        and tuple(batch.get("scene_order", [])) == FULL8_SCENES
        and tuple(batch.get("new_prediction_scene_order", [])) == NEW_SCENES
        and batch.get("frozen_implementation", {}).get("scorer", {}).get("path")
        == str(scorer_path)
        and batch.get("frozen_implementation", {}).get("scorer", {}).get("sha256")
        == sha256_file(scorer_path)
    ):
        raise ValueError("full8 batch authority contract differs")
    prereg_record = batch["preregistration"]
    prereg_path = Path(prereg_record["path"]).resolve(strict=True)
    if sha256_file(prereg_path) != prereg_record["sha256"]:
        raise ValueError("full8 preregistration changed")
    prereg = load_json(prereg_path)
    if not (
        prereg.get("artifact_type") == PREREGISTRATION_ARTIFACT_TYPE
        and prereg.get("frozen_implementation", {}).get("scorer", {}).get("sha256")
        == sha256_file(scorer_path)
    ):
        raise ValueError("full8 preregistration scorer binding differs")
    for scene in FULL8_SCENES:
        row = batch["predictions"].get(scene)
        if not isinstance(row, Mapping):
            raise ValueError(f"{scene}: missing prediction batch record")
        for name in ("prediction_receipt", "primitive_unary"):
            record = row.get(name)
            if not isinstance(record, Mapping) or sha256_file(record["path"]) != record["sha256"]:
                raise ValueError(f"{scene}: sealed {name} changed")

    pair_record = batch["authorized_pair_exact_result"]
    pair_path = Path(pair_record["path"]).resolve(strict=True)
    if sha256_file(pair_path) != pair_record["sha256"]:
        raise ValueError("authorized fern/trex exact result changed")
    pair = load_json(pair_path)
    if not (
        tuple(pair.get("scene_order", [])) == ("fern", "trex")
        and pair.get("status") == "all_predictions_verified_then_exact_scored"
    ):
        raise ValueError("authorized fern/trex result differs")

    # This entire loop, including loading all six score maps, finishes before a
    # target label path is resolved or opened below.
    verified = {
        scene: verify_new_scene_before_gt(scene, batch=batch, prereg=prereg)
        for scene in NEW_SCENES
    }

    base_run = prereg["frozen_base"]
    manifest_record = base_run["run_manifest"]
    exact_record = base_run["exact_results"]
    if sha256_file(manifest_record["path"]) != manifest_record["sha256"]:
        raise ValueError("base run manifest changed")
    if sha256_file(exact_record["path"]) != exact_record["sha256"]:
        raise ValueError("base exact result changed")
    run_manifest = load_json(manifest_record["path"])
    base_exact = load_json(exact_record["path"])
    benchmark_path = Path(run_manifest["manifest"])
    if sha256_file(benchmark_path) != run_manifest["manifest_sha256"]:
        raise ValueError("benchmark manifest changed")
    benchmark = load_json(benchmark_path)
    scene_rows = {str(row["scene_id"]): row for row in benchmark["scenes"]}

    per_scene: dict[str, Any] = {}
    for scene in ("fern", "trex"):
        pair_row = pair["per_scene"][scene]
        per_scene[scene] = {
            **pair_row,
            "reused_from_authorized_pair_exact_result": True,
        }
    for scene in NEW_SCENES:
        row = scene_rows[scene]
        frame_rows = {str(item["frame_id"]): item for item in row["frames"]}
        frames = []
        for frame_id in (str(value) for value in row["evaluation_frame_ids"]):
            frame = frame_rows[frame_id]
            target_path = Path(frame["ground_truth"])
            if sha256_file(target_path) != frame["ground_truth_sha256"]:
                raise ValueError(f"{scene}/{frame_id}: target changed")
            frames.append(
                {
                    "frame_id": frame_id,
                    **score_probability(
                        verified[scene]["scores"][frame_id],
                        load_ground_truth_mask(target_path),
                    ),
                }
            )
        baseline = float(base_exact["per_scene"][scene]["foreground_iou"])
        foreground_iou = float(np.mean([frame["foreground_iou"] for frame in frames]))
        per_scene[scene] = {
            "foreground_iou": foreground_iou,
            "baseline_foreground_iou": baseline,
            "delta_foreground_iou": foreground_iou - baseline,
            "pixel_accuracy": float(np.mean([frame["pixel_accuracy"] for frame in frames])),
            "frames": frames,
            **{name: value for name, value in verified[scene].items() if name != "scores"},
        }
    ordered = {scene: per_scene[scene] for scene in FULL8_SCENES}
    macro = float(np.mean([row["foreground_iou"] for row in ordered.values()]))
    baseline_macro = float(base_exact["aggregate"]["scene_macro_foreground_iou"])
    reference = 0.749
    return write_noclobber(
        args.output,
        {
            "schema_version": 1,
            "artifact_type": RESULT_ARTIFACT_TYPE,
            "status": "six_new_predictions_verified_then_scored_and_combined_with_authorized_pair",
            "scene_order": list(FULL8_SCENES),
            "batch_authority": {"path": str(batch_path), "sha256": args.batch_authority_sha256},
            "per_scene": ordered,
            "aggregate": {
                "scene_macro_foreground_iou": macro,
                "baseline_scene_macro_foreground_iou": baseline_macro,
                "delta_foreground_iou": macro - baseline_macro,
                "reference_foreground_iou": reference,
                "delta_to_reference_foreground_iou": macro - reference,
            },
            "safety": {
                "all_eight_receipts_and_all_six_new_score_maps_verified_before_first_new_target_mask_open": True,
                "fern_trex_reused_without_prediction_rerun": True,
                "target_rgb_used": False,
                "target_metric_used_to_fit_or_select_parameters": False,
                "all_scene_parameters_frozen_before_new_six_scoring": True,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-authority", required=True)
    parser.add_argument("--batch-authority-sha256", required=True)
    parser.add_argument("--output", required=True)
    result = run(parser.parse_args())
    print(json.dumps({"output": str(result), "sha256": sha256_file(result)}, indent=2))


if __name__ == "__main__":
    main()
