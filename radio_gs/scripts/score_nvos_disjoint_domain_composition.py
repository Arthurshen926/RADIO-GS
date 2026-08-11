#!/usr/bin/env python3
"""Verify and independently score sealed NVOS disjoint-domain predictions."""

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
from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.querying.disjoint_domain_composition import MODE
from radio_gs.querying.multiview_region_memory_runtime import RUNTIME_MODE


LIKELIHOOD_MODE = "balanced_reference_source_reconstruction_v1"
HISTORICAL_REPLAY_ENVELOPE = 5e-3
PATH_ONLY_ARGS = {"output_dir", "prediction_receipt_output", "primitive_unary_output"}
FACTOR_ARGS = {
    "registered_query_likelihood_calibration",
    "registered_disjoint_domain_composition",
    "object_multiview_region_memory",
    "object_region_memory",
    "object_region_memory_sha256",
    "source_only_correspondence_completion",
    "source_correspondence_support_graph",
    "source_correspondence_support_graph_sha256",
    "source_multiview_responsibility_cache",
    "source_multiview_responsibility_cache_sha256",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _normalized(arguments: Mapping[str, object]) -> dict[str, object]:
    result = dict(arguments)
    for name in PATH_ONLY_ARGS | FACTOR_ARGS:
        result.pop(name, None)
    return result


def _write_noclobber(path: str | Path, payload: Mapping[str, object]) -> Path:
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
            raise ValueError(f"refusing to replace different score: {output}")
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _score(score: np.ndarray, target: np.ndarray) -> dict[str, float]:
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


def _verify_partition(
    primitive: Mapping[str, object],
    likelihood_primitive: Mapping[str, object],
    memory_primitive: Mapping[str, object],
) -> dict[str, object]:
    partition = primitive.get("disjoint_domain_partition")
    compiler = primitive["compiler_contract"]
    diagnostics = compiler.get("registered_disjoint_domain_composition")
    if not isinstance(partition, Mapping) or not isinstance(diagnostics, Mapping):
        raise ValueError("disjoint partition authority is missing")
    names = ("observed_rows", "memory_rows", "abstained_rows", "hard_anchor_rows")
    masks = {name: torch.as_tensor(partition[name]).bool().reshape(-1) for name in names}
    if any(
        tensor_sha256(masks[name].contiguous()) != partition["tensor_sha256"][name]
        for name in names
    ):
        raise ValueError("disjoint partition tensor changed")
    observed = masks["observed_rows"]
    memory = masks["memory_rows"]
    abstained = masks["abstained_rows"]
    anchors = masks["hard_anchor_rows"]
    if not (
        torch.equal(partition["global_rows"], primitive["valid_rows"])
        and torch.equal(observed | memory | abstained, torch.ones_like(observed))
        and not bool(
            (observed & memory).any()
            or (observed & abstained).any()
            or (memory & abstained).any()
            or (anchors & ~observed).any()
        )
        and diagnostics.get("partition_exhaustive") is True
        and diagnostics.get("partition_pairwise_disjoint") is True
        and diagnostics.get("assignment_commutative") is True
        and diagnostics.get("same_row_double_counted") is False
        and diagnostics.get("probability_average_or_product_of_experts_used") is False
        and diagnostics.get("observed_mask_sha256")
        == tensor_sha256(observed.contiguous())
        and int(diagnostics.get("observed_rows", -1)) == int(observed.sum())
        and int(diagnostics.get("memory_rows", -1)) == int(memory.sum())
        and int(diagnostics.get("abstained_rows", -1)) == int(abstained.sum())
    ):
        raise ValueError("disjoint partition causal contract differs")
    ql_diagnostics = likelihood_primitive["compiler_contract"][
        "registered_query_likelihood"
    ]
    memory_diagnostics = memory_primitive["compiler_contract"][
        "object_multiview_region_memory"
    ]["diagnostics"]
    if not (
        int(observed.sum())
        == int(ql_diagnostics["observed_rows"])
        == int(memory_diagnostics["base_observed_rows"])
        and int((~observed).sum())
        == int(ql_diagnostics["abstained_rows"])
        == int(memory_diagnostics["base_abstained_rows"])
        and int(memory.sum()) == int(memory_diagnostics["completed_rows"])
    ):
        raise ValueError("original observed mask lineage differs")
    candidate_probability = torch.as_tensor(primitive["primitive_unary_probability"])[
        primitive["valid_rows"]
    ].float()
    learned_probability = torch.as_tensor(
        likelihood_primitive["primitive_unary_probability"]
    )[likelihood_primitive["valid_rows"]].float()
    memory_probability = torch.as_tensor(memory_primitive["primitive_unary_probability"])[
        memory_primitive["valid_rows"]
    ].float()
    learned_delta = (candidate_probability[observed] - learned_probability[observed]).abs()
    memory_delta = (candidate_probability[memory] - memory_probability[memory]).abs()
    if not (
        float(learned_delta.max()) < HISTORICAL_REPLAY_ENVELOPE
        and float(memory_delta.max()) < HISTORICAL_REPLAY_ENVELOPE
        and not bool(
            (
                (candidate_probability[observed] >= 0.5)
                != (learned_probability[observed] >= 0.5)
            ).any()
        )
        and not bool(
            (
                (candidate_probability[memory] >= 0.5)
                != (memory_probability[memory] >= 0.5)
            ).any()
        )
    ):
        raise ValueError("sealed branch decision replay differs")
    return {
        "observed_rows": int(observed.sum()),
        "memory_rows": int(memory.sum()),
        "abstained_rows": int(abstained.sum()),
        "hard_anchor_rows": int(anchors.sum()),
        "observed_mask_sha256": tensor_sha256(observed.contiguous()),
        "historical_replay_fail_safe_envelope": HISTORICAL_REPLAY_ENVELOPE,
        "maximum_historical_likelihood_probability_drift": float(learned_delta.max()),
        "maximum_historical_memory_probability_drift": float(memory_delta.max()),
        "historical_branch_decision_flips_at_0_5": 0,
    }


def run(
    *,
    preregistration: str | Path,
    preregistration_sha256: str,
    scorer_correction: str | Path,
    scorer_correction_sha256: str,
    base_run_manifest: str | Path,
    base_exact_results: str | Path,
    candidate_root: str | Path,
    scenes: Sequence[str],
    evaluator: str | Path,
    evaluator_sha256: str,
    output: str | Path,
) -> Path:
    prereg_path = Path(preregistration).resolve()
    if _sha256(prereg_path) != preregistration_sha256:
        raise ValueError("preregistration changed")
    prereg = _load_json(prereg_path)
    correction_path = Path(scorer_correction).resolve()
    if _sha256(correction_path) != scorer_correction_sha256:
        raise ValueError("scorer correction changed")
    correction = _load_json(correction_path)
    if not (
        prereg.get("artifact_type")
        == "nvos_disjoint_domain_composition_fern_trex_preregistration_v1"
        and correction.get("artifact_type")
        == "nvos_disjoint_domain_composition_scorer_correction_v1"
        and correction.get("preregistration", {}).get("path")
        == str(prereg_path)
        and correction.get("preregistration", {}).get("sha256")
        == preregistration_sha256
        and correction.get("original_scorer", {}).get("sha256")
        == prereg["frozen_implementation"]["scorer"]["sha256"]
        and correction.get("corrected_scorer", {}).get("path")
        == str(Path(__file__).resolve())
        and correction.get("corrected_scorer", {}).get("sha256")
        == _sha256(Path(__file__).resolve())
        and correction.get("target_ground_truth_opened_before_correction") is False
    ):
        raise ValueError("preregistration scorer binding differs")
    evaluator_path = Path(evaluator).resolve()
    if _sha256(evaluator_path) != evaluator_sha256:
        raise ValueError("candidate evaluator changed")
    base_manifest_path = Path(base_run_manifest).resolve()
    base_results_path = Path(base_exact_results).resolve()
    base_manifest = _load_json(base_manifest_path)
    base_results = _load_json(base_results_path)
    benchmark_path = Path(base_manifest["manifest"])
    if _sha256(benchmark_path) != base_manifest["manifest_sha256"]:
        raise ValueError("benchmark manifest changed")
    benchmark = _load_json(benchmark_path)
    scene_rows = {str(row["scene_id"]): row for row in benchmark["scenes"]}

    verified: dict[str, object] = {}
    root = Path(candidate_root).resolve()
    for scene in scenes:
        inputs = prereg["sealed_inputs"][scene]
        base_receipt = _load_json(inputs["base_prediction_receipt"]["path"])
        likelihood_receipt = _load_json(inputs["likelihood_prediction_receipt"]["path"])
        memory_receipt = _load_json(inputs["memory_prediction_receipt"]["path"])
        for name, receipt in (
            ("base", base_receipt),
            ("likelihood", likelihood_receipt),
            ("memory", memory_receipt),
        ):
            record = inputs[f"{name}_prediction_receipt"]
            if _sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"{scene}: {name} receipt changed")
        receipt_path = root / scene / "pre_metric_prediction_receipt.json"
        receipt = _load_json(receipt_path)
        if not (
            receipt.get("scene_id") == scene
            and receipt.get("sealed_before_target_ground_truth_open") is True
            and receipt.get("target_rgb_opened") is False
            and receipt.get("target_mask_opened") is False
            and receipt.get("target_metric_opened") is False
        ):
            raise ValueError(f"{scene}: candidate safety barrier differs")
        method = receipt["method_contract"]
        args = dict(method["candidate_args"])
        base_args = dict(base_receipt["method_contract"]["candidate_args"])
        if not (
            method.get("evaluator_sha256") == evaluator_sha256
            and args.get("registered_query_likelihood_calibration") == LIKELIHOOD_MODE
            and args.get("object_multiview_region_memory") == RUNTIME_MODE
            and args.get("registered_disjoint_domain_composition") == MODE
            and _normalized(args) == _normalized(base_args)
            and float(method["score_threshold"]) == 0.5
        ):
            raise ValueError(f"{scene}: candidate method differs")
        primitive_record = method["primitive_unary_artifact"]
        if _sha256(primitive_record["path"]) != primitive_record["file_sha256"]:
            raise ValueError(f"{scene}: candidate primitive changed")
        primitive = torch.load(primitive_record["path"], map_location="cpu", weights_only=True)
        ql_record = inputs["likelihood_primitive"]
        memory_record = inputs["memory_primitive"]
        if _sha256(ql_record["path"]) != ql_record["sha256"] or _sha256(
            memory_record["path"]
        ) != memory_record["sha256"]:
            raise ValueError(f"{scene}: sealed branch primitive changed")
        likelihood_primitive = torch.load(
            ql_record["path"], map_location="cpu", weights_only=True
        )
        memory_primitive = torch.load(
            memory_record["path"], map_location="cpu", weights_only=True
        )
        partition = _verify_partition(
            primitive, likelihood_primitive, memory_primitive
        )
        scores = {}
        for frame_id, record in receipt["target_scores"].items():
            if _sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"{scene}/{frame_id}: score changed")
            scores[str(frame_id)] = np.load(record["path"], allow_pickle=False)
        verified[scene] = {
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
            "primitive_unary": primitive_record,
            "partition": partition,
            "scores": scores,
            "score_records": receipt["target_scores"],
        }

    per_scene = {}
    for scene in scenes:
        row = scene_rows[scene]
        frames = []
        for frame_id in (str(value) for value in row["evaluation_frame_ids"]):
            frame = next(item for item in row["frames"] if str(item["frame_id"]) == frame_id)
            target_path = Path(frame["ground_truth"])
            if _sha256(target_path) != frame["ground_truth_sha256"]:
                raise ValueError(f"{scene}/{frame_id}: target changed")
            frames.append(
                {
                    "frame_id": frame_id,
                    **_score(
                        verified[scene]["scores"][frame_id],
                        load_ground_truth_mask(target_path),
                    ),
                }
            )
        baseline_row = base_results["per_scene"][scene]
        baseline = float(baseline_row.get("foreground_iou", baseline_row.get("iou")))
        iou = float(np.mean([frame["foreground_iou"] for frame in frames]))
        per_scene[scene] = {
            "foreground_iou": iou,
            "baseline_foreground_iou": baseline,
            "delta_foreground_iou": iou - baseline,
            "pixel_accuracy": float(np.mean([frame["pixel_accuracy"] for frame in frames])),
            "frames": frames,
            **{key: value for key, value in verified[scene].items() if key != "scores"},
        }
    macro = float(np.mean([row["foreground_iou"] for row in per_scene.values()]))
    base_macro = float(np.mean([row["baseline_foreground_iou"] for row in per_scene.values()]))
    return _write_noclobber(
        output,
        {
            "schema_version": 1,
            "artifact_type": "nvos_disjoint_domain_composition_exact_result_v1",
            "status": "all_predictions_verified_then_exact_scored",
            "scene_order": list(scenes),
            "preregistration": {"path": str(prereg_path), "sha256": preregistration_sha256},
            "scorer_correction": {
                "path": str(correction_path),
                "sha256": scorer_correction_sha256,
            },
            "per_scene": per_scene,
            "aggregate": {
                "scene_macro_foreground_iou": macro,
                "baseline_scene_macro_foreground_iou": base_macro,
                "delta_foreground_iou": macro - base_macro,
            },
            "safety": {
                "all_requested_receipts_verified_before_first_target_mask_open": True,
                "target_rgb_used": False,
                "target_metric_used_to_fit_or_select_parameters": False,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--scorer-correction", required=True)
    parser.add_argument("--scorer-correction-sha256", required=True)
    parser.add_argument("--base-run-manifest", required=True)
    parser.add_argument("--base-exact-results", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(
        preregistration=args.preregistration,
        preregistration_sha256=args.preregistration_sha256,
        scorer_correction=args.scorer_correction,
        scorer_correction_sha256=args.scorer_correction_sha256,
        base_run_manifest=args.base_run_manifest,
        base_exact_results=args.base_exact_results,
        candidate_root=args.candidate_root,
        scenes=args.scenes,
        evaluator=args.evaluator,
        evaluator_sha256=args.evaluator_sha256,
        output=args.output,
    )
    print(json.dumps({"output": str(result), "sha256": _sha256(result)}, indent=2))


if __name__ == "__main__":
    main()
