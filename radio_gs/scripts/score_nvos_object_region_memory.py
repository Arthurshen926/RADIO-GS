#!/usr/bin/env python3
"""Independently verify and score sealed NVOS object-region predictions."""

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
from radio_gs.querying.multiview_region_memory_runtime import RUNTIME_MODE


PATH_ONLY_ARGS = {"output_dir", "prediction_receipt_output", "primitive_unary_output"}
OBJECT_ARGS = {
    "object_multiview_region_memory",
    "object_region_memory",
    "object_region_memory_sha256",
}
NEW_DEFAULT_ARGS = {
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


def _normalized_base_args(values: Mapping[str, object]) -> dict[str, object]:
    result = dict(values)
    for name in PATH_ONLY_ARGS | OBJECT_ARGS | NEW_DEFAULT_ARGS:
        result.pop(name, None)
    result.setdefault("registered_query_likelihood_calibration", "none")
    return result


def _score(score: np.ndarray, target: np.ndarray) -> dict[str, float]:
    values = np.asarray(score)
    ground_truth = np.asarray(target, dtype=bool)
    if values.ndim != 2 or ground_truth.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("score and target must be finite 2D arrays")
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


def _verify_causal_change_domains(
    *,
    candidate_probability: torch.Tensor,
    base_probability: torch.Tensor,
    valid: torch.Tensor,
    memory: Mapping[str, object],
    diagnostics: Mapping[str, object],
    threshold: float = 1e-3,
) -> dict[str, object]:
    """Verify logit- and probability-domain changes without equating counts.

    The frozen evaluator measures ``materially_changed_rows`` on its in-memory
    unary logits.  The sealed primitive artifact contains sigmoid probabilities,
    so applying the same absolute threshold after sigmoid is not count
    preserving.  The logit-domain subset proof comes from the hashed evaluator:
    it initializes from the base unary and writes only ``changed_rows`` returned
    by the c==0 completion adapter.  Here we independently replay the observable
    probability-domain subset against the sealed memory authority.
    """

    candidate = torch.as_tensor(candidate_probability).float().reshape(-1)
    base = torch.as_tensor(base_probability).float().reshape(-1)
    valid_rows_mask = torch.as_tensor(valid).bool().reshape(-1)
    if candidate.shape != base.shape or candidate.shape != valid_rows_mask.shape:
        raise ValueError("primitive probability row authority differs")
    memory_valid_rows = torch.as_tensor(memory["valid_rows"]).long().reshape(-1)
    memory_confidence = (
        torch.as_tensor(memory["membership_confidence"]).float().reshape(-1)
    )
    if memory_valid_rows.shape != memory_confidence.shape:
        raise ValueError("object memory row authority differs")
    if bool((memory_valid_rows < 0).any()) or bool(
        (memory_valid_rows >= candidate.numel()).any()
    ):
        raise ValueError("object memory valid rows are out of range")
    if not torch.equal(
        valid_rows_mask[memory_valid_rows],
        torch.ones_like(memory_valid_rows, dtype=torch.bool),
    ):
        raise ValueError("object memory rows are outside the valid primitive domain")
    authorized = torch.zeros_like(valid_rows_mask)
    authorized[memory_valid_rows[memory_confidence > 0]] = True
    absolute_delta = (candidate - base).abs()
    non_memory_valid = valid_rows_mask & ~authorized
    probability_changed = absolute_delta > float(threshold)
    probability_decision_flip = (candidate >= 0.5) != (base >= 0.5)
    if bool((probability_changed & ~valid_rows_mask).any()):
        raise ValueError("probability-domain changes escape the valid primitive domain")
    if bool((probability_changed & ~authorized).any()):
        raise ValueError("probability-domain changes escape memory-authorized rows")
    if bool((probability_decision_flip & non_memory_valid).any()):
        raise ValueError("historical replay drift changes a non-memory decision")
    probability_changed_rows = int(probability_changed.sum())
    if probability_changed_rows <= 0:
        raise ValueError("object memory caused no material probability change")

    completed_rows = int(diagnostics.get("completed_rows", 0))
    logit_changed_rows = int(diagnostics.get("materially_changed_rows", 0))
    if not (0 < logit_changed_rows <= completed_rows <= int(authorized.sum())):
        raise ValueError("logit-domain change counts escape memory-authorized rows")
    return {
        "change_threshold": float(threshold),
        "logit_domain_materially_changed_rows": logit_changed_rows,
        "probability_domain_materially_changed_rows": probability_changed_rows,
        "authorized_memory_rows": int(authorized.sum()),
        "historical_replay_numerical_envelope": float(threshold),
        "maximum_probability_drift_outside_memory_authority": float(
            absolute_delta[non_memory_valid].max()
            if bool(non_memory_valid.any())
            else 0.0
        ),
        "non_memory_probability_decision_flips_at_0_5": int(
            (probability_decision_flip & non_memory_valid).sum()
        ),
        "counts_expected_to_match_across_sigmoid": False,
        "probability_changes_subset_of_valid_rows": True,
        "probability_changes_subset_of_memory_authorized_rows": True,
        "final_material_write_mask_subset_of_memory_authorized_rows": True,
        "non_memory_selected_bits_bitwise_equal": True,
        "logit_changes_subset_of_completed_authorized_rows": True,
        "maximum_primitive_probability_delta_vs_base": float(absolute_delta.max()),
    }


def run(
    *,
    base_run_manifest: str | Path,
    base_exact_results: str | Path,
    candidate_root: str | Path,
    scenes: Sequence[str],
    evaluator: str | Path,
    evaluator_sha256: str,
    runtime_module: str | Path,
    runtime_module_sha256: str,
    addendum: str | Path,
    addendum_sha256: str,
    scorer_correction: str | Path,
    scorer_correction_sha256: str,
    output: str | Path,
) -> Path:
    base_manifest_path = Path(base_run_manifest).expanduser().resolve()
    base_results_path = Path(base_exact_results).expanduser().resolve()
    evaluator_path = Path(evaluator).expanduser().resolve()
    runtime_path = Path(runtime_module).expanduser().resolve()
    addendum_path = Path(addendum).expanduser().resolve()
    correction_path = Path(scorer_correction).expanduser().resolve()
    if _sha256(evaluator_path) != evaluator_sha256:
        raise ValueError("candidate evaluator changed")
    if _sha256(runtime_path) != runtime_module_sha256:
        raise ValueError("object region-memory runtime changed")
    if _sha256(addendum_path) != addendum_sha256:
        raise ValueError("object region-memory addendum changed")
    if _sha256(correction_path) != scorer_correction_sha256:
        raise ValueError("scorer correction authority changed")
    authority = _load_json(addendum_path)
    if authority.get("artifact_type") != (
        "nvos_object_multiview_region_memory_runtime_addendum_v1"
    ):
        raise ValueError("object region-memory addendum differs")
    correction = _load_json(correction_path)
    if not (
        correction.get("artifact_type")
        == "nvos_object_multiview_region_memory_scorer_correction_v2"
        and correction.get("base_runtime_addendum", {}).get("path")
        == str(addendum_path)
        and correction.get("base_runtime_addendum", {}).get("sha256")
        == addendum_sha256
        and correction.get("corrected_scorer", {}).get("path")
        == str(Path(__file__).resolve())
        and correction.get("corrected_scorer", {}).get("sha256")
        == _sha256(Path(__file__).resolve())
        and correction.get("target_ground_truth_opened_before_correction") is False
    ):
        raise ValueError("scorer correction contract differs")
    memories = authority.get("sealed_memories")
    if not isinstance(memories, Mapping):
        raise ValueError("addendum lacks sealed memories")
    base_manifest = _load_json(base_manifest_path)
    base_results = _load_json(base_results_path)
    benchmark_path = Path(str(base_manifest["manifest"]))
    if _sha256(benchmark_path) != base_manifest["manifest_sha256"]:
        raise ValueError("frozen benchmark manifest changed")
    benchmark = _load_json(benchmark_path)
    scene_rows = {str(row["scene_id"]): row for row in benchmark["scenes"]}

    verified: dict[str, object] = {}
    root = Path(candidate_root).expanduser().resolve()
    for scene in scenes:
        base_record = base_manifest["records"][scene]
        base_receipt = _load_json(base_record["prediction_receipt"])
        receipt_path = root / scene / "pre_metric_prediction_receipt.json"
        receipt = _load_json(receipt_path)
        if not (
            receipt.get("scene_id") == scene
            and receipt.get("sealed_before_target_ground_truth_open") is True
            and receipt.get("target_rgb_opened") is False
            and receipt.get("target_mask_opened") is False
            and receipt.get("target_metric_opened") is False
        ):
            raise ValueError(f"{scene}: prediction safety barrier differs")
        method = dict(receipt["method_contract"])
        base_method = dict(base_receipt["method_contract"])
        if method.get("evaluator_sha256") != evaluator_sha256:
            raise ValueError(f"{scene}: evaluator authority differs")
        candidate_args = dict(method["candidate_args"])
        base_args = dict(base_method["candidate_args"])
        if candidate_args.get("object_multiview_region_memory") != RUNTIME_MODE:
            raise ValueError(f"{scene}: object memory mode differs")
        if _normalized_base_args(candidate_args) != _normalized_base_args(base_args):
            left = _normalized_base_args(candidate_args)
            right = _normalized_base_args(base_args)
            differing = sorted(
                name for name in set(left) | set(right) if left.get(name) != right.get(name)
            )
            raise ValueError(f"{scene}: non-single-factor args: {differing}")
        registered_memory = memories.get(scene)
        if not isinstance(registered_memory, Mapping):
            raise ValueError(f"{scene}: memory is absent from addendum")
        if (
            candidate_args.get("object_region_memory") != registered_memory.get("path")
            or candidate_args.get("object_region_memory_sha256")
            != registered_memory.get("sha256")
            or _sha256(candidate_args["object_region_memory"])
            != candidate_args["object_region_memory_sha256"]
        ):
            raise ValueError(f"{scene}: sealed memory binding differs")
        primitive_record = method["primitive_unary_artifact"]
        primitive_path = Path(primitive_record["path"])
        if _sha256(primitive_path) != primitive_record["file_sha256"]:
            raise ValueError(f"{scene}: primitive unary changed")
        candidate_primitive = torch.load(primitive_path, map_location="cpu", weights_only=True)
        base_primitive_path = Path(base_record["output_dir"]) / "primitive_unary.pt"
        base_primitive = torch.load(base_primitive_path, map_location="cpu", weights_only=True)
        completion = candidate_primitive["compiler_contract"].get(
            "object_multiview_region_memory"
        )
        diagnostics = completion.get("diagnostics", {}) if isinstance(completion, Mapping) else {}
        if not (
            isinstance(completion, Mapping)
            and completion.get("mode") == RUNTIME_MODE
            and candidate_primitive["compiler_contract"].get("graph_disabled") is True
            and candidate_primitive["compiler_contract"].get("readout") == "unary_prior"
            and candidate_primitive["compiler_contract"].get(
                "connected_selection_applied"
            )
            is False
            and diagnostics.get("observed_values_bitwise_equal") is True
            and diagnostics.get("observed_confidence_bitwise_equal") is True
            and diagnostics.get("observed_unary_bitwise_equal") is True
            and int(diagnostics.get("completed_rows", 0)) > 0
            and int(diagnostics.get("materially_changed_rows", 0)) > 0
        ):
            raise ValueError(f"{scene}: runtime causal gate failed")
        candidate_probability = torch.as_tensor(
            candidate_primitive["primitive_unary_probability"]
        ).float()
        base_probability = torch.as_tensor(base_primitive["primitive_unary_probability"]).float()
        if (
            candidate_probability.shape != base_probability.shape
            or not torch.equal(candidate_primitive["valid"], base_primitive["valid"])
        ):
            raise ValueError(f"{scene}: primitive row authority differs")
        memory_payload = torch.load(
            candidate_args["object_region_memory"],
            map_location="cpu",
            weights_only=True,
        )
        causal_replay = _verify_causal_change_domains(
            candidate_probability=candidate_probability,
            base_probability=base_probability,
            valid=candidate_primitive["valid"],
            memory=memory_payload,
            diagnostics=diagnostics,
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
            "scores": scores,
            "score_records": receipt["target_scores"],
            "diagnostics": dict(diagnostics),
            "causal_replay": causal_replay,
            "materially_changed_rows_vs_base": causal_replay[
                "probability_domain_materially_changed_rows"
            ],
            "maximum_primitive_probability_delta_vs_base": causal_replay[
                "maximum_primitive_probability_delta_vs_base"
            ],
        }

    per_scene = {}
    for scene in scenes:
        scene_row = scene_rows[scene]
        expected_frames = tuple(str(value) for value in scene_row["evaluation_frame_ids"])
        if tuple(verified[scene]["scores"]) != expected_frames:
            raise ValueError(f"{scene}: evaluation frame order differs")
        frames = []
        for frame_id in expected_frames:
            row = next(item for item in scene_row["frames"] if str(item["frame_id"]) == frame_id)
            target_path = Path(row["ground_truth"])
            if _sha256(target_path) != row["ground_truth_sha256"]:
                raise ValueError(f"{scene}/{frame_id}: target authority changed")
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
    payload = {
        "schema_version": 1,
        "artifact_type": "nvos_object_multiview_region_memory_exact_result_v1",
        "status": "sealed_predictions_verified_then_exact_scored",
        "scene_order": list(scenes),
        "base_run_manifest": {"path": str(base_manifest_path), "sha256": _sha256(base_manifest_path)},
        "base_exact_results": {"path": str(base_results_path), "sha256": _sha256(base_results_path)},
        "evaluator": {"path": str(evaluator_path), "sha256": evaluator_sha256},
        "runtime_module": {"path": str(runtime_path), "sha256": runtime_module_sha256},
        "addendum": {"path": str(addendum_path), "sha256": addendum_sha256},
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
        "evaluation_protocol": {
            "resize": "cv2.INTER_LINEAR",
            "threshold": 0.5,
            "aggregation": "per-frame then task-instance scene macro",
        },
        "safety": {
            "all_requested_receipts_verified_before_first_target_mask_open": True,
            "target_rgb_used": False,
            "target_metric_used_to_fit_or_select_parameters": False,
        },
    }
    return _write_noclobber(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-manifest", required=True)
    parser.add_argument("--base-exact-results", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--runtime-module", required=True)
    parser.add_argument("--runtime-module-sha256", required=True)
    parser.add_argument("--addendum", required=True)
    parser.add_argument("--addendum-sha256", required=True)
    parser.add_argument("--scorer-correction", required=True)
    parser.add_argument("--scorer-correction-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(
        base_run_manifest=args.base_run_manifest,
        base_exact_results=args.base_exact_results,
        candidate_root=args.candidate_root,
        scenes=args.scenes,
        evaluator=args.evaluator,
        evaluator_sha256=args.evaluator_sha256,
        runtime_module=args.runtime_module,
        runtime_module_sha256=args.runtime_module_sha256,
        addendum=args.addendum,
        addendum_sha256=args.addendum_sha256,
        scorer_correction=args.scorer_correction,
        scorer_correction_sha256=args.scorer_correction_sha256,
        output=args.output,
    )
    print(json.dumps({"output": str(result), "sha256": _sha256(result)}, indent=2))


if __name__ == "__main__":
    main()
