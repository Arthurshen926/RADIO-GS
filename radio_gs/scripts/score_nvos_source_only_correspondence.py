#!/usr/bin/env python3
"""Verify and score sealed NVOS source-only correspondence predictions."""

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
from radio_gs.querying.source_only_correspondence_completion import method_contract


MODE = "source_only_one_hop_signed_correspondence_completion_v1"
PATH_ONLY_ARGS = {"output_dir", "prediction_receipt_output", "primitive_unary_output"}
CORRESPONDENCE_ARGS = {
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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_no_clobber(path: str | Path, payload: Mapping[str, object]) -> Path:
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
    for name in PATH_ONLY_ARGS | CORRESPONDENCE_ARGS:
        result.pop(name, None)
    result.setdefault("registered_query_likelihood_calibration", "none")
    return result


def _score_frame(score: np.ndarray, target: np.ndarray) -> dict[str, float]:
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


def run(
    *,
    base_run_manifest: str | Path,
    base_exact_results: str | Path,
    candidate_root: str | Path,
    scenes: Sequence[str],
    evaluator: str | Path,
    evaluator_sha256: str,
    completion_module: str | Path,
    completion_module_sha256: str,
    preregistration: str | Path,
    preregistration_sha256: str,
    output: str | Path,
) -> Path:
    base_manifest_path = Path(base_run_manifest).expanduser().resolve()
    base_results_path = Path(base_exact_results).expanduser().resolve()
    root = Path(candidate_root).expanduser().resolve()
    evaluator_path = Path(evaluator).expanduser().resolve()
    module_path = Path(completion_module).expanduser().resolve()
    prereg_path = Path(preregistration).expanduser().resolve()
    if _sha256(evaluator_path) != evaluator_sha256:
        raise ValueError("candidate evaluator changed")
    if _sha256(module_path) != completion_module_sha256:
        raise ValueError("source correspondence module changed")
    if _sha256(prereg_path) != preregistration_sha256:
        raise ValueError("source correspondence preregistration changed")
    prereg = _load_json(prereg_path)
    if prereg.get("artifact_type") != (
        "nvos_source_only_correspondence_completion_fern_trex_preregistration_v1"
    ):
        raise ValueError("source correspondence preregistration authority differs")
    base_manifest = _load_json(base_manifest_path)
    base_results = _load_json(base_results_path)
    benchmark_path = Path(str(base_manifest["manifest"]))
    if _sha256(benchmark_path) != base_manifest["manifest_sha256"]:
        raise ValueError("frozen benchmark manifest changed")
    benchmark = _load_json(benchmark_path)
    scene_rows = {str(row["scene_id"]): row for row in benchmark["scenes"]}

    verified: dict[str, object] = {}
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
        if candidate_args.get("source_only_correspondence_completion") != MODE:
            raise ValueError(f"{scene}: correspondence mode differs")
        if _normalized_base_args(candidate_args) != _normalized_base_args(base_args):
            left = _normalized_base_args(candidate_args)
            right = _normalized_base_args(base_args)
            differing = sorted(
                name for name in set(left) | set(right) if left.get(name) != right.get(name)
            )
            raise ValueError(f"{scene}: non-single-factor args: {differing}")
        registered_assets = prereg["source_only_assets"][scene]
        if (
            candidate_args["source_correspondence_support_graph_sha256"]
            != registered_assets["typed_support_graph_sha256"]
            or candidate_args["source_multiview_responsibility_cache_sha256"]
            != registered_assets["multiview_responsibility_sha256"]
        ):
            raise ValueError(f"{scene}: source-only assets differ from preregistration")
        for path_key, sha_key in (
            ("source_correspondence_support_graph", "source_correspondence_support_graph_sha256"),
            ("source_multiview_responsibility_cache", "source_multiview_responsibility_cache_sha256"),
        ):
            if _sha256(candidate_args[path_key]) != candidate_args[sha_key]:
                raise ValueError(f"{scene}: {path_key} changed")

        primitive_record = method["primitive_unary_artifact"]
        primitive_path = Path(primitive_record["path"])
        if _sha256(primitive_path) != primitive_record["file_sha256"]:
            raise ValueError(f"{scene}: primitive unary changed")
        candidate_primitive = torch.load(primitive_path, map_location="cpu", weights_only=True)
        base_primitive_path = Path(base_record["output_dir"]) / "primitive_unary.pt"
        if _sha256(base_primitive_path) != registered_assets["base_unary_sha256"]:
            raise ValueError(f"{scene}: base primitive unary changed")
        base_primitive = torch.load(base_primitive_path, map_location="cpu", weights_only=True)
        completion = candidate_primitive["compiler_contract"].get(
            "source_only_correspondence_completion"
        )
        if not isinstance(completion, Mapping) or completion.get("mode") != MODE:
            raise ValueError(f"{scene}: completion diagnostics missing")
        if completion.get("method_contract") != method_contract():
            raise ValueError(f"{scene}: completion method contract differs")
        diagnostics = completion.get("diagnostics", {})
        if not (
            diagnostics.get("observed_unary_bitwise_equal") is True
            and int(diagnostics.get("completed_rows", 0)) > 0
            and int(diagnostics.get("observed_rows", -1))
            + int(diagnostics.get("abstained_rows", -1))
            == int(diagnostics.get("num_nodes", -2))
        ):
            raise ValueError(f"{scene}: source-only causal gate failed")
        candidate_probability = torch.as_tensor(
            candidate_primitive["primitive_unary_probability"]
        ).float()
        base_probability = torch.as_tensor(
            base_primitive["primitive_unary_probability"]
        ).float()
        if (
            candidate_probability.shape != base_probability.shape
            or not torch.equal(candidate_primitive["valid"], base_primitive["valid"])
        ):
            raise ValueError(f"{scene}: primitive row authority differs")
        absolute_delta = (candidate_probability - base_probability).abs()
        materially_changed = int((absolute_delta > 1e-6).sum())
        if materially_changed <= 0 or materially_changed > int(diagnostics["completed_rows"]):
            raise ValueError(f"{scene}: completion change-count invariant failed")

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
            "completion_diagnostics": dict(diagnostics),
            "materially_changed_rows_vs_base": materially_changed,
            "maximum_primitive_probability_delta_vs_base": float(absolute_delta.max()),
            "causal_gate": {
                "completed_abstain_rows": True,
                "observed_unary_bitwise_equal": True,
                "single_factor_args": True,
                "target_unopened": True,
            },
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
                    **_score_frame(
                        verified[scene]["scores"][frame_id],
                        load_ground_truth_mask(target_path),
                    ),
                }
            )
        baseline_row = base_results["per_scene"][scene]
        baseline = float(
            baseline_row.get("foreground_iou", baseline_row.get("iou"))
        )
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
    base_macro = float(
        np.mean([row["baseline_foreground_iou"] for row in per_scene.values()])
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "nvos_source_only_correspondence_exact_result_v1",
        "status": "sealed_predictions_verified_then_exact_scored",
        "scene_order": list(scenes),
        "base_run_manifest": {"path": str(base_manifest_path), "sha256": _sha256(base_manifest_path)},
        "base_exact_results": {"path": str(base_results_path), "sha256": _sha256(base_results_path)},
        "evaluator": {"path": str(evaluator_path), "sha256": evaluator_sha256},
        "completion_module": {"path": str(module_path), "sha256": completion_module_sha256},
        "preregistration": {"path": str(prereg_path), "sha256": preregistration_sha256},
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
    return _write_no_clobber(output, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-manifest", required=True)
    parser.add_argument("--base-exact-results", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--completion-module", required=True)
    parser.add_argument("--completion-module-sha256", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run(
        base_run_manifest=args.base_run_manifest,
        base_exact_results=args.base_exact_results,
        candidate_root=args.candidate_root,
        scenes=args.scenes,
        evaluator=args.evaluator,
        evaluator_sha256=args.evaluator_sha256,
        completion_module=args.completion_module,
        completion_module_sha256=args.completion_module_sha256,
        preregistration=args.preregistration,
        preregistration_sha256=args.preregistration_sha256,
        output=args.output,
    )
    print(json.dumps({"output": str(output), "sha256": _sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
