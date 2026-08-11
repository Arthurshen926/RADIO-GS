"""Score sealed NVOS registered-query-likelihood predictions after verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask


MODE = "balanced_reference_source_reconstruction_v1"
PATH_ONLY_ARGS = {
    "output_dir",
    "prediction_receipt_output",
    "primitive_unary_output",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def _normalized_args(values: Mapping[str, object]) -> dict[str, object]:
    result = dict(values)
    for name in PATH_ONLY_ARGS:
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
    likelihood_head: str | Path,
    likelihood_head_sha256: str,
    output: str | Path,
) -> Path:
    base_manifest_path = Path(base_run_manifest).expanduser().resolve()
    base_results_path = Path(base_exact_results).expanduser().resolve()
    candidate_root_path = Path(candidate_root).expanduser().resolve()
    evaluator_path = Path(evaluator).expanduser().resolve()
    head_path = Path(likelihood_head).expanduser().resolve()
    if _sha256(evaluator_path) != evaluator_sha256:
        raise ValueError("candidate evaluator changed")
    if _sha256(head_path) != likelihood_head_sha256:
        raise ValueError("query likelihood head changed")
    base_manifest = _load(base_manifest_path)
    base_results = _load(base_results_path)
    benchmark_path = Path(base_manifest["manifest"])
    if _sha256(benchmark_path) != base_manifest["manifest_sha256"]:
        raise ValueError("frozen benchmark manifest changed")
    benchmark = _load(benchmark_path)
    scene_rows = {str(row["scene_id"]): row for row in benchmark["scenes"]}

    # Verify every prediction and causal gate before opening any target label.
    verified: dict[str, object] = {}
    for scene in scenes:
        base_record = base_manifest["records"][scene]
        base_receipt_path = Path(base_record["prediction_receipt"])
        if _sha256(base_receipt_path) != base_record.get(
            "prediction_receipt_sha256", _sha256(base_receipt_path)
        ):
            raise ValueError(f"{scene}: base receipt changed")
        base_receipt = _load(base_receipt_path)
        receipt_path = candidate_root_path / scene / "pre_metric_prediction_receipt.json"
        receipt = _load(receipt_path)
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
        if candidate_args.get("registered_query_likelihood_calibration") != MODE:
            raise ValueError(f"{scene}: likelihood factor differs")
        candidate_normalized = _normalized_args(candidate_args)
        base_normalized = _normalized_args(base_args)
        base_normalized["registered_query_likelihood_calibration"] = MODE
        if candidate_normalized != base_normalized:
            differing = sorted(
                key
                for key in set(candidate_normalized) | set(base_normalized)
                if candidate_normalized.get(key) != base_normalized.get(key)
            )
            raise ValueError(f"{scene}: non-single-factor args: {differing}")
        diagnostics = method.get("registered_query_likelihood")
        if not isinstance(diagnostics, Mapping):
            raise ValueError(f"{scene}: likelihood diagnostics missing")
        source = diagnostics["source_reconstruction"]
        causal_gate = {
            "source_bce_decreased": source["final_balanced_bce"]
            < source["initial_balanced_bce"],
            "positive_above_negative": source[
                "positive_mass_weighted_probability"
            ]
            > source["negative_mass_weighted_probability"],
            "observed_unary_changed": diagnostics["changed_observed_unary_rows"] > 0,
            "unobserved_abstention_declared": diagnostics["abstained_rows"] > 0,
            "target_unopened": diagnostics["target_rgb_opened"] is False
            and diagnostics["target_mask_opened"] is False
            and diagnostics["target_metric_opened"] is False,
        }
        if not all(causal_gate.values()):
            raise ValueError(f"{scene}: pre-target causal gate failed")
        primitive = method["primitive_unary_artifact"]
        if _sha256(primitive["path"]) != primitive["file_sha256"]:
            raise ValueError(f"{scene}: primitive unary changed")
        scores = {}
        for frame_id, record in receipt["target_scores"].items():
            if _sha256(record["path"]) != record["sha256"]:
                raise ValueError(f"{scene}/{frame_id}: score changed")
            scores[str(frame_id)] = np.load(record["path"], allow_pickle=False)
        if float(method["score_threshold"]) != 0.5:
            raise ValueError(f"{scene}: frozen threshold differs")
        verified[scene] = {
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
            "primitive_unary": primitive,
            "scores": scores,
            "score_records": receipt["target_scores"],
            "causal_gate": causal_gate,
            "likelihood_diagnostics": diagnostics,
        }

    # Only this section opens target masks.
    per_scene = {}
    for scene in scenes:
        scene_row = scene_rows[scene]
        expected_frames = tuple(str(value) for value in scene_row["evaluation_frame_ids"])
        if tuple(verified[scene]["scores"]) != expected_frames:
            raise ValueError(f"{scene}: evaluation frame order differs")
        frames = []
        for frame_id in expected_frames:
            row = next(
                item for item in scene_row["frames"] if str(item["frame_id"]) == frame_id
            )
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
        baseline = float(base_results["per_scene"][scene]["iou"])
        iou = float(np.mean([row["foreground_iou"] for row in frames]))
        per_scene[scene] = {
            "foreground_iou": iou,
            "baseline_foreground_iou": baseline,
            "delta_foreground_iou": iou - baseline,
            "pixel_accuracy": float(np.mean([row["pixel_accuracy"] for row in frames])),
            "frames": frames,
            **{key: value for key, value in verified[scene].items() if key != "scores"},
        }
    macro = float(np.mean([row["foreground_iou"] for row in per_scene.values()]))
    base_macro = float(
        np.mean([row["baseline_foreground_iou"] for row in per_scene.values()])
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "nvos_registered_query_likelihood_exact_result_v1",
        "status": "sealed_predictions_verified_then_exact_scored",
        "scene_order": list(scenes),
        "base_run_manifest": {
            "path": str(base_manifest_path),
            "sha256": _sha256(base_manifest_path),
        },
        "base_exact_results": {
            "path": str(base_results_path),
            "sha256": _sha256(base_results_path),
        },
        "evaluator": {"path": str(evaluator_path), "sha256": evaluator_sha256},
        "likelihood_head": {"path": str(head_path), "sha256": likelihood_head_sha256},
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-manifest", required=True)
    parser.add_argument("--base-exact-results", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--likelihood-head", required=True)
    parser.add_argument("--likelihood-head-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = run(
        base_run_manifest=args.base_run_manifest,
        base_exact_results=args.base_exact_results,
        candidate_root=args.candidate_root,
        scenes=args.scenes,
        evaluator=args.evaluator,
        evaluator_sha256=args.evaluator_sha256,
        likelihood_head=args.likelihood_head,
        likelihood_head_sha256=args.likelihood_head_sha256,
        output=args.output,
    )
    print(json.dumps({"output": str(output), "sha256": _sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
