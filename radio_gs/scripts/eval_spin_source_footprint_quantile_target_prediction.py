#!/usr/bin/env python3
"""Evaluate one sealed SPIn v2 margin receipt under the frozen protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from radio_gs.evaluation.promptable_segmentation import load_ground_truth_mask
from radio_gs.scripts.build_spin_source_footprint_quantile_oof import (
    file_sha256,
    json_sha256,
)
from radio_gs.scripts.build_spin_source_footprint_quantile_target_prediction import (
    PREDICTION_RECEIPT_TYPE,
)


def _require_file(path: str | Path, expected: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    actual = file_sha256(resolved)
    if actual != str(expected):
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return resolved


def exact_four_connected_boundary(mask: np.ndarray) -> np.ndarray:
    """Return both sides of every exact 4-neighbour label transition."""

    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2 or value.size == 0:
        raise ValueError("boundary input must be a non-empty 2-D mask")
    boundary = np.zeros_like(value)
    vertical = value[:-1, :] != value[1:, :]
    boundary[:-1, :] |= vertical
    boundary[1:, :] |= vertical
    horizontal = value[:, :-1] != value[:, 1:]
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    return boundary


def _safe_mean(value: np.ndarray, selection: np.ndarray) -> float | None:
    selected = np.asarray(value)[np.asarray(selection, dtype=bool)]
    return float(selected.mean()) if selected.size else None


def _safe_rate(selection: np.ndarray, domain: np.ndarray) -> float | None:
    use = np.asarray(domain, dtype=bool)
    return float(np.asarray(selection, dtype=bool)[use].mean()) if bool(use.any()) else None


def _macro(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return float(np.mean(usable)) if usable else None


def _load_prediction_receipt(path: Path) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("artifact_type") != (
        PREDICTION_RECEIPT_TYPE
    ):
        raise ValueError("unexpected SPIn v2 prediction receipt")
    if receipt.get("sealed_before_target_ground_truth_open") is not True or any(
        receipt.get(key) is not False
        for key in ("target_rgb_opened", "target_mask_opened", "target_metric_computed")
    ):
        raise ValueError("SPIn v2 prediction receipt is not pre-metric")
    declared_content = str(receipt.get("content_sha256", ""))
    content = dict(receipt)
    content.pop("content_sha256", None)
    if json_sha256(content) != declared_content:
        raise ValueError("SPIn v2 prediction receipt content digest differs")
    method = receipt.get("method_contract")
    if not isinstance(method, Mapping) or method.get("parameter_scan") is not False:
        raise ValueError("SPIn v2 prediction method contract differs")
    if method.get("evaluation_adapter") != (
        "cv2.INTER_LINEAR_margin_to_gt_then_greater_equal_zero"
    ):
        raise ValueError("SPIn v2 evaluation adapter differs")
    return receipt


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    receipt_path = _require_file(
        args.prediction_receipt,
        args.prediction_receipt_sha256,
        "sealed prediction receipt",
    )
    receipt = _load_prediction_receipt(receipt_path)
    manifest_path = _require_file(args.manifest, args.manifest_sha256, "manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("protocol_hash") != receipt.get(
        "protocol_hash"
    ):
        raise ValueError("manifest protocol differs from prediction receipt")
    scenes = [
        scene
        for scene in manifest.get("scenes", [])
        if scene.get("scene_id") == receipt.get("scene_id")
    ]
    if len(scenes) != 1:
        raise ValueError("manifest scene identity is ambiguous")
    scene = scenes[0]
    evaluation_frames = [str(value) for value in scene["evaluation_frame_ids"]]
    frames = receipt.get("frames")
    if not isinstance(frames, Mapping) or list(frames) != sorted(frames):
        # JSON receipt is sort-key encoded; require a deterministic key set,
        # but preserve manifest order for metric aggregation below.
        if not isinstance(frames, Mapping):
            raise ValueError("prediction receipt lacks frame outputs")
    if set(frames) != set(evaluation_frames) or int(receipt.get("frame_count", -1)) != len(
        evaluation_frames
    ):
        raise ValueError("prediction receipt frame set differs from manifest")

    # Exhaustively verify and load every continuous prediction before opening
    # even one target mask.  This preserves the pre-metric separation even if a
    # later target asset fails its hash check.
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for frame_id in evaluation_frames:
        record = frames[frame_id]
        arrays: dict[str, np.ndarray] = {}
        for name in (
            "coverage",
            "score_quantile",
            "spatial_threshold_quantile",
            "continuous_margin",
        ):
            declared = record.get(name)
            if not isinstance(declared, Mapping):
                raise ValueError(f"prediction frame lacks {name}: {frame_id}")
            path = _require_file(
                declared["path"], declared["sha256"], f"{frame_id} {name}"
            )
            value = np.load(path, allow_pickle=False)
            if list(value.shape) != list(declared["shape"]) or str(value.dtype) != str(
                declared["dtype"]
            ):
                raise ValueError(f"prediction array metadata differs: {frame_id} {name}")
            if value.ndim != 2 or not np.isfinite(value).all():
                raise ValueError(f"prediction array is malformed: {frame_id} {name}")
            arrays[name] = value
        shape = arrays["continuous_margin"].shape
        if any(value.shape != shape for value in arrays.values()):
            raise ValueError(f"prediction arrays do not align: {frame_id}")
        if not np.allclose(
            arrays["continuous_margin"],
            arrays["score_quantile"] - arrays["spatial_threshold_quantile"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"prediction margin identity differs: {frame_id}")
        raw = record.get("input_raw_score")
        if not isinstance(raw, Mapping):
            raise ValueError(f"prediction frame lacks raw-score authority: {frame_id}")
        raw_path = _require_file(raw["path"], raw["sha256"], f"{frame_id} raw score")
        raw_score = np.load(raw_path, allow_pickle=False)
        if raw_score.shape != shape or not np.isfinite(raw_score).all():
            raise ValueError(f"raw score differs from prediction raster: {frame_id}")
        arrays["raw_score"] = raw_score
        loaded[frame_id] = arrays

    compact_path = _require_file(
        args.compact_baseline_result,
        args.compact_baseline_result_sha256,
        "compact baseline authority",
    )
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    compact_iou = float(
        compact["matched_comparison"]["previous_compact_full_interface_iou"]
    )
    if not 0 <= compact_iou <= 1:
        raise ValueError("compact baseline foreground IoU is invalid")

    scene_frames = {
        str(frame["frame_id"]): frame for frame in scene.get("frames", [])
    }
    if any(frame_id not in scene_frames for frame_id in evaluation_frames):
        raise ValueError("manifest lacks evaluation-frame mask authorities")
    frame_metrics: list[dict[str, object]] = []
    for frame_id in evaluation_frames:
        frame = scene_frames[frame_id]
        gt_path = _require_file(
            frame["ground_truth"],
            frame["ground_truth_sha256"],
            f"target mask {frame_id}",
        )
        gt = load_ground_truth_mask(gt_path).astype(bool)
        arrays = loaded[frame_id]
        margin = cv2.resize(
            arrays["continuous_margin"],
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        coverage = cv2.resize(
            arrays["coverage"],
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        raw_score = cv2.resize(
            arrays["raw_score"],
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        prediction = margin >= 0.0
        baseline_prediction = raw_score >= float(receipt["method_contract"].get(
            "raw_matched_baseline_threshold", 0.71
        ))
        intersection = int(np.logical_and(prediction, gt).sum())
        union = int(np.logical_or(prediction, gt).sum())
        baseline_intersection = int(np.logical_and(baseline_prediction, gt).sum())
        baseline_union = int(np.logical_or(baseline_prediction, gt).sum())
        boundary = exact_four_connected_boundary(gt)
        error = prediction != gt
        tp = prediction & gt
        fn = (~prediction) & gt
        fp = prediction & (~gt)
        tn = (~prediction) & (~gt)
        frame_metrics.append(
            {
                "frame_id": frame_id,
                "foreground_iou": float(intersection / union) if union else 1.0,
                "pixel_accuracy": float((prediction == gt).mean()),
                "matched_factorized_baseline_foreground_iou": (
                    float(baseline_intersection / baseline_union)
                    if baseline_union
                    else 1.0
                ),
                "prediction_positive_fraction": float(prediction.mean()),
                "ground_truth_positive_fraction": float(gt.mean()),
                "coverage": {
                    "all_mean": float(coverage.mean()),
                    "true_positive_mean": _safe_mean(coverage, tp),
                    "false_negative_mean": _safe_mean(coverage, fn),
                    "false_positive_mean": _safe_mean(coverage, fp),
                    "true_negative_mean": _safe_mean(coverage, tn),
                    "ground_truth_foreground_mean": _safe_mean(coverage, gt),
                    "ground_truth_background_mean": _safe_mean(coverage, ~gt),
                    "boundary_mean": _safe_mean(coverage, boundary),
                    "nonboundary_mean": _safe_mean(coverage, ~boundary),
                },
                "boundary": {
                    "definition": "both_sides_of_exact_four_neighbour_label_transition",
                    "pixel_fraction": float(boundary.mean()),
                    "error_rate": _safe_rate(error, boundary),
                    "nonboundary_error_rate": _safe_rate(error, ~boundary),
                    "foreground_boundary_recall": _safe_rate(prediction, boundary & gt),
                },
            }
        )

    foreground_iou = float(np.mean([item["foreground_iou"] for item in frame_metrics]))
    accuracy = float(np.mean([item["pixel_accuracy"] for item in frame_metrics]))
    factorized_baseline_iou = float(
        np.mean(
            [
                item["matched_factorized_baseline_foreground_iou"]
                for item in frame_metrics
            ]
        )
    )
    coverage_keys = (
        "all_mean",
        "true_positive_mean",
        "false_negative_mean",
        "false_positive_mean",
        "true_negative_mean",
        "ground_truth_foreground_mean",
        "ground_truth_background_mean",
        "boundary_mean",
        "nonboundary_mean",
    )
    coverage_macro = {
        key: _macro([item["coverage"][key] for item in frame_metrics])
        for key in coverage_keys
    }
    boundary_macro = {
        key: _macro([item["boundary"][key] for item in frame_metrics])
        for key in ("pixel_fraction", "error_rate", "nonboundary_error_rate", "foreground_boundary_recall")
    }
    report = {
        "schema_version": 1,
        "experiment": "spin_source_footprint_crossfit_quantile_calibration_v2_target",
        "status": "single_exact_evaluation_complete_no_parameter_scan",
        "scene_id": receipt["scene_id"],
        "protocol_hash": receipt["protocol_hash"],
        "prediction_receipt": str(receipt_path),
        "prediction_receipt_sha256": str(args.prediction_receipt_sha256),
        "manifest": str(manifest_path),
        "manifest_sha256": str(args.manifest_sha256),
        "compact_baseline_authority": str(compact_path),
        "compact_baseline_authority_sha256": str(args.compact_baseline_result_sha256),
        "metrics": {
            "frames": len(frame_metrics),
            "foreground_iou": foreground_iou,
            "pixel_accuracy": accuracy,
            "compact_full_interface_foreground_iou": compact_iou,
            "delta_vs_compact": foreground_iou - compact_iou,
            "matched_factorized_baseline_foreground_iou_recomputed": (
                factorized_baseline_iou
            ),
            "delta_vs_matched_factorized_baseline": (
                foreground_iou - factorized_baseline_iou
            ),
        },
        "coverage_diagnostic_frame_macro": coverage_macro,
        "boundary_diagnostic_frame_macro": {
            "definition": "both_sides_of_exact_four_neighbour_label_transition",
            **boundary_macro,
        },
        "frame_metrics": frame_metrics,
        "evaluation_contract": {
            "within_scene_aggregation": "unweighted_frame_mean",
            "margin_resize": "cv2.INTER_LINEAR",
            "comparison": "greater_equal_zero",
            "empty_union_value": 1.0,
            "parameter_scan": False,
        },
        "target_rgb_opened": False,
        "target_mask_opened": True,
        "target_metric_computed": True,
    }
    report["content_sha256"] = json_sha256(report)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"refusing to overwrite different evaluation: {output}")
    if not output.exists():
        output.write_text(encoded, encoding="utf-8")
    return {**report, "report_path": str(output), "report_sha256": file_sha256(output)}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prediction-receipt", required=True)
    result.add_argument("--prediction-receipt-sha256", required=True)
    result.add_argument("--manifest", required=True)
    result.add_argument("--manifest-sha256", required=True)
    result.add_argument("--compact-baseline-result", required=True)
    result.add_argument("--compact-baseline-result-sha256", required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> None:
    print(json.dumps(evaluate(parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
