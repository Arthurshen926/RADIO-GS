#!/usr/bin/env python3
"""Seal a rendered SPIn directional-admission candidate before target GT access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from radio_gs.querying.source_oof_transport_admission import (
    DirectionalAdmissionCalibration,
    apply_source_oof_directional_admission,
    method_contract,
)
from radio_gs.scripts.build_spin_source_footprint_quantile_oof import (
    file_sha256,
    json_sha256,
)
from radio_gs.scripts.build_spin_source_footprint_quantile_target_prediction import (
    _save_array,
)


PREDICTION_RECEIPT_TYPE = "spin_rendered_directional_admission_prediction_v1"
BASE_RECEIPT_TYPES = {
    "spin_source_footprint_quantile_target_prediction_v2",
    "spin9_factorized_source_quantile_target_prediction_v1",
}


def _require_file(path: str | Path, expected: str, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing {label}: {source}")
    actual = file_sha256(source)
    if actual != str(expected):
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return source


def _load_json_authority(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON authority must be an object")
    declared = payload.get("content_sha256")
    if declared:
        content = dict(payload)
        content.pop("content_sha256", None)
        if json_sha256(content) != declared:
            raise ValueError("JSON authority content digest differs")
    return payload


def _array(record: Mapping[str, object], label: str) -> tuple[Path, np.ndarray]:
    path = _require_file(record["path"], record["sha256"], label)
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite 2-D array")
    declared_shape = record.get("shape")
    declared_dtype = record.get("dtype")
    if declared_shape is not None and list(value.shape) != list(declared_shape):
        raise ValueError(f"{label} shape differs")
    if declared_dtype is not None and str(value.dtype) != str(declared_dtype):
        raise ValueError(f"{label} dtype differs")
    return path, value


def build_candidate_frame(
    unary_probability: np.ndarray,
    proposal_probability: np.ndarray,
    source_visible_coverage: np.ndarray,
    calibration: DirectionalAdmissionCalibration,
) -> np.ndarray:
    """Apply rendered anchor-preserving admission without target inputs."""

    unary = np.asarray(unary_probability)
    proposal = np.asarray(proposal_probability)
    coverage = np.asarray(source_visible_coverage)
    if unary.shape != proposal.shape or unary.shape != coverage.shape or unary.ndim != 2:
        raise ValueError("rendered unary, proposal, and coverage must align")
    if any(not np.isfinite(value).all() for value in (unary, proposal, coverage)):
        raise ValueError("rendered admission inputs must be finite")
    if any((value.min() < 0 or value.max() > 1) for value in (unary, proposal, coverage)):
        raise ValueError("rendered admission inputs must lie in [0,1]")
    output = apply_source_oof_directional_admission(
        torch.from_numpy(unary.astype(np.float32, copy=False)).reshape(-1),
        torch.from_numpy(proposal.astype(np.float32, copy=False)).reshape(-1),
        torch.from_numpy(coverage.astype(np.float32, copy=False)).reshape(-1),
        calibration,
        active_domain=torch.ones(unary.size, dtype=torch.bool),
    )
    return output.probability.reshape(unary.shape).numpy().astype(np.float32)


def build(args: argparse.Namespace) -> dict[str, object]:
    source_result_path = _require_file(
        args.source_result, args.source_result_sha256, "source-only admission result"
    )
    source_result = _load_json_authority(source_result_path)
    if source_result.get("source_gate_passed") is not True or source_result.get(
        "safety", {}
    ).get("target_metric_computed") is not False:
        raise ValueError("source-only admission gate is not eligible")
    scene_id = str(source_result.get("scene_id", ""))
    values = source_result.get("calibration")
    if not isinstance(values, Mapping):
        raise ValueError("source-only admission result lacks calibration")
    calibration = DirectionalAdmissionCalibration(
        expansion=float(values["expansion"]),
        contraction=float(values["contraction"]),
        leave_one_fold_expansion=tuple(float(v) for v in values["leave_one_fold_expansion"]),
        leave_one_fold_contraction=tuple(float(v) for v in values["leave_one_fold_contraction"]),
        folds=tuple(int(v) for v in values["folds"]),
        eligible_rows=int(values["eligible_rows"]),
    )

    base_path = _require_file(
        args.base_prediction_receipt,
        args.base_prediction_receipt_sha256,
        "sealed base prediction receipt",
    )
    base = _load_json_authority(base_path)
    if base.get("artifact_type") not in BASE_RECEIPT_TYPES or str(
        base.get("scene_id", "")
    ) != scene_id:
        raise ValueError("sealed base prediction receipt differs")
    if base.get("sealed_before_target_ground_truth_open") is not True or any(
        base.get(key) is not False
        for key in ("target_rgb_opened", "target_mask_opened", "target_metric_computed")
    ):
        raise ValueError("base prediction receipt is not a pre-metric authority")
    frames = base.get("frames")
    if not isinstance(frames, Mapping) or not frames:
        raise ValueError("base prediction receipt lacks frames")
    output_root = Path(args.output_dir).expanduser().resolve()
    receipt_path = output_root / "pre_metric_prediction_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite prediction receipt: {receipt_path}")

    output_frames: dict[str, dict[str, object]] = {}
    exact_full_coverage_identity = True
    candidate_positive_fractions: list[float] = []
    for frame_id in sorted(frames):
        record = frames[frame_id]
        if not isinstance(record, Mapping):
            raise ValueError(f"base frame record is malformed: {frame_id}")
        coverage_path, coverage = _array(record["coverage"], f"{frame_id} coverage")
        proposal_record = record.get("input_raw_score")
        if not isinstance(proposal_record, Mapping):
            raise ValueError(f"base frame lacks proposal score: {frame_id}")
        proposal_path = _require_file(
            proposal_record["path"], proposal_record["sha256"], f"{frame_id} proposal"
        )
        proposal = np.load(proposal_path, allow_pickle=False)
        unary_path = Path(
            str(proposal_path).replace("/stage_scores/propagated/", "/stage_scores/unary_prior/")
        )
        if unary_path == proposal_path or not unary_path.is_file():
            raise FileNotFoundError(f"missing sealed rendered unary: {unary_path}")
        unary = np.load(unary_path, allow_pickle=False)
        if unary.shape != proposal.shape or unary.shape != coverage.shape:
            raise ValueError(f"rendered score domains differ: {frame_id}")
        candidate = build_candidate_frame(unary, proposal, coverage, calibration)
        full = coverage >= 1.0 - 1e-5
        exact_full_coverage_identity = exact_full_coverage_identity and np.array_equal(
            candidate[full], unary.astype(np.float32, copy=False)[full]
        )
        threshold = np.full(candidate.shape, 0.5, dtype=np.float32)
        margin = candidate - threshold
        frame_root = output_root / "frames" / frame_id
        candidate_record = _save_array(frame_root / "candidate_probability.npy", candidate)
        threshold_record = _save_array(frame_root / "fixed_threshold.npy", threshold)
        margin_record = _save_array(frame_root / "continuous_margin.npy", margin)
        output_frames[frame_id] = {
            "coverage": {
                "path": str(coverage_path),
                "sha256": file_sha256(coverage_path),
                "shape": list(coverage.shape),
                "dtype": str(coverage.dtype),
            },
            "unary_probability": {
                "path": str(unary_path.resolve()),
                "sha256": file_sha256(unary_path),
                "shape": list(unary.shape),
                "dtype": str(unary.dtype),
            },
            "proposal_probability": {
                "path": str(proposal_path),
                "sha256": file_sha256(proposal_path),
                "shape": list(proposal.shape),
                "dtype": str(proposal.dtype),
            },
            "candidate_probability": candidate_record,
            "fixed_threshold": threshold_record,
            "continuous_margin": margin_record,
            "quantile_baseline_margin": dict(record["continuous_margin"]),
            "candidate_positive_fraction": float((candidate >= 0.5).mean()),
        }
        candidate_positive_fractions.append(float((candidate >= 0.5).mean()))
    if not exact_full_coverage_identity:
        raise RuntimeError("rendered admission failed the full-coverage identity")
    manifest_record = base.get("manifest")
    if isinstance(manifest_record, Mapping):
        manifest = dict(manifest_record)
    else:
        manifest = {
            "path": str(base.get("manifest", "")),
            "sha256": str(base.get("manifest_sha256", "")),
        }
    _require_file(manifest["path"], manifest["sha256"], "frozen manifest")
    receipt = {
        "schema_version": 1,
        "artifact_type": PREDICTION_RECEIPT_TYPE,
        "status": "sealed_before_target_ground_truth_open",
        "scene_id": scene_id,
        "protocol_hash": base["protocol_hash"],
        "source_result": {
            "path": str(source_result_path),
            "sha256": args.source_result_sha256,
        },
        "base_prediction_receipt": {
            "path": str(base_path),
            "sha256": args.base_prediction_receipt_sha256,
        },
        "manifest": manifest,
        "calibration": {
            "expansion": calibration.expansion,
            "contraction": calibration.contraction,
            "leave_one_fold_expansion": list(calibration.leave_one_fold_expansion),
            "leave_one_fold_contraction": list(calibration.leave_one_fold_contraction),
            "folds": list(calibration.folds),
            "eligible_rows": calibration.eligible_rows,
        },
        "method_contract": {
            **method_contract(),
            "application_domain": "rendered_unary_and_graph_proposal_before_resolution_adapter",
            "observation_confidence": "rendered_source_visible_coverage",
            "fixed_probability_threshold": 0.5,
            "evaluation_adapter": "cv2.INTER_LINEAR_margin_to_gt_then_greater_equal_zero",
            "parameter_scan": False,
        },
        "engineering_invariants": {
            "full_coverage_exact_unary_identity": exact_full_coverage_identity,
            "graph_is_proposal_only": True,
            "connected_selection": False,
        },
        "frames": output_frames,
        "frame_count": len(output_frames),
        "candidate_positive_fraction_frame_macro": float(
            np.mean(candidate_positive_fractions)
        ),
        "sealed_before_target_ground_truth_open": True,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
    }
    receipt["content_sha256"] = json_sha256(receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--source-result-sha256", required=True)
    parser.add_argument("--base-prediction-receipt", required=True)
    parser.add_argument("--base-prediction-receipt-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build(args)
    print(json.dumps({"receipt": result["receipt_path"], "sha256": result["receipt_sha256"]}))


if __name__ == "__main__":
    main()
