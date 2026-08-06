#!/usr/bin/env python3
"""Seal SPIn v2 target margins and source-visible coverage before GT access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from radio_gs.interfaces.prompt_responsibility_cache import tensor_sha256
from radio_gs.querying.spin_source_footprint_quantile_calibration import (
    FULL_FIT_GAUGE_ARTIFACT_TYPE,
    WeightedRightECDF,
    build_quantile_prediction_fields,
)
from radio_gs.scripts.build_spin_source_footprint_quantile_oof import (
    file_sha256,
    json_sha256,
)
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _resolve_scene_carrier_assets,
    _scene_record,
    _view_by_frame,
)
from radio_gs.config import load_config
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views


PREDICTION_RECEIPT_TYPE = "spin_source_footprint_quantile_target_prediction_v2"
NUMERIC_ADDENDUM_REGISTRATION = (
    "spin_lego_quantile_coverage_fp32_projection_addendum_v1"
)
FP32_COVERAGE_EPSILON_MULTIPLE = 16


def _require_file(path: str | Path, expected: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    actual = file_sha256(resolved)
    if actual != str(expected):
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return resolved


def _load_tensor_authority(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("artifact_type") != (
        FULL_FIT_GAUGE_ARTIFACT_TYPE
    ):
        raise ValueError("unexpected full-fit source gauge")
    hashes = payload.get("tensor_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("full-fit source gauge lacks tensor hashes")
    for name, expected in hashes.items():
        if name not in payload or tensor_sha256(torch.as_tensor(payload[name])) != expected:
            raise ValueError(f"full-fit source-gauge tensor changed: {name}")
    if any(
        payload.get(key) is not False
        for key in (
            "target_distribution_opened",
            "target_rgb_opened",
            "target_mask_opened",
            "target_metric_computed",
        )
    ):
        raise ValueError("full-fit source gauge is not target blind")
    return payload


def _load_premetric_receipt(
    path: Path,
    *,
    scene_id: str,
    protocol_hash: str,
) -> tuple[dict[str, object], dict[str, Mapping[str, str]]]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("artifact_type") != (
        "nvos_pre_metric_prediction_receipt_v1"
    ):
        raise ValueError("unexpected matched-interface pre-metric receipt")
    if receipt.get("scene_id") != scene_id or receipt.get("protocol_hash") != protocol_hash:
        raise ValueError("matched-interface receipt scene/protocol differs")
    if receipt.get("sealed_before_target_ground_truth_open") is not True or any(
        receipt.get(key) is not False
        for key in ("target_rgb_opened", "target_mask_opened", "target_metric_opened")
    ):
        raise ValueError("matched-interface score receipt is not target blind")
    stages = receipt.get("stage_target_scores")
    if not isinstance(stages, Mapping) or not isinstance(stages.get("propagated"), Mapping):
        raise ValueError("matched-interface receipt lacks propagated target scores")
    selected = dict(stages["propagated"])
    target_scores = receipt.get("target_scores")
    if not isinstance(target_scores, Mapping) or set(target_scores) != set(selected):
        raise ValueError("final and propagated target frame identities differ")
    for frame_id in selected:
        if target_scores[frame_id].get("sha256") != selected[frame_id].get("sha256"):
            raise ValueError("final and propagated target score authorities differ")
    return receipt, selected


def _save_array(path: Path, value: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = np.load(path, allow_pickle=False)
        if existing.dtype != value.dtype or existing.shape != value.shape or not np.array_equal(
            existing, value, equal_nan=False
        ):
            raise FileExistsError(f"refusing to overwrite different prediction array: {path}")
    else:
        np.save(path, value, allow_pickle=False)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    prereg_path = _require_file(
        args.preregistration, args.preregistration_sha256, "v2 preregistration"
    )
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if not isinstance(prereg, Mapping) or prereg.get("registration") != (
        "spin_source_footprint_crossfit_quantile_calibration_v2"
    ):
        raise ValueError("unexpected SPIn v2 preregistration")
    addendum_path = _require_file(
        args.numeric_addendum,
        args.numeric_addendum_sha256,
        "coverage numeric addendum",
    )
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    if not isinstance(addendum, Mapping) or addendum.get("registration") != (
        NUMERIC_ADDENDUM_REGISTRATION
    ):
        raise ValueError("unexpected coverage numeric addendum")
    correction = addendum.get("implementation_correction")
    expected_tolerance = float(
        FP32_COVERAGE_EPSILON_MULTIPLE * torch.finfo(torch.float32).eps
    )
    if not isinstance(correction, Mapping) or float(
        correction.get("absolute_tolerance", -1.0)
    ) != expected_tolerance:
        raise ValueError("coverage numeric addendum tolerance differs")
    gauge_path = _require_file(
        args.full_fit_source_gauge,
        args.full_fit_source_gauge_sha256,
        "full-fit source gauge",
    )
    gauge = _load_tensor_authority(gauge_path)
    scene_id = str(gauge.get("scene_id", ""))
    protocol_hash = str(gauge.get("protocol_hash", ""))
    premetric_path = _require_file(
        args.matched_premetric_receipt,
        args.matched_premetric_receipt_sha256,
        "matched pre-metric receipt",
    )
    _premetric, target_scores = _load_premetric_receipt(
        premetric_path,
        scene_id=scene_id,
        protocol_hash=protocol_hash,
    )
    manifest_path = _require_file(args.manifest, args.manifest_sha256, "manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("protocol_hash") != protocol_hash:
        raise ValueError("manifest protocol differs from source gauge")
    scene = _scene_record(manifest, scene_id)
    evaluation_frames = [str(value) for value in scene["evaluation_frame_ids"]]
    if set(evaluation_frames) != set(target_scores) or len(evaluation_frames) != len(
        target_scores
    ):
        raise ValueError("sealed target-score frames differ from manifest")
    queue_scene = Path(args.queue_root).expanduser().resolve() / "scenes" / scene_id
    base_scene_id = str(scene.get("base_scene_id") or scene_id)
    if not queue_scene.is_dir():
        queue_scene = Path(args.queue_root).expanduser().resolve() / "scenes" / base_scene_id
    config_path, checkpoint_path, camera_map_path = _resolve_scene_carrier_assets(
        queue_scene,
        scene_config=str(args.scene_config),
        scene_checkpoint=str(args.scene_checkpoint),
        camera_map=str(args.camera_map),
    )
    _require_file(config_path, args.scene_config_sha256, "carrier config")
    _require_file(checkpoint_path, args.scene_checkpoint_sha256, "carrier checkpoint")
    _require_file(camera_map_path, args.camera_map_sha256, "camera map")
    camera_mapping = json.loads(camera_map_path.read_text(encoding="utf-8"))
    config = load_config(str(config_path))
    views = resolve_protocol_views(
        manifest,
        scene_id=scene_id,
        scene_root=Path(str(config.scene_root)).resolve(),
        camera_mapping=camera_mapping,
    )
    device = torch.device(args.device)
    model, _codec, renderer, _sharpener, _refiner, _field_config, _is_hybrid = (
        load_render_pipeline(
            str(config_path),
            str(checkpoint_path),
            device,
            strict_checkpoint_contract=True,
            load_ply_rgb_features=False,
        )
    )
    valid = torch.as_tensor(gauge["valid"]).detach().bool().cpu().reshape(-1)
    reference = (
        torch.as_tensor(gauge["reference_weight"])
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )
    if valid.shape != reference.shape or int(model.get_xyz().shape[0]) != valid.numel():
        raise ValueError("carrier geometry differs from source-gauge primitive domain")
    source_visible = (valid & (reference > 0)).float().to(device)
    ecdf = WeightedRightECDF(
        support=torch.as_tensor(gauge["source_ecdf_support"]),
        cumulative=torch.as_tensor(gauge["source_ecdf_cumulative"]),
        total_weight=float(gauge["source_ecdf_total_weight"]),
        source_rows=int(gauge["source_ecdf_rows"]),
    )
    t_seen = float(gauge["t_seen_quantile"])
    t_completion = float(gauge["t_completion_quantile"])
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame_outputs: dict[str, dict[str, object]] = {}
    coverage_means: list[float] = []
    coverage_minimum = 1.0
    coverage_maximum = 0.0
    maximum_negative_preclamp_overshoot = 0.0
    maximum_positive_preclamp_overshoot = 0.0
    for frame_id in evaluation_frames:
        input_record = target_scores[frame_id]
        score_path = _require_file(
            input_record["path"], input_record["sha256"], f"target score {frame_id}"
        )
        raw_score = np.load(score_path, allow_pickle=False)
        if raw_score.ndim != 2 or not np.isfinite(raw_score).all() or (
            raw_score.min() < 0 or raw_score.max() > 1
        ):
            raise ValueError(f"sealed target score is malformed: {frame_id}")
        view = _view_by_frame(views, frame_id)
        pose = torch.from_numpy(view["w2c"].copy()).float().to(device)
        with torch.no_grad():
            rendered_result = renderer.render_feature_rows(
                model,
                pose,
                source_visible[:, None],
                feature_height=int(raw_score.shape[0]),
                feature_width=int(raw_score.shape[1]),
                alpha_normalize=True,
                contribution_gamma=1.0,
            )
        rendered = rendered_result["feature_map"][0]
        alpha_map = rendered_result["alpha_map"]
        coverage = rendered.detach().float().cpu()
        if coverage.dtype != torch.float32 or not bool(torch.isfinite(coverage).all()):
            raise RuntimeError(f"source-visible coverage is non-finite: {frame_id}")
        negative_overshoot = max(0.0, -float(coverage.min()))
        positive_overshoot = max(0.0, float(coverage.max()) - 1.0)
        maximum_negative_preclamp_overshoot = max(
            maximum_negative_preclamp_overshoot, negative_overshoot
        )
        maximum_positive_preclamp_overshoot = max(
            maximum_positive_preclamp_overshoot, positive_overshoot
        )
        if negative_overshoot > expected_tolerance or positive_overshoot > (
            expected_tolerance
        ):
            reconstructed_numerator = rendered * alpha_map
            raise RuntimeError(
                "source-visible coverage exceeds the registered FP32 bound: "
                + json.dumps(
                    {
                        "frame_id": frame_id,
                        "coverage_dtype": str(rendered.dtype),
                        "coverage_min": float(rendered.min()),
                        "coverage_max": float(rendered.max()),
                        "registered_absolute_tolerance": expected_tolerance,
                        "negative_overshoot": negative_overshoot,
                        "positive_overshoot": positive_overshoot,
                        "numerator_dtype": str(reconstructed_numerator.dtype),
                        "numerator_min": float(reconstructed_numerator.min()),
                        "numerator_max": float(reconstructed_numerator.max()),
                        "denominator_dtype": str(alpha_map.dtype),
                        "denominator_min": float(alpha_map.min()),
                        "denominator_max": float(alpha_map.max()),
                    },
                    sort_keys=True,
                )
            )
        coverage = coverage.clamp(0.0, 1.0)
        fields = build_quantile_prediction_fields(
            torch.from_numpy(raw_score),
            coverage,
            ecdf,
            t_seen_quantile=t_seen,
            t_completion_quantile=t_completion,
        )
        frame_root = output_root / "frames" / frame_id
        arrays = {
            "coverage": _save_array(
                frame_root / "source_visible_coverage.npy",
                coverage.numpy().astype(np.float32),
            ),
            "score_quantile": _save_array(
                frame_root / "score_quantile.npy",
                fields.score_quantile.numpy().astype(np.float64),
            ),
            "spatial_threshold_quantile": _save_array(
                frame_root / "spatial_threshold_quantile.npy",
                fields.spatial_threshold_quantile.numpy().astype(np.float64),
            ),
            "continuous_margin": _save_array(
                frame_root / "continuous_margin.npy",
                fields.continuous_margin.numpy().astype(np.float64),
            ),
            "low_resolution_prediction": _save_array(
                frame_root / "low_resolution_prediction.npy",
                fields.low_resolution_prediction.numpy().astype(np.uint8),
            ),
        }
        frame_outputs[frame_id] = {
            "input_raw_score": {
                "path": str(score_path),
                "sha256": input_record["sha256"],
            },
            **arrays,
            "coverage_mean": float(coverage.mean()),
            "coverage_min": float(coverage.min()),
            "coverage_max": float(coverage.max()),
            "low_resolution_positive_fraction": float(
                fields.low_resolution_prediction.double().mean()
            ),
        }
        coverage_means.append(float(coverage.mean()))
        coverage_minimum = min(coverage_minimum, float(coverage.min()))
        coverage_maximum = max(coverage_maximum, float(coverage.max()))
    receipt = {
        "schema_version": 1,
        "artifact_type": PREDICTION_RECEIPT_TYPE,
        "status": "sealed_before_target_ground_truth_open",
        "scene_id": scene_id,
        "protocol_hash": protocol_hash,
        "preregistration": str(prereg_path),
        "preregistration_sha256": str(args.preregistration_sha256),
        "numeric_addendum": str(addendum_path),
        "numeric_addendum_sha256": str(args.numeric_addendum_sha256),
        "full_fit_source_gauge": str(gauge_path),
        "full_fit_source_gauge_sha256": str(args.full_fit_source_gauge_sha256),
        "matched_premetric_receipt": str(premetric_path),
        "matched_premetric_receipt_sha256": str(args.matched_premetric_receipt_sha256),
        "manifest": str(manifest_path),
        "manifest_sha256": str(args.manifest_sha256),
        "carrier": {
            "config": {"path": str(config_path), "sha256": args.scene_config_sha256},
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": args.scene_checkpoint_sha256,
            },
            "camera_map": {
                "path": str(camera_map_path),
                "sha256": args.camera_map_sha256,
            },
        },
        "method_contract": {
            "source_visible": "valid and complete-source reference_weight>0",
            "coverage_compositor": "alpha_normalized_scalar_gaussian_compositor",
            "coverage_contribution_gamma": 1.0,
            "coverage_numeric_projection": {
                "finite_required": True,
                "dtype": "torch.float32",
                "absolute_tolerance": expected_tolerance,
                "operation": "fail_outside_bound_then_clamp_to_[0,1]",
                "method_semantics_changed": False,
            },
            "score_gauge": "fixed_full_source_weighted_right_ecdf",
            "raw_matched_baseline_threshold": float(gauge["t_seen_raw"]),
            "t_seen_quantile": t_seen,
            "t_completion_quantile": t_completion,
            "spatial_threshold": "coverage*t_seen+(1-coverage)*t_completion",
            "prediction_representation": "continuous_margin=F_full(raw_score)-spatial_threshold",
            "evaluation_adapter": "cv2.INTER_LINEAR_margin_to_gt_then_greater_equal_zero",
            "parameter_scan": False,
        },
        "frames": frame_outputs,
        "frame_count": len(frame_outputs),
        "coverage_summary_before_gt": {
            "frame_macro_mean": float(np.mean(coverage_means)),
            "global_min": coverage_minimum,
            "global_max": coverage_maximum,
            "maximum_negative_preclamp_overshoot": (
                maximum_negative_preclamp_overshoot
            ),
            "maximum_positive_preclamp_overshoot": (
                maximum_positive_preclamp_overshoot
            ),
        },
        "sealed_before_target_ground_truth_open": True,
        "target_rgb_opened": False,
        "target_mask_opened": False,
        "target_metric_computed": False,
        "device": str(device),
    }
    content_sha256 = json_sha256(receipt)
    receipt = {**receipt, "content_sha256": content_sha256}
    receipt_path = output_root / "pre_metric_prediction_receipt.json"
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if receipt_path.exists() and receipt_path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"refusing to overwrite different prediction receipt: {receipt_path}")
    if not receipt_path.exists():
        receipt_path.write_text(encoded, encoding="utf-8")
    return {**receipt, "receipt_path": str(receipt_path), "receipt_sha256": file_sha256(receipt_path)}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--numeric-addendum", required=True)
    result.add_argument("--numeric-addendum-sha256", required=True)
    result.add_argument("--full-fit-source-gauge", required=True)
    result.add_argument("--full-fit-source-gauge-sha256", required=True)
    result.add_argument("--matched-premetric-receipt", required=True)
    result.add_argument("--matched-premetric-receipt-sha256", required=True)
    result.add_argument("--manifest", required=True)
    result.add_argument("--manifest-sha256", required=True)
    result.add_argument("--queue-root", required=True)
    result.add_argument("--scene-config", default="")
    result.add_argument("--scene-config-sha256", required=True)
    result.add_argument("--scene-checkpoint", default="")
    result.add_argument("--scene-checkpoint-sha256", required=True)
    result.add_argument("--camera-map", default="")
    result.add_argument("--camera-map-sha256", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--device", default="cuda:0")
    return result


def main() -> None:
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
