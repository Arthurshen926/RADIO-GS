#!/usr/bin/env python3
"""Seal one full9 factorized SPIn continuous target-margin receipt before GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from radio_gs.config import load_config
from radio_gs.querying.spin_source_footprint_quantile_calibration import (
    WeightedRightECDF,
    build_quantile_prediction_fields,
)
from radio_gs.scripts.build_spin_source_footprint_quantile_oof import (
    file_sha256,
    json_sha256,
)
from radio_gs.scripts.build_spin_source_footprint_quantile_target_prediction import (
    FP32_COVERAGE_EPSILON_MULTIPLE,
    _load_premetric_receipt,
    _load_tensor_authority,
    _require_file,
    _save_array,
)
from radio_gs.scripts.eval_lerf_grounding import load_render_pipeline
from radio_gs.scripts.eval_nvos_gaussian_first import (
    _resolve_scene_carrier_assets,
    _scene_record,
    _view_by_frame,
)
from radio_gs.scripts.render_promptable_nvs_features import resolve_protocol_views


PREREGISTRATION = "spin9_factorized_source_quantile_full9_expansion_v1"
PREDICTION_RECEIPT_TYPE = "spin9_factorized_source_quantile_target_prediction_v1"
NUMERIC_ADDENDUM_REGISTRATION = (
    "spin9_factorized_complementary_coverage_numeric_addendum_v1"
)
RAW_SEEN_THRESHOLD = 0.71
COMPLETION_QUANTILE = 0.96


def _load_preregistration(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("registration") != PREREGISTRATION:
        raise ValueError("unexpected full9 factorized quantile preregistration")
    development = payload.get("development_authority")
    prediction = payload.get("prediction_contract")
    if not isinstance(development, Mapping) or (
        float(development.get("raw_seen_threshold", float("nan")))
        != RAW_SEEN_THRESHOLD
        or float(development.get("fixed_completion_quantile", float("nan")))
        != COMPLETION_QUANTILE
    ):
        raise ValueError("full9 raw/completion thresholds differ")
    if not isinstance(prediction, Mapping) or (
        prediction.get("parameter_scan") is not False
        or prediction.get("target_rgb_at_query") is not False
        or prediction.get("target_mask_or_metric_in_prediction") is not False
    ):
        raise ValueError("full9 prediction safety contract differs")
    return payload


def _paired_complementary_coverage(
    visible_mass: torch.Tensor,
    invisible_mass: torch.Tensor,
    scalar_alpha: torch.Tensor,
    *,
    legacy_tolerance: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Normalize complementary FP32 masses without clipping.

    The ratio is evaluated in FP64 and cast once to FP32.  Exact-zero paired
    mass is unsupported background; negative or non-finite compositor output
    is invalid and fails closed.
    """

    visible = torch.as_tensor(visible_mass).detach().float().cpu()
    invisible = torch.as_tensor(invisible_mass).detach().float().cpu()
    alpha = torch.as_tensor(scalar_alpha).detach().float().cpu()
    if visible.shape != invisible.shape or visible.shape != alpha.shape:
        raise ValueError("paired coverage masses and scalar alpha must align")
    if not bool(torch.isfinite(visible).all()) or not bool(
        torch.isfinite(invisible).all()
    ) or not bool(torch.isfinite(alpha).all()):
        raise RuntimeError("coverage compositor returned non-finite mass")
    if bool((visible < 0).any()) or bool((invisible < 0).any()):
        raise RuntimeError("coverage compositor returned negative paired mass")
    denominator = visible.double() + invisible.double()
    if bool((denominator < 0).any()) or not bool(torch.isfinite(denominator).all()):
        raise RuntimeError("paired coverage denominator is invalid")
    supported = denominator > 0
    oracle = torch.zeros_like(denominator)
    oracle[supported] = visible.double()[supported] / denominator[supported]
    if not bool(torch.isfinite(oracle).all()) or bool(
        ((oracle < 0) | (oracle > 1)).any()
    ):
        raise RuntimeError("FP64 complementary coverage oracle is outside [0,1]")
    coverage = oracle.float()
    if not bool(torch.isfinite(coverage).all()) or bool(
        ((coverage < 0) | (coverage > 1)).any()
    ):
        raise RuntimeError("FP32 complementary coverage is outside [0,1]")

    legacy_supported = alpha > 1e-6
    legacy = torch.zeros_like(alpha)
    legacy[legacy_supported] = visible[legacy_supported] / alpha[legacy_supported]
    legacy_finite = torch.isfinite(legacy)
    legacy_admissible = legacy_supported & legacy_finite & (
        legacy >= -float(legacy_tolerance)
    ) & (legacy <= 1.0 + float(legacy_tolerance))
    if bool(legacy_admissible.any()):
        parity = float(
            (coverage[legacy_admissible] - legacy[legacy_admissible]).abs().max()
        )
    else:
        parity = 0.0
    positive_denominator = denominator[supported]
    diagnostics: dict[str, float | int] = {
        "unsupported_zero_pixels": int((~supported).sum()),
        "supported_positive_pixels": int(supported.sum()),
        "minimum_positive_denominator": (
            float(positive_denominator.min()) if positive_denominator.numel() else 0.0
        ),
        "maximum_positive_denominator": (
            float(positive_denominator.max()) if positive_denominator.numel() else 0.0
        ),
        "legacy_minimum": float(legacy.min()),
        "legacy_maximum": float(legacy.max()),
        "legacy_positive_overshoot": max(0.0, float(legacy.max()) - 1.0),
        "legacy_negative_overshoot": max(0.0, -float(legacy.min())),
        "legacy_admissible_pixels": int(legacy_admissible.sum()),
        "maximum_corrected_vs_legacy_admissible_absolute_difference": parity,
        "maximum_fp32_vs_fp64_oracle_absolute_difference": float(
            (coverage.double() - oracle).abs().max()
        ),
        "corrected_minimum": float(coverage.min()),
        "corrected_maximum": float(coverage.max()),
    }
    return coverage, diagnostics


def build(args: argparse.Namespace) -> dict[str, object]:
    prereg_path = _require_file(
        args.preregistration, args.preregistration_sha256, "full9 preregistration"
    )
    _load_preregistration(prereg_path)
    addendum_path = _require_file(
        args.numeric_addendum,
        args.numeric_addendum_sha256,
        "coverage numeric addendum",
    )
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    legacy_tolerance = float(
        FP32_COVERAGE_EPSILON_MULTIPLE * torch.finfo(torch.float32).eps
    )
    correction = addendum.get("implementation_correction") if isinstance(
        addendum, Mapping
    ) else None
    if (
        not isinstance(addendum, Mapping)
        or addendum.get("registration") != NUMERIC_ADDENDUM_REGISTRATION
        or not isinstance(correction, Mapping)
        or correction.get("new_tunable_constant") is not False
        or correction.get("nonnegative_requirement") is not True
        or correction.get("finite_requirement") is not True
    ):
        raise ValueError("coverage numeric addendum differs")

    gauge_path = _require_file(
        args.full_fit_source_gauge,
        args.full_fit_source_gauge_sha256,
        "full9 full-fit source gauge",
    )
    gauge = _load_tensor_authority(gauge_path)
    if (
        gauge.get("status")
        != "sealed_full9_factorized_exact_w_source_only_quantile_gauge"
        or gauge.get("preregistration_sha256") != args.preregistration_sha256
        or float(gauge.get("t_seen_raw", float("nan"))) != RAW_SEEN_THRESHOLD
        or float(gauge.get("t_completion_quantile", float("nan")))
        != COMPLETION_QUANTILE
        or gauge.get("source_unary_authority")
        != "native_exact_W_adjoint_source_mask_probability"
        or gauge.get("source_weight_authority") != "native_exact_W_visible_mass"
    ):
        raise ValueError("source gauge is not the registered full9 exact-W authority")
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
    maximum_legacy_negative_overshoot = 0.0
    maximum_legacy_positive_overshoot = 0.0
    maximum_corrected_vs_legacy_admissible_difference = 0.0
    maximum_fp32_vs_fp64_oracle_difference = 0.0
    total_unsupported_zero_pixels = 0
    minimum_positive_paired_denominator = float("inf")
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
                torch.stack((source_visible, 1.0 - source_visible), dim=1),
                feature_height=int(raw_score.shape[0]),
                feature_width=int(raw_score.shape[1]),
                alpha_normalize=False,
                contribution_gamma=1.0,
            )
        paired = rendered_result["feature_map"]
        if paired.shape[0] != 2:
            raise RuntimeError("complementary coverage render did not return two channels")
        coverage, coverage_diagnostic = _paired_complementary_coverage(
            paired[0],
            paired[1],
            rendered_result["alpha_map"],
            legacy_tolerance=legacy_tolerance,
        )
        maximum_legacy_negative_overshoot = max(
            maximum_legacy_negative_overshoot,
            float(coverage_diagnostic["legacy_negative_overshoot"]),
        )
        maximum_legacy_positive_overshoot = max(
            maximum_legacy_positive_overshoot,
            float(coverage_diagnostic["legacy_positive_overshoot"]),
        )
        maximum_corrected_vs_legacy_admissible_difference = max(
            maximum_corrected_vs_legacy_admissible_difference,
            float(
                coverage_diagnostic[
                    "maximum_corrected_vs_legacy_admissible_absolute_difference"
                ]
            ),
        )
        maximum_fp32_vs_fp64_oracle_difference = max(
            maximum_fp32_vs_fp64_oracle_difference,
            float(
                coverage_diagnostic[
                    "maximum_fp32_vs_fp64_oracle_absolute_difference"
                ]
            ),
        )
        total_unsupported_zero_pixels += int(
            coverage_diagnostic["unsupported_zero_pixels"]
        )
        minimum_positive_paired_denominator = min(
            minimum_positive_paired_denominator,
            float(coverage_diagnostic["minimum_positive_denominator"]),
        )
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
            "coverage_numeric_diagnostic": coverage_diagnostic,
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
            "source_visible": "valid and native-exact-W reference_weight>0",
            "coverage_compositor": "alpha_normalized_scalar_gaussian_compositor",
            "coverage_contribution_gamma": 1.0,
            "coverage_numeric_projection": {
                "finite_required": True,
                "paired_mass_dtype": "torch.float32",
                "ratio_dtype": "torch.float64",
                "output_dtype": "torch.float32",
                "operation": (
                    "paired_visible_mass/(visible_mass+invisible_mass); no clipping"
                ),
                "unsupported_zero": "denominator==0 maps exactly to zero",
                "invalid": "negative or non-finite mass fails closed",
                "postcondition": "finite and exactly within [0,1]",
                "legacy_diagnostic_tolerance": legacy_tolerance,
                "exact_arithmetic_semantics_changed": False,
            },
            "score_gauge": "fixed_full_source_weighted_right_ecdf",
            "raw_matched_baseline_threshold": RAW_SEEN_THRESHOLD,
            "t_seen_quantile": t_seen,
            "t_completion_quantile": t_completion,
            "spatial_threshold": "coverage*t_seen+(1-coverage)*t_completion",
            "prediction_representation": (
                "continuous_margin=F_full(raw_score)-spatial_threshold"
            ),
            "evaluation_adapter": (
                "cv2.INTER_LINEAR_margin_to_gt_then_greater_equal_zero"
            ),
            "parameter_scan": False,
        },
        "frames": frame_outputs,
        "frame_count": len(frame_outputs),
        "coverage_summary_before_gt": {
            "frame_macro_mean": float(np.mean(coverage_means)),
            "global_min": coverage_minimum,
            "global_max": coverage_maximum,
            "maximum_legacy_negative_overshoot": maximum_legacy_negative_overshoot,
            "maximum_legacy_positive_overshoot": maximum_legacy_positive_overshoot,
            "maximum_corrected_vs_legacy_admissible_absolute_difference": (
                maximum_corrected_vs_legacy_admissible_difference
            ),
            "maximum_fp32_vs_fp64_oracle_absolute_difference": (
                maximum_fp32_vs_fp64_oracle_difference
            ),
            "unsupported_zero_pixels": total_unsupported_zero_pixels,
            "minimum_positive_paired_denominator": (
                minimum_positive_paired_denominator
                if np.isfinite(minimum_positive_paired_denominator)
                else 0.0
            ),
            "triggered_legacy_overshoot_before_correction": 8.344650268554688e-06,
            "corrected_coverage_clipped": False,
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
        raise FileExistsError(
            f"refusing to overwrite different full9 prediction receipt: {receipt_path}"
        )
    if not receipt_path.exists():
        receipt_path.write_text(encoded, encoding="utf-8")
    return {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
    }


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
