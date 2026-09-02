"""Scene-balanced ridge mass residuals for frozen bias-free spatial slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from radio_gs.v4.completion.scannet import load_scene_cache
from radio_gs.v4.completion.spatial_slots import TokenSpatialSupportSlots
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.training.diagnose_scannet_knn_mass_spatial_slots import (
    _load_weights_only,
    _standardize_training_features,
)
from radio_gs.v4.training.diagnose_scannet_learned_mass_calibration import (
    build_mass_features,
    target_physical_mass,
)
from radio_gs.v4.training.diagnose_scannet_oracle_mass_projection import (
    METRIC_KEYS,
    oracle_mass_project,
)
from radio_gs.v4.training.train_scannet_completion_message_passing import (
    _clamp_contract,
    _frozen_unary_probabilities,
    _load_frozen_unary_model,
    _membership_metrics,
    _posterior_to_membership,
    _prepare_runtime,
)
from radio_gs.v4.training.train_scannet_spatial_slots import (
    _full_posterior,
    build_observed_pca_geometry,
)


REPORT_SCHEMA = "radio_gs.surface_object_memory_v4.ridge_mass_spatial_slots.v1"
RIDGE_CANDIDATES = (0.1, 1.0, 10.0, 100.0, 1000.0)
BLEND_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)


def fit_scene_balanced_ridge(
    records: list[dict[str, Any]],
    *,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    if ridge <= 0 or not records:
        raise ValueError("ridge fit requires positive regularization and records")
    feature_pieces = []
    target_pieces = []
    weight_pieces = []
    for record in records:
        feature = (record["features"] - feature_mean) / feature_scale
        feature_pieces.append(feature)
        target_pieces.append(record["target_log_correction"])
        weight_pieces.append(
            torch.full(
                (feature.shape[0],), 1.0 / float(feature.shape[0]), dtype=torch.float64
            )
        )
    feature = torch.cat(feature_pieces).double()
    design = torch.cat((torch.ones(feature.shape[0], 1, dtype=torch.float64), feature), -1)
    target = torch.cat(target_pieces).double()
    weight = torch.cat(weight_pieces)
    weighted_design = design * weight.sqrt()[:, None]
    weighted_target = target * weight.sqrt()
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficient = torch.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    if not torch.isfinite(coefficient).all():
        raise RuntimeError("ridge mass coefficient became non-finite")
    return coefficient.float()


def predict_log_correction(
    features: torch.Tensor,
    coefficient: torch.Tensor,
    *,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
) -> torch.Tensor:
    standardized = (features - feature_mean) / feature_scale
    design = torch.cat((torch.ones(standardized.shape[0], 1), standardized), -1)
    return design @ coefficient


def select_ridge_and_blend(
    records: list[dict[str, Any]],
    *,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
) -> tuple[float, float, list[dict[str, float]]]:
    curve = []
    for ridge in RIDGE_CANDIDATES:
        fold_predictions = []
        for heldout_index, heldout in enumerate(records):
            coefficient = fit_scene_balanced_ridge(
                [record for index, record in enumerate(records) if index != heldout_index],
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                ridge=ridge,
            )
            fold_predictions.append(
                predict_log_correction(
                    heldout["features"],
                    coefficient,
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                )
            )
        for blend in BLEND_CANDIDATES:
            losses = []
            for record, correction in zip(records, fold_predictions):
                predicted = (
                    record["frozen_mass"] * torch.exp(float(blend) * correction)
                ).clamp_min(record["observed_mass"])
                losses.append(
                    float(
                        F.smooth_l1_loss(
                            torch.log1p(predicted), torch.log1p(record["target_mass"])
                        )
                    )
                )
            curve.append(
                {
                    "ridge": ridge,
                    "blend": blend,
                    "scene_macro_log_mass_loss": float(np.mean(losses)),
                }
            )
    selected = min(
        curve,
        key=lambda value: (
            value["scene_macro_log_mass_loss"],
            value["ridge"],
            value["blend"],
        ),
    )
    return float(selected["ridge"]), float(selected["blend"]), curve


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.feature_mode not in (
        "summary",
        "coverage_geometry",
        "summary_f71",
        "source_view_coverage",
        "source_view_coverage_f71",
    ):
        raise ValueError("unsupported ridge mass feature mode")
    base_report_path = Path(args.base_report).resolve(strict=True)
    base_checkpoint_path = Path(args.base_checkpoint).resolve(strict=True)
    slot_checkpoint_path = Path(args.slot_checkpoint).resolve(strict=True)
    base_report = json.loads(base_report_path.read_text())
    base_checkpoint = _load_weights_only(base_checkpoint_path)
    slot_checkpoint = _load_weights_only(slot_checkpoint_path)
    if slot_checkpoint.get("mode") != "spatial_only":
        raise ValueError("ridge mass requires the frozen bias-free spatial arm")

    training_ids = set(map(str, args.training_scene))
    validation_ids = set(map(str, args.validation_scene))
    if training_ids & validation_ids:
        raise ValueError("ridge mass training and validation must be disjoint")
    split = base_report["split"]
    if not set(map(str, split["training_scene_ids"])).issubset(training_ids):
        raise ValueError("ridge mass must retain the frozen training split")
    if validation_ids != set(map(str, split["validation_scene_ids"])):
        raise ValueError("ridge mass requires the frozen validation split")
    payloads = [load_scene_cache(Path(value)) for value in args.scene_cache]
    by_id = {str(value["scene_id"]): value for value in payloads}
    if set(by_id) != training_ids | validation_ids:
        raise ValueError("ridge mass caches do not equal the declared cohort")

    device = torch.device(args.device)
    frozen_model = _load_frozen_unary_model(base_checkpoint, device=device)
    temperature = float(base_report["training_configuration"]["temperature"])
    confidence_cap = float(
        base_report["training_configuration"]["completion_confidence_cap"]
    )
    training = []
    for scene_id in sorted(training_ids):
        runtime = _prepare_runtime(by_id[scene_id])
        unary = _frozen_unary_probabilities(
            frozen_model,
            runtime,
            device=device,
            element_batch_size=args.unary_element_batch_size,
            temperature=temperature,
        )
        source = build_mass_features(runtime, unary, feature_mode=args.feature_mode)
        target = target_physical_mass(runtime)
        frozen_mass = source["frozen_mass"].clamp_min(source["observed_mass"])
        training.append(
            {
                "scene_id": scene_id,
                "features": source["features"],
                "observed_mass": source["observed_mass"],
                "frozen_mass": frozen_mass,
                "target_mass": target,
                "target_log_correction": torch.log(target / frozen_mass),
            }
        )
    feature_mean, feature_scale = _standardize_training_features(training)
    selected_ridge, selected_blend, crossfit_curve = select_ridge_and_blend(
        training, feature_mean=feature_mean, feature_scale=feature_scale
    )
    coefficient = fit_scene_balanced_ridge(
        training,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        ridge=selected_ridge,
    )

    slot_configuration = slot_checkpoint["model_configuration"]
    slot_model = TokenSpatialSupportSlots(
        input_dimension=int(slot_configuration["input_dimension"]),
        hidden_dimension=int(slot_configuration["hidden_dimension"]),
        dropout=float(slot_configuration["dropout"]),
        use_token_bias=bool(slot_configuration["use_token_bias"]),
    ).to(device)
    slot_model.load_state_dict(slot_checkpoint["model_state_dict"], strict=True)
    slot_model.eval().requires_grad_(False)

    records = []
    with torch.no_grad():
        for scene_id in sorted(validation_ids):
            runtime = _prepare_runtime(by_id[scene_id])
            unary = _frozen_unary_probabilities(
                frozen_model,
                runtime,
                device=device,
                element_batch_size=args.unary_element_batch_size,
                temperature=temperature,
            )
            slot_source = build_mass_features(runtime, unary, feature_mode="summary_f71")
            slot_record = {
                "runtime": runtime,
                "unary": unary,
                "token_features": slot_source["features"],
                "pca_geometry": build_observed_pca_geometry(runtime),
            }
            slot_posterior, slot_audit = _full_posterior(
                slot_model,
                slot_record,
                device=device,
                element_batch_size=args.inference_element_batch_size,
            )
            mass_source = build_mass_features(runtime, unary, feature_mode=args.feature_mode)
            frozen_mass = mass_source["frozen_mass"].clamp_min(
                mass_source["observed_mass"]
            )
            correction = predict_log_correction(
                mass_source["features"],
                coefficient,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
            )
            predicted_mass = (
                frozen_mass * torch.exp(selected_blend * correction)
            ).clamp_min(mass_source["observed_mass"])
            clamp_mask, clamp_probabilities = _clamp_contract(runtime)
            combined, projection = oracle_mass_project(
                slot_posterior.to(device),
                clamp_mask.to(device),
                clamp_probabilities.to(device),
                predicted_mass.to(device),
                iteration_count=args.projection_iteration_count,
                damping=args.projection_damping,
            )
            frozen_mass_matched, frozen_projection = oracle_mass_project(
                unary.to(device),
                clamp_mask.to(device),
                clamp_probabilities.to(device),
                predicted_mass.to(device),
                iteration_count=args.projection_iteration_count,
                damping=args.projection_damping,
            )
            target_mass = target_physical_mass(runtime)
            memberships = {}
            for name, posterior in (
                ("frozen_aligned_pointwise", unary),
                ("spatial_only", slot_posterior),
                ("frozen_unary_ridge_mass_projection", frozen_mass_matched.cpu()),
                ("spatial_slots_ridge_mass_projection", combined.cpu()),
            ):
                membership, null = _posterior_to_membership(
                    posterior, runtime, completion_confidence_cap=confidence_cap
                )
                memberships[name] = _membership_metrics(runtime, membership, null)
            relative_error = (predicted_mass - target_mass).abs() / target_mass
            records.append(
                {
                    "scene_id": scene_id,
                    "slot_audit": slot_audit,
                    "projection": projection,
                    "frozen_unary_projection": frozen_projection,
                    "mass_prediction": {
                        "mean_absolute_relative_error": float(relative_error.mean()),
                        "median_absolute_relative_error": float(relative_error.median()),
                        "predicted_mean": float(predicted_mass.mean()),
                        "target_mean": float(target_mass.mean()),
                    },
                    **memberships,
                }
            )

    methods = (
        "frozen_aligned_pointwise",
        "spatial_only",
        "frozen_unary_ridge_mass_projection",
        "spatial_slots_ridge_mass_projection",
    )
    scene_macro = {
        method: {
            key: float(np.mean([record[method][key] for record in records]))
            for key in METRIC_KEYS
        }
        for method in methods
    }
    difference = {
        key: scene_macro["spatial_slots_ridge_mass_projection"][key]
        - scene_macro["frozen_aligned_pointwise"][key]
        for key in METRIC_KEYS
    }
    slots_minus_mass_matched_frozen = {
        key: scene_macro["spatial_slots_ridge_mass_projection"][key]
        - scene_macro["frozen_unary_ridge_mass_projection"][key]
        for key in METRIC_KEYS
    }
    source_path = Path(__file__).resolve()
    report = {
        "schema": REPORT_SCHEMA,
        "role": "training_only_crossfit_scene_balanced_ridge_mass_for_bias_free_slots",
        "feature_mode": args.feature_mode,
        "target_membership_is_model_input": False,
        "validation_used_for_selection": False,
        "training_scene_ids": sorted(training_ids),
        "validation_scene_ids": sorted(validation_ids),
        "selection": {
            "method": "leave_one_training_scene_out_scene_macro_log_mass_loss",
            "ridge_candidates": list(RIDGE_CANDIDATES),
            "blend_candidates": list(BLEND_CANDIDATES),
            "selected_ridge": selected_ridge,
            "selected_blend": selected_blend,
            "curve": crossfit_curve,
        },
        "coefficient": coefficient.tolist(),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "base_report": {"path": str(base_report_path), "sha256": sha256_file(base_report_path)},
        "base_checkpoint": {"path": str(base_checkpoint_path), "sha256": sha256_file(base_checkpoint_path)},
        "slot_checkpoint": {"path": str(slot_checkpoint_path), "sha256": sha256_file(slot_checkpoint_path)},
        "per_validation_scene": records,
        "scene_macro": scene_macro,
        "ridge_mass_slots_minus_frozen_scene_macro": difference,
        "slots_minus_mass_matched_frozen_scene_macro": slots_minus_mass_matched_frozen,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-cache", action="append", required=True)
    parser.add_argument("--training-scene", action="append", required=True)
    parser.add_argument("--validation-scene", action="append", required=True)
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--slot-checkpoint", required=True)
    parser.add_argument(
        "--feature-mode",
        choices=(
            "summary",
            "coverage_geometry",
            "summary_f71",
            "source_view_coverage",
            "source_view_coverage_f71",
        ),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unary-element-batch-size", type=int, default=4096)
    parser.add_argument("--inference-element-batch-size", type=int, default=4096)
    parser.add_argument("--projection-iteration-count", type=int, default=256)
    parser.add_argument("--projection-damping", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["ridge_mass_slots_minus_frozen_scene_macro"], indent=2))


if __name__ == "__main__":
    main()
