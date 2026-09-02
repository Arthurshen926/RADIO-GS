"""Training-only cross-fit kNN token mass for frozen bias-free spatial slots."""

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


REPORT_SCHEMA = "radio_gs.surface_object_memory_v4.knn_mass_spatial_slots.v1"
K_CANDIDATES = (4, 8, 16, 32)


def _load_weights_only(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("safe weights-only checkpoint loading is required") from error


def _standardize_training_features(
    records: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.cat([record["features"] for record in records])
    mean = values.mean(0)
    scale = values.std(0, unbiased=False).clamp_min(1e-4)
    return mean, scale


def _predict_log_ratio(
    query: torch.Tensor,
    reference_features: torch.Tensor,
    reference_log_ratio: torch.Tensor,
    *,
    neighbour_count: int,
) -> torch.Tensor:
    if reference_features.shape[0] < neighbour_count:
        raise ValueError("kNN reference set is smaller than neighbour count")
    distance = torch.cdist(query, reference_features)
    neighbours = distance.topk(neighbour_count, largest=False).indices
    return reference_log_ratio[neighbours].mean(-1)


def _select_k_training_crossfit(
    records: list[dict[str, Any]], mean: torch.Tensor, scale: torch.Tensor
) -> tuple[int, list[dict[str, float]]]:
    curve = []
    for neighbour_count in K_CANDIDATES:
        scene_losses = []
        for heldout in records:
            references = [record for record in records if record is not heldout]
            reference_features = torch.cat(
                [(record["features"] - mean) / scale for record in references]
            )
            reference_log_ratio = torch.cat(
                [record["target_log_ratio"] for record in references]
            )
            predicted_log_ratio = _predict_log_ratio(
                (heldout["features"] - mean) / scale,
                reference_features,
                reference_log_ratio,
                neighbour_count=neighbour_count,
            )
            predicted_mass = heldout["observed_mass"] * predicted_log_ratio.exp()
            scene_losses.append(
                float(
                    F.smooth_l1_loss(
                        torch.log1p(predicted_mass),
                        torch.log1p(heldout["target_mass"]),
                    )
                )
            )
        curve.append(
            {
                "neighbour_count": neighbour_count,
                "scene_macro_log_mass_loss": float(np.mean(scene_losses)),
            }
        )
    selected = min(
        curve,
        key=lambda value: (
            value["scene_macro_log_mass_loss"],
            value["neighbour_count"],
        ),
    )
    return int(selected["neighbour_count"]), curve


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.feature_mode not in ("summary", "summary_f71"):
        raise ValueError("unsupported kNN mass feature mode")
    base_report_path = Path(args.base_report).resolve(strict=True)
    base_checkpoint_path = Path(args.base_checkpoint).resolve(strict=True)
    slot_checkpoint_path = Path(args.slot_checkpoint).resolve(strict=True)
    base_report = json.loads(base_report_path.read_text())
    base_checkpoint = _load_weights_only(base_checkpoint_path)
    slot_checkpoint = _load_weights_only(slot_checkpoint_path)
    if slot_checkpoint.get("mode") != "spatial_only":
        raise ValueError("kNN mass requires the frozen bias-free spatial arm")

    training_ids = set(map(str, args.training_scene))
    validation_ids = set(map(str, args.validation_scene))
    if training_ids & validation_ids:
        raise ValueError("kNN mass training and validation must be disjoint")
    split = base_report["split"]
    if not set(map(str, split["training_scene_ids"])).issubset(training_ids):
        raise ValueError("kNN mass must retain the frozen training split")
    if validation_ids != set(map(str, split["validation_scene_ids"])):
        raise ValueError("kNN mass requires the frozen validation split")
    payloads = [load_scene_cache(Path(value)) for value in args.scene_cache]
    by_id = {str(value["scene_id"]): value for value in payloads}
    if set(by_id) != training_ids | validation_ids:
        raise ValueError("kNN mass caches do not equal the declared cohort")

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
        observed = source["observed_mass"]
        training.append(
            {
                "scene_id": scene_id,
                "features": source["features"],
                "observed_mass": observed,
                "target_mass": target,
                "target_log_ratio": torch.log(target / observed),
            }
        )
    feature_mean, feature_scale = _standardize_training_features(training)
    selected_k, crossfit_curve = _select_k_training_crossfit(
        training, feature_mean, feature_scale
    )
    reference_features = torch.cat(
        [(record["features"] - feature_mean) / feature_scale for record in training]
    )
    reference_log_ratio = torch.cat(
        [record["target_log_ratio"] for record in training]
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
            mass_source = build_mass_features(
                runtime, unary, feature_mode=args.feature_mode
            )
            predicted_log_ratio = _predict_log_ratio(
                (mass_source["features"] - feature_mean) / feature_scale,
                reference_features,
                reference_log_ratio,
                neighbour_count=selected_k,
            )
            predicted_mass = mass_source["observed_mass"] * predicted_log_ratio.exp()
            clamp_mask, clamp_probabilities = _clamp_contract(runtime)
            combined, projection = oracle_mass_project(
                slot_posterior.to(device),
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
                ("spatial_slots_knn_mass_projection", combined.cpu()),
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
        "spatial_slots_knn_mass_projection",
    )
    scene_macro = {
        method: {
            key: float(np.mean([record[method][key] for record in records]))
            for key in METRIC_KEYS
        }
        for method in methods
    }
    difference = {
        key: scene_macro["spatial_slots_knn_mass_projection"][key]
        - scene_macro["frozen_aligned_pointwise"][key]
        for key in METRIC_KEYS
    }
    source_path = Path(__file__).resolve()
    report = {
        "schema": REPORT_SCHEMA,
        "role": "training_only_crossfit_nonparametric_mass_for_bias_free_spatial_slots",
        "feature_mode": args.feature_mode,
        "target_membership_is_model_input": False,
        "validation_used_for_k_selection": False,
        "training_scene_ids": sorted(training_ids),
        "validation_scene_ids": sorted(validation_ids),
        "k_selection": {
            "method": "leave_one_training_scene_out_scene_macro_log_mass_loss",
            "candidates": list(K_CANDIDATES),
            "curve": crossfit_curve,
            "selected_k": selected_k,
        },
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "base_report": {"path": str(base_report_path), "sha256": sha256_file(base_report_path)},
        "base_checkpoint": {"path": str(base_checkpoint_path), "sha256": sha256_file(base_checkpoint_path)},
        "slot_checkpoint": {"path": str(slot_checkpoint_path), "sha256": sha256_file(slot_checkpoint_path)},
        "per_validation_scene": records,
        "scene_macro": scene_macro,
        "knn_mass_slots_minus_frozen_scene_macro": difference,
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
    parser.add_argument("--feature-mode", choices=("summary", "summary_f71"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unary-element-batch-size", type=int, default=4096)
    parser.add_argument("--inference-element-batch-size", type=int, default=4096)
    parser.add_argument("--projection-iteration-count", type=int, default=256)
    parser.add_argument("--projection-damping", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["knn_mass_slots_minus_frozen_scene_macro"], indent=2))


if __name__ == "__main__":
    main()
