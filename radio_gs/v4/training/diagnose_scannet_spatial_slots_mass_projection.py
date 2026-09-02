"""Combine learned spatial-slot ranking with learned source-only token mass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radio_gs.v4.completion.scannet import load_scene_cache
from radio_gs.v4.completion.spatial_slots import TokenSpatialSupportSlots
from radio_gs.v4.contracts.geometry_receipt import sha256_file
from radio_gs.v4.training.diagnose_scannet_learned_mass_calibration import (
    UnaryMassCalibrator,
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


REPORT_SCHEMA = (
    "radio_gs.surface_object_memory_v4.spatial_slots_learned_mass_projection.v1"
)
ORACLE_REPORT_SCHEMA = (
    "radio_gs.surface_object_memory_v4.spatial_slots_oracle_mass_projection.v1"
)


def _load_weights_only(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("safe weights-only checkpoint loading is required") from error


def _load_mass_calibrator(
    report: dict[str, Any], *, device: torch.device
) -> UnaryMassCalibrator:
    configuration = report["training_configuration"]
    model = UnaryMassCalibrator(
        int(configuration["input_dimension"]),
        int(configuration["hidden_dimension"]),
    )
    state = {
        key: torch.as_tensor(value) for key, value in report["model_state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mass_mode == "learned" and not args.mass_report:
        raise ValueError("learned mass mode requires --mass-report")
    base_report_path = Path(args.base_report).resolve(strict=True)
    base_checkpoint_path = Path(args.base_checkpoint).resolve(strict=True)
    slot_checkpoint_path = Path(args.slot_checkpoint).resolve(strict=True)
    mass_report_path = (
        Path(args.mass_report).resolve(strict=True)
        if args.mass_mode == "learned"
        else None
    )
    base_report = json.loads(base_report_path.read_text())
    base_checkpoint = _load_weights_only(base_checkpoint_path)
    slot_checkpoint = _load_weights_only(slot_checkpoint_path)
    mass_report = (
        json.loads(mass_report_path.read_text())
        if mass_report_path is not None
        else None
    )
    if slot_checkpoint.get("mode") not in ("spatial_slots", "spatial_only"):
        raise ValueError("combination requires the learned spatial-slot arm")
    if mass_report is not None:
        if mass_report.get("feature_mode") != "summary":
            raise ValueError("combination requires the preregistered summary mass arm")
        if mass_report.get("validation_used_for_model_selection") is not False:
            raise ValueError("mass calibrator validation-selection contract changed")

    validation_ids = set(map(str, args.validation_scene))
    expected = set(map(str, base_report["split"]["validation_scene_ids"]))
    if validation_ids != expected:
        raise ValueError("combination requires the frozen validation split")
    payloads = [load_scene_cache(Path(value)) for value in args.scene_cache]
    by_id = {str(value["scene_id"]): value for value in payloads}
    if set(by_id) != validation_ids:
        raise ValueError("combination caches must exactly equal frozen validation")

    device = torch.device(args.device)
    frozen_model = _load_frozen_unary_model(base_checkpoint, device=device)
    slot_configuration = slot_checkpoint["model_configuration"]
    slot_model = TokenSpatialSupportSlots(
        input_dimension=int(slot_configuration["input_dimension"]),
        hidden_dimension=int(slot_configuration["hidden_dimension"]),
        dropout=float(slot_configuration["dropout"]),
        use_token_bias=bool(slot_configuration.get("use_token_bias", True)),
    ).to(device)
    slot_model.load_state_dict(slot_checkpoint["model_state_dict"], strict=True)
    slot_model.eval().requires_grad_(False)
    mass_model = (
        _load_mass_calibrator(mass_report, device=device)
        if mass_report is not None
        else None
    )
    blend_fraction = (
        float(mass_report["training_configuration"]["applied_blend_fraction"])
        if mass_report is not None
        else None
    )
    temperature = float(base_report["training_configuration"]["temperature"])
    confidence_cap = float(
        base_report["training_configuration"]["completion_confidence_cap"]
    )

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
            source = build_mass_features(runtime, unary, feature_mode="summary_f71")
            slot_record = {
                "runtime": runtime,
                "unary": unary,
                "token_features": source["features"],
                "pca_geometry": build_observed_pca_geometry(runtime),
            }
            slot_posterior, slot_audit = _full_posterior(
                slot_model,
                slot_record,
                device=device,
                element_batch_size=args.inference_element_batch_size,
            )

            if mass_model is not None:
                mass_source = build_mass_features(runtime, unary, feature_mode="summary")
                raw_mass = mass_model(
                    mass_source["features"].to(device),
                    mass_source["observed_mass"].to(device),
                    mass_source["frozen_mass"].to(device),
                )
                predicted_mass = (
                    mass_source["frozen_mass"].to(device)
                    + float(blend_fraction)
                    * (raw_mass - mass_source["frozen_mass"].to(device))
                ).clamp_min(mass_source["observed_mass"].to(device))
            else:
                predicted_mass = target_physical_mass(runtime).to(device)
            clamp_mask, clamp_probabilities = _clamp_contract(runtime)
            combined, projection = oracle_mass_project(
                slot_posterior.to(device),
                clamp_mask.to(device),
                clamp_probabilities.to(device),
                predicted_mass,
                iteration_count=args.projection_iteration_count,
                damping=args.projection_damping,
            )

            memberships = {}
            for name, posterior in (
                ("frozen_aligned_pointwise", unary),
                ("spatial_slot_completion", slot_posterior),
                ("spatial_slots_learned_mass_projection", combined.cpu()),
            ):
                membership, null = _posterior_to_membership(
                    posterior, runtime, completion_confidence_cap=confidence_cap
                )
                memberships[name] = _membership_metrics(runtime, membership, null)
            records.append(
                {
                    "scene_id": scene_id,
                    "element_count": int(runtime["centres"].shape[0]),
                    "token_count": int(runtime["partial"].positive.shape[1]),
                    "slot_audit": slot_audit,
                    "predicted_mass_mean": float(predicted_mass.mean()),
                    "projection": projection,
                    **memberships,
                }
            )

    methods = (
        "frozen_aligned_pointwise",
        "spatial_slot_completion",
        "spatial_slots_learned_mass_projection",
    )
    scene_macro = {
        method: {
            key: float(np.mean([record[method][key] for record in records]))
            for key in METRIC_KEYS
        }
        for method in methods
    }
    combined_minus_frozen = {
        key: scene_macro["spatial_slots_learned_mass_projection"][key]
        - scene_macro["frozen_aligned_pointwise"][key]
        for key in METRIC_KEYS
    }
    combined_minus_slots = {
        key: scene_macro["spatial_slots_learned_mass_projection"][key]
        - scene_macro["spatial_slot_completion"][key]
        for key in METRIC_KEYS
    }
    source_path = Path(__file__).resolve()
    report = {
        "schema": REPORT_SCHEMA if mass_model is not None else ORACLE_REPORT_SCHEMA,
        "role": (
            "source_only_orthogonal_spatial_allocation_and_global_mass_diagnostic"
            if mass_model is not None
            else "target_only_oracle_mass_spatial_ranking_upper_bound"
        ),
        "mass_mode": args.mass_mode,
        "target_membership_is_model_input": mass_model is None,
        "validation_used_for_model_selection": False,
        "validation_scene_ids": sorted(validation_ids),
        "mass_blend_fraction_from_training_only_crossfit": blend_fraction,
        "projection_configuration": {
            "iteration_count": args.projection_iteration_count,
            "damping": args.projection_damping,
            "hard_threshold": False,
        },
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "base_report": {"path": str(base_report_path), "sha256": sha256_file(base_report_path)},
        "base_checkpoint": {"path": str(base_checkpoint_path), "sha256": sha256_file(base_checkpoint_path)},
        "slot_checkpoint": {"path": str(slot_checkpoint_path), "sha256": sha256_file(slot_checkpoint_path)},
        "mass_report": (
            {"path": str(mass_report_path), "sha256": sha256_file(mass_report_path)}
            if mass_report_path is not None
            else None
        ),
        "per_validation_scene": records,
        "scene_macro": scene_macro,
        "combined_minus_frozen_scene_macro": combined_minus_frozen,
        "combined_minus_slots_scene_macro": combined_minus_slots,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-cache", action="append", required=True)
    parser.add_argument("--validation-scene", action="append", required=True)
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--slot-checkpoint", required=True)
    parser.add_argument("--mass-mode", choices=("learned", "oracle"), default="learned")
    parser.add_argument("--mass-report", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unary-element-batch-size", type=int, default=4096)
    parser.add_argument("--inference-element-batch-size", type=int, default=4096)
    parser.add_argument("--projection-iteration-count", type=int, default=256)
    parser.add_argument("--projection-damping", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["combined_minus_frozen_scene_macro"], indent=2))


if __name__ == "__main__":
    main()
