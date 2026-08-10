#!/usr/bin/env python3
"""Freeze a scene0001+scene0002 source-only missing-core selector v2."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import (
    source_multiscene_monotone_missing_core_selector as multiscene_api,
)
from radio_gs.interfaces.source_monotone_missing_core_selector import (
    OOF_FOLDS,
    SELECTOR_FEATURE_NAMES,
    SOURCE_UNIT_FEATURE_INDICES,
    tie_invariant_average_precision,
)
from radio_gs.interfaces.source_multiscene_monotone_missing_core_selector import (
    MODEL_SCHEMA,
    fit_source_multiscene_monotone_selector_oof,
    packed_scene_region_groups,
    scene_region_balanced_weights,
    select_largest_multiscene_safe_oof_threshold,
    validate_multiscene_selector_model_payload,
)
from radio_gs.utils.immutable_artifacts import (
    canonical_json_sha256,
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


AUTHORITY_SCHEMA = (
    "radio_gs.source_multiscene_monotone_missing_core_selector_authority.v2"
)
REPORT_SCHEMA = "radio_gs.source_multiscene_monotone_missing_core_selector_report.v2"
SOURCE_SCENES = ("scene0001_00", "scene0002_00")


def source_access() -> dict[str, bool]:
    return {
        "scene0001_source_membership_opened": True,
        "scene0002_source_membership_opened": True,
        "scene0003_membership_opened": False,
        "scene0004_membership_opened": False,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_metrics_opened": False,
        "target_metrics_computed": False,
    }


def fixed_fit() -> dict[str, Any]:
    return {
        "model": "six_feature_monotone_additive_logistic",
        "feature_names": list(SELECTOR_FEATURE_NAMES),
        "source_unit_feature_indices": list(SOURCE_UNIT_FEATURE_INDICES),
        "feature_orientation": "larger_is_safer_after_fixed_transform",
        "normalization": "per_fold_training_median_and_1p4826_MAD",
        "training_loss": (
            "equal_scene_mass_then_equal_region_mass_within_scene_BCE_plus_L2"
        ),
        "l2_strength": 0.01,
        "maximum_LBFGS_iterations": 100,
        "positive_weight_parameterization": "softplus_nonnegative",
        "fold_assignment": "splitmix64_complete_packed_scene_region_groups",
        "fold_count": OOF_FOLDS,
        "threshold_selection": (
            "largest_tie_complete_global_OOF_population_passing_overall_and_every_scene_gates"
        ),
        "threshold_inclusive": True,
        "minimum_selected_per_scene": 256,
        "minimum_overall_hard_precision_Wilson95_lower": 0.80,
        "minimum_each_scene_hard_precision_Wilson95_lower": 0.75,
        "minimum_signed_utility_mean_exclusive": 0.0,
        "require_overall_and_each_scene_signed_utility_above_unconditional": True,
        "require_selected_hard_positive_and_negative_per_scene": True,
        "target_probability": "minimum_probability_across_three_fold_models",
        "scene_or_query_identifiers_as_features": False,
        "scene_identifier_use": "fold_routing_and_sample_balance_only",
        "instance_labels_as_features": False,
        "deep_network": False,
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact file record")
    path = validate_file_record(value, label=label)
    return {"path": str(path), "sha256": str(value["sha256"])}


def validate_execution_authority(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "source_scenes",
        "implementation",
        "selector_interface",
        "source_inputs",
        "fixed_fit",
        "outputs",
        "source_access",
        "source_validation_execution_authorized",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("source multi-scene selector authority fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 2
        or authority.get("status")
        != "sealed_after_scene0001_scene0002_source_tables_before_multiscene_fit"
        or authority.get("source_scenes") != list(SOURCE_SCENES)
        or authority.get("fixed_fit") != fixed_fit()
        or authority.get("source_access") != source_access()
        or authority.get("source_validation_execution_authorized") is not False
        or authority.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("source multi-scene selector authority header differs")
    for name in ("implementation", "selector_interface"):
        authority[name] = _record(authority[name], label=name)
    rows = authority.get("source_inputs")
    if not isinstance(rows, list) or len(rows) != len(SOURCE_SCENES):
        raise ValueError("source multi-scene selector source inputs differ")
    normalized = []
    expected_roles = ("mechanism_train", "external_source_train")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "scene_id",
            "role",
            "authority",
            "report",
            "unit_table",
        }:
            raise ValueError("source multi-scene selector input fields differ")
        if (
            row.get("scene_id") != SOURCE_SCENES[index]
            or row.get("role") != expected_roles[index]
        ):
            raise ValueError("source multi-scene selector input identity differs")
        normalized.append(
            {
                "scene_id": SOURCE_SCENES[index],
                "role": expected_roles[index],
                "authority": _record(row["authority"], label=f"source_{index}_authority"),
                "report": _record(row["report"], label=f"source_{index}_report"),
                "unit_table": _record(
                    row["unit_table"], label=f"source_{index}_unit_table"
                ),
            }
        )
    authority["source_inputs"] = normalized
    outputs = authority.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"model", "report"}:
        raise ValueError("source multi-scene selector outputs differ")
    authority["outputs"] = {
        name: str(Path(path).expanduser().resolve()) for name, path in outputs.items()
    }
    return authority


def _model_payload(model: object) -> dict[str, torch.Tensor]:
    return {
        "location": model.location,
        "scale": model.scale,
        "positive_weights": model.positive_weights,
        "bias": model.bias,
    }


def _validate_source_report(
    scene_index: int,
    report: Mapping[str, Any],
    authority_record: Mapping[str, str],
    unit_record: Mapping[str, str],
) -> None:
    if scene_index == 0:
        valid = (
            report.get("status") == "scene0001_same_axis_O0_mechanism_gate_passed"
            and report.get("execution_authority") == authority_record
            and report.get("unit_table") == unit_record
            and report.get("heldout_scene0004_membership_opened") is False
            and report.get("benchmark_execution_authorized") is False
            and report.get("target_execution_performed") is False
        )
    else:
        valid = (
            report.get("status") == "scene0002_frozen_selector_external_gate_passed"
            and report.get("execution_authority") == authority_record
            and report.get("unit_table") == unit_record
            and report.get("benchmark_execution_authorized") is False
            and report.get("target_execution_performed") is False
        )
    if not valid:
        raise ValueError(f"source scene {scene_index} report does not authorize fit")


def _load_source_unit(
    scene_index: int, row: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    report, _, _ = load_json_object(
        row["report"]["path"],
        expected_sha256=row["report"]["sha256"],
        label=f"source scene {scene_index} report",
    )
    _validate_source_report(
        scene_index, report, row["authority"], row["unit_table"]
    )
    unit, _, _ = load_torch_mapping(
        row["unit_table"]["path"],
        expected_sha256=row["unit_table"]["sha256"],
        map_location="cpu",
        label=f"source scene {scene_index} missing-core unit table",
    )
    expected_schema = (
        "radio_gs.source_same_axis_o0_mechanism_audit.v1"
        if scene_index == 0
        else "radio_gs.source_monotone_missing_core_scene0002_validation.v1"
    )
    features = torch.as_tensor(unit.get("features")).detach().float().cpu()
    labels = torch.as_tensor(unit.get("hard_labels")).detach().bool().cpu()
    utility = torch.as_tensor(unit.get("signed_utility")).detach().float().cpu()
    groups = torch.as_tensor(unit.get("unit_region_indices")).detach().long().cpu()
    if (
        unit.get("schema") != expected_schema
        or unit.get("scene_id") != SOURCE_SCENES[scene_index]
        or unit.get("execution_authority") != row["authority"]
        or unit.get("feature_names") is None
        or features.ndim != 2
        or features.shape[1] <= max(SOURCE_UNIT_FEATURE_INDICES)
        or labels.shape != (features.shape[0],)
        or utility.shape != labels.shape
        or groups.shape != labels.shape
        or features.shape[0] < fixed_fit()["minimum_selected_per_scene"]
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(utility).all())
        or int(labels.sum()) <= 0
        or int((~labels).sum()) <= 0
    ):
        raise ValueError(f"source scene {scene_index} unit-table axes differ")
    return features, labels, utility, groups


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_authority, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="source multi-scene monotone selector authority",
    )
    authority = validate_execution_authority(raw_authority)
    if authority["implementation"] != file_record(Path(__file__).resolve()):
        raise ValueError("source multi-scene selector implementation changed")
    if authority["selector_interface"] != file_record(
        Path(multiscene_api.__file__).resolve()
    ):
        raise ValueError("source multi-scene selector interface changed")
    model_output = Path(authority["outputs"]["model"])
    report_output = Path(authority["outputs"]["report"])
    if any(path.exists() or path.is_symlink() for path in (model_output, report_output)):
        raise FileExistsError("source multi-scene selector outputs must both be new")

    source_rows = authority["source_inputs"]
    loaded = [_load_source_unit(i, row) for i, row in enumerate(source_rows)]
    reference_names = None
    for row in source_rows:
        unit, _, _ = load_torch_mapping(
            row["unit_table"]["path"],
            expected_sha256=row["unit_table"]["sha256"],
            map_location="cpu",
            label="source feature-name contract",
        )
        names = list(unit["feature_names"])
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise ValueError("source unit feature names differ across scenes")

    features = torch.cat([row[0] for row in loaded], dim=0)
    labels = torch.cat([row[1] for row in loaded], dim=0)
    utility = torch.cat([row[2] for row in loaded], dim=0)
    groups = torch.cat([row[3] for row in loaded], dim=0)
    scenes = torch.cat(
        [torch.full((row[0].shape[0],), i, dtype=torch.long) for i, row in enumerate(loaded)]
    )

    fit = fit_source_multiscene_monotone_selector_oof(
        features,
        labels,
        scenes,
        groups,
        l2_strength=fixed_fit()["l2_strength"],
        maximum_iterations=fixed_fit()["maximum_LBFGS_iterations"],
    )
    threshold = select_largest_multiscene_safe_oof_threshold(
        fit.oof_probability,
        labels,
        utility,
        scenes,
        minimum_selected_per_scene=fixed_fit()["minimum_selected_per_scene"],
        minimum_overall_wilson_lower=fixed_fit()[
            "minimum_overall_hard_precision_Wilson95_lower"
        ],
        minimum_scene_wilson_lower=fixed_fit()[
            "minimum_each_scene_hard_precision_Wilson95_lower"
        ],
    )
    for scene_row in threshold["per_scene"]:
        scene_row["scene_id"] = SOURCE_SCENES[int(scene_row["scene_index"])]

    packed = packed_scene_region_groups(scenes, groups)
    fold_reports = []
    for fold, model in enumerate(fit.fold_models):
        heldout = fit.fold_ids == fold
        training = ~heldout
        train_groups = set(packed[training].tolist())
        heldout_groups = set(packed[heldout].tolist())
        overlap = len(train_groups.intersection(heldout_groups))
        training_scenes = scenes[training]
        training_regions = groups[training]
        training_packed = packed[training]
        training_balance = scene_region_balanced_weights(
            training_scenes, training_regions
        )
        scene_rows = []
        for scene_index, scene_id in enumerate(SOURCE_SCENES):
            train_scene = training & (scenes == scene_index)
            heldout_scene = heldout & (scenes == scene_index)
            local_scene = training_scenes == scene_index
            region_masses = torch.stack(
                [
                    training_balance[training_packed == group].sum()
                    for group in training_packed[local_scene].unique()
                ]
            )
            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "training_units": int(train_scene.sum()),
                    "heldout_units": int(heldout_scene.sum()),
                    "training_regions": int(groups[train_scene].unique().numel()),
                    "heldout_regions": int(groups[heldout_scene].unique().numel()),
                    "training_sample_weight": float(
                        training_balance[local_scene].sum()
                    ),
                    "minimum_training_region_sample_weight": float(
                        region_masses.min()
                    ),
                    "maximum_training_region_sample_weight": float(
                        region_masses.max()
                    ),
                }
            )
        if overlap != 0:
            raise RuntimeError("source multi-scene fold leaked a complete region")
        fold_reports.append(
            {
                "fold": fold,
                "training_units": int(training.sum()),
                "heldout_units": int(heldout.sum()),
                "packed_region_overlap": overlap,
                "scenes": scene_rows,
                "positive_weights": model.positive_weights.tolist(),
                "bias": float(model.bias),
            }
        )

    scene_metrics = []
    for scene_index, scene_id in enumerate(SOURCE_SCENES):
        mask = scenes == scene_index
        selected = fit.oof_probability[mask] >= float(
            threshold["threshold_inclusive"]
        )
        scene_metrics.append(
            {
                "scene_id": scene_id,
                "units": int(mask.sum()),
                "regions": int(groups[mask].unique().numel()),
                "hard_positive": int(labels[mask].sum()),
                "hard_negative": int((~labels[mask]).sum()),
                "OOF_average_precision": tie_invariant_average_precision(
                    fit.oof_probability[mask], labels[mask]
                ),
                "unit_O0_score_average_precision": tie_invariant_average_precision(
                    features[mask, 0], labels[mask]
                ),
                "selected": int(selected.sum()),
            }
        )

    provenance = {
        "source_scene_count": len(SOURCE_SCENES),
        "source_scene_ids_sha256": canonical_json_sha256(list(SOURCE_SCENES)),
        "scene_identifier_used_for_balancing_and_folds_only": True,
        "query_identifier_used_as_feature": False,
        "scene_identifier_used_as_feature": False,
    }
    model_payload: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "schema_version": 2,
        "feature_names": list(SELECTOR_FEATURE_NAMES),
        "source_unit_feature_indices": list(SOURCE_UNIT_FEATURE_INDICES),
        "fold_models": [_model_payload(model) for model in fit.fold_models],
        "threshold_inclusive": float(threshold["threshold_inclusive"]),
        "target_probability": fixed_fit()["target_probability"],
        "execution_authority": {"path": str(authority_path), "sha256": authority_sha},
        "training_provenance": provenance,
    }
    validate_multiscene_selector_model_payload(model_payload)
    write_torch_noclobber(model_output, model_payload)

    gate = {
        "maximum_coverage_threshold_found": True,
        "overall_Wilson95_lower_at_least_0p80": float(
            threshold["hard_precision_wilson95_lower"]
        )
        >= 0.80,
        "every_scene_Wilson95_lower_at_least_0p75": all(
            float(row["hard_precision_wilson95_lower"]) >= 0.75
            for row in threshold["per_scene"]
        ),
        "overall_signed_utility_positive_and_above_unconditional": float(
            threshold["signed_utility_mean"]
        )
        > max(0.0, float(threshold["unconditional_signed_utility_mean"])),
        "every_scene_signed_utility_positive_and_above_unconditional": all(
            float(row["signed_utility_mean"])
            > max(0.0, float(row["unconditional_signed_utility_mean"]))
            for row in threshold["per_scene"]
        ),
        "all_fold_packed_region_overlap_zero": all(
            int(row["packed_region_overlap"]) == 0 for row in fold_reports
        ),
        "query_or_scene_identifier_used_as_feature": False,
    }
    gate["passed"] = all(
        value for key, value in gate.items() if key != "query_or_scene_identifier_used_as_feature"
    ) and gate["query_or_scene_identifier_used_as_feature"] is False
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": 2,
        "status": (
            "scene0001_scene0002_multiscene_selector_v2_gate_passed"
            if gate["passed"]
            else "scene0001_scene0002_multiscene_selector_v2_gate_failed"
        ),
        "execution_authority": {"path": str(authority_path), "sha256": authority_sha},
        "model": file_record(model_output),
        "fixed_fit": fixed_fit(),
        "source_scene_metrics": scene_metrics,
        "fold_reports": fold_reports,
        "metrics": {
            "overall_OOF_average_precision": tie_invariant_average_precision(
                fit.oof_probability, labels
            ),
            "overall_unit_O0_score_average_precision": tie_invariant_average_precision(
                features[:, 0], labels
            ),
            "threshold_selection": threshold,
        },
        "gate": gate,
        "source_access": source_access(),
        "source_validation_execution_performed": False,
        "benchmark_execution_authorized": False,
        "target_execution_performed": False,
    }
    report["content_authority_sha256"] = canonical_json_sha256(report)
    write_frozen_json(report_output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    report = run(build_parser().parse_args(argv))
    print(report["status"])
    print(report["metrics"])


if __name__ == "__main__":
    main()
