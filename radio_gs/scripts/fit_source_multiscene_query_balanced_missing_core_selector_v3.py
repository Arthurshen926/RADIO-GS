#!/usr/bin/env python3
"""Fit the preregistered scene/query/region-balanced source selector v3."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import source_multiscene_query_balanced_missing_core_selector as selector
from radio_gs.interfaces.source_monotone_missing_core_selector import (
    OOF_FOLDS,
    SELECTOR_FEATURE_NAMES,
    SOURCE_UNIT_FEATURE_INDICES,
    tie_invariant_average_precision,
)
from radio_gs.scripts.audit_source_same_axis_o0_missing_core_mechanism import (
    FEATURE_NAMES,
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
    "radio_gs.source_multiscene_query_balanced_missing_core_selector_authority.v3"
)
REPORT_SCHEMA = (
    "radio_gs.source_multiscene_query_balanced_missing_core_selector_report.v3"
)
SOURCE_SCENES = ("scene0001_00", "scene0002_00")
IMPLEMENTATION = Path(__file__).resolve()
INTERFACE = Path(selector.__file__).resolve()


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
        "model": "same_six_feature_monotone_additive_logistic",
        "feature_names": list(SELECTOR_FEATURE_NAMES),
        "source_unit_feature_indices": list(SOURCE_UNIT_FEATURE_INDICES),
        "feature_orientation": "larger_is_safer_after_fixed_transform",
        "normalization": "per_fold_training_median_and_1p4826_MAD",
        "training_loss": (
            "equal_scene_then_equal_query_then_equal_region_mass_BCE_plus_L2"
        ),
        "l2_strength": 0.01,
        "maximum_LBFGS_iterations": 100,
        "positive_weight_parameterization": "softplus_nonnegative",
        "fold_assignment": "splitmix64_complete_packed_scene_region_groups",
        "fold_count": OOF_FOLDS,
        "threshold_selection": (
            "largest_tie_complete_OOF_population_passing_precision_coverage_"
            "query_macro_and_lower_tail_gates"
        ),
        "threshold_inclusive": True,
        "minimum_selected_per_scene": 256,
        "minimum_rejected_per_scene": 256,
        "maximum_selected_fraction_per_scene": 0.90,
        "minimum_overall_hard_precision_Wilson95_lower": 0.80,
        "minimum_each_scene_hard_precision_Wilson95_lower": 0.75,
        "minimum_candidate_units_per_query": 64,
        "minimum_selected_units_per_query": 16,
        "minimum_evaluable_query_fraction": 0.80,
        "minimum_evaluable_queries_per_scene": 8,
        "lower_tail_query_CVaR_fraction": 0.20,
        "require_every_evaluable_query_selected_utility_nonnegative": True,
        "require_each_scene_query_macro_selected_utility_above_unconditional": True,
        "require_each_scene_lower_tail_selected_utility_CVaR_nonnegative": True,
        "require_each_scene_lower_tail_utility_gain_CVaR_nonnegative": False,
        "diagnostic_only_each_scene_lower_tail_utility_gain_CVaR_reported": True,
        "target_probability": "minimum_probability_across_three_fold_models",
        "query_identifier_use": "balancing_and_threshold_gate_only",
        "scene_identifier_use": "sample_balance_and_fold_group_only",
        "query_or_scene_identifiers_as_features": False,
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
        "schema", "schema_version", "status", "source_scenes", "implementation",
        "selector_interface", "source_inputs", "fixed_fit", "outputs",
        "source_access", "source_validation_execution_authorized",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("v3 selector authority fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 3
        or authority.get("status")
        != "preregistered_v3b_after_strict_tail_failure_before_refit"
        or authority.get("source_scenes") != list(SOURCE_SCENES)
        or authority.get("implementation") != file_record(IMPLEMENTATION)
        or authority.get("selector_interface") != file_record(INTERFACE)
        or authority.get("fixed_fit") != fixed_fit()
        or authority.get("source_access") != source_access()
        or authority.get("source_validation_execution_authorized") is not False
        or authority.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("v3 selector authority header differs")
    rows = authority.get("source_inputs")
    roles = ("mechanism_train", "external_source_train")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("v3 selector source inputs differ")
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "scene_id", "role", "authority", "report", "unit_table"
        }:
            raise ValueError("v3 selector source input fields differ")
        if row.get("scene_id") != SOURCE_SCENES[index] or row.get("role") != roles[index]:
            raise ValueError("v3 selector source identity differs")
        normalized.append(
            {
                "scene_id": SOURCE_SCENES[index],
                "role": roles[index],
                "authority": _record(row["authority"], label=f"source{index} authority"),
                "report": _record(row["report"], label=f"source{index} report"),
                "unit_table": _record(row["unit_table"], label=f"source{index} unit"),
            }
        )
    authority["source_inputs"] = normalized
    outputs = authority.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"model", "report"}:
        raise ValueError("v3 selector outputs differ")
    authority["outputs"] = {
        name: str(Path(path).expanduser().resolve()) for name, path in outputs.items()
    }
    return authority


def _validate_source_report(
    index: int,
    report: Mapping[str, Any],
    authority: Mapping[str, str],
    unit: Mapping[str, str],
) -> None:
    expected_status = (
        "scene0001_same_axis_O0_mechanism_gate_passed"
        if index == 0
        else "scene0002_frozen_selector_external_gate_passed"
    )
    if (
        report.get("status") != expected_status
        or report.get("execution_authority") != authority
        or report.get("unit_table") != unit
        or report.get("benchmark_execution_authorized") is not False
        or report.get("target_execution_performed") is not False
        or (index == 0 and report.get("heldout_scene0004_membership_opened") is not False)
    ):
        raise ValueError(f"v3 source scene {index} report differs")


def _load_source_unit(
    index: int, row: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    report, _, _ = load_json_object(
        row["report"]["path"],
        expected_sha256=row["report"]["sha256"],
        label=f"v3 source scene {index} report",
    )
    _validate_source_report(index, report, row["authority"], row["unit_table"])
    unit, _, _ = load_torch_mapping(
        row["unit_table"]["path"],
        expected_sha256=row["unit_table"]["sha256"],
        map_location="cpu",
        label=f"v3 source scene {index} unit table",
    )
    expected_schema = (
        "radio_gs.source_same_axis_o0_mechanism_audit.v1"
        if index == 0
        else "radio_gs.source_monotone_missing_core_scene0002_validation.v1"
    )
    features = torch.as_tensor(unit.get("features")).detach().float().cpu()
    labels = torch.as_tensor(unit.get("hard_labels")).detach().bool().cpu()
    utility = torch.as_tensor(unit.get("signed_utility")).detach().float().cpu()
    regions = torch.as_tensor(unit.get("unit_region_indices")).detach().long().cpu()
    queries = torch.as_tensor(unit.get("unit_query_indices")).detach().long().cpu()
    if (
        unit.get("schema") != expected_schema
        or unit.get("scene_id") != SOURCE_SCENES[index]
        or unit.get("execution_authority") != row["authority"]
        or unit.get("feature_names") != list(FEATURE_NAMES)
        or features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or labels.shape != (features.shape[0],)
        or utility.shape != labels.shape
        or regions.shape != labels.shape
        or queries.shape != labels.shape
        or features.shape[0] < fixed_fit()["minimum_selected_per_scene"]
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(utility).all())
        or bool((regions < 0).any())
        or bool((queries < 0).any())
        or int(labels.sum()) <= 0
        or int((~labels).sum()) <= 0
    ):
        raise ValueError(f"v3 source scene {index} unit axes differ")
    return features, labels, utility, regions, queries


def _model_payload(model: object) -> dict[str, torch.Tensor]:
    return {
        "location": model.location,
        "scale": model.scale,
        "positive_weights": model.positive_weights,
        "bias": model.bias,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="v3 selector execution authority",
    )
    authority = validate_execution_authority(raw)
    model_output = Path(authority["outputs"]["model"])
    report_output = Path(authority["outputs"]["report"])
    if any(path.exists() or path.is_symlink() for path in (model_output, report_output)):
        raise FileExistsError("v3 selector outputs must both be new")
    loaded = [_load_source_unit(i, row) for i, row in enumerate(authority["source_inputs"])]
    features = torch.cat([row[0] for row in loaded])
    labels = torch.cat([row[1] for row in loaded])
    utility = torch.cat([row[2] for row in loaded])
    regions = torch.cat([row[3] for row in loaded])
    queries = torch.cat([row[4] for row in loaded])
    scenes = torch.cat(
        [torch.full((row[0].shape[0],), index, dtype=torch.long) for index, row in enumerate(loaded)]
    )
    fit = selector.fit_source_multiscene_query_balanced_selector_oof(
        features,
        labels,
        scenes,
        queries,
        regions,
        l2_strength=fixed_fit()["l2_strength"],
        maximum_iterations=fixed_fit()["maximum_LBFGS_iterations"],
    )
    threshold = selector.select_largest_query_safe_oof_threshold(
        fit.oof_probability,
        labels,
        utility,
        scenes,
        queries,
        minimum_selected_per_scene=fixed_fit()["minimum_selected_per_scene"],
        minimum_rejected_per_scene=fixed_fit()["minimum_rejected_per_scene"],
        maximum_selected_fraction_per_scene=fixed_fit()["maximum_selected_fraction_per_scene"],
        minimum_overall_wilson_lower=fixed_fit()["minimum_overall_hard_precision_Wilson95_lower"],
        minimum_scene_wilson_lower=fixed_fit()["minimum_each_scene_hard_precision_Wilson95_lower"],
        minimum_candidate_units_per_query=fixed_fit()["minimum_candidate_units_per_query"],
        minimum_selected_units_per_query=fixed_fit()["minimum_selected_units_per_query"],
        minimum_evaluable_query_fraction=fixed_fit()["minimum_evaluable_query_fraction"],
        minimum_evaluable_queries=fixed_fit()["minimum_evaluable_queries_per_scene"],
        lower_tail_fraction=fixed_fit()["lower_tail_query_CVaR_fraction"],
        require_lower_tail_utility_gain_cvar=fixed_fit()[
            "require_each_scene_lower_tail_utility_gain_CVaR_nonnegative"
        ],
    )
    for row in threshold["per_scene"]:
        row["scene_id"] = SOURCE_SCENES[int(row["scene_index"])]
    packed = selector.packed_scene_region_groups(scenes, regions)
    fold_reports = []
    for fold, model in enumerate(fit.fold_models):
        heldout = fit.fold_ids == fold
        training = ~heldout
        overlap = len(set(packed[training].tolist()) & set(packed[heldout].tolist()))
        weights = selector.scene_query_region_balanced_weights(
            scenes[training], queries[training], regions[training]
        )
        scene_rows = []
        for scene_index, scene_id in enumerate(SOURCE_SCENES):
            local = scenes[training] == scene_index
            query_masses = []
            for query in torch.unique(queries[training][local], sorted=True):
                query_masses.append(float(weights[local & (queries[training] == query)].sum()))
            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "training_units": int((training & (scenes == scene_index)).sum()),
                    "heldout_units": int((heldout & (scenes == scene_index)).sum()),
                    "training_queries": len(query_masses),
                    "minimum_training_query_mass": min(query_masses),
                    "maximum_training_query_mass": max(query_masses),
                    "training_sample_weight": float(weights[local].sum()),
                }
            )
        if overlap:
            raise RuntimeError("v3 fold leaked a complete physical scene/region group")
        fold_reports.append(
            {
                "fold": fold,
                "training_units": int(training.sum()),
                "heldout_units": int(heldout.sum()),
                "packed_scene_region_overlap": overlap,
                "scenes": scene_rows,
                "positive_weights": model.positive_weights.tolist(),
                "bias": float(model.bias),
            }
        )
    provenance = {
        "source_scene_count": 2,
        "scene_identifier_used_for_balancing_and_groups_only": True,
        "query_identifier_used_for_balancing_and_threshold_gate_only": True,
        "query_identifier_used_as_feature": False,
        "scene_identifier_used_as_feature": False,
    }
    model_payload = {
        "schema": selector.MODEL_SCHEMA,
        "schema_version": 3,
        "feature_names": list(SELECTOR_FEATURE_NAMES),
        "source_unit_feature_indices": list(SOURCE_UNIT_FEATURE_INDICES),
        "fold_models": [_model_payload(model) for model in fit.fold_models],
        "threshold_inclusive": float(threshold["threshold_inclusive"]),
        "target_probability": fixed_fit()["target_probability"],
        "execution_authority": {"path": str(authority_path), "sha256": authority_sha},
        "training_provenance": provenance,
    }
    selector.validate_query_balanced_selector_model_payload(model_payload)
    write_torch_noclobber(model_output, model_payload)
    scene_metrics = []
    for scene_index, scene_id in enumerate(SOURCE_SCENES):
        mask = scenes == scene_index
        selected = fit.oof_probability[mask] >= float(threshold["threshold_inclusive"])
        scene_metrics.append(
            {
                "scene_id": scene_id,
                "units": int(mask.sum()),
                "queries": int(torch.unique(queries[mask]).numel()),
                "regions": int(torch.unique(regions[mask]).numel()),
                "selected": int(selected.sum()),
                "coverage_fraction": float(selected.float().mean()),
                "OOF_average_precision": tie_invariant_average_precision(
                    fit.oof_probability[mask], labels[mask]
                ),
                "unit_O0_score_average_precision": tie_invariant_average_precision(
                    features[mask, 0], labels[mask]
                ),
            }
        )
    gate = {
        "maximum_coverage_query_safe_threshold_found": True,
        "every_scene_coverage_at_most_0p90": all(
            float(row["coverage_fraction"]) <= 0.90 for row in threshold["per_scene"]
        ),
        "every_scene_query_macro_utility_above_unconditional": all(
            row["query_gate"]["query_macro_selected_utility_strictly_above_unconditional"]
            for row in threshold["per_scene"]
        ),
        "every_scene_query_lower_tail_gates_passed": all(
            row["query_gate"]["passed"] for row in threshold["per_scene"]
        ),
        "all_fold_physical_scene_region_overlap_zero": all(
            row["packed_scene_region_overlap"] == 0 for row in fold_reports
        ),
        "query_or_scene_identifier_used_as_feature": False,
    }
    gate["passed"] = all(
        value for key, value in gate.items()
        if key != "query_or_scene_identifier_used_as_feature"
    ) and gate["query_or_scene_identifier_used_as_feature"] is False
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": 3,
        "status": (
            "scene0001_scene0002_query_balanced_selector_v3b_gate_passed"
            if gate["passed"]
            else "scene0001_scene0002_query_balanced_selector_v3b_gate_failed"
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


__all__ = [
    "AUTHORITY_SCHEMA", "REPORT_SCHEMA", "fixed_fit", "run", "source_access",
    "validate_execution_authority",
]
