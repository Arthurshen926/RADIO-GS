#!/usr/bin/env python3
"""Freeze the scene0001 monotone missing-core selector before heldout access."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import source_monotone_missing_core_selector as selector_api
from radio_gs.interfaces.source_monotone_missing_core_selector import (
    OOF_FOLDS,
    SELECTOR_FEATURE_NAMES,
    SOURCE_UNIT_FEATURE_INDICES,
    fit_source_monotone_selector_oof,
    select_largest_safe_oof_threshold,
    tie_invariant_average_precision,
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


AUTHORITY_SCHEMA = "radio_gs.source_monotone_missing_core_selector_authority.v1"
RESULT_SCHEMA = "radio_gs.source_monotone_missing_core_selector.v1"


def source_access() -> dict[str, bool]:
    return {
        "source_train_instance_labels_opened": True,
        "source_validation_instance_labels_opened": False,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_metrics_opened": False,
        "target_heldout_opened": False,
        "target_metrics_computed": False,
    }


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact file record")
    path = validate_file_record(value, label=label)
    return {"path": str(path), "sha256": str(value["sha256"])}


def fixed_fit() -> dict[str, Any]:
    return {
        "model": "six_feature_monotone_additive_logistic",
        "feature_names": list(SELECTOR_FEATURE_NAMES),
        "source_unit_feature_indices": list(SOURCE_UNIT_FEATURE_INDICES),
        "feature_orientation": "larger_is_safer_after_fixed_transform",
        "normalization": "per_fold_training_median_and_1p4826_MAD",
        "training_loss": "region_group_balanced_binary_cross_entropy_plus_L2",
        "l2_strength": 0.01,
        "maximum_LBFGS_iterations": 100,
        "positive_weight_parameterization": "softplus_nonnegative",
        "fold_assignment": "splitmix64_complete_region_groups",
        "fold_count": OOF_FOLDS,
        "threshold_selection": (
            "largest_tie_complete_OOF_population_satisfying_all_safety_gates"
        ),
        "threshold_inclusive": True,
        "minimum_selected": 256,
        "minimum_hard_precision_Wilson95_lower": 0.80,
        "minimum_signed_utility_mean_exclusive": 0.0,
        "require_selected_hard_positive_and_negative": True,
        "require_OOF_AP_strictly_above_unit_O0_score_AP": True,
        "target_probability": "minimum_probability_across_three_fold_models",
        "scene_or_query_identifiers_as_features": False,
        "instance_labels_as_features": False,
        "deep_network": False,
    }


def validate_execution_authority(value: object) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "split",
        "implementation",
        "selector_interface",
        "mechanism_audit_authority",
        "mechanism_audit_report",
        "unit_table",
        "fixed_fit",
        "outputs",
        "source_access",
        "heldout_scene_execution_authorized",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("source monotone selector authority fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "sealed_after_scene0001_mechanism_gate_before_selector_fit"
        or authority.get("scene_id") != "scene0001_00"
        or authority.get("split") != "source_train"
        or authority.get("fixed_fit") != fixed_fit()
        or authority.get("source_access") != source_access()
        or authority.get("heldout_scene_execution_authorized") is not False
        or authority.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("source monotone selector authority header differs")
    for name in (
        "implementation",
        "selector_interface",
        "mechanism_audit_authority",
        "mechanism_audit_report",
        "unit_table",
    ):
        authority[name] = _record(authority[name], label=name)
    outputs = authority.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"model", "report"}:
        raise ValueError("source monotone selector outputs differ")
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_authority, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="source monotone missing-core selector authority",
    )
    authority = validate_execution_authority(raw_authority)
    if authority["implementation"] != file_record(Path(__file__).resolve()):
        raise ValueError("source monotone selector implementation changed")
    if authority["selector_interface"] != file_record(
        Path(selector_api.__file__).resolve()
    ):
        raise ValueError("source monotone selector interface changed")
    model_output = Path(authority["outputs"]["model"])
    report_output = Path(authority["outputs"]["report"])
    if any(path.exists() or path.is_symlink() for path in (model_output, report_output)):
        raise FileExistsError("source monotone selector outputs must both be new")
    audit_report, _, _ = load_json_object(
        authority["mechanism_audit_report"]["path"],
        expected_sha256=authority["mechanism_audit_report"]["sha256"],
        label="scene0001 mechanism audit report",
    )
    if (
        audit_report.get("status")
        != "scene0001_same_axis_O0_mechanism_gate_passed"
        or audit_report.get("heldout_scene0004_membership_opened") is not False
        or audit_report.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("scene0001 mechanism audit does not authorize selector fit")
    unit, _, _ = load_torch_mapping(
        authority["unit_table"]["path"],
        expected_sha256=authority["unit_table"]["sha256"],
        map_location="cpu",
        label="scene0001 missing-core unit table",
    )
    features = torch.as_tensor(unit.get("features")).detach().float().cpu()
    labels = torch.as_tensor(unit.get("hard_labels")).detach().bool().cpu()
    utility = torch.as_tensor(unit.get("signed_utility")).detach().float().cpu()
    groups = torch.as_tensor(unit.get("unit_region_indices")).detach().long().cpu()
    if (
        unit.get("feature_names") is None
        or features.ndim != 2
        or features.shape[1] <= max(SOURCE_UNIT_FEATURE_INDICES)
        or labels.shape != (features.shape[0],)
        or utility.shape != labels.shape
        or groups.shape != labels.shape
        or features.shape[0] < fixed_fit()["minimum_selected"]
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(utility).all())
    ):
        raise ValueError("scene0001 missing-core unit-table axes differ")
    fit = fit_source_monotone_selector_oof(
        features,
        labels,
        groups,
        l2_strength=fixed_fit()["l2_strength"],
        maximum_iterations=fixed_fit()["maximum_LBFGS_iterations"],
    )
    threshold = select_largest_safe_oof_threshold(
        fit.oof_probability,
        labels,
        utility,
        minimum_selected=fixed_fit()["minimum_selected"],
        minimum_wilson_lower=fixed_fit()[
            "minimum_hard_precision_Wilson95_lower"
        ],
    )
    oof_ap = tie_invariant_average_precision(fit.oof_probability, labels)
    o0_ap = tie_invariant_average_precision(features[:, 0], labels)
    selected = fit.oof_probability >= float(threshold["threshold_inclusive"])
    gate = {
        "selected_at_least_256": int(selected.sum()) >= 256,
        "selected_hard_positive_and_negative_both_evaluated": bool(
            labels[selected].any() and (~labels[selected]).any()
        ),
        "hard_precision_Wilson95_lower_at_least_0p80": float(
            threshold["hard_precision_wilson95_lower"]
        )
        >= 0.80,
        "signed_utility_mean_positive": float(threshold["signed_utility_mean"])
        > 0.0,
        "OOF_AP_strictly_above_unit_O0_score_AP": oof_ap > o0_ap,
    }
    gate["passed"] = all(gate.values())
    model_payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "scene_id": "scene0001_00",
        "feature_names": list(SELECTOR_FEATURE_NAMES),
        "source_unit_feature_indices": list(SOURCE_UNIT_FEATURE_INDICES),
        "fold_models": [_model_payload(model) for model in fit.fold_models],
        "fold_ids": fit.fold_ids,
        "oof_probability": fit.oof_probability,
        "threshold_inclusive": float(threshold["threshold_inclusive"]),
        "target_probability": fixed_fit()["target_probability"],
        "execution_authority": {
            "path": str(authority_path),
            "sha256": authority_sha,
        },
    }
    model_payload["channel_sha256"] = {
        "fold_ids": tensor_sha256(fit.fold_ids),
        "oof_probability": tensor_sha256(fit.oof_probability),
    }
    write_torch_noclobber(model_output, model_payload)
    fold_reports = []
    for fold, model in enumerate(fit.fold_models):
        heldout = fit.fold_ids == fold
        fold_reports.append(
            {
                "fold": fold,
                "training_units": int((~heldout).sum()),
                "heldout_units": int(heldout.sum()),
                "training_regions": int(groups[~heldout].unique().numel()),
                "heldout_regions": int(groups[heldout].unique().numel()),
                "heldout_positive": int(labels[heldout].sum()),
                "heldout_negative": int((~labels[heldout]).sum()),
                "positive_weights": model.positive_weights.tolist(),
                "bias": float(model.bias),
            }
        )
    report = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "status": (
            "scene0001_monotone_selector_gate_passed"
            if gate["passed"]
            else "scene0001_monotone_selector_gate_failed"
        ),
        "execution_authority": {
            "path": str(authority_path),
            "sha256": authority_sha,
        },
        "model": file_record(model_output),
        "fixed_fit": fixed_fit(),
        "fold_reports": fold_reports,
        "metrics": {
            "OOF_average_precision": oof_ap,
            "unit_O0_score_average_precision": o0_ap,
            "OOF_AP_gain_over_unit_O0_score": oof_ap - o0_ap,
            "threshold_selection": threshold,
        },
        "gate": gate,
        "heldout_scene0004_membership_opened": False,
        "source_access": source_access(),
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
