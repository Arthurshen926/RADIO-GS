#!/usr/bin/env python3
"""Materialize a no-GT source-frozen conditional FIX6b score cache."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from radio_gs.interfaces import (
    lerf_o0_conditional_missing_core_completion as formal,
)
from radio_gs.interfaces.factorized_primitive_state import (
    load_factorized_primitive_state,
)
from radio_gs.interfaces.source_monotone_missing_core_selector import (
    MonotoneAdditiveLogistic,
    SELECTOR_FEATURE_NAMES,
    SOURCE_UNIT_FEATURE_INDICES,
)
from radio_gs.interfaces.surface_region_full_scalar_contract import (
    aggregate_surface_region_full_scalars,
)
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.scripts import (
    audit_source_same_axis_o0_missing_core_mechanism as source_feature_api,
)
from radio_gs.scripts.materialize_region_capability_descriptors_v2 import (
    validate_region_capability_descriptor_authority,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


AUTHORITY_SCHEMA = (
    "radio_gs.lerf_o0_conditional_missing_core_completion_execution.v1"
)
REPORT_SCHEMA = "radio_gs.lerf_o0_conditional_missing_core_completion_report.v1"
IMPLEMENTATION = Path(__file__).resolve()
INTERFACE = Path(formal.__file__).resolve()
FROZEN_THRESHOLD = 0.5480217337608337


def _record(path: object, digest: object, *, label: str) -> dict[str, str]:
    raw = str(path)
    canonical = str(Path(raw).expanduser().resolve())
    value = {"path": canonical, "sha256": str(digest)}
    if raw != canonical:
        raise ValueError(f"{label} path must be canonical absolute")
    validate_file_record(value, label=label)
    return value


def _new_path(value: object, *, label: str) -> Path:
    raw = str(value)
    path = Path(raw).expanduser().resolve()
    if raw != str(path) or path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be a new canonical path")
    return path


def _build_access() -> dict[str, bool]:
    return {
        "source_train_and_heldout_PASS_records_opened": True,
        "input_target_file_records_validated": True,
        "target_payloads_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
    }


def _execute_access() -> dict[str, bool]:
    return {
        "source_train_and_heldout_PASS_records_opened": True,
        "exact_O0_and_query_independent_target_geometry_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "threshold_scan_or_refit": False,
    }


def fixed_no_gt_gates() -> dict[str, Any]:
    return {
        "candidate_units_nonempty": True,
        "selected_and_rejected_units_both_nonempty": True,
        "selected_unique_cells_strict_subset_of_unconditional_FIX6_cells": True,
        "changed_cells_exactly_equal_selected_unique_cells": True,
        "outside_selected_cells_bitwise_exact_O0": True,
        "invalid_primitives_bitwise_exact_O0": True,
        "pointwise_non_decreasing": True,
        "maximum_per_query_membership_expansion": formal.MAXIMUM_MEMBERSHIP_EXPANSION,
        "all_target_axes_and_lineage_exact": True,
    }


def _load_json(record: Mapping[str, str], *, label: str) -> dict[str, Any]:
    value, digest, source = load_json_object(
        record["path"], expected_sha256=record["sha256"], label=label
    )
    if {"path": str(source), "sha256": digest} != dict(record):
        raise ValueError(f"{label} record differs")
    return dict(value)


def _load_torch(record: Mapping[str, str], *, label: str) -> dict[str, Any]:
    value, digest, source = load_torch_mapping(
        record["path"],
        expected_sha256=record["sha256"],
        map_location="cpu",
        label=label,
    )
    if {"path": str(source), "sha256": digest} != dict(record):
        raise ValueError(f"{label} record differs")
    return dict(value)


def _validate_source_gate(inputs: Mapping[str, Mapping[str, str]]) -> float:
    source_authority = _load_json(
        inputs["source_selector_authority"], label="source selector authority"
    )
    source_report = _load_json(
        inputs["source_selector_report"], label="source selector report"
    )
    heldout_authority = _load_json(
        inputs["heldout_selector_authority"], label="heldout selector authority"
    )
    heldout_report = _load_json(
        inputs["heldout_selector_report"], label="heldout selector report"
    )
    model = _load_torch(inputs["source_selector_model"], label="source selector model")
    if (
        source_authority.get("status")
        != "sealed_after_scene0001_mechanism_gate_before_selector_fit"
        or source_authority.get("outputs", {}).get("model")
        != inputs["source_selector_model"]["path"]
        or source_authority.get("outputs", {}).get("report")
        != inputs["source_selector_report"]["path"]
        or source_report.get("status") != "scene0001_monotone_selector_gate_passed"
        or source_report.get("execution_authority")
        != inputs["source_selector_authority"]
        or source_report.get("model") != inputs["source_selector_model"]
        or model.get("schema") != "radio_gs.source_monotone_missing_core_selector.v1"
        or model.get("execution_authority") != inputs["source_selector_authority"]
        or model.get("feature_names") != list(SELECTOR_FEATURE_NAMES)
        or model.get("source_unit_feature_indices")
        != list(SOURCE_UNIT_FEATURE_INDICES)
        or model.get("target_probability")
        != "minimum_probability_across_three_fold_models"
        or float(model.get("threshold_inclusive", float("nan")))
        != FROZEN_THRESHOLD
        or heldout_authority.get("status")
        != "sealed_after_scene0004_raw_O0_before_membership_open"
        or heldout_authority.get("frozen_selector_model")
        != inputs["source_selector_model"]
        or heldout_authority.get("frozen_selector_report")
        != inputs["source_selector_report"]
        or heldout_authority.get("outputs", {}).get("report")
        != inputs["heldout_selector_report"]["path"]
        or heldout_authority.get("outputs", {}).get("unit_table")
        != inputs["heldout_selector_unit_table"]["path"]
        or heldout_report.get("status")
        != "scene0004_frozen_selector_heldout_gate_passed"
        or heldout_report.get("execution_authority")
        != inputs["heldout_selector_authority"]
        or heldout_report.get("unit_table") != inputs["heldout_selector_unit_table"]
        or float(heldout_report.get("frozen_threshold_inclusive", float("nan")))
        != FROZEN_THRESHOLD
        or heldout_report.get("selector_gate", {}).get("checks", {}).get("passed")
        is not True
        or heldout_report.get("target_execution_performed") is not False
    ):
        raise ValueError("source-frozen conditional selector gate differs")
    return FROZEN_THRESHOLD


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = _new_path(args.output_authority, label="execution authority")
    output_cache = _new_path(args.output_cache, label="external query cache")
    output_report = _new_path(args.output_report, label="no-GT report")
    inputs: dict[str, dict[str, str]] = {}
    for name in (
        "exact_o0_cache",
        "target_accepted_v2",
        "target_capability_descriptor",
        "factorized_primitive_state",
        "source_selector_authority",
        "source_selector_model",
        "source_selector_report",
        "heldout_selector_authority",
        "heldout_selector_report",
        "heldout_selector_unit_table",
    ):
        inputs[name] = _record(
            getattr(args, name),
            getattr(args, "expected_" + name + "_sha256"),
            label=name,
        )
    threshold = _validate_source_gate(inputs)
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": 1,
        "status": "authorized_after_strict_source_heldout_PASS_for_no_GT_cache_only",
        "scene_id": str(args.scene_id),
        "implementation": file_record(IMPLEMENTATION),
        "interface": file_record(INTERFACE),
        "source_feature_implementation": file_record(
            Path(source_feature_api.__file__).resolve()
        ),
        "contract": formal.completion_contract(),
        "contract_sha256": formal.CONTRACT_SHA256,
        "frozen_threshold_inclusive": threshold,
        "fixed_no_GT_gates": fixed_no_gt_gates(),
        "input_authority": inputs,
        "outputs": {"cache": str(output_cache), "report": str(output_report)},
        "target_score_cache_authorized": True,
        "target_metric_execution_authorized": False,
        "access_audit": _build_access(),
    }
    write_frozen_json(output, authority)
    return {
        "status": "authority_built_after_source_heldout_PASS_without_target_payload_open",
        "authority": file_record(output),
    }


def validate_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("conditional completion authority must be a mapping")
    authority = dict(value)
    required = {
        "schema",
        "schema_version",
        "status",
        "scene_id",
        "implementation",
        "interface",
        "source_feature_implementation",
        "contract",
        "contract_sha256",
        "frozen_threshold_inclusive",
        "fixed_no_GT_gates",
        "input_authority",
        "outputs",
        "target_score_cache_authorized",
        "target_metric_execution_authorized",
        "access_audit",
    }
    if (
        set(authority) != required
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 1
        or authority.get("status")
        != "authorized_after_strict_source_heldout_PASS_for_no_GT_cache_only"
        or authority.get("implementation") != file_record(IMPLEMENTATION)
        or authority.get("interface") != file_record(INTERFACE)
        or authority.get("source_feature_implementation")
        != file_record(Path(source_feature_api.__file__).resolve())
        or authority.get("contract") != formal.completion_contract()
        or authority.get("contract_sha256") != formal.CONTRACT_SHA256
        or authority.get("frozen_threshold_inclusive") != FROZEN_THRESHOLD
        or authority.get("fixed_no_GT_gates") != fixed_no_gt_gates()
        or authority.get("target_score_cache_authorized") is not True
        or authority.get("target_metric_execution_authorized") is not False
        or authority.get("access_audit") != _build_access()
    ):
        raise ValueError("conditional completion authority header differs")
    expected_inputs = {
        "exact_o0_cache",
        "target_accepted_v2",
        "target_capability_descriptor",
        "factorized_primitive_state",
        "source_selector_authority",
        "source_selector_model",
        "source_selector_report",
        "heldout_selector_authority",
        "heldout_selector_report",
        "heldout_selector_unit_table",
    }
    if set(authority.get("input_authority", {})) != expected_inputs:
        raise ValueError("conditional completion input set differs")
    for name, record in authority["input_authority"].items():
        validate_file_record(record, label=name)
    if set(authority.get("outputs", {})) != {"cache", "report"}:
        raise ValueError("conditional completion outputs differ")
    return authority


def _fold_models(raw: Mapping[str, Any]) -> tuple[MonotoneAdditiveLogistic, ...]:
    rows = raw.get("fold_models")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("conditional selector fold-model axis differs")
    result = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "location",
            "scale",
            "positive_weights",
            "bias",
        }:
            raise ValueError("conditional selector fold-model fields differ")
        result.append(
            MonotoneAdditiveLogistic(
                location=torch.as_tensor(row["location"]).float().cpu(),
                scale=torch.as_tensor(row["scale"]).float().cpu(),
                positive_weights=torch.as_tensor(row["positive_weights"])
                .float()
                .cpu(),
                bias=torch.as_tensor(row["bias"]).float().cpu().reshape(()),
            )
        )
    return tuple(result)


def _spatial_radius(rows: torch.Tensor, mask: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(rows.shape[0], dtype=torch.float32)
    for region in range(rows.shape[0]):
        cloud = xyz[rows[region, mask[region]]]
        center = cloud.mean(dim=0, keepdim=True)
        result[region] = ((cloud - center).square().sum(1).mean()).sqrt()
    return result.contiguous()


def execute(args: argparse.Namespace) -> dict[str, Any]:
    raw, digest, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="conditional completion execution authority",
    )
    authority = validate_authority(raw)
    authority["verified_record"] = {
        "path": str(authority_path),
        "sha256": digest,
    }
    cache_path = Path(authority["outputs"]["cache"])
    report_path = Path(authority["outputs"]["report"])
    if any(path.exists() or path.is_symlink() for path in (cache_path, report_path)):
        raise FileExistsError("conditional completion outputs must both be new")
    inputs = authority["input_authority"]
    threshold = _validate_source_gate(inputs)
    model = _load_torch(inputs["source_selector_model"], label="source selector model")
    o0 = _load_torch(inputs["exact_o0_cache"], label="exact O0 cache")
    accepted = validate_target_accepted_v2_authority(
        _load_torch(inputs["target_accepted_v2"], label="target AcceptedV2")
    )
    capability = validate_region_capability_descriptor_authority(
        _load_torch(
            inputs["target_capability_descriptor"],
            label="target capability descriptor",
        )
    )
    state_record = inputs["factorized_primitive_state"]
    state = load_factorized_primitive_state(
        state_record["path"], expected_sha256=state_record["sha256"]
    )
    scores = torch.as_tensor(o0.get("query_scores")).detach().float().cpu()
    valid = torch.as_tensor(o0.get("valid")).detach().bool().cpu()
    xyz = torch.as_tensor(o0.get("xyz")).detach().float().cpu()
    selection = o0.get("selection", {})
    names = [str(value) for value in o0.get("metadata", {}).get("query_names", ())]
    selected_scales = torch.as_tensor(selection.get("selected_scale_indices"))
    accepted_record = capability.get("input_authority", {}).get("accepted_v2")
    accepted_geometry = accepted["input_authority"]["geometry_authority"]
    if (
        o0.get("schema")
        != "radio_gs.lerf_o0_anchored_graph_residual_external_scores.v1"
        or o0.get("metadata", {}).get("canonical_capability")
        != "exact_frozen_O0_canonical_negative_VALA_peak_scale"
        or any(len(value) for value in selection.get("selected_region_rows", ()))
        or scores.ndim != 2
        or valid.shape != (scores.shape[0],)
        or xyz.shape != (scores.shape[0], 3)
        or len(names) != scores.shape[1]
        or selected_scales.dtype not in {torch.int32, torch.int64}
        or selected_scales.shape != (scores.shape[1],)
        or accepted["scene_id"] != authority["scene_id"]
        or capability["scene_id"] != authority["scene_id"]
        or accepted_record != inputs["target_accepted_v2"]
        or not torch.equal(capability["region_rows"], accepted["region_rows"])
        or not torch.equal(capability["token_mask"], accepted["token_mask"])
        or capability["region_fingerprints"] != accepted["region_fingerprints"]
        or not torch.equal(
            capability["canonical_region_indices"],
            accepted["canonical_region_indices"],
        )
        or accepted_geometry["factorized_primitive_state_file_sha256"]
        != state_record["sha256"]
        or state.xyz.shape != xyz.shape
        or not torch.equal(state.xyz.float().cpu(), xyz)
    ):
        raise ValueError("conditional completion target lineage differs")
    summary = aggregate_surface_region_full_scalars(
        state,
        accepted["accepted_base_valid"],
        accepted["region_rows"],
        accepted["token_mask"],
        accepted["anchor_index"],
    )
    _, scalar_ood = source_feature_api._source_robust_normalize(
        summary.summary, summary.use_full_scalar_mask
    )
    radius = _spatial_radius(accepted["region_rows"], accepted["token_mask"], xyz)
    result = formal.conditional_missing_core_completion(
        o0_scores=scores,
        region_rows=accepted["region_rows"],
        core_mask=accepted["token_mask"],
        primitive_valid_mask=valid,
        appearance_concentration=capability["appearance_concentration"],
        boundary_concentration=capability["boundary_concentration"],
        core_spatial_rms_radius=radius,
        selected_query_scale_indices=selected_scales,
        full_scalar_source_robust_ood_linf=scalar_ood,
        fold_models=_fold_models(model),
        threshold_inclusive=threshold,
    )
    o0_membership = (scores > formal.O0_SCORE_MINIMUM) & valid[:, None]
    final_membership = (result.final_scores > formal.O0_SCORE_MINIMUM) & valid[:, None]
    o0_count = o0_membership.sum(0)
    final_count = final_membership.sum(0)
    expansion = final_count.float() / o0_count.clamp_min(1).float()
    candidate_units = int(result.unit_region_indices.numel())
    selected_units = int(result.selected_unit_mask.sum())
    rejected_units = candidate_units - selected_units
    selected_cells = int(result.selected_cell_mask.sum())
    unconditional_cells = int(result.unconditional_cell_mask.sum())
    checks = {
        "candidate_units_nonempty": candidate_units > 0,
        "selected_and_rejected_units_both_nonempty": selected_units > 0
        and rejected_units > 0,
        "selected_unique_cells_strict_subset_of_unconditional_FIX6_cells": (
            0 < selected_cells < unconditional_cells
        ),
        "changed_cells_exactly_equal_selected_unique_cells": torch.equal(
            result.changed_mask, result.selected_cell_mask
        ),
        "outside_selected_cells_bitwise_exact_O0": torch.equal(
            result.final_scores[~result.selected_cell_mask],
            scores[~result.selected_cell_mask],
        ),
        "invalid_primitives_bitwise_exact_O0": torch.equal(
            result.final_scores[~valid], scores[~valid]
        ),
        "pointwise_non_decreasing": bool((result.final_scores >= scores).all()),
        "maximum_per_query_membership_expansion": float(expansion.max())
        <= formal.MAXIMUM_MEMBERSHIP_EXPANSION,
        "all_target_axes_and_lineage_exact": True,
    }
    checks["passed"] = bool(all(checks.values()))
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": 1,
        "status": (
            "conditional_missing_core_no_GT_gate_passed"
            if checks["passed"]
            else "conditional_missing_core_no_GT_gate_failed"
        ),
        "execution_authority": authority["verified_record"],
        "scene_id": authority["scene_id"],
        "frozen_threshold_inclusive": threshold,
        "source_heldout_status": "scene0004_frozen_selector_heldout_gate_passed",
        "candidate_units": candidate_units,
        "selected_units": selected_units,
        "rejected_units": rejected_units,
        "qualified_anchor_region_query_pairs": int(
            result.qualified_anchor_mask.sum()
        ),
        "unconditional_FIX6_unique_cells": unconditional_cells,
        "selected_unique_cells": selected_cells,
        "strictly_changed_cells": int(result.changed_mask.sum()),
        "threshold_membership_flips": int((~o0_membership & final_membership).sum()),
        "per_query": {
            "query_names": names,
            "O0_membership_counts": o0_count.long().tolist(),
            "final_membership_counts": final_count.long().tolist(),
            "membership_expansion": expansion.tolist(),
            "selected_unit_counts": torch.bincount(
                result.unit_query_indices[result.selected_unit_mask],
                minlength=scores.shape[1],
            )
            .long()
            .tolist(),
        },
        "no_GT_safety_gate": checks,
        "access_audit": _execute_access(),
        "target_metric_execution_authorized": False,
    }
    if not checks["passed"]:
        write_frozen_json(report_path, report)
        raise RuntimeError("conditional missing-core no-GT safety gate failed")
    external = formal.build_external_query_score_cache(
        result=result,
        o0_valid=valid,
        o0_xyz=xyz,
        query_names=names,
        scene_id=authority["scene_id"],
        input_authority=inputs,
        threshold_inclusive=threshold,
    )
    write_torch_noclobber(cache_path, external)
    report["output_cache"] = file_record(cache_path)
    write_frozen_json(report_path, report)
    return {
        "status": report["status"],
        "cache": file_record(cache_path),
        "report": file_record(report_path),
        "selected_units": selected_units,
        "rejected_units": rejected_units,
        "threshold_membership_flips": report["threshold_membership_flips"],
        "maximum_membership_expansion": float(expansion.max()),
    }


def _add_record(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument(
        "--expected-" + name.replace("_", "-") + "-sha256", required=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    for name in (
        "exact_o0_cache",
        "target_accepted_v2",
        "target_capability_descriptor",
        "factorized_primitive_state",
        "source_selector_authority",
        "source_selector_model",
        "source_selector_report",
        "heldout_selector_authority",
        "heldout_selector_report",
        "heldout_selector_unit_table",
    ):
        _add_record(build, name)
    build.add_argument("--scene-id", required=True)
    build.add_argument("--output-cache", required=True)
    build.add_argument("--output-report", required=True)
    build.add_argument("--output-authority", required=True)
    build.set_defaults(handler=build_authority)
    run = commands.add_parser("execute")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--expected-execution-authority-sha256", required=True)
    run.set_defaults(handler=execute)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHORITY_SCHEMA",
    "FROZEN_THRESHOLD",
    "REPORT_SCHEMA",
    "build_authority",
    "execute",
    "fixed_no_gt_gates",
    "validate_authority",
]
