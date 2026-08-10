#!/usr/bin/env python3
"""Materialize target-blind FIX6c scores after the full source v2 gate.

This wrapper keeps the FIX6b completion operator unchanged.  It replaces the
old single-scene selector lineage with the two-scene v2 model and requires a
strict, external scene0003 PASS before a target score cache can be authorized.
The selector threshold is always read from the validated model payload.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import lerf_o0_conditional_missing_core_completion_v2 as formal
from radio_gs.interfaces.factorized_primitive_state import load_factorized_primitive_state
from radio_gs.interfaces.source_monotone_missing_core_selector import (
    SELECTOR_FEATURE_NAMES,
    SOURCE_UNIT_FEATURE_INDICES,
    target_consensus_probability,
)
from radio_gs.interfaces.source_multiscene_monotone_missing_core_selector import (
    MODEL_SCHEMA,
    validate_multiscene_selector_model_payload,
)
from radio_gs.interfaces.surface_region_full_scalar_contract import (
    aggregate_surface_region_full_scalars,
)
from radio_gs.interfaces.surface_region_target_accepted_v2 import (
    validate_target_accepted_v2_authority,
)
from radio_gs.scripts import audit_source_same_axis_o0_missing_core_mechanism as source_feature_api
from radio_gs.scripts import materialize_lerf_o0_conditional_missing_core_completion as fix6b
from radio_gs.scripts.materialize_region_capability_descriptors_v2 import (
    validate_region_capability_descriptor_authority,
)
from radio_gs.scripts.validate_source_multiscene_monotone_missing_core_selector_scene0003 import (
    evaluate_selector_gate,
)
from radio_gs.utils.immutable_artifacts import (
    file_record,
    load_json_object,
    load_torch_mapping,
    validate_file_record,
    write_frozen_json,
    write_torch_noclobber,
)


AUTHORITY_SCHEMA = "radio_gs.lerf_o0_conditional_missing_core_completion_execution.v2"
REPORT_SCHEMA = "radio_gs.lerf_o0_conditional_missing_core_completion_report.v2"
IMPLEMENTATION = Path(__file__).resolve()
INTERFACE = Path(formal.__file__).resolve()
SOURCE_AUTHORITY_SCHEMA = (
    "radio_gs.source_multiscene_monotone_missing_core_selector_authority.v2"
)
SOURCE_REPORT_SCHEMA = (
    "radio_gs.source_multiscene_monotone_missing_core_selector_report.v2"
)
SCENE0003_AUTHORITY_SCHEMA = (
    "radio_gs.source_multiscene_monotone_missing_core_scene0003_result.v2"
)
SCENE0003_REPORT_SCHEMA = (
    "radio_gs.source_multiscene_monotone_missing_core_scene0003_validation.v2"
)
SOURCE_INPUT_NAMES = formal.SOURCE_INPUT_NAMES
TARGET_INPUT_NAMES = formal.TARGET_INPUT_NAMES


def completion_contract_v2() -> dict[str, Any]:
    return formal.completion_contract()


CONTRACT_SHA256 = formal.CONTRACT_SHA256


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


def _load_json(record: Mapping[str, str], *, label: str) -> dict[str, Any]:
    value, digest, source = load_json_object(
        record["path"], expected_sha256=record["sha256"], label=label
    )
    if {"path": str(source), "sha256": digest} != dict(record) or not isinstance(
        value, Mapping
    ):
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


def _source_access_is_target_blind(value: object) -> bool:
    required = {
        "benchmark_images_opened", "benchmark_labels_opened",
        "benchmark_masks_opened", "benchmark_metrics_opened",
        "benchmark_queries_opened", "target_metrics_computed",
    }
    return (
        isinstance(value, Mapping)
        and required.issubset(value)
        and all(value[name] is False for name in required)
    )


def _validate_source_unit_table(
    record: Mapping[str, str],
    *,
    expected_schema: str,
    expected_scene_id: str,
    expected_execution_authority: Mapping[str, str],
    require_selected: bool,
) -> dict[str, Any]:
    unit = _load_torch(record, label=f"{expected_scene_id} source unit table")
    features = torch.as_tensor(unit.get("features"))
    labels = torch.as_tensor(unit.get("hard_labels"))
    signed = torch.as_tensor(unit.get("signed_utility"))
    regions = torch.as_tensor(unit.get("unit_region_indices"))
    queries = torch.as_tensor(unit.get("unit_query_indices"))
    primitives = torch.as_tensor(unit.get("unit_primitive_rows"))
    rows = int(features.shape[0]) if features.ndim == 2 else -1
    selected = torch.as_tensor(unit.get("selected")) if require_selected else None
    hashes = unit.get("channel_sha256")
    common_hashes = {
        "features", "hard_labels", "signed_utility", "unit_region_indices",
        "unit_query_indices", "unit_primitive_rows",
    }
    expected_hashes = (
        common_hashes | {"selector_probability", "selected"}
        if require_selected
        else common_hashes
        | {
            "missing_counts", "positive_fraction", "qualified_region_mask",
            "selected_query_scale_indices", "soft_target_mass_fraction",
        }
    )
    selector_probability = (
        torch.as_tensor(unit.get("selector_probability")) if require_selected else None
    )
    if (
        unit.get("schema") != expected_schema
        or unit.get("scene_id") != expected_scene_id
        or unit.get("execution_authority") != dict(expected_execution_authority)
        or unit.get("feature_names") != list(source_feature_api.FEATURE_NAMES)
        or features.ndim != 2
        or features.shape[1] != len(source_feature_api.FEATURE_NAMES)
        or rows <= 0
        or labels.shape != (rows,)
        or signed.shape != (rows,)
        or regions.shape != (rows,)
        or queries.shape != (rows,)
        or primitives.shape != (rows,)
        or (require_selected and selected is not None and selected.shape != (rows,))
        or not isinstance(hashes, Mapping)
        or set(hashes) != expected_hashes
        or features.dtype != torch.float32
        or labels.dtype != torch.bool
        or signed.dtype != torch.float32
        or regions.dtype != torch.int64
        or queries.dtype != torch.int64
        or primitives.dtype != torch.int64
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(signed).all())
        or bool((regions < 0).any())
        or bool((queries < 0).any())
        or bool((primitives < 0).any())
        or (
            require_selected
            and (
                selected is None
                or selected.dtype != torch.bool
                or selector_probability is None
                or selector_probability.dtype != torch.float32
                or selector_probability.shape != (rows,)
                or not bool(torch.isfinite(selector_probability).all())
                or bool((selector_probability < 0.0).any())
                or bool((selector_probability > 1.0).any())
            )
        )
    ):
        raise ValueError(f"{expected_scene_id} source unit table contract differs")
    for name, digest in hashes.items():
        if name not in unit or tensor_sha256(torch.as_tensor(unit[name])) != digest:
            raise ValueError(f"{expected_scene_id} source unit channel changed")
    return unit


def validate_source_gate_v2(inputs: Mapping[str, Mapping[str, str]]) -> float:
    """Validate the exact six source records and return the model threshold."""

    if not isinstance(inputs, Mapping) or not set(SOURCE_INPUT_NAMES).issubset(inputs):
        raise ValueError("FIX6c source input set differs")
    source_authority_record = inputs["multisource_selector_authority"]
    model_record = inputs["multisource_selector_model"]
    source_report_record = inputs["multisource_selector_report"]
    scene3_authority_record = inputs["scene0003_pass_authority"]
    scene3_report_record = inputs["scene0003_pass_report"]
    scene3_unit_record = inputs["scene0003_pass_unit_table"]
    source_authority = _load_json(
        source_authority_record, label="multi-source v2 selector authority"
    )
    source_report = _load_json(
        source_report_record, label="multi-source v2 selector report"
    )
    model = validate_multiscene_selector_model_payload(
        _load_torch(model_record, label="multi-source v2 selector model")
    )
    scene3_authority = _load_json(
        scene3_authority_record, label="scene0003 external PASS authority"
    )
    scene3_report = _load_json(
        scene3_report_record, label="scene0003 external PASS report"
    )
    source_rows = source_authority.get("source_inputs")
    if (
        not isinstance(source_rows, list)
        or len(source_rows) != 2
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"scene_id", "role", "authority", "report", "unit_table"}
            for row in source_rows
        )
    ):
        raise ValueError("FIX6c multi-source training input records differ")
    expected_source_rows = (
        ("scene0001_00", "mechanism_train", "radio_gs.source_same_axis_o0_mechanism_audit.v1", False),
        ("scene0002_00", "external_source_train", "radio_gs.source_monotone_missing_core_scene0002_validation.v1", True),
    )
    for row, (scene_id, role, schema, require_selected) in zip(
        source_rows, expected_source_rows
    ):
        if row.get("scene_id") != scene_id or row.get("role") != role:
            raise ValueError("FIX6c multi-source training scene/role differs")
        for name in ("authority", "report", "unit_table"):
            validate_file_record(row[name], label=f"{scene_id} {name}")
        _validate_source_unit_table(
            row["unit_table"],
            expected_schema=schema,
            expected_scene_id=scene_id,
            expected_execution_authority=row["authority"],
            require_selected=require_selected,
        )
    scene3_unit = _validate_source_unit_table(
        scene3_unit_record,
        expected_schema=SCENE0003_REPORT_SCHEMA,
        expected_scene_id="scene0003_00",
        expected_execution_authority=scene3_authority.get("execution_authority", {}),
        require_selected=True,
    )
    threshold = float(model["threshold_inclusive"])
    recomputed_probability = target_consensus_probability(
        fix6b._fold_models(model), scene3_unit["features"]
    )
    recomputed_selected = recomputed_probability >= threshold
    recomputed_outcomes, recomputed_checks = evaluate_selector_gate(
        labels=scene3_unit["hard_labels"],
        signed_utility=scene3_unit["signed_utility"],
        probability=recomputed_probability,
        unit_o0_score=scene3_unit["features"][:, 0],
        selected_mask=recomputed_selected,
    )
    sample_passed = scene3_report.get("sample_gate", {}).get("outcomes", {}).get(
        "passed"
    )
    recomputed_checks["passed"] = bool(
        sample_passed is True
        and all(value for key, value in recomputed_checks.items() if key != "passed")
    )
    source_threshold = source_report.get("metrics", {}).get(
        "threshold_selection", {}
    ).get("threshold_inclusive")
    selector_checks = scene3_report.get("selector_gate", {}).get("checks", {})
    if (
        model.get("schema") != MODEL_SCHEMA
        or model.get("execution_authority") != source_authority_record
        or model.get("feature_names") != list(SELECTOR_FEATURE_NAMES)
        or model.get("source_unit_feature_indices")
        != list(SOURCE_UNIT_FEATURE_INDICES)
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
        or source_authority.get("schema") != SOURCE_AUTHORITY_SCHEMA
        or source_authority.get("schema_version") != 2
        or source_authority.get("status")
        != "sealed_after_scene0001_scene0002_source_tables_before_multiscene_fit"
        or source_authority.get("source_scenes")
        != ["scene0001_00", "scene0002_00"]
        or source_authority.get("outputs")
        != {"model": model_record["path"], "report": source_report_record["path"]}
        or source_authority.get("source_validation_execution_authorized") is not False
        or source_authority.get("benchmark_execution_authorized") is not False
        or not _source_access_is_target_blind(source_authority.get("source_access"))
        or source_report.get("schema") != SOURCE_REPORT_SCHEMA
        or source_report.get("schema_version") != 2
        or source_report.get("status")
        != "scene0001_scene0002_multiscene_selector_v2_gate_passed"
        or source_report.get("execution_authority") != source_authority_record
        or source_report.get("model") != model_record
        or source_report.get("gate", {}).get("passed") is not True
        or source_report.get("source_validation_execution_performed") is not False
        or source_report.get("target_execution_performed") is not False
        or source_report.get("benchmark_execution_authorized") is not False
        or source_threshold != threshold
        or not _source_access_is_target_blind(source_report.get("source_access"))
        or scene3_authority.get("schema") != SCENE0003_AUTHORITY_SCHEMA
        or scene3_authority.get("schema_version") != 2
        or scene3_authority.get("status")
        != "scene0003_frozen_multiscene_selector_external_gate_passed"
        or scene3_authority.get("scene_id") != "scene0003_00"
        or scene3_authority.get("frozen_selector_authority")
        != source_authority_record
        or scene3_authority.get("frozen_selector_model") != model_record
        or scene3_authority.get("frozen_selector_report") != source_report_record
        or scene3_authority.get("unit_table") != scene3_unit_record
        or scene3_authority.get("validation_report") != scene3_report_record
        or scene3_authority.get("all_formal_gates_passed") is not True
        or scene3_authority.get("benchmark_execution_authorized") is not False
        or not _source_access_is_target_blind(scene3_authority.get("source_access"))
        or scene3_report.get("schema") != SCENE0003_REPORT_SCHEMA
        or scene3_report.get("schema_version") != 2
        or scene3_report.get("status")
        != "scene0003_frozen_multiscene_selector_external_gate_passed"
        or scene3_report.get("frozen_selector_model") != model_record
        or scene3_report.get("unit_table") != scene3_unit_record
        or scene3_report.get("execution_authority")
        != scene3_authority.get("execution_authority")
        or scene3_unit.get("execution_authority")
        != scene3_authority.get("execution_authority")
        or not torch.equal(
            recomputed_probability,
            torch.as_tensor(scene3_unit.get("selector_probability")),
        )
        or not torch.equal(
            recomputed_selected, torch.as_tensor(scene3_unit.get("selected"))
        )
        or recomputed_outcomes
        != scene3_report.get("selector_gate", {}).get("outcomes")
        or recomputed_checks != selector_checks
        or recomputed_outcomes != scene3_authority.get("validation_outcomes")
        or scene3_report.get("frozen_threshold_inclusive") != threshold
        or not isinstance(selector_checks, Mapping)
        or selector_checks.get("passed") is not True
        or any(value is not True for value in selector_checks.values())
        or scene3_report.get("benchmark_execution_authorized") is not False
        or scene3_report.get("target_execution_performed") is not False
        or not _source_access_is_target_blind(scene3_report.get("source_access"))
    ):
        raise ValueError("FIX6c multi-source/external selector gate differs")
    return threshold


def _build_access() -> dict[str, bool]:
    return {
        "multisource_v2_and_scene0003_external_PASS_records_opened": True,
        "input_target_file_records_validated": True,
        "target_payloads_opened": False,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
    }


def _execute_access() -> dict[str, bool]:
    return {
        "multisource_v2_and_scene0003_external_PASS_records_opened": True,
        "exact_O0_and_query_independent_target_geometry_opened": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "threshold_scan_or_refit": False,
    }


def fixed_no_gt_gates() -> dict[str, Any]:
    return fix6b.fixed_no_gt_gates()


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    output = _new_path(args.output_authority, label="FIX6c execution authority")
    output_cache = _new_path(args.output_cache, label="FIX6c external query cache")
    output_report = _new_path(args.output_report, label="FIX6c no-GT report")
    inputs = {
        name: _record(
            getattr(args, name),
            getattr(args, "expected_" + name + "_sha256"),
            label=name,
        )
        for name in (*TARGET_INPUT_NAMES, *SOURCE_INPUT_NAMES)
    }
    threshold = validate_source_gate_v2(inputs)
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": 2,
        "status": "authorized_after_multisource_v2_and_scene0003_external_PASS_for_no_GT_cache_only",
        "scene_id": str(args.scene_id),
        "implementation": file_record(IMPLEMENTATION),
        "interface": file_record(INTERFACE),
        "source_feature_implementation": file_record(
            Path(source_feature_api.__file__).resolve()
        ),
        "contract": completion_contract_v2(),
        "contract_sha256": CONTRACT_SHA256,
        "frozen_threshold_inclusive": threshold,
        "threshold_source": inputs["multisource_selector_model"],
        "fixed_no_GT_gates": fixed_no_gt_gates(),
        "input_authority": inputs,
        "outputs": {"cache": str(output_cache), "report": str(output_report)},
        "target_score_cache_authorized": True,
        "target_metric_execution_authorized": False,
        "access_audit": _build_access(),
    }
    write_frozen_json(output, authority)
    return {
        "status": "FIX6c_authority_built_after_full_source_PASS_without_target_payload_open",
        "authority": file_record(output),
    }


def validate_authority(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("FIX6c authority must be a mapping")
    authority = dict(value)
    required = {
        "schema", "schema_version", "status", "scene_id", "implementation",
        "interface", "source_feature_implementation", "contract", "contract_sha256",
        "frozen_threshold_inclusive", "threshold_source", "fixed_no_GT_gates",
        "input_authority", "outputs", "target_score_cache_authorized",
        "target_metric_execution_authorized", "access_audit",
    }
    if set(authority) != required:
        raise ValueError("FIX6c authority fields differ")
    inputs = authority.get("input_authority")
    if not isinstance(inputs, Mapping) or set(inputs) != set((*TARGET_INPUT_NAMES, *SOURCE_INPUT_NAMES)):
        raise ValueError("FIX6c authority input set differs")
    for name, record in inputs.items():
        validate_file_record(record, label=f"FIX6c {name}")
    threshold = validate_source_gate_v2(inputs)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 2
        or authority.get("status")
        != "authorized_after_multisource_v2_and_scene0003_external_PASS_for_no_GT_cache_only"
        or authority.get("implementation") != file_record(IMPLEMENTATION)
        or authority.get("interface") != file_record(INTERFACE)
        or authority.get("source_feature_implementation")
        != file_record(Path(source_feature_api.__file__).resolve())
        or authority.get("contract") != completion_contract_v2()
        or authority.get("contract_sha256") != CONTRACT_SHA256
        or authority.get("frozen_threshold_inclusive") != threshold
        or authority.get("threshold_source") != inputs["multisource_selector_model"]
        or authority.get("fixed_no_GT_gates") != fixed_no_gt_gates()
        or authority.get("target_score_cache_authorized") is not True
        or authority.get("target_metric_execution_authorized") is not False
        or authority.get("access_audit") != _build_access()
        or not isinstance(authority.get("scene_id"), str)
        or not authority["scene_id"]
        or not isinstance(authority.get("outputs"), Mapping)
        or set(authority["outputs"]) != {"cache", "report"}
    ):
        raise ValueError("FIX6c authority header differs")
    return authority


def execute(args: argparse.Namespace) -> dict[str, Any]:
    raw, digest, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="FIX6c execution authority",
    )
    authority = validate_authority(raw)
    verified_record = {"path": str(authority_path), "sha256": digest}
    cache_path = Path(authority["outputs"]["cache"])
    report_path = Path(authority["outputs"]["report"])
    if any(path.exists() or path.is_symlink() for path in (cache_path, report_path)):
        raise FileExistsError("FIX6c outputs must both be new")
    inputs = authority["input_authority"]
    threshold = validate_source_gate_v2(inputs)
    model = _load_torch(inputs["multisource_selector_model"], label="multi-source v2 selector model")
    o0 = _load_torch(inputs["exact_o0_cache"], label="exact O0 cache")
    accepted = validate_target_accepted_v2_authority(
        _load_torch(inputs["target_accepted_v2"], label="target AcceptedV2")
    )
    capability = validate_region_capability_descriptor_authority(
        _load_torch(inputs["target_capability_descriptor"], label="target capability descriptor")
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
        o0.get("schema") != "radio_gs.lerf_o0_anchored_graph_residual_external_scores.v1"
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
        or not torch.equal(capability["canonical_region_indices"], accepted["canonical_region_indices"])
        or accepted_geometry["factorized_primitive_state_file_sha256"] != state_record["sha256"]
        or state.xyz.shape != xyz.shape
        or not torch.equal(state.xyz.float().cpu(), xyz)
    ):
        raise ValueError("FIX6c target lineage differs")
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
    radius = fix6b._spatial_radius(accepted["region_rows"], accepted["token_mask"], xyz)
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
        fold_models=fix6b._fold_models(model),
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
    unit_flat = result.unit_primitive_rows * scores.shape[1] + result.unit_query_indices
    unique_unit_cells, unit_cell_inverse, unit_cell_multiplicity = torch.unique(
        unit_flat, sorted=True, return_inverse=True, return_counts=True
    )
    selected_per_unit_cell = torch.zeros(unique_unit_cells.numel(), dtype=torch.long)
    if unit_flat.numel():
        selected_per_unit_cell.scatter_add_(
            0, unit_cell_inverse, result.selected_unit_mask.long()
        )
    mixed_unit_cell = (selected_per_unit_cell > 0) & (
        selected_per_unit_cell < unit_cell_multiplicity
    )
    selected_cell_union_count = int((selected_per_unit_cell > 0).sum())
    checks = {
        "candidate_units_nonempty": candidate_units > 0,
        "selected_and_rejected_units_both_nonempty": selected_units > 0 and rejected_units > 0,
        "selected_unique_cells_strict_subset_of_unconditional_FIX6_cells": 0 < selected_cells < unconditional_cells,
        "changed_cells_exactly_equal_selected_unique_cells": torch.equal(result.changed_mask, result.selected_cell_mask),
        "outside_selected_cells_bitwise_exact_O0": torch.equal(result.final_scores[~result.selected_cell_mask], scores[~result.selected_cell_mask]),
        "invalid_primitives_bitwise_exact_O0": torch.equal(result.final_scores[~valid], scores[~valid]),
        "pointwise_non_decreasing": bool((result.final_scores >= scores).all()),
        "maximum_per_query_membership_expansion": float(expansion.max()) <= formal.MAXIMUM_MEMBERSHIP_EXPANSION,
        "all_target_axes_and_lineage_exact": True,
        "selected_unique_cells_equal_any_selected_proposal_union": (
            selected_cells == selected_cell_union_count
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": 2,
        "status": "conditional_missing_core_v2_no_GT_gate_passed" if checks["passed"] else "conditional_missing_core_v2_no_GT_gate_failed",
        "execution_authority": verified_record,
        "scene_id": authority["scene_id"],
        "frozen_threshold_inclusive": threshold,
        "threshold_source": inputs["multisource_selector_model"],
        "source_external_status": "scene0003_frozen_multiscene_selector_external_gate_passed",
        "candidate_units": candidate_units,
        "selected_units": selected_units,
        "rejected_units": rejected_units,
        "qualified_anchor_region_query_pairs": int(result.qualified_anchor_mask.sum()),
        "unconditional_FIX6_unique_cells": unconditional_cells,
        "selected_unique_cells": selected_cells,
        "strictly_changed_cells": int(result.changed_mask.sum()),
        "unit_to_unique_cell_overlap": {
            "candidate_units": int(unit_flat.numel()),
            "candidate_unique_cells": int(unique_unit_cells.numel()),
            "unique_cells_with_multiple_proposals": int((unit_cell_multiplicity > 1).sum()),
            "extra_proposals_beyond_unique_cells": int(unit_flat.numel() - unique_unit_cells.numel()),
            "mixed_selected_rejected_unique_cells": int(mixed_unit_cell.sum()),
            "mixed_selected_rejected_proposals": int(unit_cell_multiplicity[mixed_unit_cell].sum()),
            "selection_semantics": "any_selected_proposal_selects_unique_cell_union",
            "completion_semantics": "amax_over_selected_proposals_only",
        },
        "threshold_membership_flips": int((~o0_membership & final_membership).sum()),
        "per_query": {
            "query_names": names,
            "O0_membership_counts": o0_count.long().tolist(),
            "final_membership_counts": final_count.long().tolist(),
            "membership_expansion": expansion.tolist(),
            "selected_unit_counts": torch.bincount(
                result.unit_query_indices[result.selected_unit_mask], minlength=scores.shape[1]
            ).long().tolist(),
        },
        "no_GT_safety_gate": checks,
        "access_audit": _execute_access(),
        "target_metric_execution_authorized": False,
    }
    if not checks["passed"]:
        write_frozen_json(report_path, report)
        raise RuntimeError("FIX6c no-GT safety gate failed")
    external = formal.build_external_query_score_cache(
        result=result,
        o0_valid=valid,
        o0_xyz=xyz,
        query_names=names,
        scene_id=authority["scene_id"],
        input_authority=inputs,
        threshold_inclusive=threshold,
        threshold_source=inputs["multisource_selector_model"],
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
    parser.add_argument("--expected-" + name.replace("_", "-") + "-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    for name in (*TARGET_INPUT_NAMES, *SOURCE_INPUT_NAMES):
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
    "AUTHORITY_SCHEMA", "CONTRACT_SHA256", "REPORT_SCHEMA", "SOURCE_INPUT_NAMES",
    "TARGET_INPUT_NAMES", "build_authority", "completion_contract_v2", "execute",
    "fixed_no_gt_gates", "validate_authority", "validate_source_gate_v2",
]
