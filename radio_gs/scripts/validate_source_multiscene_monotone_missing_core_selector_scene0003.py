#!/usr/bin/env python3
"""One-shot scene0003 external validation of the frozen source selector v2.

The execution authority must be sealed after raw same-axis O0 materialization
and before this program is allowed to open the scene0003 membership payload.
No model, feature, threshold, or gate is fitted in this validator.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from radio_gs.evaluation.text_response_fidelity import tensor_sha256
from radio_gs.interfaces import (
    source_multiscene_monotone_missing_core_selector as selector_v2_api,
)
from radio_gs.interfaces.source_missing_core_conditional_utility import (
    source_missing_core_conditional_utility,
)
from radio_gs.interfaces.source_monotone_missing_core_selector import (
    MonotoneAdditiveLogistic,
    SELECTOR_FEATURE_NAMES,
    SOURCE_UNIT_FEATURE_INDICES,
    target_consensus_probability,
    tie_invariant_average_precision,
)
from radio_gs.interfaces.source_multiscene_monotone_missing_core_selector import (
    MODEL_SCHEMA,
    validate_multiscene_selector_model_payload,
)
from radio_gs.scripts import (
    audit_source_same_axis_o0_missing_core_mechanism as feature_api,
)
from radio_gs.scripts import (
    build_lerf_o0_anchored_graph_residual_cache as exact_o0_api,
)
from radio_gs.scripts.audit_source_same_axis_o0_missing_core_mechanism import (
    FEATURE_NAMES,
    KNN_CHUNK_SIZE,
    build_unit_feature_table,
)
from radio_gs.scripts.build_lerf_o0_anchored_graph_residual_cache import (
    exact_o0_readout,
)
from radio_gs.scripts.calibrate_source_only_graph_confidence_v1 import (
    one_sided_wilson_lower,
)
from radio_gs.scripts.train_source_region_comembership_v2 import (
    load_scene_authority_v2,
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
    "radio_gs.source_multiscene_monotone_missing_core_scene0003_authority.v2"
)
RESULT_SCHEMA = (
    "radio_gs.source_multiscene_monotone_missing_core_scene0003_validation.v2"
)
RESULT_AUTHORITY_SCHEMA = (
    "radio_gs.source_multiscene_monotone_missing_core_scene0003_result.v2"
)
SCENE_ID = "scene0003_00"
EXTERNAL_SPLIT = "source_external_validation_multisource_selector_only"
REGISTERED_MEMBERSHIP_SPLIT = "source_train"


def source_access() -> dict[str, bool]:
    return {
        "multiscene_selector_frozen_before_scene0003_labels_opened": True,
        "scene0003_membership_payload_opened_after_raw_O0_frozen": True,
        "scene0003_membership_payload_opened_exactly_once_by_validator": True,
        "scene0003_not_used_for_selector_fit_or_threshold_selection": True,
        "scene0003_accepted_v2_opened": True,
        "scene0003_query_independent_capability_reliability_opened": True,
        "scene0003_full_scalar_shard_opened": True,
        "scene0004_membership_opened": False,
        "benchmark_images_opened": False,
        "benchmark_labels_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_queries_opened": False,
        "benchmark_metrics_opened": False,
        "target_benchmark_opened": False,
        "target_metrics_computed": False,
    }


def fixed_validation() -> dict[str, Any]:
    return {
        "canonical_negative_count": 4,
        "canonical_negative_probability_logit_scale": 10.0,
        "knn": 10,
        "knn_chunk_size": KNN_CHUNK_SIZE,
        "scale_selection": "highest_raw_smoothed_peak_lower_scale_tie_break",
        "per_scale_query_minmax": True,
        "o0_positive": "strictly_greater_than_0p6",
        "qualified_anchor": "valid_core_positive_fraction_at_least_0p75",
        "missing_core": "valid_core_O0_score_less_than_or_equal_to_0p6",
        "selector_model_schema": MODEL_SCHEMA,
        "selector_feature_names": list(SELECTOR_FEATURE_NAMES),
        "selector_source_unit_feature_indices": list(SOURCE_UNIT_FEATURE_INDICES),
        "selector_probability": "minimum_probability_across_three_fold_models",
        "selector_threshold": (
            "exact_frozen_scene0001_scene0002_v2_threshold_inclusive"
        ),
        "sample_gate": {
            "minimum_qualified_anchor_query_pairs": 32,
            "minimum_missing_core_units": 256,
            "minimum_hard_positive_missing_units": 32,
            "minimum_hard_negative_missing_units": 32,
        },
        "selector_gate": {
            "minimum_selected": 256,
            "require_selected_hard_positive_and_negative": True,
            "minimum_hard_precision_Wilson95_lower": 0.75,
            "minimum_signed_utility_mean_exclusive": 0.0,
            "require_selected_signed_utility_above_unconditional": True,
            "require_selector_AP_strictly_above_unit_O0_score_AP": True,
        },
        "threshold_or_model_refit_on_scene0003": False,
        "positive_query_count": "bound_from_label_blind_subset_authority",
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
        "scene_id",
        "split",
        "registered_membership_split",
        "implementation",
        "selector_v2_interface",
        "feature_implementation",
        "exact_o0_implementation",
        "raw_pre_membership_authority",
        "subset_execution_authority",
        "dense_stream_execution_authority",
        "combined_text_subset",
        "raw_combined_multiscale_scores",
        "accepted_v2",
        "source_membership_authority",
        "source_full_scalar_shard",
        "source_capability_descriptor",
        "frozen_selector_authority",
        "frozen_selector_model",
        "frozen_selector_report",
        "fixed_validation",
        "outputs",
        "source_access",
        "benchmark_execution_authorized",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("scene0003 selector validation authority fields differ")
    authority = dict(value)
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("schema_version") != 2
        or authority.get("status")
        != "sealed_after_scene0003_raw_O0_before_membership_payload_open"
        or authority.get("scene_id") != SCENE_ID
        or authority.get("split") != EXTERNAL_SPLIT
        or authority.get("registered_membership_split")
        != REGISTERED_MEMBERSHIP_SPLIT
        or authority.get("fixed_validation") != fixed_validation()
        or authority.get("source_access") != source_access()
        or authority.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("scene0003 selector validation authority header differs")
    for name in (
        "implementation",
        "selector_v2_interface",
        "feature_implementation",
        "exact_o0_implementation",
        "raw_pre_membership_authority",
        "subset_execution_authority",
        "dense_stream_execution_authority",
        "combined_text_subset",
        "raw_combined_multiscale_scores",
        "accepted_v2",
        "source_membership_authority",
        "source_full_scalar_shard",
        "source_capability_descriptor",
        "frozen_selector_authority",
        "frozen_selector_model",
        "frozen_selector_report",
    ):
        authority[name] = _record(authority[name], label=name)
    outputs = authority.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "unit_table",
        "report",
        "result_authority",
    }:
        raise ValueError("scene0003 selector validation outputs differ")
    authority["outputs"] = {
        name: str(Path(path).expanduser().resolve()) for name, path in outputs.items()
    }
    return authority


def _validate_dependencies(authority: Mapping[str, Any]) -> None:
    expected = {
        "implementation": Path(__file__).resolve(),
        "selector_v2_interface": Path(selector_v2_api.__file__).resolve(),
        "feature_implementation": Path(feature_api.__file__).resolve(),
        "exact_o0_implementation": Path(exact_o0_api.__file__).resolve(),
    }
    for key, path in expected.items():
        if authority[key] != file_record(path):
            raise ValueError(f"scene0003 selector dependency changed: {key}")


def validate_raw_pre_membership_value(
    raw: object, authority: Mapping[str, Any]
) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError("scene0003 raw pre-membership authority must be a mapping")
    source_assets = raw.get("source_assets", {})
    deferred_membership = source_assets.get(
        "source_membership_authority_deferred", {}
    )
    if (
        raw.get("schema")
        != "radio_gs.source_same_axis_o0_scene0003_raw_pre_membership_authority.v1"
        or raw.get("status")
        != "sealed_after_scene0003_raw_O0_before_multisource_selector_and_membership_payload_open"
        or raw.get("scene_id") != SCENE_ID
        or raw.get("split") != EXTERNAL_SPLIT
        or raw.get("registered_membership_split") != REGISTERED_MEMBERSHIP_SPLIT
        or raw.get("subset_authority") != authority["subset_execution_authority"]
        or raw.get("dense_stream_authority")
        != authority["dense_stream_execution_authority"]
        or {
            "path": raw.get("combined_text_subset", {}).get("path"),
            "sha256": raw.get("combined_text_subset", {}).get("sha256"),
        }
        != authority["combined_text_subset"]
        or {
            "path": raw.get("raw_combined_multiscale_scores", {}).get("path"),
            "sha256": raw.get("raw_combined_multiscale_scores", {}).get("sha256"),
        }
        != authority["raw_combined_multiscale_scores"]
        or source_assets.get("accepted_v2") != authority["accepted_v2"]
        or source_assets.get("source_full_scalar_shard")
        != authority["source_full_scalar_shard"]
        or source_assets.get("source_capability_descriptor")
        != authority["source_capability_descriptor"]
        or {
            "path": deferred_membership.get("path"),
            "sha256": deferred_membership.get("sha256"),
        }
        != authority["source_membership_authority"]
        or deferred_membership.get("payload_opened") is not False
        or raw.get("source_access", {}).get("scene0003_membership_payload_opened")
        is not False
        or raw.get("source_access", {}).get("multisource_selector_opened") is not False
        or raw.get("benchmark_execution_authorized") is not False
    ):
        raise ValueError("scene0003 raw pre-membership chain differs")


def _validate_raw_pre_membership_chain(authority: Mapping[str, Any]) -> None:
    raw, _, _ = load_json_object(
        authority["raw_pre_membership_authority"]["path"],
        expected_sha256=authority["raw_pre_membership_authority"]["sha256"],
        label="scene0003 raw pre-membership authority",
    )
    validate_raw_pre_membership_value(raw, authority)


def _load_fold_models(raw: Mapping[str, Any]) -> tuple[MonotoneAdditiveLogistic, ...]:
    validate_multiscene_selector_model_payload(dict(raw))
    result = []
    for row in raw["fold_models"]:
        result.append(
            MonotoneAdditiveLogistic(
                location=torch.as_tensor(row["location"]).detach().float().cpu(),
                scale=torch.as_tensor(row["scale"]).detach().float().cpu(),
                positive_weights=torch.as_tensor(row["positive_weights"])
                .detach()
                .float()
                .cpu(),
                bias=torch.as_tensor(row["bias"]).detach().float().cpu().reshape(()),
            )
        )
    return tuple(result)


def evaluate_selector_gate(
    *,
    labels: torch.Tensor,
    signed_utility: torch.Tensor,
    probability: torch.Tensor,
    unit_o0_score: torch.Tensor,
    selected_mask: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Evaluate the fixed scene0003 gate without fitting or threshold search."""

    truth = torch.as_tensor(labels).detach().bool().cpu().reshape(-1)
    signed = torch.as_tensor(signed_utility).detach().float().cpu().reshape(-1)
    score = torch.as_tensor(probability).detach().float().cpu().reshape(-1)
    o0 = torch.as_tensor(unit_o0_score).detach().float().cpu().reshape(-1)
    selected = torch.as_tensor(selected_mask).detach().bool().cpu().reshape(-1)
    if (
        truth.shape != signed.shape
        or score.shape != truth.shape
        or o0.shape != truth.shape
        or selected.shape != truth.shape
        or truth.numel() <= 0
        or not bool(torch.isfinite(signed).all())
        or not bool(torch.isfinite(score).all())
        or not bool(torch.isfinite(o0).all())
    ):
        raise ValueError("scene0003 selector gate inputs differ")
    selected_total = int(selected.sum())
    selected_positive = int(truth[selected].sum())
    selected_negative = selected_total - selected_positive
    selected_signed = float(signed[selected].mean()) if selected_total else 0.0
    unconditional_signed = float(signed.mean())
    selected_wilson = (
        one_sided_wilson_lower(selected_positive, selected_total)
        if selected_total
        else 0.0
    )
    selector_ap = tie_invariant_average_precision(score, truth)
    o0_ap = tie_invariant_average_precision(o0, truth)
    outcomes = {
        "selected": selected_total,
        "selected_hard_positive": selected_positive,
        "selected_hard_negative": selected_negative,
        "selected_hard_precision": selected_positive / max(selected_total, 1),
        "selected_hard_precision_Wilson95_lower": selected_wilson,
        "selected_signed_utility_mean": selected_signed,
        "unconditional_signed_utility_mean": unconditional_signed,
        "selector_average_precision": selector_ap,
        "unit_O0_score_average_precision": o0_ap,
    }
    gate = fixed_validation()["selector_gate"]
    checks = {
        "selected_at_least_minimum": selected_total >= gate["minimum_selected"],
        "selected_hard_positive_and_negative_both_evaluated": (
            selected_positive > 0 and selected_negative > 0
        ),
        "hard_precision_Wilson95_lower_at_least_minimum": selected_wilson
        >= gate["minimum_hard_precision_Wilson95_lower"],
        "selected_signed_utility_mean_positive": selected_signed
        > gate["minimum_signed_utility_mean_exclusive"],
        "selected_signed_utility_above_unconditional": selected_signed
        > unconditional_signed,
        "selector_AP_strictly_above_unit_O0_score_AP": selector_ap > o0_ap,
    }
    checks["passed"] = all(checks.values())
    return outcomes, checks


def _validate_frozen_selector(
    authority: Mapping[str, Any],
) -> tuple[float, tuple[MonotoneAdditiveLogistic, ...]]:
    source_authority, _, _ = load_json_object(
        authority["frozen_selector_authority"]["path"],
        expected_sha256=authority["frozen_selector_authority"]["sha256"],
        label="frozen multi-scene selector authority",
    )
    source_report, _, _ = load_json_object(
        authority["frozen_selector_report"]["path"],
        expected_sha256=authority["frozen_selector_report"]["sha256"],
        label="frozen multi-scene selector report",
    )
    selector, _, _ = load_torch_mapping(
        authority["frozen_selector_model"]["path"],
        expected_sha256=authority["frozen_selector_model"]["sha256"],
        map_location="cpu",
        label="frozen multi-scene selector model",
    )
    if (
        source_authority.get("status")
        != "sealed_after_scene0001_scene0002_source_tables_before_multiscene_fit"
        or source_authority.get("source_scenes")
        != ["scene0001_00", "scene0002_00"]
        or source_authority.get("outputs", {}).get("model")
        != authority["frozen_selector_model"]["path"]
        or source_authority.get("outputs", {}).get("report")
        != authority["frozen_selector_report"]["path"]
        or source_authority.get("source_validation_execution_authorized") is not False
        or source_report.get("status")
        != "scene0001_scene0002_multiscene_selector_v2_gate_passed"
        or source_report.get("execution_authority")
        != authority["frozen_selector_authority"]
        or source_report.get("model") != authority["frozen_selector_model"]
        or source_report.get("gate", {}).get("passed") is not True
        or source_report.get("source_validation_execution_performed") is not False
        or source_report.get("source_access", {}).get(
            "scene0003_membership_opened"
        )
        is not False
        or source_report.get("source_access", {}).get(
            "scene0004_membership_opened"
        )
        is not False
        or selector.get("execution_authority")
        != authority["frozen_selector_authority"]
        or selector.get("training_provenance", {}).get("source_scene_ids_sha256")
        != canonical_json_sha256(["scene0001_00", "scene0002_00"])
    ):
        raise ValueError("frozen multi-scene selector chain differs")
    fold_models = _load_fold_models(selector)
    threshold = float(selector.get("threshold_inclusive", float("nan")))
    report_threshold = float(
        source_report.get("metrics", {})
        .get("threshold_selection", {})
        .get("threshold_inclusive", float("nan"))
    )
    if not 0.0 <= threshold <= 1.0 or threshold != report_threshold:
        raise ValueError("frozen multi-scene selector threshold differs")
    return threshold, fold_models


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw_authority, authority_sha, authority_path = load_json_object(
        args.execution_authority,
        expected_sha256=args.expected_execution_authority_sha256,
        label="scene0003 selector validation authority",
    )
    authority = validate_execution_authority(raw_authority)
    _validate_dependencies(authority)
    _validate_raw_pre_membership_chain(authority)
    output_paths = [Path(value) for value in authority["outputs"].values()]
    if any(path.exists() or path.is_symlink() for path in output_paths):
        raise FileExistsError("scene0003 selector validation outputs must all be new")

    threshold, fold_models = _validate_frozen_selector(authority)
    subset, _, _ = load_torch_mapping(
        authority["combined_text_subset"]["path"],
        expected_sha256=authority["combined_text_subset"]["sha256"],
        map_location="cpu",
        label="scene0003 combined text subset",
    )
    raw_scores, _, _ = load_torch_mapping(
        authority["raw_combined_multiscale_scores"]["path"],
        expected_sha256=authority["raw_combined_multiscale_scores"]["sha256"],
        map_location="cpu",
        label="scene0003 raw combined scores",
    )
    positive_count = int(subset.get("positive_query_count", -1))
    query_names = [str(value) for value in subset.get("queries", [])]
    raw_tensor = torch.as_tensor(raw_scores.get("features")).detach().float().cpu()
    valid = torch.as_tensor(raw_scores.get("valid")).detach().bool().cpu()
    xyz = torch.as_tensor(raw_scores.get("xyz")).detach().float().cpu()
    metadata = raw_scores.get("metadata", {})
    if (
        subset.get("scene_id") != SCENE_ID
        or subset.get("split") != EXTERNAL_SPLIT
        or positive_count <= 0
        or int(subset.get("canonical_negative_query_count", -1)) != 4
        or len(query_names) != positive_count + 4
        or raw_tensor.ndim != 3
        or raw_tensor.shape[1:] != (3, len(query_names))
        or valid.shape != (raw_tensor.shape[0],)
        or xyz.shape != (raw_tensor.shape[0], 3)
        or metadata.get("query_names") != query_names
        or metadata.get("feature_space")
        != "primitive_text_query_scores_multiscale_unreduced"
        or metadata.get("scale_aggregation") != "none_frozen_downstream_only"
        or metadata.get("completion")
        != {
            "applied": False,
            "reason": "frozen_direct3d_requires_raw_unreduced_scale_scores",
        }
    ):
        raise ValueError("scene0003 raw same-axis O0 contract differs")
    o0 = exact_o0_readout(
        positive_scores=raw_tensor[:, :, :positive_count],
        negative_scores=raw_tensor[:, :, positive_count:],
        xyz=xyz,
        valid=valid,
        chunk_size=KNN_CHUNK_SIZE,
    )
    accepted, _, _ = load_torch_mapping(
        authority["accepted_v2"]["path"],
        expected_sha256=authority["accepted_v2"]["sha256"],
        map_location="cpu",
        label="scene0003 AcceptedV2",
    )

    # This is the first and only semantic membership-payload open.  Exact raw
    # O0, the v2 model, threshold, and every gate are immutable above.
    scene = load_scene_authority_v2(
        authority["source_membership_authority"],
        expected_scene_id=SCENE_ID,
        expected_split=REGISTERED_MEMBERSHIP_SPLIT,
    )
    if (
        not torch.equal(scene.region_rows, accepted["region_rows"])
        or not torch.equal(scene.token_mask, accepted["token_mask"])
    ):
        raise ValueError("scene0003 source region axes differ")
    utility = source_missing_core_conditional_utility(
        o0_scores=o0.final_scores,
        region_rows=scene.region_rows,
        core_mask=scene.token_mask,
        primitive_valid_mask=valid,
        region_query_indices=subset["region_dominant_positive_subset_index"],
        region_dominant_instance_ids=scene.dominant_instance_ids,
        primitive_instance_mass=scene.primitive_instance_mass,
    )
    shard, _, _ = load_torch_mapping(
        authority["source_full_scalar_shard"]["path"],
        expected_sha256=authority["source_full_scalar_shard"]["sha256"],
        map_location="cpu",
        label="scene0003 source full-scalar shard",
    )
    capability, _, _ = load_torch_mapping(
        authority["source_capability_descriptor"]["path"],
        expected_sha256=authority["source_capability_descriptor"]["sha256"],
        map_location="cpu",
        label="scene0003 source capability descriptor",
    )
    if (
        torch.as_tensor(shard.get("accepted_v2_e0")).shape != (4096, 1536)
        or not torch.equal(
            torch.as_tensor(shard["accepted_v2_e0"]),
            torch.as_tensor(accepted["accepted_v2_e0"]),
        )
        or torch.as_tensor(shard.get("raw_full_scalar_summary")).shape != (4096, 18)
        or torch.as_tensor(shard.get("eligible")).shape != (4096,)
        or not torch.equal(torch.as_tensor(capability.get("region_rows")), scene.region_rows)
        or not torch.equal(torch.as_tensor(capability.get("token_mask")), scene.token_mask)
    ):
        raise ValueError("scene0003 source feature axes differ")
    features = build_unit_feature_table(
        utility=utility,
        o0_scores=o0.final_scores,
        primitive_valid_mask=valid,
        region_query_indices=subset["region_dominant_positive_subset_index"],
        region_rows=scene.region_rows,
        token_mask=scene.token_mask,
        xyz=xyz,
        region_scale_indices=accepted["scale_indices"],
        selected_query_scale_indices=o0.selected_scale_indices,
        appearance_concentration=capability["appearance_concentration"],
        boundary_concentration=capability["boundary_concentration"],
        raw_full_scalar_summary=shard["raw_full_scalar_summary"],
        full_scalar_eligible=shard["eligible"],
    )
    labels = utility.unit_hard_labels
    signed = utility.unit_signed_utility
    probability = target_consensus_probability(fold_models, features)
    selected = probability >= threshold

    sample_thresholds = fixed_validation()["sample_gate"]
    total = int(labels.numel())
    hard_positive = int(labels.sum())
    hard_negative = total - hard_positive
    sample = {
        "qualified_anchor_query_pairs": int(utility.qualified_region_mask.sum()),
        "missing_core_units": total,
        "hard_positive_missing_units": hard_positive,
        "hard_negative_missing_units": hard_negative,
    }
    sample["passed"] = (
        sample["qualified_anchor_query_pairs"]
        >= sample_thresholds["minimum_qualified_anchor_query_pairs"]
        and total >= sample_thresholds["minimum_missing_core_units"]
        and hard_positive >= sample_thresholds["minimum_hard_positive_missing_units"]
        and hard_negative >= sample_thresholds["minimum_hard_negative_missing_units"]
    )
    outcomes, checks = evaluate_selector_gate(
        labels=labels,
        signed_utility=signed,
        probability=probability,
        unit_o0_score=features[:, 0],
        selected_mask=selected,
    )
    gate = fixed_validation()["selector_gate"]
    checks["passed"] = bool(sample["passed"] and all(
        value for key, value in checks.items() if key != "passed"
    ))

    unit_output = Path(authority["outputs"]["unit_table"])
    report_output = Path(authority["outputs"]["report"])
    result_output = Path(authority["outputs"]["result_authority"])
    unit_payload = {
        "schema": RESULT_SCHEMA,
        "schema_version": 2,
        "scene_id": SCENE_ID,
        "feature_names": list(FEATURE_NAMES),
        "features": features,
        "hard_labels": labels,
        "signed_utility": signed,
        "selector_probability": probability,
        "selected": selected,
        "unit_region_indices": utility.unit_region_indices,
        "unit_query_indices": utility.unit_query_indices,
        "unit_primitive_rows": utility.unit_primitive_rows,
        "execution_authority": {"path": str(authority_path), "sha256": authority_sha},
    }
    unit_payload["channel_sha256"] = {
        name: tensor_sha256(unit_payload[name])
        for name in (
            "features",
            "hard_labels",
            "signed_utility",
            "selector_probability",
            "selected",
            "unit_region_indices",
            "unit_query_indices",
            "unit_primitive_rows",
        )
    }
    write_torch_noclobber(unit_output, unit_payload)
    report = {
        "schema": RESULT_SCHEMA,
        "schema_version": 2,
        "status": (
            "scene0003_frozen_multiscene_selector_external_gate_passed"
            if checks["passed"]
            else "scene0003_frozen_multiscene_selector_external_gate_failed"
        ),
        "execution_authority": {"path": str(authority_path), "sha256": authority_sha},
        "frozen_selector_model": authority["frozen_selector_model"],
        "unit_table": file_record(unit_output),
        "same_axis_O0": {
            "valid_primitives": int(valid.sum()),
            "positive_queries": positive_count,
            "selected_scale_counts": {
                str(index): int((o0.selected_scale_indices == index).sum())
                for index in range(3)
            },
        },
        "frozen_threshold_inclusive": threshold,
        "sample_gate": {"thresholds": sample_thresholds, "outcomes": sample},
        "selector_gate": {"thresholds": gate, "outcomes": outcomes, "checks": checks},
        "source_access": source_access(),
        "benchmark_execution_authorized": False,
        "target_execution_performed": False,
    }
    report["content_authority_sha256"] = canonical_json_sha256(report)
    write_frozen_json(report_output, report)
    result = {
        "schema": RESULT_AUTHORITY_SCHEMA,
        "schema_version": 2,
        "status": report["status"],
        "scene_id": SCENE_ID,
        "execution_authority": {"path": str(authority_path), "sha256": authority_sha},
        "frozen_selector_authority": authority["frozen_selector_authority"],
        "frozen_selector_model": authority["frozen_selector_model"],
        "frozen_selector_report": authority["frozen_selector_report"],
        "unit_table": file_record(unit_output),
        "validation_report": file_record(report_output),
        "validation_outcomes": outcomes,
        "all_formal_gates_passed": checks["passed"],
        "source_access": source_access(),
        "benchmark_execution_authorized": False,
    }
    result["content_authority_sha256"] = canonical_json_sha256(result)
    write_frozen_json(result_output, result)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-authority", required=True)
    parser.add_argument("--expected-execution-authority-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    report = run(build_parser().parse_args(argv))
    print(report["status"])
    print(report["selector_gate"])


if __name__ == "__main__":
    main()
