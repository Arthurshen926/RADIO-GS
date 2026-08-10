#!/usr/bin/env python3
"""Materialize one globally selected reliability-conditioned LERF candidate.

This is a query-free, metric-closed bridge from an AcceptedV2 multiscale O0
descriptor and a teacher-agreement-v2 payload to ordinary raw positive and
canonical-negative score caches.  The source-only selector contributes one
global angular ceiling; it cannot contribute a scene- or query-specific
parameter.  Candidate descriptors exist only for one bounded row batch.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.interfaces.lerf_reliability_geodesic_budget import (
    CONSERVATIVE_ANGLE_RADIANS,
    MAXIMUM_ANGLE_RADIANS,
    RELIABILITY_GEODESIC_BUDGET_CONTRACT_SHA256,
    VIEW_AGREEMENT_SCALAR,
    VIEW_AGREEMENT_SHA256_FIELD,
    reliability_conditioned_geodesic_fusion,
)
from radio_gs.scripts import materialize_lerf_o1_o2_streaming as _o1o2
from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2 as _agreement_v2,
)
from radio_gs.scripts import (
    materialize_lerf_o1_o2_teacher_agreement_streaming_v2_lowmem as _agreement_lowmem,
)
from radio_gs.scripts import (
    select_lerf_source_only_global_reliability_ceiling as _selector,
)
from radio_gs.scripts import (
    execute_lerf_source_only_global_ceiling_lowmem_lineage_compatibility as _selector_compat,
)
from radio_gs.scripts.materialize_lerf_teacher_view_oracle_matrix import (
    geodesic_project,
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
    "radio_gs.lerf_reliability_conditioned_candidate_execution.v1"
)
RESULT_SCHEMA = "radio_gs.lerf_reliability_conditioned_candidate_result.v1"
SCHEMA_VERSION = 1
ROW_BATCH_SIZE = 256
SCALE_COUNT = 3
DESCRIPTOR_DIMENSION = 1536
SUPPORTED_PHYSICAL_GPUS = (0, 1)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_ACCESS_AUDIT = {
    "base_descriptor_opened": True,
    "teacher_agreement_v2_opened": True,
    "source_only_global_selector_opened": True,
    "exact_query_axis_opened": True,
    "exact_o0_pair_opened": True,
    "target_images_opened": False,
    "target_ground_truth_opened": False,
    "target_masks_opened": False,
    "target_metrics_opened": False,
    "target_quality_readout_executed": False,
}


def method_contract() -> dict[str, Any]:
    return {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "immutable_backbone": "accepted_v2_o0_descriptor_per_scale",
        "teacher_payload_schema": _agreement_v2.MEAN_SCHEMA,
        "accepted_teacher_payload_producers": [
            dict(_agreement_v2.ENTRYPOINT_IMPLEMENTATION),
            dict(_agreement_lowmem.ENTRYPOINT_IMPLEMENTATION),
        ],
        "accepted_teacher_payload_method_contract_sha256": [
            _agreement_v2.METHOD_CONTRACT_SHA256,
            _agreement_lowmem.METHOD_CONTRACT_SHA256,
        ],
        "selector_schema": _selector.OUTPUT_SCHEMA,
        "selector_method_contract_sha256": _selector.METHOD_CONTRACT_SHA256,
        "accepted_selector_authorities": [
            {
                "schema": _selector.OUTPUT_SCHEMA,
                "implementation": file_record(Path(_selector.__file__).resolve()),
                "contract_sha256": _selector.METHOD_CONTRACT_SHA256,
            },
            {
                "schema": _selector_compat.OUTPUT_SCHEMA,
                "implementation": file_record(
                    Path(_selector_compat.__file__).resolve()
                ),
                "contract_sha256": (
                    _selector_compat.COMPATIBILITY_CONTRACT_SHA256
                ),
                "post_result_lineage_compatibility_only": True,
            },
        ],
        "reliability_geodesic_budget_contract_sha256": (
            RELIABILITY_GEODESIC_BUDGET_CONTRACT_SHA256
        ),
        "candidate_grid_radians": list(_selector.CANDIDATE_GRID_RADIANS),
        "global_ceiling_application": (
            "canonical_field_attenuation=(selected-0.15)/(0.75-0.15)"
        ),
        "fusion": "existing_reliability_conditioned_geodesic_budget_v1",
        "scale_axis": "three_O0_scales_preserved_independently",
        "fallback": {
            "teacher_invalid": "bitwise_O0_score_cache",
            "single_retained_view": "exact_conservative_O1_no_expansion",
            "same_direction": "bitwise_O0_score_cache",
            "antipodal_route": "bitwise_O0_score_cache",
        },
        "conservative_0p15_route": (
            "exact_existing_O1_geodesic_project_for_nonfallback_scales"
        ),
        "descriptor_materialization": "row_batch_only",
        "row_batch_size": ROW_BATCH_SIZE,
        "full_n_by_3_by_d_candidate_descriptor_allocated": False,
        "descriptor_hash": "typed_float32_row_order_stream_sha256",
        "score_dtype": "torch.float32",
        "score_semantics": "raw_independent_normalized_cosine",
        "one_global_ceiling": True,
        "scene_or_query_specific_parameters": False,
        "query_independent_fusion": True,
        "target_data_or_metric_access": False,
        "metric_execution_authorized": False,
    }


METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())


def _record(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} file record differs")
    result = {"path": str(value["path"]), "sha256": str(value["sha256"])}
    validate_file_record(result, label=label)
    return result


def _record_args(path: str, digest: str, *, label: str) -> dict[str, str]:
    result = {"path": str(Path(path).expanduser().resolve()), "sha256": digest}
    validate_file_record(result, label=label)
    return result


def _new(path: str | Path, *, label: str) -> Path:
    raw = str(path)
    result = Path(raw).expanduser().resolve()
    if raw != str(result):
        raise ValueError(f"{label} must be canonical absolute")
    if result.exists() or result.is_symlink():
        raise FileExistsError(f"{label} already exists: {result}")
    return result


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_selector_candidate(
    path: str | Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, str], float]:
    payload, digest, source = load_json_object(
        path,
        expected_sha256=expected_sha256,
        label="source-only global ceiling selector authority",
    )
    original_required = {
        "schema", "schema_version", "status", "selector_implementation",
        "preregistration", "method_contract", "method_contract_sha256",
        "source_scene_ids", "source_count", "inputs", "candidate_grid",
        "selection", "access_audit", "query_independent",
        "metric_execution_authorized", "metric_executed", "candidate_role",
        "next_gate",
    }
    original_access = {
        "teacher_agreement_payload_or_summary_opened": True,
        "source_only_loo_summary_opened": True,
        "target_images_opened": False,
        "target_labels_or_masks_opened": False,
        "target_metrics_opened": False,
    }
    compatibility_required = {
        "schema", "schema_version", "status", "compatibility_executor",
        "compatibility_contract", "compatibility_contract_sha256",
        "created_after_source_results_for_lineage_compatibility_only",
        "selection_rule_preregistered_before_results", "original_selector",
        "original_preregistration", "allocation_lineage", "source_scene_ids",
        "source_count", "inputs", "candidate_grid", "selection", "access_audit",
        "query_independent", "metric_execution_authorized", "metric_executed",
        "candidate_role", "next_gate",
    }
    compatibility_access = {
        "lowmem_teacher_agreement_payloads_opened": True,
        "source_only_loo_summaries_opened": True,
        "target_images_opened": False,
        "target_labels_or_masks_opened": False,
        "target_metrics_opened": False,
    }
    compatibility = payload.get("schema") == _selector_compat.OUTPUT_SCHEMA
    if compatibility:
        lineage = _selector_compat.validate_local_lineage()
        expected_allocation = {
            "lowmem_implementation": lineage["lowmem_implementation"],
            "lowmem_method_contract_sha256": _selector_compat.LOWMEM_METHOD_SHA256,
            "lowmem_tests": lineage["lowmem_tests"],
            "agreement_v2_implementation": lineage["agreement_v2_implementation"],
            "agreement_v2_method_contract_sha256": (
                _selector_compat.AGREEMENT_V2_METHOD_SHA256
            ),
            "agreement_and_loo_reused_without_change": True,
            "teacher_mean_chunking_bitwise_equivalence_test_bound": True,
            "allocation_schedule_only": True,
        }
        if (
            set(payload) != compatibility_required
            or payload.get("schema_version") != _selector_compat.SCHEMA_VERSION
            or payload.get("status")
            != "complete_post_result_allocation_lineage_compatibility_selection"
            or payload.get("compatibility_executor")
            != file_record(Path(_selector_compat.__file__).resolve())
            or payload.get("compatibility_contract")
            != _selector_compat.compatibility_contract()
            or payload.get("compatibility_contract_sha256")
            != _selector_compat.COMPATIBILITY_CONTRACT_SHA256
            or payload.get(
                "created_after_source_results_for_lineage_compatibility_only"
            ) is not True
            or payload.get("selection_rule_preregistered_before_results") is not True
            or payload.get("original_selector") != {
                "implementation": lineage["original_selector_implementation"],
                "method_contract_sha256": _selector.METHOD_CONTRACT_SHA256,
            }
            or payload.get("allocation_lineage") != expected_allocation
            or payload.get("access_audit") != compatibility_access
            or payload.get("candidate_role")
            != "source_only_global_ceiling_candidate_authority_via_lineage_compatibility"
        ):
            raise ValueError("source-only compatibility selector header differs")
        preregistration_value = payload.get("original_preregistration")
        allowed_source_formats = {
            "teacher_payload_v2_lowmem_allocation_compatible"
        }
    else:
        if (
            set(payload) != original_required
            or payload.get("schema") != _selector.OUTPUT_SCHEMA
            or payload.get("schema_version") != _selector.SCHEMA_VERSION
            or payload.get("status")
            != "source_only_candidate_selected_metric_not_authorized"
            or payload.get("selector_implementation")
            != file_record(Path(_selector.__file__).resolve())
            or payload.get("method_contract") != _selector.method_contract()
            or payload.get("method_contract_sha256")
            != _selector.METHOD_CONTRACT_SHA256
            or payload.get("access_audit") != original_access
            or payload.get("candidate_role")
            != "source_only_global_ceiling_candidate_authority"
        ):
            raise ValueError("source-only selector authority header differs")
        preregistration_value = payload.get("preregistration")
        allowed_source_formats = {
            "teacher_payload_v2", "compact_source_summary_v1"
        }
    if (
        payload.get("query_independent") is not True
        or payload.get("metric_execution_authorized") is not False
        or payload.get("metric_executed") is not False
        or payload.get("next_gate")
        != "source_only_candidate_execution_authority_before_frozen_target_metric"
    ):
        raise ValueError("source-only selector metric boundary differs")
    preregistration = _record(
        preregistration_value, label="global ceiling preregistration"
    )
    _selector.validate_preregistration(
        preregistration["path"], preregistration["sha256"]
    )
    scene_ids = payload.get("source_scene_ids")
    inputs = payload.get("inputs")
    source_count = payload.get("source_count")
    if (
        not isinstance(scene_ids, list)
        or not all(isinstance(item, str) and item for item in scene_ids)
        or scene_ids != sorted(scene_ids)
        or len(set(scene_ids)) != len(scene_ids)
        or not isinstance(source_count, int)
        or isinstance(source_count, bool)
        or source_count < _selector.MINIMUM_DISTINCT_SCENES
        or source_count != len(scene_ids)
        or not isinstance(inputs, list)
        or len(inputs) != source_count
        or [item.get("scene_id") if isinstance(item, Mapping) else None for item in inputs]
        != scene_ids
    ):
        raise ValueError("source-only selector scene set differs")
    selector_input_keys = {
        "scene_id", "source_format", "source", "teacher_payload",
        "execution_authority", "source_only_loo_audit_sha256",
    }
    if any(
        not isinstance(item, Mapping)
        or set(item) != selector_input_keys
        or item.get("source_format") not in allowed_source_formats
        or not isinstance(item.get("source_only_loo_audit_sha256"), str)
        or _SHA256.fullmatch(item["source_only_loo_audit_sha256"]) is None
        for item in inputs
    ):
        raise ValueError("source-only selector input lineage differs")
    rows = payload.get("candidate_grid")
    if not isinstance(rows, list) or len(rows) != len(
        _selector.CANDIDATE_GRID_RADIANS
    ):
        raise ValueError("source-only selector candidate grid differs")
    eligible_angles: list[float] = []
    candidate_keys = {
        "maximum_angle_radians",
        "pooled_delta_cosine_sum_vs_o1_0p15",
        "pooled_heldout_scale_observations",
        "pooled_mean_delta_cosine_vs_o1_0p15",
        "pooled_improvement_strict",
        "every_source_scene_nonregression",
        "eligible",
        "per_scene",
    }
    scene_statistic_keys = {
        "scene_id",
        "heldout_scale_observations",
        "delta_cosine_sum_vs_o1_0p15",
        "mean_delta_cosine_vs_o1_0p15",
        "nonregression",
    }
    for angle, row in zip(_selector.CANDIDATE_GRID_RADIANS, rows):
        if not isinstance(row, Mapping) or set(row) != candidate_keys:
            raise ValueError("source-only selector candidate row differs")
        per_scene = row.get("per_scene")
        pooled_delta = _finite(
            row.get("pooled_delta_cosine_sum_vs_o1_0p15"),
            label="selector pooled delta",
        )
        pooled_observations = row.get("pooled_heldout_scale_observations")
        pooled_mean = _finite(
            row.get("pooled_mean_delta_cosine_vs_o1_0p15"),
            label="selector pooled mean delta",
        )
        if (
            row.get("maximum_angle_radians") != angle
            or not isinstance(per_scene, list)
            or len(per_scene) != source_count
            or [item.get("scene_id") if isinstance(item, Mapping) else None for item in per_scene]
            != scene_ids
            or not isinstance(pooled_observations, int)
            or isinstance(pooled_observations, bool)
            or pooled_observations <= 0
            or not math.isclose(
                pooled_mean,
                pooled_delta / pooled_observations,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("source-only selector pooled statistic differs")
        every_nonregression = True
        observed_delta = 0.0
        observed_count = 0
        for scene_row in per_scene:
            if (
                not isinstance(scene_row, Mapping)
                or set(scene_row) != scene_statistic_keys
            ):
                raise ValueError("source-only selector scene statistic differs")
            delta = _finite(
                scene_row.get("delta_cosine_sum_vs_o1_0p15"),
                label="selector source delta",
            )
            observations = scene_row.get("heldout_scale_observations")
            if (
                not isinstance(observations, int)
                or isinstance(observations, bool)
                or observations <= 0
                or not math.isclose(
                    _finite(
                        scene_row.get("mean_delta_cosine_vs_o1_0p15"),
                        label="selector source mean delta",
                    ),
                    delta / observations,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or scene_row.get("nonregression") is not (delta >= 0.0)
            ):
                raise ValueError("source-only selector scene statistic differs")
            observed_delta += delta
            observed_count += observations
            every_nonregression &= delta >= 0.0
        pooled_improvement = pooled_mean > 0.0
        eligible = pooled_improvement and every_nonregression
        if (
            not math.isclose(observed_delta, pooled_delta, rel_tol=1e-12, abs_tol=1e-12)
            or observed_count != pooled_observations
            or row.get("pooled_improvement_strict") is not pooled_improvement
            or row.get("every_source_scene_nonregression") is not every_nonregression
            or row.get("eligible") is not eligible
        ):
            raise ValueError("source-only selector eligibility differs")
        if eligible:
            eligible_angles.append(angle)
    selection = payload.get("selection")
    selection_keys = {
        "global_maximum_angle_radians", "selection_rule",
        "baseline_fallback_used", "one_global_ceiling",
        "per_scene_or_per_query_override_authorized",
    }
    expected_selected = max(eligible_angles, default=_selector.BASELINE_RADIANS)
    if (
        not isinstance(selection, Mapping)
        or set(selection) != selection_keys
        or selection.get("global_maximum_angle_radians") != expected_selected
        or selection.get("selection_rule")
        != "largest_eligible_angle_else_0.15"
        or selection.get("baseline_fallback_used") is not (not eligible_angles)
        or selection.get("one_global_ceiling") is not True
        or selection.get("per_scene_or_per_query_override_authorized") is not False
    ):
        raise ValueError("source-only selector global selection differs")
    return (
        payload,
        {"path": str(source), "sha256": digest},
        expected_selected,
    )


def _validate_teacher_payload(
    path: str | Path,
    expected_sha256: str,
    *,
    scene_id: str,
    base_descriptor_record: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    payload, digest, source = load_torch_mapping(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        label="teacher-agreement v2 payload",
    )
    producer = payload.get("producer")
    if producer == _agreement_v2.ENTRYPOINT_IMPLEMENTATION:
        _agreement_v2.validate_teacher_payload_v2(payload)
    elif producer == _agreement_lowmem.ENTRYPOINT_IMPLEMENTATION:
        _agreement_lowmem.validate_teacher_payload_lowmem(payload)
    else:
        raise ValueError("teacher-agreement v2 producer differs")
    input_authority = payload.get("input_authority")
    if (
        payload.get("scene_id") != scene_id
        or not isinstance(input_authority, Mapping)
        or input_authority.get("base_descriptor") != dict(base_descriptor_record)
        or payload.get("access_audit") != _o1o2.access_audit()
    ):
        raise ValueError("teacher-agreement v2 target lineage differs")
    authority_record = _record(
        payload.get("execution_authority"),
        label="teacher-agreement v2 execution authority",
    )
    authority, _, _ = load_json_object(
        authority_record["path"],
        expected_sha256=authority_record["sha256"],
        label="teacher-agreement v2 execution authority",
    )
    if (
        authority.get("schema") != _agreement_v2.AUTHORITY_SCHEMA
        or authority.get("schema_version") != _agreement_v2.SCHEMA_VERSION
        or authority.get("scene_id") != scene_id
        or authority.get("implementation") != producer
        or authority.get("method_contract_sha256")
        != payload.get("method_contract_sha256")
        or authority.get("query_free_materialization_authorized") is not True
        or authority.get("metric_execution_authorized") is not False
        or authority.get("access_audit") != _o1o2.access_audit()
    ):
        raise ValueError("teacher-agreement v2 execution lineage differs")
    return payload, {"path": str(source), "sha256": digest}


def _typed_stream_hasher(shape: tuple[int, ...]) -> "hashlib._Hash":
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(torch.float32), "shape": list(shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    return digest


def _batch_invariant_cosine_scores(
    descriptor: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Use an explicit D-axis reduction independent of the row batch shape."""

    return (
        descriptor[:, :, None, :] * text_embeddings[None, None, :, :]
    ).sum(dim=-1)


def reliability_candidate_descriptor_batch(
    base: torch.Tensor,
    teacher_mean: torch.Tensor,
    teacher_valid: torch.Tensor,
    retained_view_count: torch.Tensor,
    directional_resultant: torch.Tensor,
    *,
    global_ceiling_radians: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return one row batch and the scales authorized to replace O0 scores."""

    ceiling = _finite(global_ceiling_radians, label="global ceiling")
    if ceiling not in _selector.CANDIDATE_GRID_RADIANS:
        raise ValueError("global ceiling is outside the frozen grid")
    base_unit = F.normalize(torch.as_tensor(base).float(), dim=-1)
    if base_unit.ndim != 3 or base_unit.shape[1] != SCALE_COUNT:
        raise ValueError("candidate base descriptor axis differs")
    attenuation_value = (
        (ceiling - CONSERVATIVE_ANGLE_RADIANS)
        / (MAXIMUM_ANGLE_RADIANS - CONSERVATIVE_ANGLE_RADIANS)
    )
    attenuation = torch.full(
        (base_unit.shape[0],),
        attenuation_value,
        dtype=base_unit.dtype,
        device=base_unit.device,
    )
    fusion = reliability_conditioned_geodesic_fusion(
        base_unit,
        torch.as_tensor(teacher_mean, device=base_unit.device).float(),
        teacher_valid=torch.as_tensor(teacher_valid, device=base_unit.device),
        retained_view_count=torch.as_tensor(
            retained_view_count, device=base_unit.device
        ),
        teacher_view_directional_resultant=torch.as_tensor(
            directional_resultant, device=base_unit.device
        ).float(),
        canonical_field_reliability=attenuation,
    )
    # Count=1 has zero expansion reliability, but the ordinary 0.15-radian
    # O1 trust region remains available exactly as declared by the interface.
    replace = fusion.teacher_applied
    candidate = fusion.descriptor
    if ceiling == CONSERVATIVE_ANGLE_RADIANS:
        teacher = torch.as_tensor(teacher_mean, device=base_unit.device).float()
        exact_o1 = torch.stack(
            [
                geodesic_project(
                    base_unit[:, scale], teacher, CONSERVATIVE_ANGLE_RADIANS
                )
                for scale in range(SCALE_COUNT)
            ],
            dim=1,
        )
        candidate = exact_o1
    candidate = torch.where(replace[..., None], candidate, base_unit)
    audit = {
        "angular_budget_radians": fusion.angular_budget_radians,
        "reliability_score": fusion.reliability_score,
        "expanded_budget": fusion.expanded_budget,
        "interface_fallback": fusion.fallback_to_o0,
    }
    return candidate.contiguous().float(), replace.contiguous(), audit


@torch.inference_mode()
def materialize_scores_lowmem(
    *,
    base_features_by_scale: torch.Tensor,
    global_rows: torch.Tensor,
    teacher_mean: torch.Tensor,
    teacher_valid: torch.Tensor,
    retained_view_count: torch.Tensor,
    directional_resultant: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    o0_positive_scores: torch.Tensor,
    o0_negative_scores: torch.Tensor,
    global_ceiling_radians: float,
    device: torch.device,
    row_batch_size: int = ROW_BATCH_SIZE,
) -> dict[str, Any]:
    """Score a candidate without allocating a scene-wide descriptor tensor."""

    base = torch.as_tensor(base_features_by_scale)
    rows = torch.as_tensor(global_rows)
    n_rows = int(rows.numel())
    positive_text = F.normalize(torch.as_tensor(positive_embeddings).float(), dim=-1)
    negative_text = F.normalize(torch.as_tensor(negative_embeddings).float(), dim=-1)
    positive = torch.as_tensor(o0_positive_scores)
    negative = torch.as_tensor(o0_negative_scores)
    if (
        base.ndim != 3
        or base.shape != (n_rows, SCALE_COUNT, base.shape[-1])
        or base.shape[-1] <= 1
        or not base.is_floating_point()
        or not bool(torch.isfinite(base).all())
        or rows.shape != (n_rows,)
        or rows.dtype != torch.int64
        or (n_rows > 1 and not bool((rows[1:] > rows[:-1]).all()))
        or not isinstance(row_batch_size, int)
        or row_batch_size < 1
        or teacher_mean.shape != (n_rows, base.shape[-1])
        or teacher_valid.shape != (n_rows,)
        or teacher_valid.dtype != torch.bool
        or retained_view_count.shape != (n_rows,)
        or directional_resultant.shape != (n_rows,)
        or positive_text.ndim != 2
        or negative_text.ndim != 2
        or positive_text.shape[1] != base.shape[-1]
        or negative_text.shape[1] != base.shape[-1]
        or positive.ndim != 3
        or negative.ndim != 3
        or positive.shape[:2] != (negative.shape[0], SCALE_COUNT)
        or negative.shape[1] != SCALE_COUNT
        or positive.shape[2] != positive_text.shape[0]
        or negative.shape[2] != negative_text.shape[0]
        or positive.dtype != torch.float32
        or negative.dtype != torch.float32
        or (n_rows > 0 and int(rows[-1]) >= positive.shape[0])
        or not bool(torch.isfinite(positive).all())
        or not bool(torch.isfinite(negative).all())
    ):
        raise ValueError("candidate score materialization axes differ")
    candidate_positive = positive.clone()
    candidate_negative = negative.clone()
    positive_text = positive_text.to(device)
    negative_text = negative_text.to(device)
    descriptor_hasher = _typed_stream_hasher(
        (n_rows, SCALE_COUNT, int(base.shape[-1]))
    )
    rows_with_replacement = 0
    scales_replaced = 0
    expanded_rows = 0
    maximum_batch_rows_observed = 0
    for start in range(0, n_rows, row_batch_size):
        stop = min(n_rows, start + row_batch_size)
        maximum_batch_rows_observed = max(maximum_batch_rows_observed, stop - start)
        base_batch = base[start:stop].to(device)
        teacher_batch = teacher_mean[start:stop].to(device)
        valid_batch = teacher_valid[start:stop].to(device)
        descriptor, replace, audit = reliability_candidate_descriptor_batch(
            base_batch,
            teacher_batch,
            valid_batch,
            retained_view_count[start:stop].to(device),
            directional_resultant[start:stop].to(device),
            global_ceiling_radians=global_ceiling_radians,
        )
        descriptor_cpu = descriptor.cpu()
        descriptor_hasher.update(descriptor_cpu.numpy().tobytes(order="C"))
        replace_cpu = replace.cpu()
        output_rows = rows[start:stop]
        if global_ceiling_radians == CONSERVATIVE_ANGLE_RADIANS:
            # Reuse the frozen O1 scorer and its active-row einsum shape so a
            # 0.15 candidate is byte-identical to existing O1 wherever the
            # interface authorizes replacement.  Single-view rows receive
            # ordinary O1; only invalid and ambiguous routes retain O0.
            exact_o1, _ = _o1o2._score_descriptors(
                base=base_batch,
                teacher_mean=teacher_batch,
                teacher_valid=valid_batch,
            )
            positive_device = torch.zeros(
                stop - start,
                SCALE_COUNT,
                positive_text.shape[0],
                dtype=torch.float32,
                device=device,
            )
            negative_device = torch.zeros(
                stop - start,
                SCALE_COUNT,
                negative_text.shape[0],
                dtype=torch.float32,
                device=device,
            )
            if bool(valid_batch.any()):
                positive_device[valid_batch] = torch.einsum(
                    "bsd,qd->bsq", exact_o1[valid_batch], positive_text
                )
                negative_device[valid_batch] = torch.einsum(
                    "bsd,qd->bsq", exact_o1[valid_batch], negative_text
                )
            positive_batch = positive_device.cpu()
            negative_batch = negative_device.cpu()
        else:
            positive_batch = _batch_invariant_cosine_scores(
                descriptor, positive_text
            ).cpu()
            negative_batch = _batch_invariant_cosine_scores(
                descriptor, negative_text
            ).cpu()
        candidate_positive[output_rows] = torch.where(
            replace_cpu[..., None],
            positive_batch,
            candidate_positive[output_rows],
        )
        candidate_negative[output_rows] = torch.where(
            replace_cpu[..., None],
            negative_batch,
            candidate_negative[output_rows],
        )
        rows_with_replacement += int(replace_cpu.any(dim=1).sum())
        scales_replaced += int(replace_cpu.sum())
        expanded_rows += int(
            (
                audit["expanded_budget"]
                & replace.any(dim=1)
            ).sum().cpu()
        )
        del descriptor, descriptor_cpu, replace, replace_cpu, audit
        del base_batch, teacher_batch, valid_batch
    return {
        "positive_scores": candidate_positive.contiguous(),
        "negative_scores": candidate_negative.contiguous(),
        "descriptor_sha256": descriptor_hasher.hexdigest(),
        "rows_with_score_replacement": rows_with_replacement,
        "scales_with_score_replacement": scales_replaced,
        "rows_with_expanded_budget": expanded_rows,
        "maximum_batch_rows_observed": maximum_batch_rows_observed,
    }


def _candidate_cache(
    template: Mapping[str, Any],
    scores: torch.Tensor,
    *,
    teacher_record: Mapping[str, str],
    selector_record: Mapping[str, str],
    text_record: Mapping[str, str],
    descriptor_sha256: str,
    global_ceiling_radians: float,
) -> dict[str, Any]:
    payload = {key: value for key, value in template.items() if key != "authority"}
    payload["query_scores"] = scores.contiguous().float()
    authority = copy.deepcopy(template["authority"])
    authority["contract"] = _o1o2.RAW_AUTHORITY_CONTRACT
    authority["score_semantics"] = "raw_independent_normalized_cosine"
    # Keep the frozen FP32 cache formula token byte-identical for the shared
    # evaluator; the descriptor-axis provenance records how the descriptor
    # itself was formed.
    authority["score_formula"] = (
        "l2_normalize(descriptor) @ l2_normalize(text_embedding).T"
    )
    authority["score_implementation"] = str(Path(__file__).resolve())
    authority["score_dtype"] = "torch.float32"
    authority["query_scores_sha256"] = _o1o2.tensor_sha256_typed(
        payload["query_scores"]
    )
    authority["descriptor_axis"]["features_by_scale_sha256"] = descriptor_sha256
    authority["descriptor_axis"]["oracle"] = (
        "source_only_global_reliability_conditioned_O1_to_O2"
    )
    authority["descriptor_axis"]["execution_representation"] = (
        "row_streamed_three_scale_candidate_no_full_descriptor"
    )
    authority["source_artifacts"]["descriptor_cache"] = dict(teacher_record)
    authority["source_artifacts"]["text_query_cache"] = dict(text_record)
    authority["source_artifacts"]["global_ceiling_selector_authority"] = dict(
        selector_record
    )
    authority["source_artifacts"]["materializer_source"] = file_record(
        Path(__file__).resolve()
    )
    authority["calibration_constraints"]["benchmark_metrics_opened"] = False
    authority["global_reliability_ceiling_radians"] = global_ceiling_radians
    authority["reliability_geodesic_budget_contract_sha256"] = (
        RELIABILITY_GEODESIC_BUDGET_CONTRACT_SHA256
    )
    payload["authority"] = authority
    return payload


def prepare_inputs(
    authority_path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    raw, digest, source = load_json_object(
        authority_path,
        expected_sha256=expected_sha256,
        label="reliability candidate execution authority",
    )
    required = {
        "schema", "schema_version", "status", "scene_id", "implementation",
        "method_contract", "method_contract_sha256", "inputs", "outputs",
        "execution", "query_free_materialization_authorized",
        "metric_execution_authorized", "access_audit",
    }
    if (
        set(raw) != required
        or raw.get("schema") != AUTHORITY_SCHEMA
        or raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("status")
        != "authorized_source_only_global_reliability_candidate"
        or raw.get("implementation") != file_record(Path(__file__).resolve())
        or raw.get("method_contract") != method_contract()
        or raw.get("method_contract_sha256") != METHOD_CONTRACT_SHA256
        or raw.get("query_free_materialization_authorized") is not True
        or raw.get("metric_execution_authorized") is not False
        or raw.get("access_audit") != EXPECTED_ACCESS_AUDIT
    ):
        raise ValueError("reliability candidate execution authority differs")
    scene_id = raw.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id or scene_id.strip() != scene_id:
        raise ValueError("reliability candidate scene id differs")
    execution = raw.get("execution")
    physical_gpu = execution.get("physical_gpu") if isinstance(execution, Mapping) else None
    expected_execution = {
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": str(physical_gpu),
        "program_device": "cuda:0",
        "row_batch_size": ROW_BATCH_SIZE,
        "thermal_safety_owner": "external_300s_hard88_guard",
        "maximum_temperature_c": 88,
    }
    if physical_gpu not in SUPPORTED_PHYSICAL_GPUS or execution != expected_execution:
        raise ValueError("reliability candidate execution device differs")
    expected_inputs = {
        "base_descriptor", "teacher_agreement_v2", "global_ceiling_selector",
        "positive_text", "negative_text", "o0_positive", "o0_negative",
    }
    inputs = raw.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise ValueError("reliability candidate input records differ")
    records = {
        name: _record(inputs[name], label=f"candidate {name}")
        for name in sorted(expected_inputs)
    }
    base, rows = _o1o2._validate_base_descriptor_general(
        Path(records["base_descriptor"]["path"]),
        records["base_descriptor"]["sha256"],
    )
    teacher, teacher_record = _validate_teacher_payload(
        records["teacher_agreement_v2"]["path"],
        records["teacher_agreement_v2"]["sha256"],
        scene_id=scene_id,
        base_descriptor_record=records["base_descriptor"],
    )
    if not torch.equal(teacher["global_rows"], rows):
        raise ValueError("teacher/base accepted row lineage differs")
    selector, selector_record, ceiling = _validate_selector_candidate(
        records["global_ceiling_selector"]["path"],
        records["global_ceiling_selector"]["sha256"],
    )
    positive_raw, _, _ = load_torch_mapping(
        records["positive_text"]["path"],
        expected_sha256=records["positive_text"]["sha256"],
        map_location="cpu",
        label="candidate positive text bank",
    )
    negative_raw, _, _ = load_torch_mapping(
        records["negative_text"]["path"],
        expected_sha256=records["negative_text"]["sha256"],
        map_location="cpu",
        label="candidate negative text bank",
    )
    positive_queries = list(positive_raw.get("queries", []))
    positive_embeddings = _o1o2._validate_text_bank(
        positive_raw, expected_queries=positive_queries
    )
    negative_embeddings = _o1o2._validate_text_bank(
        negative_raw, expected_queries=list(_o1o2.NEGATIVE_QUERIES)
    )
    o0_positive, _, _ = load_torch_mapping(
        records["o0_positive"]["path"],
        expected_sha256=records["o0_positive"]["sha256"],
        map_location="cpu",
        label="candidate O0 positive cache",
    )
    o0_negative, _, _ = load_torch_mapping(
        records["o0_negative"]["path"],
        expected_sha256=records["o0_negative"]["sha256"],
        map_location="cpu",
        label="candidate O0 negative cache",
    )
    renderer_sha = o0_positive.get("renderer_geometry_checkpoint_sha256")
    if not isinstance(renderer_sha, str) or _SHA256.fullmatch(renderer_sha) is None:
        raise ValueError("candidate O0 renderer lineage differs")
    _o1o2._validate_o0_pair(
        o0_positive,
        o0_negative,
        base=base,
        positive_queries=positive_queries,
        renderer_sha256=renderer_sha,
    )
    outputs = raw.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "positive", "negative", "result"
    }:
        raise ValueError("reliability candidate outputs differ")
    resolved_outputs = {}
    for name, path in outputs.items():
        resolved = str(Path(str(path)).expanduser().resolve())
        if str(path) != resolved:
            raise ValueError(f"candidate {name} output must be canonical absolute")
        resolved_outputs[name] = resolved
    return {
        "authority": dict(raw),
        "authority_record": {"path": str(source), "sha256": digest},
        "records": records,
        "base": base,
        "rows": rows,
        "teacher": teacher,
        "teacher_record": teacher_record,
        "selector": selector,
        "selector_record": selector_record,
        "global_ceiling_radians": ceiling,
        "positive_embeddings": positive_embeddings,
        "negative_embeddings": negative_embeddings,
        "o0_positive": o0_positive,
        "o0_negative": o0_negative,
        "outputs": resolved_outputs,
    }


def build_authority(args: argparse.Namespace) -> dict[str, Any]:
    authority_output = _new(args.authority_output, label="candidate authority")
    output_root = Path(args.output_dir).expanduser().resolve()
    if str(output_root) != str(args.output_dir):
        raise ValueError("candidate output directory must be canonical absolute")
    outputs = {
        "positive": str(_new(output_root / f"{args.scene_id}_reliability_candidate_positive.pt", label="candidate positive output")),
        "negative": str(_new(output_root / f"{args.scene_id}_reliability_candidate_negative.pt", label="candidate negative output")),
        "result": str(_new(output_root / f"{args.scene_id}_reliability_candidate_result.json", label="candidate result output")),
    }
    names = (
        "base_descriptor", "teacher_agreement_v2", "global_ceiling_selector",
        "positive_text", "negative_text", "o0_positive", "o0_negative",
    )
    inputs = {
        name: _record_args(
            getattr(args, name), getattr(args, f"{name}_sha256"), label=name
        )
        for name in names
    }
    if args.physical_gpu not in SUPPORTED_PHYSICAL_GPUS:
        raise ValueError("candidate physical GPU differs")
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "authorized_source_only_global_reliability_candidate",
        "scene_id": args.scene_id,
        "implementation": file_record(Path(__file__).resolve()),
        "method_contract": method_contract(),
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "inputs": inputs,
        "outputs": outputs,
        "execution": {
            "physical_gpu": args.physical_gpu,
            "cuda_visible_devices": str(args.physical_gpu),
            "program_device": "cuda:0",
            "row_batch_size": ROW_BATCH_SIZE,
            "thermal_safety_owner": "external_300s_hard88_guard",
            "maximum_temperature_c": 88,
        },
        "query_free_materialization_authorized": True,
        "metric_execution_authorized": False,
        "access_audit": dict(EXPECTED_ACCESS_AUDIT),
    }
    write_frozen_json(authority_output, authority)
    record = file_record(authority_output)
    prepare_inputs(record["path"], expected_sha256=record["sha256"])
    return {"status": "authorized", "authority": record, "outputs": outputs}


def validate_runtime_device(
    execution: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    cuda_available: bool | None = None,
) -> torch.device:
    physical = execution.get("physical_gpu")
    expected = str(physical)
    environment = os.environ if environ is None else environ
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    if (
        physical not in SUPPORTED_PHYSICAL_GPUS
        or execution.get("cuda_visible_devices") != expected
        or execution.get("program_device") != "cuda:0"
        or environment.get("CUDA_VISIBLE_DEVICES") != expected
        or available is not True
    ):
        raise RuntimeError("candidate runtime CUDA device authority differs")
    return torch.device("cuda:0")


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    prepared = prepare_inputs(
        args.execution_authority,
        expected_sha256=args.execution_authority_sha256,
    )
    for name, path in prepared["outputs"].items():
        _new(path, label=f"candidate {name} output")
    device = validate_runtime_device(prepared["authority"]["execution"])
    teacher = prepared["teacher"]
    scored = materialize_scores_lowmem(
        base_features_by_scale=prepared["base"]["features_by_scale"],
        global_rows=prepared["rows"],
        teacher_mean=teacher["teacher_mean"],
        teacher_valid=teacher["teacher_valid"],
        retained_view_count=teacher["retained_view_count"],
        directional_resultant=teacher[VIEW_AGREEMENT_SCALAR],
        positive_embeddings=prepared["positive_embeddings"],
        negative_embeddings=prepared["negative_embeddings"],
        o0_positive_scores=prepared["o0_positive"]["query_scores"],
        o0_negative_scores=prepared["o0_negative"]["query_scores"],
        global_ceiling_radians=prepared["global_ceiling_radians"],
        device=device,
        row_batch_size=ROW_BATCH_SIZE,
    )
    positive = _candidate_cache(
        prepared["o0_positive"],
        scored["positive_scores"],
        teacher_record=prepared["teacher_record"],
        selector_record=prepared["selector_record"],
        text_record=prepared["records"]["positive_text"],
        descriptor_sha256=scored["descriptor_sha256"],
        global_ceiling_radians=prepared["global_ceiling_radians"],
    )
    negative = _candidate_cache(
        prepared["o0_negative"],
        scored["negative_scores"],
        teacher_record=prepared["teacher_record"],
        selector_record=prepared["selector_record"],
        text_record=prepared["records"]["negative_text"],
        descriptor_sha256=scored["descriptor_sha256"],
        global_ceiling_radians=prepared["global_ceiling_radians"],
    )
    write_torch_noclobber(prepared["outputs"]["positive"], positive)
    positive_record = file_record(prepared["outputs"]["positive"])
    write_torch_noclobber(prepared["outputs"]["negative"], negative)
    negative_record = file_record(prepared["outputs"]["negative"])
    result = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_source_only_global_reliability_candidate",
        "scene_id": prepared["authority"]["scene_id"],
        "execution_authority": prepared["authority_record"],
        "teacher_agreement_v2": prepared["teacher_record"],
        "global_ceiling_selector": prepared["selector_record"],
        "global_ceiling_radians": prepared["global_ceiling_radians"],
        "outputs": {"positive": positive_record, "negative": negative_record},
        "accepted_rows": int(prepared["rows"].numel()),
        "rows_with_score_replacement": scored["rows_with_score_replacement"],
        "scales_with_score_replacement": scored["scales_with_score_replacement"],
        "rows_with_expanded_budget": scored["rows_with_expanded_budget"],
        "maximum_batch_rows_observed": scored["maximum_batch_rows_observed"],
        "descriptor_sha256": scored["descriptor_sha256"],
        "elapsed_seconds": time.monotonic() - started,
        "method_contract_sha256": METHOD_CONTRACT_SHA256,
        "access_audit": dict(EXPECTED_ACCESS_AUDIT),
        "metric_execution_authorized": False,
        "metric_executed": False,
        "next_gate": "separate_one_shot_frozen_metric_execution_authority",
    }
    write_frozen_json(prepared["outputs"]["result"], result)
    return {**result, "result": file_record(prepared["outputs"]["result"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-authority")
    build.add_argument("--scene-id", required=True)
    for name in (
        "base_descriptor", "teacher_agreement_v2", "global_ceiling_selector",
        "positive_text", "negative_text", "o0_positive", "o0_negative",
    ):
        option = name.replace("_", "-")
        build.add_argument(f"--{option}", required=True)
        build.add_argument(f"--{option}-sha256", required=True)
    build.add_argument("--physical-gpu", type=int, choices=SUPPORTED_PHYSICAL_GPUS, required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--authority-output", required=True)
    build.set_defaults(handler=build_authority)
    run = commands.add_parser("materialize")
    run.add_argument("--execution-authority", required=True)
    run.add_argument("--execution-authority-sha256", required=True)
    run.set_defaults(handler=materialize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHORITY_SCHEMA",
    "EXPECTED_ACCESS_AUDIT",
    "METHOD_CONTRACT_SHA256",
    "RESULT_SCHEMA",
    "ROW_BATCH_SIZE",
    "build_authority",
    "materialize_scores_lowmem",
    "method_contract",
    "prepare_inputs",
    "reliability_candidate_descriptor_batch",
    "validate_runtime_device",
]
