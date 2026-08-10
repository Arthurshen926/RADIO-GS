#!/usr/bin/env python3
"""Materialize the O1/O2 stream plus durable top-view agreement.

This is an independent, hash-bound v2 entrypoint.  It delegates input
validation, projection, top-view selection, O1/O2 descriptor construction,
and score materialization to :mod:`materialize_lerf_o1_o2_streaming`.  The
only numerical addition is one query-free scalar computed after the existing
top-4 view axis is canonicalized and before the teacher sum is normalized::

    ||sum_j unit(top4_teacher_j)||_2 / retained_view_count

The scalar is retained as float32 and receives a typed tensor hash.  Existing
v1 entrypoints and their authorities/outputs are not imported through or
modified by this file.  The reliability-conditioned 0.75-radian ceiling is
explicitly *not* authorized here; selecting such a ceiling still requires a
separate source-only preregistration gate.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from radio_gs.interfaces.lerf_reliability_geodesic_budget import (
    RELIABILITY_GEODESIC_BUDGET_CONTRACT_SHA256,
    VIEW_AGREEMENT_SCALAR,
    VIEW_AGREEMENT_SHA256_FIELD,
)
from radio_gs.scripts import materialize_lerf_o1_o2_streaming as _core
from radio_gs.utils.immutable_artifacts import canonical_json_sha256, file_record


AUTHORITY_SCHEMA = "radio_gs.lerf_o1_o2_teacher_agreement_streaming_execution.v2"
MEAN_SCHEMA = "radio_gs.lerf_source_teacher_mean_siglip.v2"
RESULT_SCHEMA = "radio_gs.lerf_o1_o2_teacher_agreement_streaming_result.v2"
SCHEMA_VERSION = 2
PACING_SECONDS_PER_PROJECTION_BATCH = 0.0
AGREEMENT_DTYPE = torch.float32
AGREEMENT_ROW_CHUNK = 2048
LOO_AUDIT_FIELD = "source_only_leave_one_view_out_ceiling_audit"
LOO_AUDIT_SHA256_FIELD = f"{LOO_AUDIT_FIELD}_sha256"
LOO_CEILING_CANDIDATES_RADIANS = (0.15, 0.3, 0.45, 0.6, 0.75)

# Capture the untouched v1 callables and file identity before the v2
# installation changes the delegated module globals inside this process.
CORE_IMPLEMENTATION = file_record(Path(_core.__file__).resolve())
ENTRYPOINT_IMPLEMENTATION = file_record(Path(__file__).resolve())
_CORE_METHOD_CONTRACT = _core.method_contract
_CORE_PROJECT_VIEW = _core._project_view
_CORE_CANONICALIZE_VIEW_AXIS = _core._canonicalize_view_axis
_CORE_RAW_CACHE = _core._raw_cache
_CORE_MATERIALIZE = _core.materialize
_CORE_PREPARE_INPUTS = _core.prepare_inputs
_CORE_WRITE_TORCH_NOCLOBBER = _core.write_torch_noclobber

# Populated exactly once per materialization, after top-view canonicalization.
# It is process-local transient state; no per-view descriptor is durable.
_agreement_state: dict[str, Any] | None = None
_base_descriptor_state: torch.Tensor | None = None


def method_contract() -> dict[str, Any]:
    """Return unchanged O1/O2 semantics plus the minimal v2 scalar contract."""

    contract = copy.deepcopy(_CORE_METHOD_CONTRACT())
    contract.update(
        {
            "schema": AUTHORITY_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "streaming_entrypoint_implementation": dict(ENTRYPOINT_IMPLEMENTATION),
            "streaming_core_implementation": dict(CORE_IMPLEMENTATION),
            "projection_pacing_seconds_per_batch": (
                PACING_SECONDS_PER_PROJECTION_BATCH
            ),
            "projection_pacing_affects_method_numerics": False,
            "thermal_safety_owner": "external_300s_hard88_guard",
            "execution_device_authority": {
                "implemented_physical_gpu": 0,
                "required_cuda_visible_devices": "0",
                "program_device": "cuda:0",
                "other_physical_gpu_authorized": False,
            },
            "teacher_payload": {
                "schema": MEAN_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "additional_tensor": VIEW_AGREEMENT_SCALAR,
                "additional_tensor_dtype": str(AGREEMENT_DTYPE),
                "additional_tensor_shape": "accepted_rows",
                "additional_tensor_invalid_value": "exact_zero",
                "additional_tensor_hash": VIEW_AGREEMENT_SHA256_FIELD,
                "additional_tensor_hash_semantics": "typed_tensor_sha256",
                "formula": (
                    "l2_norm(sum(top4_retained_unit_teacher_descriptors))"
                    "/retained_view_count"
                ),
                "calculation_point": (
                    "after_top4_axis_canonicalization_before_teacher_sum_normalization"
                ),
                "top4_descriptors_durable": False,
                "query_independent": True,
            },
            "source_only_leave_one_view_out_ceiling_audit": {
                "field": LOO_AUDIT_FIELD,
                "hash_field": LOO_AUDIT_SHA256_FIELD,
                "hash_semantics": "canonical_json_sha256",
                "candidate_maximum_angles_radians": list(
                    LOO_CEILING_CANDIDATES_RADIANS
                ),
                "prediction": (
                    "per_retained_view_direction_from_normalized_mean_of_other"
                    "_retained_unit_teacher_directions"
                ),
                "candidate_budget": (
                    "0.15+(candidate_ceiling-0.15)*sqrt("
                    "loo_count_sufficiency*loo_directional_resultant)"
                ),
                "evaluation_unit": "heldout_source_view_x_o0_scale",
                "baseline": "same_prediction_with_0.15_radian_ceiling",
                "durable_per_view_descriptors": False,
                "query_independent": True,
                "source_only_diagnostic": True,
                "target_candidate_authorization": False,
            },
            "reliability_geodesic_budget_contract_sha256": (
                RELIABILITY_GEODESIC_BUDGET_CONTRACT_SHA256
            ),
            "reliability_budget_candidate_materialized": False,
            "reliability_budget_maximum_angle_authorized": False,
            "reliability_budget_next_gate": (
                "source_only_global_ceiling_preregistration_without_target_metrics"
            ),
        }
    )
    return contract


METHOD_CONTRACT_SHA256 = canonical_json_sha256(method_contract())


def directional_resultant_from_canonical_top_views(
    top_descriptors: torch.Tensor,
    top_frame_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return resultant and count for an already canonicalized top-4 axis.

    The projected descriptors are quantized to FP16 by the v1 core.  They are
    normalized again in float32 here so the persisted statistic has the exact
    declared unit-direction semantics while the existing teacher mean remains
    byte-for-byte on its original computation path.
    """

    descriptors = torch.as_tensor(top_descriptors)
    frame_ids = torch.as_tensor(top_frame_ids)
    if (
        descriptors.ndim != 3
        or descriptors.shape[1] != _core.TOP_VIEW_COUNT
        or descriptors.shape[2] != 1536
        or not descriptors.is_floating_point()
        or not bool(torch.isfinite(descriptors).all())
        or frame_ids.shape != descriptors.shape[:2]
        or frame_ids.dtype == torch.bool
    ):
        raise ValueError("canonical top-view tensors differ")
    mask = frame_ids >= 0
    counts = mask.sum(dim=1)
    # Do not promote the entire durable top-4 tensor to float32 at once.  For
    # a large scene that would add several GiB of transient host memory on top
    # of the v1 buffers.  This row chunking changes neither the reduction axis
    # nor its order within a row.
    resultant = torch.zeros(descriptors.shape[0], dtype=AGREEMENT_DTYPE)
    for start in range(0, descriptors.shape[0], AGREEMENT_ROW_CHUNK):
        stop = min(descriptors.shape[0], start + AGREEMENT_ROW_CHUNK)
        selected_raw = descriptors[start:stop]
        selected_mask = mask[start:stop]
        if bool((selected_raw[~selected_mask] != 0).any()):
            raise ValueError("unretained teacher descriptor must be exact zero")
        selected = selected_raw.float()
        selected_norm = torch.linalg.vector_norm(selected, dim=-1)
        if bool((selected_norm[selected_mask] <= 0.0).any()):
            raise ValueError("retained teacher descriptor must have nonzero norm")
        unit = F.normalize(selected, dim=-1)
        direction_sum = (unit * selected_mask[:, :, None]).sum(dim=1)
        resultant[start:stop] = torch.linalg.vector_norm(
            direction_sum, dim=-1
        ) / counts[start:stop].clamp_min(1).float()
    # Roundoff can exceed one by a few ulps for perfectly aligned views.
    resultant = resultant.clamp(0.0, 1.0).to(AGREEMENT_DTYPE).contiguous()
    resultant[counts == 0] = 0.0
    return resultant, counts.to(torch.uint8).contiguous()


def source_only_leave_one_view_out_ceiling_audit(
    top_descriptors: torch.Tensor,
    top_frame_ids: torch.Tensor,
    o0_descriptor_by_scale: torch.Tensor,
) -> dict[str, Any]:
    """Summarize a query-free LOO ceiling grid while views are transient.

    For each retained teacher direction, the other retained directions predict
    it.  Every candidate uses the deployed reliability formula, but the
    ceiling values here are diagnostics only.  Cosine sums and counts are
    retained so a later authority can compute a pooled source result without
    reopening per-view descriptors; per-scene means support the mandatory
    every-source-scene nonregression check.
    """

    descriptors = torch.as_tensor(top_descriptors)
    frame_ids = torch.as_tensor(top_frame_ids)
    base = torch.as_tensor(o0_descriptor_by_scale)
    if (
        descriptors.ndim != 3
        or descriptors.shape[1] != _core.TOP_VIEW_COUNT
        or not descriptors.is_floating_point()
        or frame_ids.shape != descriptors.shape[:2]
        or frame_ids.dtype == torch.bool
        or base.shape != (descriptors.shape[0], 3, descriptors.shape[2])
        or not base.is_floating_point()
        or not bool(torch.isfinite(descriptors).all())
        or not bool(torch.isfinite(base).all())
    ):
        raise ValueError("source-only LOO audit tensors differ")
    mask = frame_ids >= 0
    counts = mask.sum(dim=1)
    candidate_sums = torch.zeros(
        len(LOO_CEILING_CANDIDATES_RADIANS), dtype=torch.float64
    )
    heldout_predictions = 0
    heldout_scale_observations = 0
    rows_with_loo = 0
    rows_with_expansion_evidence = 0
    loo_cosine_sum = 0.0
    loo_angular_error_sum = 0.0
    eps = 8.0 * torch.finfo(torch.float32).eps

    for start in range(0, descriptors.shape[0], AGREEMENT_ROW_CHUNK):
        stop = min(descriptors.shape[0], start + AGREEMENT_ROW_CHUNK)
        selected_raw = descriptors[start:stop]
        selected_mask = mask[start:stop]
        if bool((selected_raw[~selected_mask] != 0).any()):
            raise ValueError("unretained teacher descriptor must be exact zero")
        unit_views = F.normalize(selected_raw.float(), dim=-1)
        base_unit = F.normalize(base[start:stop].float(), dim=-1)
        base_norm = torch.linalg.vector_norm(base[start:stop].float(), dim=-1)
        if bool((base_norm <= 0.0).any()):
            raise ValueError("O0 descriptor must have nonzero norm")
        selected_counts = counts[start:stop]
        view_sum = (unit_views * selected_mask[:, :, None]).sum(dim=1)
        row_has_loo = torch.zeros(stop - start, dtype=torch.bool)

        for heldout_index in range(_core.TOP_VIEW_COUNT):
            heldout_valid = selected_mask[:, heldout_index] & (
                selected_counts >= 2
            )
            if not bool(heldout_valid.any()):
                continue
            heldout = unit_views[:, heldout_index]
            loo_count = (selected_counts - 1).clamp_min(1)
            loo_sum = view_sum - heldout
            loo_norm = torch.linalg.vector_norm(loo_sum, dim=-1)
            prediction_valid = heldout_valid & (loo_norm > eps)
            if not bool(prediction_valid.any()):
                continue
            row_has_loo |= prediction_valid
            loo_direction = loo_sum / loo_norm[:, None].clamp_min(eps)
            loo_to_heldout = (loo_direction * heldout).sum(dim=-1).clamp(-1.0, 1.0)
            valid_loo_cosine = loo_to_heldout[prediction_valid]
            prediction_count = int(prediction_valid.sum())
            heldout_predictions += prediction_count
            loo_cosine_sum += float(valid_loo_cosine.double().sum())
            loo_angular_error_sum += float(
                torch.acos(valid_loo_cosine).double().sum()
            )

            loo_agreement = (
                loo_norm / loo_count.float()
            ).clamp(0.0, 1.0)
            count_sufficiency = (
                (loo_count.float() - 1.0) / (_core.TOP_VIEW_COUNT - 1)
            ).clamp(0.0, 1.0)
            reliability = torch.sqrt(
                (count_sufficiency * loo_agreement).clamp_min(0.0)
            )
            dot_base_loo = (
                base_unit * loo_direction[:, None, :]
            ).sum(dim=-1).clamp(-1.0, 1.0)
            dot_base_heldout = (
                base_unit * heldout[:, None, :]
            ).sum(dim=-1).clamp(-1.0, 1.0)
            requested = torch.acos(dot_base_loo)
            tangent_norm = torch.sqrt(
                (1.0 - dot_base_loo.square()).clamp_min(0.0)
            )
            tangent_dot_heldout = (
                loo_to_heldout[:, None]
                - dot_base_loo * dot_base_heldout
            ) / tangent_norm.clamp_min(eps)
            stable_route = tangent_norm > eps
            for candidate_index, ceiling in enumerate(
                LOO_CEILING_CANDIDATES_RADIANS
            ):
                budget = 0.15 + (float(ceiling) - 0.15) * reliability
                step = torch.minimum(requested, budget[:, None])
                predicted_cosine = (
                    torch.cos(step) * dot_base_heldout
                    + torch.sin(step) * tangent_dot_heldout
                ).clamp(-1.0, 1.0)
                # The deployed interface falls back to O0 for an ambiguous
                # antipodal route; a same-direction route is also unchanged.
                predicted_cosine = torch.where(
                    stable_route, predicted_cosine, dot_base_heldout
                )
                candidate_sums[candidate_index] += predicted_cosine[
                    prediction_valid
                ].double().sum()
            heldout_scale_observations += prediction_count * base.shape[1]

        rows_with_loo += int(row_has_loo.sum())
        rows_with_expansion_evidence += int(
            (row_has_loo & (selected_counts >= 3)).sum()
        )

    if heldout_predictions <= 0 or heldout_scale_observations <= 0:
        raise ValueError("source-only LOO audit has no valid prediction")
    baseline_sum = float(candidate_sums[0])
    candidates = []
    for ceiling, cosine_sum_tensor in zip(
        LOO_CEILING_CANDIDATES_RADIANS, candidate_sums
    ):
        cosine_sum = float(cosine_sum_tensor)
        delta_sum = cosine_sum - baseline_sum
        candidates.append(
            {
                "maximum_angle_radians": float(ceiling),
                "heldout_scale_observations": heldout_scale_observations,
                "cosine_sum": cosine_sum,
                "mean_cosine": cosine_sum / heldout_scale_observations,
                "delta_cosine_sum_vs_o1_0p15": delta_sum,
                "mean_delta_cosine_vs_o1_0p15": (
                    delta_sum / heldout_scale_observations
                ),
                "scene_nonregression_vs_o1_0p15": delta_sum >= 0.0,
            }
        )
    return {
        "schema": "radio_gs.source_only_loo_reliability_ceiling_audit.v1",
        "schema_version": 1,
        "query_independent": True,
        "target_images_labels_masks_metrics_opened": False,
        "candidate_role": "source_only_diagnostic_not_target_authorization",
        "target_candidate_authorized": False,
        "retained_view_capacity": _core.TOP_VIEW_COUNT,
        "rows": int(descriptors.shape[0]),
        "rows_with_valid_loo_prediction": rows_with_loo,
        "rows_with_expansion_evidence": rows_with_expansion_evidence,
        "heldout_predictions": heldout_predictions,
        "heldout_scale_observations": heldout_scale_observations,
        "loo_direction_mean_cosine": loo_cosine_sum / heldout_predictions,
        "loo_direction_mean_angular_error_radians": (
            loo_angular_error_sum / heldout_predictions
        ),
        "candidate_baseline_maximum_angle_radians": 0.15,
        "candidates": candidates,
        "cross_scene_gate": {
            "pooled_statistic": (
                "sum(delta_cosine_sum_vs_o1_0p15)/"
                "sum(heldout_scale_observations)"
            ),
            "pooled_improvement_required": True,
            "every_source_scene_nonregression_required": True,
            "one_global_ceiling_required": True,
            "preregister_before_target_metric": True,
        },
    }


def validate_source_only_loo_ceiling_audit(value: Mapping[str, Any]) -> None:
    """Validate the complete durable LOO summary without reopening views."""

    required = {
        "schema",
        "schema_version",
        "query_independent",
        "target_images_labels_masks_metrics_opened",
        "candidate_role",
        "target_candidate_authorized",
        "retained_view_capacity",
        "rows",
        "rows_with_valid_loo_prediction",
        "rows_with_expansion_evidence",
        "heldout_predictions",
        "heldout_scale_observations",
        "loo_direction_mean_cosine",
        "loo_direction_mean_angular_error_radians",
        "candidate_baseline_maximum_angle_radians",
        "candidates",
        "cross_scene_gate",
    }
    cross_scene_gate = {
        "pooled_statistic": (
            "sum(delta_cosine_sum_vs_o1_0p15)/"
            "sum(heldout_scale_observations)"
        ),
        "pooled_improvement_required": True,
        "every_source_scene_nonregression_required": True,
        "one_global_ceiling_required": True,
        "preregister_before_target_metric": True,
    }
    rows = value.get("rows")
    rows_loo = value.get("rows_with_valid_loo_prediction")
    rows_expansion = value.get("rows_with_expansion_evidence")
    heldout = value.get("heldout_predictions")
    observations = value.get("heldout_scale_observations")
    candidates = value.get("candidates")
    loo_cosine = value.get("loo_direction_mean_cosine")
    loo_angle = value.get("loo_direction_mean_angular_error_radians")
    if (
        set(value) != required
        or value.get("schema")
        != "radio_gs.source_only_loo_reliability_ceiling_audit.v1"
        or value.get("schema_version") != 1
        or value.get("query_independent") is not True
        or value.get("target_images_labels_masks_metrics_opened") is not False
        or value.get("candidate_role")
        != "source_only_diagnostic_not_target_authorization"
        or value.get("target_candidate_authorized") is not False
        or value.get("retained_view_capacity") != _core.TOP_VIEW_COUNT
        or not isinstance(rows, int)
        or not isinstance(rows_loo, int)
        or not isinstance(rows_expansion, int)
        or not isinstance(heldout, int)
        or not isinstance(observations, int)
        or not (0 < rows_loo <= rows)
        or not (0 <= rows_expansion <= rows_loo)
        or heldout <= 0
        or observations != heldout * 3
        or not isinstance(loo_cosine, float)
        or not math.isfinite(loo_cosine)
        or not -1.0 <= loo_cosine <= 1.0
        or not isinstance(loo_angle, float)
        or not math.isfinite(loo_angle)
        or not 0.0 <= loo_angle <= math.pi
        or value.get("candidate_baseline_maximum_angle_radians") != 0.15
        or value.get("cross_scene_gate") != cross_scene_gate
        or not isinstance(candidates, list)
        or len(candidates) != len(LOO_CEILING_CANDIDATES_RADIANS)
    ):
        raise ValueError("source-only LOO ceiling audit contract differs")
    candidate_keys = {
        "maximum_angle_radians",
        "heldout_scale_observations",
        "cosine_sum",
        "mean_cosine",
        "delta_cosine_sum_vs_o1_0p15",
        "mean_delta_cosine_vs_o1_0p15",
        "scene_nonregression_vs_o1_0p15",
    }
    baseline_sum = candidates[0].get("cosine_sum")
    if not isinstance(baseline_sum, float) or not math.isfinite(baseline_sum):
        raise ValueError("source-only LOO ceiling audit baseline differs")
    for expected_angle, candidate in zip(
        LOO_CEILING_CANDIDATES_RADIANS, candidates
    ):
        if not isinstance(candidate, Mapping) or set(candidate) != candidate_keys:
            raise ValueError("source-only LOO ceiling candidate fields differ")
        cosine_sum = candidate.get("cosine_sum")
        mean_cosine = candidate.get("mean_cosine")
        delta_sum = candidate.get("delta_cosine_sum_vs_o1_0p15")
        mean_delta = candidate.get("mean_delta_cosine_vs_o1_0p15")
        if (
            candidate.get("maximum_angle_radians") != expected_angle
            or candidate.get("heldout_scale_observations") != observations
            or not isinstance(cosine_sum, float)
            or not math.isfinite(cosine_sum)
            or not -observations <= cosine_sum <= observations
            or not isinstance(mean_cosine, float)
            or mean_cosine != cosine_sum / observations
            or not isinstance(delta_sum, float)
            or delta_sum != cosine_sum - baseline_sum
            or not isinstance(mean_delta, float)
            or mean_delta != delta_sum / observations
            or candidate.get("scene_nonregression_vs_o1_0p15")
            is not (delta_sum >= 0.0)
        ):
            raise ValueError("source-only LOO ceiling candidate contract differs")


def _canonicalize_view_axis_v2(
    top_descriptors: torch.Tensor,
    top_mass: torch.Tensor,
    top_frame_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Delegate canonicalization and capture only the minimal row statistic."""

    global _agreement_state, _base_descriptor_state
    canonical = _CORE_CANONICALIZE_VIEW_AXIS(
        top_descriptors, top_mass, top_frame_ids
    )
    resultant, counts = directional_resultant_from_canonical_top_views(
        canonical[0], canonical[2]
    )
    if _base_descriptor_state is None:
        raise RuntimeError("O0 descriptor was not captured for source-only LOO audit")
    loo_audit = source_only_leave_one_view_out_ceiling_audit(
        canonical[0], canonical[2], _base_descriptor_state
    )
    _base_descriptor_state = None
    _agreement_state = {
        "resultant": resultant,
        "counts": counts,
        "loo_audit": loo_audit,
    }
    return canonical


def validate_teacher_payload_v2(payload: Mapping[str, Any]) -> None:
    """Fail closed on the durable v2 teacher payload contract."""

    required = {
        "schema",
        "schema_version",
        "scene_id",
        "global_rows",
        "teacher_mean",
        "teacher_valid",
        "retained_view_count",
        VIEW_AGREEMENT_SCALAR,
        "producer",
        "execution_authority",
        "input_authority",
        "method_contract_sha256",
        "teacher_mean_sha256",
        VIEW_AGREEMENT_SHA256_FIELD,
        LOO_AUDIT_FIELD,
        LOO_AUDIT_SHA256_FIELD,
        "access_audit",
    }
    if set(payload) != required:
        raise ValueError("teacher-agreement v2 payload fields differ")
    rows = payload.get("global_rows")
    mean = payload.get("teacher_mean")
    valid = payload.get("teacher_valid")
    count = payload.get("retained_view_count")
    agreement = payload.get(VIEW_AGREEMENT_SCALAR)
    loo_audit = payload.get(LOO_AUDIT_FIELD)
    if not isinstance(loo_audit, Mapping):
        raise ValueError("teacher-agreement v2 LOO audit differs")
    validate_source_only_loo_ceiling_audit(loo_audit)
    if (
        payload.get("schema") != MEAN_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or not torch.is_tensor(rows)
        or rows.ndim != 1
        or rows.dtype != torch.int64
        or not torch.is_tensor(mean)
        or mean.shape != (rows.numel(), 1536)
        or mean.dtype != torch.float16
        or not bool(torch.isfinite(mean).all())
        or not torch.is_tensor(valid)
        or valid.shape != (rows.numel(),)
        or valid.dtype != torch.bool
        or not torch.is_tensor(count)
        or count.shape != (rows.numel(),)
        or count.dtype != torch.uint8
        or not torch.equal(valid, count > 0)
        or bool((count > _core.TOP_VIEW_COUNT).any())
        or not torch.is_tensor(agreement)
        or agreement.shape != (rows.numel(),)
        or agreement.dtype != AGREEMENT_DTYPE
        or not bool(torch.isfinite(agreement).all())
        or bool((agreement < 0.0).any())
        or bool((agreement > 1.0).any())
        or bool((agreement[~valid] != 0.0).any())
        or bool((mean[~valid] != 0.0).any())
        or payload.get("teacher_mean_sha256") != _core.tensor_sha256_typed(mean)
        or payload.get(VIEW_AGREEMENT_SHA256_FIELD)
        != _core.tensor_sha256_typed(agreement)
        or loo_audit.get("rows") != rows.numel()
        or loo_audit.get("heldout_scale_observations", 0) <= 0
        or payload.get(LOO_AUDIT_SHA256_FIELD)
        != canonical_json_sha256(loo_audit)
        or payload.get("method_contract_sha256") != METHOD_CONTRACT_SHA256
        or payload.get("producer") != ENTRYPOINT_IMPLEMENTATION
    ):
        raise ValueError("teacher-agreement v2 payload contract differs")


def augment_teacher_payload_v2(
    payload: Mapping[str, Any],
    *,
    resultant: torch.Tensor,
    expected_counts: torch.Tensor,
    loo_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Add agreement and its typed hash without changing v1 numeric tensors."""

    output = dict(payload)
    count = output.get("retained_view_count")
    value = torch.as_tensor(resultant).detach().cpu().to(AGREEMENT_DTYPE).contiguous()
    expected = torch.as_tensor(expected_counts).detach().cpu().to(torch.uint8)
    if not torch.is_tensor(count) or not torch.equal(count.detach().cpu(), expected):
        raise ValueError("captured top-view counts differ from teacher payload")
    output[VIEW_AGREEMENT_SCALAR] = value
    output[VIEW_AGREEMENT_SHA256_FIELD] = _core.tensor_sha256_typed(value)
    output[LOO_AUDIT_FIELD] = copy.deepcopy(dict(loo_audit))
    output[LOO_AUDIT_SHA256_FIELD] = canonical_json_sha256(
        output[LOO_AUDIT_FIELD]
    )
    validate_teacher_payload_v2(output)
    return output


def _write_torch_noclobber_v2(path: str | Path, payload: Mapping[str, Any]) -> None:
    global _agreement_state
    output: Mapping[str, Any] = payload
    if payload.get("schema") == MEAN_SCHEMA:
        if _agreement_state is None:
            raise RuntimeError("teacher agreement was not captured")
        output = augment_teacher_payload_v2(
            payload,
            resultant=_agreement_state["resultant"],
            expected_counts=_agreement_state["counts"],
            loo_audit=_agreement_state["loo_audit"],
        )
        _agreement_state = None
    _CORE_WRITE_TORCH_NOCLOBBER(path, output)


def _project_view_unpaced(**kwargs: Any):
    kwargs["pace"] = False
    return _CORE_PROJECT_VIEW(**kwargs)


def _raw_cache_v2(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _CORE_RAW_CACHE(*args, **kwargs)
    payload["authority"]["descriptor_axis"]["execution_representation"] = (
        "source_teacher_mean_with_directional_resultant_streaming_v2"
    )
    return payload


def _prepare_inputs_v2(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Delegate validation and retain a zero-copy O0 reference for LOO audit."""

    global _base_descriptor_state
    prepared = _CORE_PREPARE_INPUTS(*args, **kwargs)
    base = prepared["base"]["features_by_scale"]
    if not torch.is_tensor(base):
        raise ValueError("O0 descriptor axis is missing for source-only LOO audit")
    _base_descriptor_state = base
    return prepared


def _materialize_v2(args: Any) -> dict[str, Any]:
    global _agreement_state, _base_descriptor_state
    _agreement_state = None
    _base_descriptor_state = None
    result = _CORE_MATERIALIZE(args)
    if _agreement_state is not None:
        raise RuntimeError("teacher agreement was captured but not persisted")
    if _base_descriptor_state is not None:
        raise RuntimeError("O0 descriptor was captured but LOO audit did not consume it")
    return result


def _install_v2_contract() -> None:
    """Install v2 hooks only in this entrypoint's private Python process."""

    _core.AUTHORITY_SCHEMA = AUTHORITY_SCHEMA
    _core.MEAN_SCHEMA = MEAN_SCHEMA
    _core.RESULT_SCHEMA = RESULT_SCHEMA
    _core.SCHEMA_VERSION = SCHEMA_VERSION
    _core.PACING_SECONDS_PER_PROJECTION_BATCH = PACING_SECONDS_PER_PROJECTION_BATCH
    _core.method_contract = method_contract
    _core.METHOD_CONTRACT_SHA256 = METHOD_CONTRACT_SHA256
    _core._canonicalize_view_axis = _canonicalize_view_axis_v2
    _core._project_view = _project_view_unpaced
    _core._raw_cache = _raw_cache_v2
    _core.prepare_inputs = _prepare_inputs_v2
    _core.materialize = _materialize_v2
    _core.write_torch_noclobber = _write_torch_noclobber_v2
    _core.__file__ = str(Path(__file__).resolve())


def main() -> None:
    _install_v2_contract()
    _core.main()


if __name__ == "__main__":
    main()


__all__ = [
    "AGREEMENT_DTYPE",
    "AGREEMENT_ROW_CHUNK",
    "AUTHORITY_SCHEMA",
    "CORE_IMPLEMENTATION",
    "ENTRYPOINT_IMPLEMENTATION",
    "LOO_AUDIT_FIELD",
    "LOO_AUDIT_SHA256_FIELD",
    "LOO_CEILING_CANDIDATES_RADIANS",
    "MEAN_SCHEMA",
    "METHOD_CONTRACT_SHA256",
    "PACING_SECONDS_PER_PROJECTION_BATCH",
    "RESULT_SCHEMA",
    "SCHEMA_VERSION",
    "augment_teacher_payload_v2",
    "directional_resultant_from_canonical_top_views",
    "main",
    "method_contract",
    "source_only_leave_one_view_out_ceiling_audit",
    "validate_source_only_loo_ceiling_audit",
    "validate_teacher_payload_v2",
]
