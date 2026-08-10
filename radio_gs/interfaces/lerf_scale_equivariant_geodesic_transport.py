"""Scale-equivariant geodesic transport for a multiscale LERF field.

The three AcceptedV2 O0 scale descriptors are treated as one small frame.
One shortest-plane rotation moves the normalized frame mean toward a
query-free source-view teacher, and the *same* orthogonal map is applied to
every scale.  Consequently the per-scale norms and the complete scale Gram
matrix are preserved up to floating-point roundoff.

The reliability budget is inherited from the existing query-free interface,
but an angular ceiling selected for the older independent-scale projection
does not authorize this transport.  A separate source-only LOO statistic is
provided below so callers can validate the transport before target metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from radio_gs.interfaces.lerf_reliability_geodesic_budget import (
    CONSERVATIVE_ANGLE_RADIANS,
    MAXIMUM_ANGLE_RADIANS,
    RETAINED_VIEW_CAPACITY,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SCALE_COUNT = 3
CEILING_GRID_RADIANS = (0.15, 0.3, 0.45, 0.6, 0.75)
SOURCE_LOO_SCHEMA = "radio_gs.source_only_loo_scale_equivariant_transport.v1"


def transport_contract() -> dict[str, Any]:
    return {
        "schema": "radio_gs.lerf_scale_equivariant_geodesic_transport.v1",
        "schema_version": 1,
        "immutable_backbone": "accepted_v2_o0_descriptor_three_scale_frame",
        "frame_anchor": "normalize(mean_s(o0_scale_s_as_stored))",
        "teacher": "normalized_equal_view_source_teacher_mean",
        "plane": "span(frame_anchor,teacher)",
        "rotation": (
            "x+(cos(phi)-1)*(<x,a>*a+<x,e2>*e2)"
            "+sin(phi)*(<x,a>*e2-<x,e2>*a)"
        ),
        "same_orthogonal_map_for_all_scales": True,
        "invariants": [
            "per_scale_l2_norm",
            "three_by_three_pairwise_gram",
        ],
        "post_rotation_renormalization": False,
        "conservative_angle_radians": CONSERVATIVE_ANGLE_RADIANS,
        "maximum_interface_angle_radians": MAXIMUM_ANGLE_RADIANS,
        "candidate_grid_radians": list(CEILING_GRID_RADIANS),
        "reliability": (
            "sqrt(count_sufficiency*teacher_view_directional_resultant)"
            "*global_ceiling_attenuation"
        ),
        "global_ceiling_attenuation": (
            "(ceiling-0.15)/(0.75-0.15)"
        ),
        "angular_budget": "0.15+(0.75-0.15)*reliability",
        "single_view_policy": "common_rotation_with_exact_0.15_budget",
        "invalid_teacher_policy": "bitwise_o0_fallback",
        "undefined_frame_mean_policy": "bitwise_o0_fallback",
        "same_direction_policy": "bitwise_o0_fallback",
        "antipodal_policy": "bitwise_o0_fallback",
        "query_independent": True,
        "scene_or_query_specific_parameters": False,
        "source_only_validation": {
            "prediction": "leave_one_retained_source_view_out",
            "evaluation_unit": "heldout_source_view_x_o0_scale",
            "baseline": "same_transport_with_0.15_ceiling",
            "pooled_improvement_required": True,
            "every_source_scene_nonregression_required": True,
            "one_global_ceiling_required": True,
        },
        "old_independent_scale_selector_directly_authorizes_transport": False,
        "target_labels_masks_metrics_consumed": False,
    }


TRANSPORT_CONTRACT_SHA256 = canonical_json_sha256(transport_contract())


@dataclass(frozen=True)
class ScaleEquivariantTransportOutput:
    descriptor: torch.Tensor
    teacher_applied: torch.Tensor
    fallback_to_o0: torch.Tensor
    reliability_score: torch.Tensor
    angular_budget_radians: torch.Tensor
    requested_angle_radians: torch.Tensor
    angular_step_radians: torch.Tensor
    expanded_budget: torch.Tensor
    frame_mean_valid: torch.Tensor
    same_direction: torch.Tensor
    antipodal: torch.Tensor


def _validate_inputs(
    base: torch.Tensor,
    teacher: torch.Tensor,
    valid: torch.Tensor,
    count: torch.Tensor,
    agreement: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base = torch.as_tensor(base)
    teacher = torch.as_tensor(teacher, device=base.device)
    valid = torch.as_tensor(valid, device=base.device)
    count = torch.as_tensor(count, device=base.device)
    agreement = torch.as_tensor(agreement, device=base.device)
    if (
        base.ndim < 3
        or base.shape[-2] != SCALE_COUNT
        or base.shape[-1] <= 1
        or not base.is_floating_point()
        or not bool(torch.isfinite(base).all())
    ):
        raise ValueError("transport base must be finite floating [...,3,D]")
    prefix = tuple(base.shape[:-2])
    if (
        teacher.shape != (*prefix, base.shape[-1])
        or not teacher.is_floating_point()
        or not bool(torch.isfinite(teacher).all())
        or valid.shape != prefix
        or valid.dtype != torch.bool
        or count.shape != prefix
        or count.dtype == torch.bool
        or agreement.shape != prefix
        or not agreement.is_floating_point()
        or not bool(torch.isfinite(agreement).all())
        or bool((agreement < 0).any())
        or bool((agreement > 1).any())
    ):
        raise ValueError("transport teacher/reliability axes differ")
    count_f = count.float()
    if (
        not bool(torch.isfinite(count_f).all())
        or bool((count_f < 0).any())
        or bool((count_f > RETAINED_VIEW_CAPACITY).any())
        or not torch.equal(count_f, count_f.round())
        or bool((valid != (count_f > 0)).any())
        or bool((agreement[~valid] != 0).any())
        or bool((teacher[~valid] != 0).any())
    ):
        raise ValueError("transport validity/count contract differs")
    base_norm = torch.linalg.vector_norm(base.float(), dim=-1)
    teacher_norm = torch.linalg.vector_norm(teacher.float(), dim=-1)
    if bool((base_norm <= 0).any()) or not torch.allclose(
        base_norm, torch.ones_like(base_norm), rtol=0.0, atol=2e-3
    ):
        raise ValueError("transport base must use the unit L2 gauge")
    if bool(valid.any()) and not torch.allclose(
        teacher_norm[valid],
        torch.ones_like(teacher_norm[valid]),
        rtol=0.0,
        atol=2e-3,
    ):
        raise ValueError("transport teacher must use the unit L2 gauge")
    return base, teacher, valid, count_f, agreement


def scale_equivariant_geodesic_transport(
    o0_descriptor_by_scale: torch.Tensor,
    teacher_mean: torch.Tensor,
    *,
    teacher_valid: torch.Tensor,
    retained_view_count: torch.Tensor,
    teacher_view_directional_resultant: torch.Tensor,
    maximum_angle_radians: float,
) -> ScaleEquivariantTransportOutput:
    """Apply one reliability-bounded shortest-plane rotation to all scales."""

    if (
        isinstance(maximum_angle_radians, bool)
        or not isinstance(maximum_angle_radians, (int, float))
        or not math.isfinite(float(maximum_angle_radians))
        or float(maximum_angle_radians) not in CEILING_GRID_RADIANS
    ):
        raise ValueError("transport maximum angle is outside the frozen grid")
    base, teacher, valid, count, agreement = _validate_inputs(
        o0_descriptor_by_scale,
        teacher_mean,
        teacher_valid,
        retained_view_count,
        teacher_view_directional_resultant,
    )
    # The accepted cache is already in the unit gauge (within its FP16
    # storage tolerance).  Do not renormalize here: applying the orthogonal
    # map to the promoted cache values preserves their *actual* norms and Gram
    # matrix, whereas another normalization would slightly change both.
    base_unit = base.float()
    teacher_unit = F.normalize(teacher.float(), dim=-1)
    eps = 8.0 * torch.finfo(torch.float32).eps

    mean_raw = base_unit.mean(dim=-2)
    mean_norm = torch.linalg.vector_norm(mean_raw, dim=-1)
    mean_valid = mean_norm > eps
    anchor = mean_raw / mean_norm[..., None].clamp_min(eps)
    cosine = (anchor * teacher_unit).sum(dim=-1).clamp(-1.0, 1.0)
    tangent = teacher_unit - cosine[..., None] * anchor
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
    same = valid & mean_valid & (tangent_norm <= eps) & (cosine >= 0)
    antipodal = valid & mean_valid & (tangent_norm <= eps) & (cosine < 0)
    applied = valid & mean_valid & ~same & ~antipodal
    e2 = tangent / tangent_norm[..., None].clamp_min(eps)

    count_sufficiency = ((count - 1.0) / (RETAINED_VIEW_CAPACITY - 1)).clamp(
        0.0, 1.0
    )
    ceiling_attenuation = (
        (float(maximum_angle_radians) - CONSERVATIVE_ANGLE_RADIANS)
        / (MAXIMUM_ANGLE_RADIANS - CONSERVATIVE_ANGLE_RADIANS)
    )
    reliability = (
        torch.sqrt((count_sufficiency * agreement.float()).clamp_min(0.0))
        * ceiling_attenuation
    ).clamp(0.0, 1.0)
    reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
    budget = CONSERVATIVE_ANGLE_RADIANS + (
        MAXIMUM_ANGLE_RADIANS - CONSERVATIVE_ANGLE_RADIANS
    ) * reliability
    requested = torch.acos(cosine)
    step = torch.minimum(requested, budget)
    step = torch.where(applied, step, torch.zeros_like(step))

    p = (base_unit * anchor[..., None, :]).sum(dim=-1)
    q = (base_unit * e2[..., None, :]).sum(dim=-1)
    plane = p[..., None] * anchor[..., None, :] + q[..., None] * e2[..., None, :]
    skew = p[..., None] * e2[..., None, :] - q[..., None] * anchor[..., None, :]
    rotated = (
        base_unit
        + (torch.cos(step) - 1.0)[..., None, None] * plane
        + torch.sin(step)[..., None, None] * skew
    )
    descriptor = torch.where(applied[..., None, None], rotated, base_unit)
    return ScaleEquivariantTransportOutput(
        descriptor=descriptor.contiguous().float(),
        teacher_applied=applied[..., None].expand(*applied.shape, SCALE_COUNT).contiguous(),
        fallback_to_o0=(~applied)[..., None].expand(*applied.shape, SCALE_COUNT).contiguous(),
        reliability_score=reliability.contiguous(),
        angular_budget_radians=budget.contiguous(),
        requested_angle_radians=requested.contiguous(),
        angular_step_radians=step.contiguous(),
        expanded_budget=(valid & (reliability > 0)).contiguous(),
        frame_mean_valid=mean_valid.contiguous(),
        same_direction=same.contiguous(),
        antipodal=antipodal.contiguous(),
    )


def source_only_leave_one_view_out_transport_audit(
    top_descriptors: torch.Tensor,
    top_frame_ids: torch.Tensor,
    o0_descriptor_by_scale: torch.Tensor,
    *,
    row_chunk: int = 2048,
) -> dict[str, Any]:
    """Evaluate the transport ceiling grid on held-out source directions."""

    descriptors = torch.as_tensor(top_descriptors)
    frame_ids = torch.as_tensor(top_frame_ids)
    base = torch.as_tensor(o0_descriptor_by_scale)
    if (
        descriptors.ndim != 3
        or descriptors.shape[1] != RETAINED_VIEW_CAPACITY
        or frame_ids.shape != descriptors.shape[:2]
        or frame_ids.dtype == torch.bool
        or base.shape != (descriptors.shape[0], SCALE_COUNT, descriptors.shape[2])
        or not descriptors.is_floating_point()
        or not base.is_floating_point()
        or not bool(torch.isfinite(descriptors).all())
        or not bool(torch.isfinite(base).all())
        or not isinstance(row_chunk, int)
        or row_chunk < 1
    ):
        raise ValueError("transport source-only LOO tensors differ")
    mask = frame_ids >= 0
    counts = mask.sum(dim=1)
    candidate_sums = torch.zeros(len(CEILING_GRID_RADIANS), dtype=torch.float64)
    heldout_predictions = 0
    rows_with_loo = 0
    rows_with_expansion = 0
    eps = 8.0 * torch.finfo(torch.float32).eps
    for start in range(0, descriptors.shape[0], row_chunk):
        stop = min(descriptors.shape[0], start + row_chunk)
        selected = descriptors[start:stop]
        selected_mask = mask[start:stop]
        if bool((selected[~selected_mask] != 0).any()):
            raise ValueError("unretained transport LOO view must be exact zero")
        unit_views = F.normalize(selected.float(), dim=-1)
        base_unit = base[start:stop].float()
        selected_counts = counts[start:stop]
        view_sum = (unit_views * selected_mask[..., None]).sum(dim=1)
        row_has_loo = torch.zeros(stop - start, dtype=torch.bool)
        for heldout_index in range(RETAINED_VIEW_CAPACITY):
            heldout_valid = selected_mask[:, heldout_index] & (selected_counts >= 2)
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
            teacher = loo_sum / loo_norm[..., None].clamp_min(eps)
            teacher = torch.where(prediction_valid[..., None], teacher, torch.zeros_like(teacher))
            effective_count = torch.where(
                prediction_valid, loo_count, torch.zeros_like(loo_count)
            )
            agreement = (loo_norm / loo_count.float()).clamp(0.0, 1.0)
            agreement = torch.where(prediction_valid, agreement, torch.zeros_like(agreement))
            for index, ceiling in enumerate(CEILING_GRID_RADIANS):
                transported = scale_equivariant_geodesic_transport(
                    base_unit,
                    teacher,
                    teacher_valid=prediction_valid,
                    retained_view_count=effective_count,
                    teacher_view_directional_resultant=agreement,
                    maximum_angle_radians=ceiling,
                ).descriptor
                predicted = (transported * heldout[:, None, :]).sum(dim=-1)
                candidate_sums[index] += predicted[prediction_valid].double().sum()
            heldout_predictions += int(prediction_valid.sum())
        rows_with_loo += int(row_has_loo.sum())
        rows_with_expansion += int((row_has_loo & (selected_counts >= 3)).sum())
    observations = heldout_predictions * SCALE_COUNT
    if heldout_predictions <= 0:
        raise ValueError("transport source-only LOO has no valid prediction")
    baseline = float(candidate_sums[0])
    candidates = []
    for ceiling, value in zip(CEILING_GRID_RADIANS, candidate_sums):
        cosine_sum = float(value)
        delta = cosine_sum - baseline
        candidates.append(
            {
                "maximum_angle_radians": ceiling,
                "heldout_scale_observations": observations,
                "cosine_sum": cosine_sum,
                "mean_cosine": cosine_sum / observations,
                "delta_cosine_sum_vs_transport_0p15": delta,
                "mean_delta_cosine_vs_transport_0p15": delta / observations,
                "scene_nonregression_vs_transport_0p15": delta >= 0.0,
            }
        )
    result = {
        "schema": SOURCE_LOO_SCHEMA,
        "schema_version": 1,
        "transport_contract_sha256": TRANSPORT_CONTRACT_SHA256,
        "query_independent": True,
        "target_images_labels_masks_metrics_opened": False,
        "candidate_role": "source_only_transport_diagnostic_not_target_authorization",
        "target_candidate_authorized": False,
        "retained_view_capacity": RETAINED_VIEW_CAPACITY,
        "rows": int(descriptors.shape[0]),
        "rows_with_valid_loo_prediction": rows_with_loo,
        "rows_with_expansion_evidence": rows_with_expansion,
        "heldout_predictions": heldout_predictions,
        "heldout_scale_observations": observations,
        "candidate_baseline_maximum_angle_radians": CONSERVATIVE_ANGLE_RADIANS,
        "candidates": candidates,
        "cross_scene_gate": {
            "pooled_statistic": (
                "sum(delta_cosine_sum_vs_transport_0p15)"
                "/sum(heldout_scale_observations)"
            ),
            "pooled_improvement_required": True,
            "every_source_scene_nonregression_required": True,
            "one_global_ceiling_required": True,
        },
    }
    validate_source_only_transport_audit(result)
    return result


def validate_source_only_transport_audit(value: Mapping[str, Any]) -> None:
    required = {
        "schema", "schema_version", "transport_contract_sha256",
        "query_independent", "target_images_labels_masks_metrics_opened",
        "candidate_role", "target_candidate_authorized",
        "retained_view_capacity", "rows", "rows_with_valid_loo_prediction",
        "rows_with_expansion_evidence", "heldout_predictions",
        "heldout_scale_observations",
        "candidate_baseline_maximum_angle_radians", "candidates",
        "cross_scene_gate",
    }
    candidates = value.get("candidates")
    observations = value.get("heldout_scale_observations")
    if (
        set(value) != required
        or value.get("schema") != SOURCE_LOO_SCHEMA
        or value.get("schema_version") != 1
        or value.get("transport_contract_sha256") != TRANSPORT_CONTRACT_SHA256
        or value.get("query_independent") is not True
        or value.get("target_images_labels_masks_metrics_opened") is not False
        or value.get("candidate_role")
        != "source_only_transport_diagnostic_not_target_authorization"
        or value.get("target_candidate_authorized") is not False
        or value.get("retained_view_capacity") != RETAINED_VIEW_CAPACITY
        or not isinstance(value.get("rows"), int)
        or not isinstance(value.get("rows_with_valid_loo_prediction"), int)
        or not isinstance(value.get("rows_with_expansion_evidence"), int)
        or not isinstance(value.get("heldout_predictions"), int)
        or not isinstance(observations, int)
        or observations != value.get("heldout_predictions") * SCALE_COUNT
        or value.get("candidate_baseline_maximum_angle_radians")
        != CONSERVATIVE_ANGLE_RADIANS
        or not isinstance(candidates, list)
        or len(candidates) != len(CEILING_GRID_RADIANS)
    ):
        raise ValueError("transport source-only LOO contract differs")
    keys = {
        "maximum_angle_radians", "heldout_scale_observations", "cosine_sum",
        "mean_cosine", "delta_cosine_sum_vs_transport_0p15",
        "mean_delta_cosine_vs_transport_0p15",
        "scene_nonregression_vs_transport_0p15",
    }
    baseline = candidates[0].get("cosine_sum")
    if not isinstance(baseline, float) or not math.isfinite(baseline):
        raise ValueError("transport source-only LOO baseline differs")
    for ceiling, row in zip(CEILING_GRID_RADIANS, candidates):
        if not isinstance(row, Mapping) or set(row) != keys:
            raise ValueError("transport source-only LOO candidate fields differ")
        cosine_sum = row.get("cosine_sum")
        mean = row.get("mean_cosine")
        delta = row.get("delta_cosine_sum_vs_transport_0p15")
        mean_delta = row.get("mean_delta_cosine_vs_transport_0p15")
        if (
            row.get("maximum_angle_radians") != ceiling
            or row.get("heldout_scale_observations") != observations
            or not isinstance(cosine_sum, float)
            or not math.isfinite(cosine_sum)
            or not isinstance(mean, float)
            or mean != cosine_sum / observations
            or not isinstance(delta, float)
            or delta != cosine_sum - baseline
            or not isinstance(mean_delta, float)
            or mean_delta != delta / observations
            or row.get("scene_nonregression_vs_transport_0p15")
            is not (delta >= 0.0)
        ):
            raise ValueError("transport source-only LOO candidate contract differs")


__all__ = [
    "CEILING_GRID_RADIANS",
    "SOURCE_LOO_SCHEMA",
    "ScaleEquivariantTransportOutput",
    "TRANSPORT_CONTRACT_SHA256",
    "scale_equivariant_geodesic_transport",
    "source_only_leave_one_view_out_transport_audit",
    "transport_contract",
    "validate_source_only_transport_audit",
]
