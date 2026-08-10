"""Query-free reliability-conditioned geodesic teacher fusion for LERF.

The accepted multiscale O0 descriptor is the immutable backbone.  For every
scale independently, this interface can move its direction toward a shared
source-view teacher direction.  The ordinary O1 trust region (0.15 radians)
is always available.  A larger trust region is possible only when the
durable teacher payload contains query-free multiview agreement.

The v1 streaming payload stores the normalized teacher mean and retained-view
count, but not the norm of the *unnormalized* mean.  Count alone cannot
distinguish agreeing views from cancelling views, so v1 necessarily receives
the conservative 0.15-radian budget.  A future streamer v2 needs only one
additional row scalar, ``teacher_view_directional_resultant``::

    ||sum_j unit_teacher_j||_2 / retained_view_count

It can be computed before the current normalization and does not require
retaining any per-view descriptor.  Optional responsibility and canonical
field reliabilities may only attenuate, never amplify, this view evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import canonical_json_sha256


CONSERVATIVE_ANGLE_RADIANS = 0.15
# This is an interface safety ceiling inherited from the pre-existing V2.1B
# residual design, not a LERF metric-selected or benchmark-authorized value.
# A source-only preregistration gate must bind the deployed ceiling before a
# target benchmark is opened.
MAXIMUM_ANGLE_RADIANS = 0.75
RETAINED_VIEW_CAPACITY = 4
VIEW_AGREEMENT_SCALAR = "teacher_view_directional_resultant"
VIEW_AGREEMENT_SHA256_FIELD = "teacher_view_directional_resultant_sha256"


def reliability_geodesic_budget_contract() -> dict[str, Any]:
    """Return the scene-general, query-free interface contract."""

    return {
        "schema": "radio_gs.lerf_reliability_geodesic_budget.v1",
        "schema_version": 1,
        "immutable_backbone": "accepted_v2_o0_descriptor_per_scale",
        "teacher": "normalized_equal_view_source_teacher_mean",
        "conservative_angle_radians": CONSERVATIVE_ANGLE_RADIANS,
        "maximum_angle_radians": MAXIMUM_ANGLE_RADIANS,
        "maximum_angle_status": (
            "interface_safety_ceiling_not_yet_source_preregistered_for_lerf"
        ),
        "maximum_angle_origin": (
            "preexisting_v21b_query_free_interface_not_ramen_or_teatime_metric"
        ),
        "retained_view_capacity": RETAINED_VIEW_CAPACITY,
        "count_sufficiency": "clamp((retained_view_count-1)/(4-1),0,1)",
        "required_for_expansion": {
            "name": VIEW_AGREEMENT_SCALAR,
            "formula": (
                "l2_norm(sum_of_retained_unit_teacher_directions)"
                "/retained_view_count"
            ),
            "range": [0.0, 1.0],
            "query_independent": True,
        },
        "streamer_v2_minimal_record_change": {
            "schema_action": "bump_teacher_payload_schema_and_version",
            "tensor_field": VIEW_AGREEMENT_SCALAR,
            "tensor_shape": "accepted_rows",
            "invalid_value": "exact_zero",
            "hash_field": VIEW_AGREEMENT_SHA256_FIELD,
            "hash": "typed_tensor_sha256",
            "calculation_point": "before_teacher_sum_l2_normalization",
            "per_view_descriptors_retained": False,
            "access_audit": (
                "source_features_and_exact_responsibility_only_unchanged"
            ),
        },
        "v1_missing_agreement_policy": (
            "reliability_zero_and_exact_0.15_radian_budget"
        ),
        "optional_attenuators": [
            "responsibility_reliability",
            "canonical_field_reliability",
        ],
        "reliability": (
            "sqrt(count_sufficiency*teacher_view_directional_resultant)"
            "*responsibility_reliability*canonical_field_reliability"
        ),
        "missing_optional_attenuator": "multiplicative_identity_one",
        "angular_budget": "0.15+(0.75-0.15)*reliability",
        "projection": "per_scale_unit_sphere_shortest_geodesic",
        "single_view_expansion_allowed": False,
        "antipodal_teacher_policy": "bitwise_o0_fallback",
        "teacher_invalid_policy": "bitwise_o0_fallback",
        "scene_or_query_id_consumed": False,
        "per_scene_parameters": False,
        "benchmark_labels_consumed": False,
        "benchmark_metrics_consumed": False,
        "benchmark_candidate_authorized": False,
        "future_activation_gate": {
            "inputs": "source_views_only_without_queries_labels_or_metrics",
            "validation": (
                "deterministic_leave_one_source_view_out_direction_prediction"
            ),
            "scope": "one_global_ceiling_across_source_scenes",
            "requirement": (
                "source_pooled_improvement_and_every_source_scene_nonregression"
            ),
            "then": "freeze_ceiling_and_contract_before_target_metric",
        },
        "query_independent": True,
    }


RELIABILITY_GEODESIC_BUDGET_CONTRACT_SHA256 = canonical_json_sha256(
    reliability_geodesic_budget_contract()
)


@dataclass(frozen=True)
class ReliabilityGeodesicBudgetOutput:
    """Descriptor and complete audit state for one bounded fusion call."""

    descriptor: torch.Tensor
    reliability_score: torch.Tensor
    count_sufficiency: torch.Tensor
    view_agreement: torch.Tensor
    responsibility_attenuation: torch.Tensor
    field_attenuation: torch.Tensor
    angular_budget_radians: torch.Tensor
    requested_angle_radians: torch.Tensor
    angular_step_radians: torch.Tensor
    expanded_budget: torch.Tensor
    teacher_applied: torch.Tensor
    fallback_to_o0: torch.Tensor
    agreement_available: bool


def _aligned_scalar(
    value: torch.Tensor,
    *,
    prefix: tuple[int, ...],
    label: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device)
    if (
        result.shape != prefix
        or not result.is_floating_point()
        or not bool(torch.isfinite(result).all())
        or bool((result < 0.0).any())
        or bool((result > 1.0).any())
    ):
        raise ValueError(f"{label} must be finite floating {prefix} in [0,1]")
    return result.to(dtype=dtype)


def _validate_view_count(
    value: torch.Tensor,
    *,
    prefix: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    count = torch.as_tensor(value, device=device)
    if count.shape != prefix or count.dtype == torch.bool:
        raise ValueError(f"retained_view_count must align as {prefix}")
    count_float = count.to(dtype=torch.float32)
    if (
        not bool(torch.isfinite(count_float).all())
        or bool((count_float < 0).any())
        or bool((count_float > RETAINED_VIEW_CAPACITY).any())
        or not torch.equal(count_float, count_float.round())
    ):
        raise ValueError("retained_view_count must be integral in [0,4]")
    return count_float.to(dtype=dtype)


def reliability_conditioned_geodesic_fusion(
    o0_descriptor_by_scale: torch.Tensor,
    teacher_mean: torch.Tensor,
    *,
    teacher_valid: torch.Tensor,
    retained_view_count: torch.Tensor,
    teacher_view_directional_resultant: torch.Tensor | None = None,
    responsibility_reliability: torch.Tensor | None = None,
    canonical_field_reliability: torch.Tensor | None = None,
) -> ReliabilityGeodesicBudgetOutput:
    """Move each O0 scale toward the teacher under a query-free angle budget.

    Args:
        o0_descriptor_by_scale: Unit O0 directions shaped ``[..., S, D]``.
        teacher_mean: Unit source-view teacher direction shaped ``[..., D]``.
        teacher_valid: Whether a row has at least one retained teacher view.
        retained_view_count: Integer count in ``[0, 4]`` shaped ``[...]``.
        teacher_view_directional_resultant: Optional v2 agreement scalar.  Its
            absence deliberately disables every angle beyond 0.15 radians.
        responsibility_reliability: Optional query-free attenuation in [0,1].
        canonical_field_reliability: Optional query-free attenuation in [0,1].

    Returns:
        A same-shaped descriptor plus row/scale diagnostics.  Invalid teacher
        rows and ambiguous antipodal routes are bitwise O0 fallbacks.
    """

    base = torch.as_tensor(o0_descriptor_by_scale)
    teacher = torch.as_tensor(teacher_mean, device=base.device)
    if (
        base.ndim < 3
        or base.shape[-2] <= 0
        or base.shape[-1] <= 1
        or not base.is_floating_point()
        or not bool(torch.isfinite(base).all())
    ):
        raise ValueError("O0 descriptor must be finite floating [...,S,D]")
    prefix = tuple(base.shape[:-2])
    if (
        teacher.shape != (*prefix, base.shape[-1])
        or not teacher.is_floating_point()
        or not bool(torch.isfinite(teacher).all())
    ):
        raise ValueError("teacher_mean must be finite floating [...,D]")
    valid = torch.as_tensor(teacher_valid, device=base.device)
    if valid.shape != prefix or valid.dtype != torch.bool:
        raise ValueError(f"teacher_valid must be boolean {prefix}")
    count = _validate_view_count(
        retained_view_count,
        prefix=prefix,
        device=base.device,
        dtype=base.dtype,
    )
    if bool((valid != (count > 0)).any()):
        raise ValueError("teacher_valid must equal retained_view_count > 0")

    base_norm = torch.linalg.vector_norm(base.float(), dim=-1)
    teacher_norm = torch.linalg.vector_norm(teacher.float(), dim=-1)
    if not torch.allclose(
        base_norm,
        torch.ones_like(base_norm),
        rtol=0.0,
        atol=2e-3,
    ):
        raise ValueError("O0 descriptors must use the unit L2 gauge")
    if bool(valid.any()) and not torch.allclose(
        teacher_norm[valid],
        torch.ones_like(teacher_norm[valid]),
        rtol=0.0,
        atol=2e-3,
    ):
        raise ValueError("valid teacher means must use the unit L2 gauge")
    if bool((teacher[~valid] != 0).any()):
        raise ValueError("invalid teacher means must be exact zero")

    count_sufficiency = ((count - 1.0) / (RETAINED_VIEW_CAPACITY - 1)).clamp(
        0.0, 1.0
    )
    agreement_available = teacher_view_directional_resultant is not None
    if agreement_available:
        agreement = _aligned_scalar(
            teacher_view_directional_resultant,
            prefix=prefix,
            label=VIEW_AGREEMENT_SCALAR,
            device=base.device,
            dtype=base.dtype,
        )
        if bool((agreement[~valid] != 0).any()):
            raise ValueError("invalid teacher agreement must be exact zero")
    else:
        agreement = torch.zeros(prefix, device=base.device, dtype=base.dtype)

    responsibility = (
        torch.ones(prefix, device=base.device, dtype=base.dtype)
        if responsibility_reliability is None
        else _aligned_scalar(
            responsibility_reliability,
            prefix=prefix,
            label="responsibility_reliability",
            device=base.device,
            dtype=base.dtype,
        )
    )
    field = (
        torch.ones(prefix, device=base.device, dtype=base.dtype)
        if canonical_field_reliability is None
        else _aligned_scalar(
            canonical_field_reliability,
            prefix=prefix,
            label="canonical_field_reliability",
            device=base.device,
            dtype=base.dtype,
        )
    )
    reliability = (
        torch.sqrt((count_sufficiency * agreement).clamp_min(0.0))
        * responsibility
        * field
    ).clamp(0.0, 1.0)
    reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
    budget = CONSERVATIVE_ANGLE_RADIANS + (
        MAXIMUM_ANGLE_RADIANS - CONSERVATIVE_ANGLE_RADIANS
    ) * reliability

    # Calculate in float32 even for the FP16 durable payload, then cast the
    # active result back.  Exact fallback rows select the untouched input.
    base_unit = torch.nn.functional.normalize(base.float(), dim=-1)
    teacher_unit = torch.nn.functional.normalize(teacher.float(), dim=-1)
    dot = (base_unit * teacher_unit[..., None, :]).sum(dim=-1).clamp(-1.0, 1.0)
    requested = torch.acos(dot)
    tangent = teacher_unit[..., None, :] - dot[..., None] * base_unit
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
    eps = 8.0 * torch.finfo(torch.float32).eps
    same_direction = (tangent_norm <= eps) & (dot >= 0.0)
    antipodal = (tangent_norm <= eps) & (dot < 0.0)
    row_valid = valid[..., None]
    teacher_applied = row_valid & ~same_direction & ~antipodal
    step = torch.minimum(requested, budget.float()[..., None])
    step = torch.where(teacher_applied, step, torch.zeros_like(step))
    unit_tangent = tangent / tangent_norm[..., None].clamp_min(eps)
    projected = (
        torch.cos(step)[..., None] * base_unit
        + torch.sin(step)[..., None] * unit_tangent
    )
    projected = torch.nn.functional.normalize(projected, dim=-1).to(base.dtype)
    descriptor = torch.where(teacher_applied[..., None], projected, base)
    fallback = ~teacher_applied
    return ReliabilityGeodesicBudgetOutput(
        descriptor=descriptor,
        reliability_score=reliability,
        count_sufficiency=count_sufficiency,
        view_agreement=agreement,
        responsibility_attenuation=responsibility,
        field_attenuation=field,
        angular_budget_radians=budget,
        requested_angle_radians=requested,
        angular_step_radians=step,
        expanded_budget=valid & (reliability > 0.0),
        teacher_applied=teacher_applied,
        fallback_to_o0=fallback,
        agreement_available=agreement_available,
    )


__all__ = [
    "CONSERVATIVE_ANGLE_RADIANS",
    "MAXIMUM_ANGLE_RADIANS",
    "RELIABILITY_GEODESIC_BUDGET_CONTRACT_SHA256",
    "RETAINED_VIEW_CAPACITY",
    "ReliabilityGeodesicBudgetOutput",
    "VIEW_AGREEMENT_SCALAR",
    "VIEW_AGREEMENT_SHA256_FIELD",
    "reliability_conditioned_geodesic_fusion",
    "reliability_geodesic_budget_contract",
]
