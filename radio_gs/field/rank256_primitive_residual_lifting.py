"""Query-free lifting of rank-256 region residuals onto O0 primitives.

The rank in the module name refers to the source-promoted V2.1B model, not
the descriptor dimension.  In production both region and O0 descriptors use
the 1536-dimensional official SigLIP2 gauge; keeping the final dimension
generic makes the geometric contract testable with small tensors.

This module owns no text query, label, mask, renderer, or metric.  It treats
the frozen O0 multiscale primitive descriptors as the capability carrier and
uses the rank-256 region field only as a bounded, query-independent spherical
residual.  Every fallback is implemented by cloning O0 and writing only the
explicitly updated entries, so uncovered and invalid primitives retain their
original bytes (including signed zero).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SCHEMA = "radio_gs.rank256_region_residual_to_o0_primitive_multiscale.v1"
_UNIT_ATOL = 2e-4
_GEOMETRY_EPS = 1e-12


def lifting_contract() -> dict[str, Any]:
    """Return the static, query-free lifting contract."""

    return {
        "schema": SCHEMA,
        "canonical_capability": "immutable_o0_multiscale_primitive_descriptor",
        "region_residual": "unit_sphere_log_map_rank256_base_to_semantic",
        "region_gate": (
            "active_and_not_ood_and_reliability_at_least_source_fixed_minimum"
        ),
        "overlap_aggregation": (
            "canonical_region_index_ordered_fp64_reliability_weighted_sum_"
            "divided_by_contributor_count"
        ),
        "primitive_transport": (
            "ambient_overlap_residual_projected_to_each_o0_scale_tangent_plane"
        ),
        "primitive_update": "unit_sphere_exponential_map_with_hard_angular_cap",
        "scale_axis": "original_o0_scale_order_preserved_without_reduction",
        "fallback": (
            "uncovered_invalid_zero_tangent_inactive_or_ood_is_bitwise_o0"
        ),
        "canonical_order": "unique_nonnegative_canonical_region_index",
        "query_conditioned_parameters": False,
        "scene_conditioned_parameters": False,
        "target_metrics_used": False,
    }


CONTRACT_SHA256 = canonical_json_sha256(lifting_contract())


@dataclass(frozen=True)
class Rank256PrimitiveLiftingConfig:
    """Numerical choices that an external source-only authority must bind."""

    max_angle_radians: float
    minimum_region_reliability: float = 0.0

    def __post_init__(self) -> None:
        angle = float(self.max_angle_radians)
        threshold = float(self.minimum_region_reliability)
        if not math.isfinite(angle) or not 0.0 < angle < math.pi / 2.0:
            raise ValueError("max_angle_radians must lie strictly in (0,pi/2)")
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_region_reliability must lie in [0,1]")

    def to_dict(self) -> dict[str, float]:
        return {
            "max_angle_radians": float(self.max_angle_radians),
            "minimum_region_reliability": float(
                self.minimum_region_reliability
            ),
        }


@dataclass(frozen=True)
class Rank256PrimitiveLiftingOutput:
    """Lifted descriptors and compact audit diagnostics.

    ``primitive_descriptor`` has the same dtype, shape, device (CPU),
    primitive order, and scale order as O0.  All floating diagnostics are
    FP64 so the aggregation and spherical geometry can be audited without
    inferring them from a possibly FP16 descriptor artifact.
    """

    primitive_descriptor: torch.Tensor
    region_contribution_mask: torch.Tensor
    coverage_count: torch.Tensor
    aggregate_reliability: torch.Tensor
    aggregate_residual_norm: torch.Tensor
    angular_step_radians: torch.Tensor
    updated_mask: torch.Tensor
    fallback_mask: torch.Tensor


def _validate_inputs(
    *,
    o0_primitive_descriptor: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
    region_base_descriptor: torch.Tensor,
    region_semantic_descriptor: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    region_reliability: torch.Tensor,
    region_active_mask: torch.Tensor,
    region_ood_mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    o0 = torch.as_tensor(o0_primitive_descriptor).detach()
    valid = torch.as_tensor(primitive_valid_mask).detach()
    region_base = torch.as_tensor(region_base_descriptor).detach()
    region_semantic = torch.as_tensor(region_semantic_descriptor).detach()
    rows = torch.as_tensor(region_rows).detach()
    tokens = torch.as_tensor(token_mask).detach()
    canonical = torch.as_tensor(canonical_region_indices).detach()
    reliability = torch.as_tensor(region_reliability).detach()
    active = torch.as_tensor(region_active_mask).detach()
    ood = torch.as_tensor(region_ood_mask).detach()

    tensors = (
        o0,
        valid,
        region_base,
        region_semantic,
        rows,
        tokens,
        canonical,
        reliability,
        active,
        ood,
    )
    if any(value.device.type != "cpu" for value in tensors):
        raise ValueError("canonical primitive lifting requires CPU tensors")
    if (
        o0.ndim != 3
        or min(o0.shape) <= 0
        or not o0.is_floating_point()
        or not bool(torch.isfinite(o0).all())
    ):
        raise ValueError("O0 descriptors must be finite floating CPU [N,S,D]")
    primitive_count, _scale_count, descriptor_dim = o0.shape
    if descriptor_dim < 2:
        raise ValueError("descriptor dimension must be at least two")
    if valid.dtype != torch.bool or valid.shape != (primitive_count,):
        raise ValueError("primitive_valid_mask must be boolean [N]")
    if bool(valid.any()):
        valid_norm = torch.linalg.vector_norm(o0.double(), dim=-1)[valid]
        if not torch.allclose(
            valid_norm,
            torch.ones_like(valid_norm),
            rtol=0.0,
            atol=_UNIT_ATOL,
        ):
            raise ValueError("valid O0 descriptors must use the unit L2 gauge")

    if (
        region_base.ndim != 2
        or region_base.shape[1] != descriptor_dim
        or region_base.shape != region_semantic.shape
        or not region_base.is_floating_point()
        or not region_semantic.is_floating_point()
        or not bool(torch.isfinite(region_base).all())
        or not bool(torch.isfinite(region_semantic).all())
    ):
        raise ValueError("region descriptors must be finite floating [R,D]")
    region_count = int(region_base.shape[0])
    if region_count <= 0:
        raise ValueError("region axis must be nonempty")
    for value, label in (
        (region_base, "region base"),
        (region_semantic, "region semantic"),
    ):
        norm = torch.linalg.vector_norm(value.double(), dim=-1)
        if not torch.allclose(
            norm, torch.ones_like(norm), rtol=0.0, atol=_UNIT_ATOL
        ):
            raise ValueError(f"{label} descriptors must use the unit L2 gauge")

    if (
        rows.ndim != 2
        or rows.shape[0] != region_count
        or rows.dtype not in {torch.int32, torch.int64}
        or tokens.dtype != torch.bool
        or tokens.shape != rows.shape
        or not bool(tokens.any(dim=1).all())
    ):
        raise ValueError("region_rows/token_mask must align as nonempty [R,T]")
    active_rows = rows[tokens]
    if bool((active_rows < 0).any()) or bool(
        (active_rows >= primitive_count).any()
    ):
        raise ValueError("active region token contains an out-of-range primitive")
    for region_index in range(region_count):
        member = rows[region_index, tokens[region_index]].long()
        if int(torch.unique(member).numel()) != int(member.numel()):
            raise ValueError("one region contains a duplicate primitive")

    if (
        canonical.dtype != torch.int64
        or canonical.shape != (region_count,)
        or bool((canonical < 0).any())
        or int(torch.unique(canonical).numel()) != region_count
    ):
        raise ValueError(
            "canonical_region_indices must be unique nonnegative int64 [R]"
        )
    if (
        reliability.shape != (region_count,)
        or not reliability.is_floating_point()
        or not bool(torch.isfinite(reliability).all())
        or bool((reliability < 0.0).any())
        or bool((reliability > 1.0).any())
    ):
        raise ValueError("region_reliability must be finite floating [R] in [0,1]")
    if (
        active.dtype != torch.bool
        or active.shape != (region_count,)
        or ood.dtype != torch.bool
        or ood.shape != (region_count,)
    ):
        raise ValueError("region active/OOD masks must be boolean [R]")
    return (
        o0.contiguous(),
        valid.contiguous(),
        region_base.contiguous(),
        region_semantic.contiguous(),
        rows.long().contiguous(),
        tokens.contiguous(),
        canonical.contiguous(),
        reliability.contiguous(),
        active.contiguous(),
        ood.contiguous(),
    )


def _sphere_log_map(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the non-antipodal unit-sphere log map in FP64."""

    base64 = base.double()
    target64 = target.double()
    cosine = (base64 * target64).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    tangent = target64 - cosine * base64
    sine = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    if bool(((cosine.squeeze(-1) < 0.0) & (sine.squeeze(-1) <= 1e-8)).any()):
        raise ValueError("region base/semantic descriptors must be non-antipodal")
    angle = torch.atan2(sine, cosine)
    scale = torch.where(
        sine > _GEOMETRY_EPS,
        angle / sine.clamp_min(_GEOMETRY_EPS),
        torch.ones_like(sine),
    )
    return (scale * tangent).contiguous()


def _canonical_fp64_overlap_aggregate(
    *,
    primitive_count: int,
    region_log_residual: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    reliability: torch.Tensor,
    contribution_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate overlaps in canonical region order with FP64 arithmetic."""

    descriptor_dim = int(region_log_residual.shape[1])
    order = torch.argsort(canonical_region_indices)
    ordered_rows = region_rows[order]
    ordered_tokens = token_mask[order]
    ordered_reliability = reliability[order].double()
    ordered_contribution = contribution_mask[order]
    active_tokens = ordered_tokens & ordered_contribution[:, None]
    if not bool(active_tokens.any()):
        return (
            torch.zeros((primitive_count, descriptor_dim), dtype=torch.float64),
            torch.zeros(primitive_count, dtype=torch.int64),
            torch.zeros(primitive_count, dtype=torch.float64),
        )

    flat_rows = ordered_rows[active_tokens].long()
    flat_region = (
        torch.arange(len(order), dtype=torch.long)[:, None]
        .expand_as(ordered_rows)[active_tokens]
    )
    flat_weight = ordered_reliability[:, None].expand_as(ordered_rows)[
        active_tokens
    ]
    # Coalescing canonicalizes sparse entries to (primitive, canonical region)
    # order.  Duplicate (primitive, region) pairs were rejected at validation.
    incidence = torch.sparse_coo_tensor(
        torch.stack((flat_rows, flat_region)),
        flat_weight,
        size=(primitive_count, len(order)),
        dtype=torch.float64,
        device="cpu",
    ).coalesce()
    ordered_log = region_log_residual[order].double().contiguous()
    weighted_sum = torch.sparse.mm(incidence, ordered_log)
    coverage = torch.bincount(flat_rows, minlength=primitive_count).long()
    reliability_sum = torch.bincount(
        flat_rows,
        weights=flat_weight,
        minlength=primitive_count,
    ).double()
    denominator = coverage.clamp_min(1).double()
    aggregate = weighted_sum / denominator[:, None]
    aggregate_reliability = reliability_sum / denominator
    return (
        aggregate.contiguous(),
        coverage.contiguous(),
        aggregate_reliability.contiguous(),
    )


def lift_rank256_region_residual_to_o0_multiscale(
    *,
    o0_primitive_descriptor: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
    region_base_descriptor: torch.Tensor,
    region_semantic_descriptor: torch.Tensor,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    region_reliability: torch.Tensor,
    region_active_mask: torch.Tensor,
    region_ood_mask: torch.Tensor,
    config: Rank256PrimitiveLiftingConfig,
) -> Rank256PrimitiveLiftingOutput:
    """Lift a query-free region residual into every native O0 scale.

    Region residuals are first represented by a unit-sphere log map.  Usable
    overlapping regions are then accumulated in canonical order and FP64.
    Division is by contributor count, rather than reliability sum, so
    reliability attenuates both isolated and overlapping evidence instead of
    cancelling out for a single contributor.  The aggregate ambient vector is
    projected independently onto each O0 scale's tangent plane and applied by
    a hard-bounded exponential map.
    """

    if not isinstance(config, Rank256PrimitiveLiftingConfig):
        raise TypeError("config must be Rank256PrimitiveLiftingConfig")
    (
        o0,
        valid,
        region_base,
        region_semantic,
        rows,
        tokens,
        canonical,
        reliability,
        active,
        ood,
    ) = _validate_inputs(
        o0_primitive_descriptor=o0_primitive_descriptor,
        primitive_valid_mask=primitive_valid_mask,
        region_base_descriptor=region_base_descriptor,
        region_semantic_descriptor=region_semantic_descriptor,
        region_rows=region_rows,
        token_mask=token_mask,
        canonical_region_indices=canonical_region_indices,
        region_reliability=region_reliability,
        region_active_mask=region_active_mask,
        region_ood_mask=region_ood_mask,
    )
    contribution = (
        active
        & ~ood
        & (reliability > 0.0)
        & (reliability >= float(config.minimum_region_reliability))
    )
    region_log = _sphere_log_map(region_base, region_semantic)
    aggregate, coverage, aggregate_reliability = (
        _canonical_fp64_overlap_aggregate(
            primitive_count=int(o0.shape[0]),
            region_log_residual=region_log,
            region_rows=rows,
            token_mask=tokens,
            canonical_region_indices=canonical,
            reliability=reliability,
            contribution_mask=contribution,
        )
    )

    base64 = o0.double()
    ambient = aggregate[:, None, :]
    tangent = ambient - (ambient * base64).sum(dim=-1, keepdim=True) * base64
    requested_angle = torch.linalg.vector_norm(tangent, dim=-1)
    permitted = valid[:, None] & (coverage[:, None] > 0)
    angle = requested_angle.clamp_max(float(config.max_angle_radians))
    updated = permitted & (angle > _GEOMETRY_EPS)
    unit_tangent = tangent / requested_angle[..., None].clamp_min(_GEOMETRY_EPS)
    candidate64 = (
        torch.cos(angle)[..., None] * base64
        + torch.sin(angle)[..., None] * unit_tangent
    )
    candidate64 = candidate64 / torch.linalg.vector_norm(
        candidate64, dim=-1, keepdim=True
    ).clamp_min(_GEOMETRY_EPS)

    # Assignment, rather than a whole-tensor where, is the bitwise fallback
    # authority for signed zero and low-precision O0 caches.
    output = o0.clone()
    candidate = candidate64.to(dtype=o0.dtype)
    output[updated] = candidate[updated]
    fallback = ~updated
    if not torch.equal(output[fallback].view(torch.uint8), o0[fallback].view(torch.uint8)):
        raise RuntimeError("primitive lifting changed an O0 fallback byte")
    output_norm = torch.linalg.vector_norm(output.double(), dim=-1)
    if bool(valid.any()) and not torch.allclose(
        output_norm[valid],
        torch.ones_like(output_norm[valid]),
        rtol=0.0,
        atol=8e-4,
    ):
        raise RuntimeError("lifted primitive descriptor left the unit L2 gauge")
    if bool((angle[updated] > float(config.max_angle_radians) + 1e-12).any()):
        raise RuntimeError("primitive lifting exceeded the angular cap")

    return Rank256PrimitiveLiftingOutput(
        primitive_descriptor=output.contiguous(),
        region_contribution_mask=contribution.contiguous(),
        coverage_count=coverage,
        aggregate_reliability=aggregate_reliability,
        aggregate_residual_norm=torch.linalg.vector_norm(
            aggregate, dim=-1
        ).contiguous(),
        angular_step_radians=torch.where(
            updated, angle, torch.zeros_like(angle)
        ).contiguous(),
        updated_mask=updated.contiguous(),
        fallback_mask=fallback.contiguous(),
    )


__all__ = [
    "CONTRACT_SHA256",
    "Rank256PrimitiveLiftingConfig",
    "Rank256PrimitiveLiftingOutput",
    "SCHEMA",
    "lift_rank256_region_residual_to_o0_multiscale",
    "lifting_contract",
]
