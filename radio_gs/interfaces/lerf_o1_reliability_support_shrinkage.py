"""Query-free, row-streamable reliability support shrinkage for LERF O1.

O1 preserves three physical scale slots and moves every primitive a bounded
distance toward its own multi-view teacher mean.  This interface addresses a
different error source: an uncertain primitive may remain a weak local
estimate even when reliable primitives in the same query-independent
SurfaceRegion agree on a semantic direction.

The implementation is deliberately two-stage.  Planning stores only one
region index and four scalars per primitive.  Materialization recomputes the
chosen region's leave-one-out spherical mean for an explicit output row batch.
It never allocates another full ``[N,3,D]`` target or output tensor.  The
consumer may therefore stream batches directly into a new immutable cache.

This is conservative *within-region unary/support repair*, not a multi-region
query selector.  It has no query, text, dataset, renderer, label, mask, or
metric entry point.  Region reliability must come from a separately sealed
source-only authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from radio_gs.utils.immutable_artifacts import canonical_json_sha256


SCHEMA = "radio_gs.lerf_o1_reliability_support_shrinkage.v2"
SCALE_SLOTS = 3
MAXIMUM_ROTATION_RADIANS = 0.15
_EPS = 1.0e-8
_VALIDATION_ROWS = 8192


def support_shrinkage_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "input_field": "O1_three_scale_unit_direction_float16_or_float32",
        "support": "query_independent_SurfaceRegion_semantic_core",
        "method_scope": "within_region_unary_support_repair_not_multi_region_selector",
        "region_reliability": "sealed_source_only_query_free_probability",
        "estimator": "reliability_weighted_leave_one_out_spherical_mean",
        "support_strength": (
            "primitive_uncertainty_times_region_reliability_times_"
            "leave_one_out_resultant_times_positive_cosine"
        ),
        "region_choice": (
            "maximum_minimum_support_across_three_scales_then_lower_"
            "canonical_region_index"
        ),
        "rotation": (
            "independent_per_scale_geodesic_step_bounded_by_"
            "0p15_times_per_scale_support"
        ),
        "memory": {
            "planning": "O(N_times_three)_scalars_no_N_times_three_times_D_target",
            "materialization": "explicit_output_row_batch_only",
            "full_output_clone": False,
        },
        "fallback": (
            "invalid_singleton_zero_weight_nonpositive_agreement_or_"
            "cross_scale_inconsistent_rows_are_bitwise_O1"
        ),
        "query_axis_consumed": False,
        "scale_axis_collapsed": False,
        "scene_or_query_parameters": False,
        "target_images_masks_labels_or_metrics": False,
    }


CONTRACT_SHA256 = canonical_json_sha256(support_shrinkage_contract())


@dataclass(frozen=True)
class ReliabilitySupportShrinkagePlan:
    support_strength: torch.Tensor
    per_scale_support_strength: torch.Tensor
    selected_region_rows: torch.Tensor
    selected_canonical_region_indices: torch.Tensor


@dataclass(frozen=True)
class ReliabilitySupportShrinkageBatch:
    primitive_rows: torch.Tensor
    descriptors: torch.Tensor
    rotation_radians: torch.Tensor
    changed_scale_mask: torch.Tensor


def _validated_inputs(
    *,
    o1_descriptors: torch.Tensor,
    primitive_reliability: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    region_reliability: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    validate_all_descriptor_rows: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    descriptor = torch.as_tensor(o1_descriptors).detach()
    primitive_precision = torch.as_tensor(primitive_reliability).detach()
    primitive_valid = torch.as_tensor(primitive_valid_mask).detach()
    rows = torch.as_tensor(region_rows).detach()
    core = torch.as_tensor(core_mask).detach()
    region_precision = torch.as_tensor(region_reliability).detach()
    canonical = torch.as_tensor(canonical_region_indices).detach()
    if (
        descriptor.device.type != "cpu"
        or descriptor.dtype not in {torch.float16, torch.float32}
        or descriptor.ndim != 3
        or descriptor.shape[1] != SCALE_SLOTS
        or descriptor.shape[0] <= 0
        or descriptor.shape[2] <= 1
        or not descriptor.is_contiguous()
    ):
        raise ValueError("O1 descriptors must be CPU float16/float32 [N,3,D]")
    primitive_count = int(descriptor.shape[0])
    if validate_all_descriptor_rows:
        for start in range(0, primitive_count, _VALIDATION_ROWS):
            chunk = descriptor[start : start + _VALIDATION_ROWS].float()
            norm = torch.linalg.vector_norm(chunk, dim=-1)
            if not bool(torch.isfinite(chunk).all()) or not torch.allclose(
                norm, torch.ones_like(norm), atol=3.0e-3, rtol=0.0
            ):
                raise ValueError("O1 descriptor directions must be finite unit vectors")
    if (
        primitive_precision.device.type != "cpu"
        or primitive_precision.dtype != torch.float32
        or primitive_precision.shape != (primitive_count,)
        or not bool(torch.isfinite(primitive_precision).all())
        or bool((primitive_precision < 0.0).any())
        or bool((primitive_precision > 1.0).any())
        or primitive_valid.device.type != "cpu"
        or primitive_valid.dtype != torch.bool
        or primitive_valid.shape != (primitive_count,)
        or bool(primitive_precision[~primitive_valid].count_nonzero())
    ):
        raise ValueError("primitive reliability/validity axes differ")
    region_count = int(rows.shape[0]) if rows.ndim == 2 else -1
    if (
        rows.device.type != "cpu"
        or rows.dtype not in {torch.int32, torch.int64}
        or region_count <= 0
        or core.device.type != "cpu"
        or core.dtype != torch.bool
        or core.shape != rows.shape
        or not bool(core.any(dim=1).all())
        or bool((rows[core] < 0).any())
        or bool((rows[core] >= primitive_count).any())
        or region_precision.device.type != "cpu"
        or region_precision.dtype != torch.float32
        or region_precision.shape != (region_count,)
        or not bool(torch.isfinite(region_precision).all())
        or bool((region_precision < 0.0).any())
        or bool((region_precision > 1.0).any())
        or canonical.device.type != "cpu"
        or canonical.dtype != torch.int64
        or canonical.shape != (region_count,)
        or bool((canonical < 0).any())
        or int(torch.unique(canonical).numel()) != region_count
    ):
        raise ValueError("SurfaceRegion support axes differ")
    for index in range(region_count):
        active = rows[index, core[index]].long()
        if int(torch.unique(active).numel()) != int(active.numel()):
            raise ValueError("a SurfaceRegion core contains duplicate primitives")
    return (
        descriptor,
        primitive_precision.contiguous(),
        primitive_valid.contiguous(),
        rows.long().contiguous(),
        core.contiguous(),
        region_precision.contiguous(),
        canonical.contiguous(),
    )


def _region_leave_one_out_support(
    *,
    descriptor: torch.Tensor,
    primitive_precision: torch.Tensor,
    primitive_valid: torch.Tensor,
    rows: torch.Tensor,
    core: torch.Tensor,
    region_precision: torch.Tensor,
    region: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    member = rows[region, core[region]].long()
    member = member[primitive_valid[member]]
    empty_mean = torch.empty(0, SCALE_SLOTS, descriptor.shape[2], dtype=torch.float32)
    empty_scale = torch.empty(0, SCALE_SLOTS, dtype=torch.float32)
    empty_joint = torch.empty(0, dtype=torch.float32)
    if member.numel() < 2 or float(region_precision[region]) <= 0.0:
        return member, empty_mean, empty_scale, empty_joint
    directions = descriptor[member].float()
    if not bool(torch.isfinite(directions).all()) or not torch.allclose(
        torch.linalg.vector_norm(directions, dim=-1),
        torch.ones(member.numel(), SCALE_SLOTS),
        atol=3.0e-3,
        rtol=0.0,
    ):
        raise ValueError("SurfaceRegion contains an invalid O1 direction")
    weights = primitive_precision[member]
    total_weight = weights.sum()
    weighted_sum = (directions * weights[:, None, None]).sum(dim=0)
    loo_weight = total_weight - weights
    loo_sum = weighted_sum[None] - weights[:, None, None] * directions
    loo_norm = torch.linalg.vector_norm(loo_sum, dim=-1)
    available = (loo_weight[:, None] > _EPS) & (loo_norm > _EPS)
    loo_mean = loo_sum / loo_norm.clamp_min(_EPS)[..., None]
    resultant = loo_norm / loo_weight[:, None].clamp_min(_EPS)
    cosine = (directions * loo_mean).sum(dim=-1).clamp(0.0, 1.0)
    per_scale = (
        (1.0 - weights)[:, None]
        * region_precision[region]
        * resultant.clamp(0.0, 1.0)
        * cosine
    )
    per_scale = torch.where(available, per_scale, torch.zeros_like(per_scale))
    return member, loo_mean, per_scale, per_scale.amin(dim=1)


def plan_reliability_conditioned_support_shrinkage(
    *,
    o1_descriptors: torch.Tensor,
    primitive_reliability: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    region_reliability: torch.Tensor,
    canonical_region_indices: torch.Tensor,
) -> ReliabilitySupportShrinkagePlan:
    """Choose support regions without retaining any full descriptor target."""

    (
        descriptor,
        primitive_precision,
        primitive_valid,
        rows,
        core,
        region_precision,
        canonical,
    ) = _validated_inputs(
        o1_descriptors=o1_descriptors,
        primitive_reliability=primitive_reliability,
        primitive_valid_mask=primitive_valid_mask,
        region_rows=region_rows,
        core_mask=core_mask,
        region_reliability=region_reliability,
        canonical_region_indices=canonical_region_indices,
        validate_all_descriptor_rows=True,
    )
    primitive_count = int(descriptor.shape[0])
    best_strength = torch.zeros(primitive_count, dtype=torch.float32)
    best_per_scale = torch.zeros(primitive_count, SCALE_SLOTS, dtype=torch.float32)
    best_row = torch.full((primitive_count,), -1, dtype=torch.long)
    best_canonical = torch.full(
        (primitive_count,), torch.iinfo(torch.long).max, dtype=torch.long
    )
    for region in range(rows.shape[0]):
        member, _loo_mean, per_scale, joint = _region_leave_one_out_support(
            descriptor=descriptor,
            primitive_precision=primitive_precision,
            primitive_valid=primitive_valid,
            rows=rows,
            core=core,
            region_precision=region_precision,
            region=region,
        )
        if joint.numel() == 0:
            continue
        better = joint > best_strength[member]
        tied = (joint == best_strength[member]) & (joint > 0.0)
        lower_index = canonical[region] < best_canonical[member]
        update = better | (tied & lower_index)
        if not bool(update.any()):
            continue
        selected = member[update]
        best_strength[selected] = joint[update]
        best_per_scale[selected] = per_scale[update]
        best_row[selected] = int(region)
        best_canonical[selected] = canonical[region]
    selected = best_row >= 0
    best_canonical[~selected] = -1
    if (
        bool(selected[~primitive_valid].any())
        or bool((best_strength[~selected] != 0.0).any())
        or bool((best_per_scale[~selected] != 0.0).any())
        or bool((best_strength[selected] <= 0.0).any())
        or not torch.equal(
            best_strength[selected], best_per_scale[selected].amin(dim=1)
        )
        or not torch.equal(best_canonical[selected], canonical[best_row[selected]])
    ):
        raise RuntimeError("O1 reliability support plan invariant failed")
    return ReliabilitySupportShrinkagePlan(
        support_strength=best_strength.contiguous(),
        per_scale_support_strength=best_per_scale.contiguous(),
        selected_region_rows=best_row.contiguous(),
        selected_canonical_region_indices=best_canonical.contiguous(),
    )


def _validated_plan(
    plan: ReliabilitySupportShrinkagePlan,
    *,
    primitive_valid: torch.Tensor,
    canonical: torch.Tensor,
) -> ReliabilitySupportShrinkagePlan:
    if not isinstance(plan, ReliabilitySupportShrinkagePlan):
        raise TypeError("plan must be ReliabilitySupportShrinkagePlan")
    count = int(primitive_valid.numel())
    strength = torch.as_tensor(plan.support_strength)
    per_scale = torch.as_tensor(plan.per_scale_support_strength)
    selected = torch.as_tensor(plan.selected_region_rows)
    selected_canonical = torch.as_tensor(plan.selected_canonical_region_indices)
    active = selected >= 0
    if (
        strength.device.type != "cpu"
        or strength.dtype != torch.float32
        or strength.shape != (count,)
        or per_scale.device.type != "cpu"
        or per_scale.dtype != torch.float32
        or per_scale.shape != (count, SCALE_SLOTS)
        or selected.device.type != "cpu"
        or selected.dtype != torch.int64
        or selected.shape != (count,)
        or selected_canonical.device.type != "cpu"
        or selected_canonical.dtype != torch.int64
        or selected_canonical.shape != (count,)
        or bool((selected < -1).any())
        or bool((selected >= canonical.numel()).any())
        or bool(active[~primitive_valid].any())
        or bool((strength[~active] != 0.0).any())
        or bool((per_scale[~active] != 0.0).any())
        or bool((selected_canonical[~active] != -1).any())
        or bool((strength[active] <= 0.0).any())
        or not torch.equal(strength[active], per_scale[active].amin(dim=1))
        or not torch.equal(selected_canonical[active], canonical[selected[active]])
    ):
        raise ValueError("O1 reliability support plan differs")
    return plan


def apply_reliability_conditioned_support_shrinkage_rows(
    *,
    o1_descriptors: torch.Tensor,
    primitive_reliability: torch.Tensor,
    primitive_valid_mask: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    region_reliability: torch.Tensor,
    canonical_region_indices: torch.Tensor,
    plan: ReliabilitySupportShrinkagePlan,
    output_rows: torch.Tensor,
) -> ReliabilitySupportShrinkageBatch:
    """Materialize only the requested primitive rows from a frozen plan."""

    (
        descriptor,
        primitive_precision,
        primitive_valid,
        rows,
        core,
        region_precision,
        canonical,
    ) = _validated_inputs(
        o1_descriptors=o1_descriptors,
        primitive_reliability=primitive_reliability,
        primitive_valid_mask=primitive_valid_mask,
        region_rows=region_rows,
        core_mask=core_mask,
        region_reliability=region_reliability,
        canonical_region_indices=canonical_region_indices,
        validate_all_descriptor_rows=False,
    )
    checked = _validated_plan(
        plan, primitive_valid=primitive_valid, canonical=canonical
    )
    requested = torch.as_tensor(output_rows).detach()
    if (
        requested.device.type != "cpu"
        or requested.dtype != torch.int64
        or requested.ndim != 1
        or requested.numel() <= 0
        or int(torch.unique(requested).numel()) != int(requested.numel())
        or int(requested.min()) < 0
        or int(requested.max()) >= descriptor.shape[0]
    ):
        raise ValueError("output_rows must be unique in-range CPU int64 rows")
    requested_descriptor = descriptor[requested].float()
    if not bool(torch.isfinite(requested_descriptor).all()) or not torch.allclose(
        torch.linalg.vector_norm(requested_descriptor, dim=-1),
        torch.ones(requested.numel(), SCALE_SLOTS),
        atol=3.0e-3,
        rtol=0.0,
    ):
        raise ValueError("requested O1 descriptor rows must be finite unit vectors")
    output = descriptor[requested].clone()
    rotation = torch.zeros(requested.numel(), SCALE_SLOTS, dtype=torch.float32)
    requested_region = checked.selected_region_rows[requested]
    for region_tensor in torch.unique(requested_region[requested_region >= 0]):
        region = int(region_tensor)
        batch_positions = torch.where(requested_region == region)[0]
        selected_rows = requested[batch_positions]
        member, loo_mean, per_scale, joint = _region_leave_one_out_support(
            descriptor=descriptor,
            primitive_precision=primitive_precision,
            primitive_valid=primitive_valid,
            rows=rows,
            core=core,
            region_precision=region_precision,
            region=region,
        )
        position_by_row = {int(row): index for index, row in enumerate(member.tolist())}
        if any(int(row) not in position_by_row for row in selected_rows.tolist()):
            raise RuntimeError("support plan selected a row outside its region")
        member_positions = torch.tensor(
            [position_by_row[int(row)] for row in selected_rows.tolist()],
            dtype=torch.long,
        )
        if not torch.equal(
            per_scale[member_positions],
            checked.per_scale_support_strength[selected_rows],
        ) or not torch.equal(
            joint[member_positions], checked.support_strength[selected_rows]
        ):
            raise RuntimeError("support plan replay differs")
        source = descriptor[selected_rows].float()
        target = loo_mean[member_positions]
        angle = torch.acos((source * target).sum(dim=-1).clamp(-1.0, 1.0))
        maximum_step = (
            MAXIMUM_ROTATION_RADIANS * checked.per_scale_support_strength[selected_rows]
        )
        step = torch.minimum(angle, maximum_step)
        fraction = torch.where(
            angle > _EPS, step / angle.clamp_min(_EPS), torch.zeros_like(angle)
        )
        sin_angle = torch.sin(angle)
        regular = sin_angle.abs() > _EPS
        source_weight = torch.where(
            regular,
            torch.sin((1.0 - fraction) * angle) / sin_angle.clamp_min(_EPS),
            1.0 - fraction,
        )
        target_weight = torch.where(
            regular,
            torch.sin(fraction * angle) / sin_angle.clamp_min(_EPS),
            fraction,
        )
        fused = source_weight[..., None] * source + target_weight[..., None] * target
        fused = fused / torch.linalg.vector_norm(fused, dim=-1, keepdim=True).clamp_min(
            _EPS
        )
        output[batch_positions] = fused.to(dtype=descriptor.dtype)
        rotation[batch_positions] = step
    changed = rotation > 0.0
    inactive = requested_region < 0
    if (
        not torch.equal(output[inactive], descriptor[requested[inactive]])
        or bool(changed[~primitive_valid[requested]].any())
        or float(rotation.max()) > MAXIMUM_ROTATION_RADIANS + 1.0e-7
    ):
        raise RuntimeError("O1 reliability support batch invariant failed")
    return ReliabilitySupportShrinkageBatch(
        primitive_rows=requested.contiguous(),
        descriptors=output.contiguous(),
        rotation_radians=rotation.contiguous(),
        changed_scale_mask=changed.contiguous(),
    )


def access_audit() -> dict[str, bool]:
    return {
        "o1_query_free_descriptor_opened": True,
        "primitive_reliability_opened": True,
        "query_independent_region_support_opened": True,
        "source_only_region_reliability_opened": True,
        "query_ids_or_text_opened": False,
        "benchmark_images_masks_labels_or_metrics_opened": False,
        "scene_specific_parameters": False,
    }


__all__ = [
    "CONTRACT_SHA256",
    "MAXIMUM_ROTATION_RADIANS",
    "ReliabilitySupportShrinkageBatch",
    "ReliabilitySupportShrinkagePlan",
    "SCHEMA",
    "SCALE_SLOTS",
    "access_audit",
    "apply_reliability_conditioned_support_shrinkage_rows",
    "plan_reliability_conditioned_support_shrinkage",
    "support_shrinkage_contract",
]
