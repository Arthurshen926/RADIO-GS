from __future__ import annotations

import math

import pytest
import torch

from radio_gs.interfaces.lerf_o1_reliability_support_shrinkage import (
    MAXIMUM_ROTATION_RADIANS,
    access_audit,
    apply_reliability_conditioned_support_shrinkage_rows,
    plan_reliability_conditioned_support_shrinkage,
    support_shrinkage_contract,
)


def _direction(angle: float) -> torch.Tensor:
    return torch.tensor([math.cos(angle), math.sin(angle), 0.0], dtype=torch.float32)


def _descriptors(
    angles: list[float], *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    values = torch.stack([_direction(value) for value in angles])
    return values[:, None, :].repeat(1, 3, 1).to(dtype=dtype).contiguous()


def _inputs(
    descriptors: torch.Tensor,
    reliability: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
    core: torch.Tensor | None = None,
    region_reliability: torch.Tensor | None = None,
    canonical: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    rows = torch.tensor([[0, 1, 2]], dtype=torch.long) if rows is None else rows
    core = torch.ones_like(rows, dtype=torch.bool) if core is None else core
    region_count = rows.shape[0]
    return {
        "o1_descriptors": descriptors,
        "primitive_reliability": reliability,
        "primitive_valid_mask": (
            torch.ones(descriptors.shape[0], dtype=torch.bool)
            if valid is None
            else valid
        ),
        "region_rows": rows,
        "core_mask": core,
        "region_reliability": (
            torch.ones(region_count, dtype=torch.float32)
            if region_reliability is None
            else region_reliability
        ),
        "canonical_region_indices": (
            torch.arange(region_count, dtype=torch.long)
            if canonical is None
            else canonical
        ),
    }


def _plan_and_apply(inputs: dict[str, torch.Tensor], output_rows: torch.Tensor):
    plan = plan_reliability_conditioned_support_shrinkage(**inputs)
    batch = apply_reliability_conditioned_support_shrinkage_rows(
        **inputs, plan=plan, output_rows=output_rows
    )
    return plan, batch


def test_low_reliability_direction_borrows_only_bounded_peer_support() -> None:
    descriptor = _descriptors([0.60, 0.0, 0.0])
    inputs = _inputs(descriptor, torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32))
    plan, batch = _plan_and_apply(inputs, torch.arange(3, dtype=torch.long))
    assert plan.selected_region_rows.tolist() == [0, -1, -1]
    assert bool(batch.changed_scale_mask[0].all())
    assert not bool(batch.changed_scale_mask[1:].any())
    assert torch.equal(batch.descriptors[1:], descriptor[1:])
    assert 0.0 < float(batch.rotation_radians[0, 0]) <= MAXIMUM_ROTATION_RADIANS
    assert float(batch.descriptors[0, 0, 0]) > float(descriptor[0, 0, 0])


def test_row_batches_are_exactly_equivalent_to_one_full_requested_batch() -> None:
    descriptor = _descriptors([0.60, 0.45, 0.0, 0.0], dtype=torch.float16)
    inputs = _inputs(
        descriptor,
        torch.tensor([0.0, 0.25, 1.0, 1.0], dtype=torch.float32),
        rows=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
    )
    plan = plan_reliability_conditioned_support_shrinkage(**inputs)
    full = apply_reliability_conditioned_support_shrinkage_rows(
        **inputs, plan=plan, output_rows=torch.arange(4, dtype=torch.long)
    )
    first = apply_reliability_conditioned_support_shrinkage_rows(
        **inputs, plan=plan, output_rows=torch.tensor([0, 1], dtype=torch.long)
    )
    second = apply_reliability_conditioned_support_shrinkage_rows(
        **inputs, plan=plan, output_rows=torch.tensor([2, 3], dtype=torch.long)
    )
    assert full.descriptors.dtype == torch.float16
    assert torch.equal(
        full.descriptors, torch.cat([first.descriptors, second.descriptors])
    )
    assert torch.equal(
        full.rotation_radians,
        torch.cat([first.rotation_radians, second.rotation_radians]),
    )
    assert torch.equal(
        full.changed_scale_mask,
        torch.cat([first.changed_scale_mask, second.changed_scale_mask]),
    )


def test_zero_region_reliability_and_fully_reliable_rows_are_bitwise_o1() -> None:
    descriptor = _descriptors([0.4, 0.0, 0.0])
    cases = (
        _inputs(
            descriptor,
            torch.zeros(3, dtype=torch.float32),
            region_reliability=torch.zeros(1, dtype=torch.float32),
        ),
        _inputs(descriptor, torch.ones(3, dtype=torch.float32)),
    )
    for inputs in cases:
        plan, batch = _plan_and_apply(inputs, torch.arange(3, dtype=torch.long))
        assert torch.equal(batch.descriptors, descriptor)
        assert not bool(batch.changed_scale_mask.any())
        assert plan.selected_region_rows.tolist() == [-1, -1, -1]


def test_singleton_cannot_self_confirm() -> None:
    descriptor = _descriptors([0.4, 0.0])
    inputs = _inputs(
        descriptor,
        torch.zeros(2, dtype=torch.float32),
        rows=torch.tensor([[0, -1]], dtype=torch.long),
        core=torch.tensor([[True, False]]),
    )
    plan, batch = _plan_and_apply(inputs, torch.tensor([0], dtype=torch.long))
    assert int(plan.selected_region_rows[0]) == -1
    assert torch.equal(batch.descriptors, descriptor[[0]])


def test_cross_scale_disagreement_forces_conservative_fallback() -> None:
    descriptor = _descriptors([0.4, 0.0, 0.0])
    descriptor[0, 2] = _direction(math.pi)
    inputs = _inputs(descriptor, torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32))
    plan, batch = _plan_and_apply(inputs, torch.arange(3, dtype=torch.long))
    assert torch.equal(batch.descriptors, descriptor)
    assert plan.selected_region_rows.tolist() == [-1, -1, -1]
    assert not bool(batch.changed_scale_mask.any())


def test_overlapping_region_tie_uses_lower_canonical_index() -> None:
    descriptor = _descriptors([0.4, 0.0, 0.0])
    rows = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    inputs = _inputs(
        descriptor,
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32),
        rows=rows,
        core=torch.ones_like(rows, dtype=torch.bool),
        canonical=torch.tensor([9, 3], dtype=torch.long),
    )
    plan = plan_reliability_conditioned_support_shrinkage(**inputs)
    assert int(plan.selected_region_rows[0]) == 1
    assert int(plan.selected_canonical_region_indices[0]) == 3


def test_invalid_rows_are_exact_fallback_and_need_zero_reliability() -> None:
    descriptor = _descriptors([0.4, 0.0, 0.0])
    valid = torch.tensor([False, True, True])
    inputs = _inputs(
        descriptor,
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32),
        valid=valid,
    )
    plan, batch = _plan_and_apply(inputs, torch.tensor([0], dtype=torch.long))
    assert torch.equal(batch.descriptors[0], descriptor[0])
    assert int(plan.selected_region_rows[0]) == -1
    with pytest.raises(ValueError, match="primitive reliability"):
        plan_reliability_conditioned_support_shrinkage(
            **_inputs(
                descriptor,
                torch.tensor([0.1, 1.0, 1.0], dtype=torch.float32),
                valid=valid,
            )
        )


def test_contract_has_no_full_target_query_or_metric_surface() -> None:
    contract = support_shrinkage_contract()
    audit = access_audit()
    assert contract["method_scope"].endswith("not_multi_region_selector")
    assert contract["memory"]["full_output_clone"] is False
    assert contract["query_axis_consumed"] is False
    assert contract["scale_axis_collapsed"] is False
    assert contract["target_images_masks_labels_or_metrics"] is False
    assert audit["query_ids_or_text_opened"] is False
    assert audit["benchmark_images_masks_labels_or_metrics_opened"] is False
