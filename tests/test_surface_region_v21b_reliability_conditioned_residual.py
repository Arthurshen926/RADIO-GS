from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from radio_gs.interfaces.surface_region_full_scalar_contract import (
    SURFACE_REGION_FULL_SCALAR_NAMES,
)
from radio_gs.interfaces.surface_region_typed_context import (
    TYPED_CONTEXT_STATISTIC_NAMES,
)
from radio_gs.models.surface_region_v21b_reliability_conditioned_residual import (
    SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B,
)


_DISPERSION = SURFACE_REGION_FULL_SCALAR_NAMES.index(
    "legacy_reliability_weighted_mean_directional_dispersion"
)
_EVIDENCE = SURFACE_REGION_FULL_SCALAR_NAMES.index(
    "legacy_reliability_weighted_mean_observation_evidence"
)
_PURITY = SURFACE_REGION_FULL_SCALAR_NAMES.index(
    "legacy_reliability_weighted_mean_visibility_purity_value"
)
_PURITY_KNOWN = SURFACE_REGION_FULL_SCALAR_NAMES.index(
    "legacy_reliability_weighted_mean_visibility_purity_known"
)
_CONTEXT_RELIABILITY = TYPED_CONTEXT_STATISTIC_NAMES.index(
    "context_reliability_mean"
)
_CONTEXT_RESULTANT = TYPED_CONTEXT_STATISTIC_NAMES.index(
    "context_weighted_directional_resultant_length"
)


def _set_reliability(
    full_scalar: torch.Tensor,
    context_statistics: torch.Tensor,
    row: int,
    value: float,
) -> None:
    score = float(value)
    full_scalar[row, _DISPERSION] = 1.0 - score
    full_scalar[row, _EVIDENCE] = score
    full_scalar[row, _PURITY] = score
    # Keep the known fraction one so the purity contribution is linear.
    full_scalar[row, _PURITY_KNOWN] = 1.0
    context_statistics[row, _CONTEXT_RELIABILITY] = score
    context_statistics[row, _CONTEXT_RESULTANT] = score


def _inputs(rows: int = 5):
    generator = torch.Generator().manual_seed(20260807)
    base = F.normalize(torch.randn(rows, 1536, generator=generator), dim=-1)
    active = torch.tensor([True, False, True, True, False])[:rows]
    context = torch.zeros(rows, 1280)
    context[active] = F.normalize(
        torch.randn(int(active.sum()), 1280, generator=generator),
        dim=-1,
    )
    full_scalar = torch.zeros(rows, 18)
    statistics = torch.zeros(rows, 12)
    for row in torch.where(active)[0].tolist():
        _set_reliability(full_scalar, statistics, row, 0.25 + 0.1 * row)
    return base, context, full_scalar, statistics, active


def test_fixed_reliability_score_and_continuous_budget_boundaries() -> None:
    full_scalar = torch.zeros(3, 18)
    statistics = torch.zeros(3, 12)
    _set_reliability(full_scalar, statistics, 0, 0.0)
    _set_reliability(full_scalar, statistics, 1, 0.5)
    _set_reliability(full_scalar, statistics, 2, 1.0)
    score = (
        SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B
        .reliability_score(full_scalar, statistics)
    )
    budget = (
        SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B
        .angular_budget(score)
    )
    assert torch.allclose(score, torch.tensor([0.0, 0.5, 1.0]), atol=1e-7)
    assert torch.allclose(budget, torch.tensor([0.15, 0.45, 0.75]), atol=1e-7)

    dense_score = torch.linspace(0.0, 1.0, 101)
    dense_budget = (
        SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B
        .angular_budget(dense_score)
    )
    assert bool((dense_budget[1:] > dense_budget[:-1]).all())
    assert torch.allclose(
        dense_budget[1:] - dense_budget[:-1],
        torch.full((100,), 0.006),
        atol=1e-7,
    )


def test_zero_initialization_is_bitwise_accepted_v2_and_has_gradient() -> None:
    model = SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B()
    base, context, full_scalar, statistics, active = _inputs()
    result = model.forward_with_diagnostics(
        base,
        context,
        full_scalar,
        statistics,
        active_mask=active,
    )
    assert torch.equal(result.semantic_descriptor, base)
    assert not bool(result.tangent_update.count_nonzero())
    assert not bool(result.tangent_gain.count_nonzero())

    target = F.normalize(torch.roll(base, shifts=1, dims=-1), dim=-1)
    loss = -(result.semantic_descriptor[active] * target[active]).sum()
    loss.backward()
    gradient = model.residual_projection.weight.grad
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0


def test_rank256_capacity_is_fixed_and_has_no_scene_parameter() -> None:
    model = SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B()
    assert model.HIDDEN_RANK == 256
    assert model.trainable_parameter_count() == 1_451_520
    assert not any("scene" in name.lower() for name, _ in model.named_parameters())
    architecture = model.architecture()
    assert architecture["hidden_rank"] == 256
    assert architecture["scene_parameters"] is False
    assert architecture["per_scene_hyperparameters"] is False
    assert architecture["reliability_learned"] is False


def test_nonzero_update_is_tangent_and_obeys_per_row_geodesic_budget() -> None:
    model = SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B()
    base = F.normalize(torch.randn(1, 1536), dim=-1).repeat(4, 1)
    context = F.normalize(torch.randn(1, 1280), dim=-1).repeat(4, 1)
    context[2].zero_()
    full_scalar = torch.zeros(4, 18)
    statistics = torch.zeros(4, 12)
    _set_reliability(full_scalar, statistics, 0, 0.0)
    _set_reliability(full_scalar, statistics, 1, 1.0)
    _set_reliability(full_scalar, statistics, 3, 1.0)
    active = torch.tensor([True, True, False, True])
    ood = torch.tensor([False, False, False, True])
    with torch.no_grad():
        model.residual_projection.bias.copy_(torch.linspace(-12.0, 12.0, 1536))
    result = model.forward_with_diagnostics(
        base,
        context,
        full_scalar,
        statistics,
        active_mask=active,
        ood_mask=ood,
    )

    effective = active & ~ood
    assert torch.equal(result.semantic_descriptor[~effective], base[~effective])
    assert not bool(result.tangent_update[~effective].count_nonzero())
    tangent_dot = (result.tangent_update[effective] * base[effective]).sum(dim=-1)
    assert torch.allclose(tangent_dot, torch.zeros_like(tangent_dot), atol=2e-5)
    output_norm = torch.linalg.vector_norm(result.semantic_descriptor, dim=-1)
    assert torch.allclose(output_norm, torch.ones_like(output_norm), atol=2e-6)
    cosine = (result.semantic_descriptor[effective] * base[effective]).sum(dim=-1)
    angles = torch.acos(cosine.clamp(-1.0, 1.0))
    budgets = result.angular_budget_radians[effective]
    assert bool((angles <= budgets + 3e-5).all())
    assert float(angles[1] - angles[0]) > 0.50
    assert math.isclose(float(budgets[0]), 0.15, abs_tol=1e-7)
    assert math.isclose(float(budgets[1]), 0.75, abs_tol=1e-7)
    assert bool(
        (result.tangent_gain[effective]
         <= model.MAX_TANGENT_GAIN + 1e-7).all()
    )


def test_fallback_rows_have_no_residual_carrier_gradient() -> None:
    model = SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B()
    base, context, full_scalar, statistics, active = _inputs()
    ood = torch.tensor([False, False, True, False, False])
    context.requires_grad_()
    full_scalar.requires_grad_()
    statistics.requires_grad_()
    with torch.no_grad():
        model.residual_projection.bias.copy_(torch.linspace(-0.2, 0.2, 1536))
    output = model(
        base,
        context,
        full_scalar,
        statistics,
        active_mask=active,
        ood_mask=ood,
    )
    output.sum().backward()
    effective = active & ~ood
    for gradient in (context.grad, full_scalar.grad, statistics.grad):
        assert gradient is not None
        assert not bool(gradient[~effective].count_nonzero())


def test_invalid_and_inactive_carriers_fail_closed() -> None:
    model = SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B()
    base, context, full_scalar, statistics, active = _inputs()
    context[1, 0] = 1.0
    with pytest.raises(ValueError, match="inactive"):
        model(
            base,
            context,
            full_scalar,
            statistics,
            active_mask=active,
        )
    context[1, 0] = 0.0
    statistics[0, 0] = math.nan
    with pytest.raises(ValueError, match="statistics"):
        model(
            base,
            context,
            full_scalar,
            statistics,
            active_mask=active,
        )
