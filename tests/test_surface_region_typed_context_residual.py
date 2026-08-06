import math

import pytest
import torch
from torch.nn import functional as F

from radio_gs.models.surface_region_typed_context_residual import (
    SurfaceRegionAcceptedV2TypedContextResidualV1,
)


def _inputs():
    torch.manual_seed(7)
    base = F.normalize(torch.randn(5, 1536), dim=-1)
    active = torch.tensor([True, False, True, True, False])
    context = torch.zeros(5, 1280)
    context[active] = F.normalize(torch.randn(int(active.sum()), 1280), dim=-1)
    full_scalar = torch.randn(5, 18)
    statistics = torch.zeros(5, 12)
    statistics[active] = torch.randn(int(active.sum()), 12)
    return base, context, full_scalar, statistics, active


def test_zero_initialization_is_bitwise_e0_for_every_row() -> None:
    model = SurfaceRegionAcceptedV2TypedContextResidualV1()
    base, context, scalar, statistics, active = _inputs()
    output = model(
        base,
        context,
        scalar,
        statistics,
        active_mask=active,
    )
    assert torch.equal(output, base)
    assert model.architecture()["trainable_parameter_count"] > 0


def test_nonzero_residual_keeps_inactive_and_ood_bitwise_base_and_is_bounded() -> None:
    model = SurfaceRegionAcceptedV2TypedContextResidualV1()
    base, context, scalar, statistics, active = _inputs()
    with torch.no_grad():
        model.residual_projection.bias.copy_(torch.linspace(-0.5, 0.5, 1536))
    ood = torch.tensor([False, False, True, False, False])
    result = model.forward_with_diagnostics(
        base,
        context,
        scalar,
        statistics,
        active_mask=active,
        ood_mask=ood,
    )
    effective = active & ~ood
    assert torch.equal(result.semantic_descriptor[~effective], base[~effective])
    assert not torch.equal(result.semantic_descriptor[effective], base[effective])
    assert not bool(result.tangent_update[~effective].count_nonzero())
    tangent_dot = (result.tangent_update[effective] * base[effective]).sum(dim=-1)
    assert torch.allclose(tangent_dot, torch.zeros_like(tangent_dot), atol=2e-6)
    cosine = (result.semantic_descriptor[effective] * base[effective]).sum(dim=-1)
    angles = torch.acos(cosine.clamp(-1.0, 1.0))
    assert bool((angles <= 0.15 + 2e-6).all())
    assert bool((result.alpha[effective] <= 0.25 + 1e-7).all())


def test_inactive_carrier_and_input_nan_fail_closed() -> None:
    model = SurfaceRegionAcceptedV2TypedContextResidualV1()
    base, context, scalar, statistics, active = _inputs()
    context[1, 0] = 1.0
    with pytest.raises(ValueError, match="inactive"):
        model(
            base,
            context,
            scalar,
            statistics,
            active_mask=active,
        )
    context[1, 0] = 0.0
    scalar[0, 0] = math.nan
    with pytest.raises(ValueError, match="scalar"):
        model(
            base,
            context,
            scalar,
            statistics,
            active_mask=active,
        )
