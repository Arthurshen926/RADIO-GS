import math

import pytest
import torch
from torch.nn import functional as F

from radio_gs.models.surface_region_dual_descriptor import (
    SurfaceRegionAcceptedV2FullScalarResidualV1,
    SurfaceRegionFullScalarResidualOutput,
)


def _inputs(
    *, descriptor_dim: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1201)
    base = F.normalize(torch.randn(3, descriptor_dim), dim=-1)
    scalars = torch.randn(3, 18)
    return base, scalars


def _nonzero_residual(
    model: SurfaceRegionAcceptedV2FullScalarResidualV1,
) -> None:
    with torch.no_grad():
        model.residual_projection.weight.zero_()
        model.residual_projection.bias.copy_(
            torch.linspace(-3.0, 4.0, model.descriptor_dim)
        )


def test_full_scalar_zero_init_is_bitwise_base_with_first_step_gradient() -> None:
    base, scalars = _inputs()
    model = SurfaceRegionAcceptedV2FullScalarResidualV1(
        descriptor_dim=base.shape[-1]
    )

    output = model.forward_with_diagnostics(base, scalars)

    assert isinstance(output, SurfaceRegionFullScalarResidualOutput)
    assert torch.equal(output.base_descriptor, base)
    assert torch.equal(output.semantic_descriptor, base)
    assert torch.count_nonzero(output.tangent_update) == 0
    assert torch.count_nonzero(output.alpha) == 0
    assert torch.count_nonzero(model.residual_projection.weight) == 0
    assert torch.count_nonzero(model.residual_projection.bias) == 0
    expected_parameters = (
        8 * 64
        + 18 * 64 + 64
        + 3 * 64 * 64 + 64
        + 64 * 8 + 8
    )
    assert model.trainable_parameter_count() == expected_parameters

    output.semantic_descriptor[:, 0].sum().backward()
    assert model.residual_projection.bias.grad is not None
    assert torch.count_nonzero(model.residual_projection.bias.grad) > 0


def test_full_scalar_hidden_residual_is_conditioned_on_semantic_content() -> None:
    base, scalars = _inputs()
    scalars[1] = scalars[0]
    captured: list[torch.Tensor] = []
    model = SurfaceRegionAcceptedV2FullScalarResidualV1(
        descriptor_dim=base.shape[-1]
    )

    def capture_hidden(_module, inputs):
        captured.append(inputs[0].detach().clone())

    handle = model.residual_projection.register_forward_pre_hook(capture_hidden)
    try:
        model(base[:2], scalars[:2])
    finally:
        handle.remove()
    assert len(captured) == 1
    assert not torch.equal(captured[0][0], captured[0][1])
    assert model.architecture()["conditioning"] == (
        "content_scalar_concat_with_multiplicative_interaction"
    )


def test_full_scalar_update_is_tangent_to_accepted_base() -> None:
    base, scalars = _inputs()
    model = SurfaceRegionAcceptedV2FullScalarResidualV1(
        descriptor_dim=base.shape[-1],
        max_alpha=0.4,
    )
    _nonzero_residual(model)

    output = model.forward_with_diagnostics(base, scalars)
    radial = (output.tangent_update * base).sum(dim=-1)

    torch.testing.assert_close(radial, torch.zeros_like(radial), atol=2e-7, rtol=0.0)
    assert not torch.equal(output.semantic_descriptor, base)
    torch.testing.assert_close(
        output.semantic_descriptor.norm(dim=-1),
        torch.ones(base.shape[0]),
        atol=1e-6,
        rtol=0.0,
    )


def test_full_scalar_angle_and_alpha_are_bounded() -> None:
    base, scalars = _inputs()
    max_angle = 0.12
    max_alpha = 0.3
    model = SurfaceRegionAcceptedV2FullScalarResidualV1(
        descriptor_dim=base.shape[-1],
        max_angle_radians=max_angle,
        max_alpha=max_alpha,
    )
    with torch.no_grad():
        model.residual_projection.weight.fill_(100.0)
        model.residual_projection.bias.copy_(
            torch.linspace(-1000.0, 1000.0, base.shape[-1])
        )

    output = model.forward_with_diagnostics(base, scalars)
    cosine = (base * output.semantic_descriptor).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)

    assert bool((output.alpha <= max_alpha).all())
    assert bool((angle <= max_angle + 2e-6).all())


def test_full_scalar_ood_mask_is_bitwise_gradient_free_fallback() -> None:
    base, scalars = _inputs()
    model = SurfaceRegionAcceptedV2FullScalarResidualV1(
        descriptor_dim=base.shape[-1]
    )
    _nonzero_residual(model)
    ood = torch.tensor([True, False, True])

    output = model.forward_with_diagnostics(base, scalars, ood_mask=ood)

    assert torch.equal(output.semantic_descriptor[ood], base[ood])
    assert torch.count_nonzero(output.tangent_update[ood]) == 0
    assert torch.count_nonzero(output.alpha[ood]) == 0
    assert not torch.equal(output.semantic_descriptor[~ood], base[~ood])
    assert torch.equal(output.ood_fallback, ood)

    model.zero_grad(set_to_none=True)
    output.semantic_descriptor[ood].sum().backward()
    assert model.residual_projection.bias.grad is None or torch.count_nonzero(
        model.residual_projection.bias.grad
    ) == 0


def test_full_scalar_shape_normalization_and_ood_contract_fail_closed() -> None:
    base, scalars = _inputs()
    median = torch.linspace(-1.0, 1.0, 18)
    mad = torch.linspace(0.5, 2.0, 18)
    model = SurfaceRegionAcceptedV2FullScalarResidualV1(
        descriptor_dim=base.shape[-1],
        scalar_median=median,
        scalar_robust_scale=mad,
    )

    assert torch.equal(model.scalar_median, median)
    assert torch.equal(model.scalar_robust_scale, mad)
    architecture = model.architecture()
    assert architecture["scalar_dim"] == 18
    assert architecture["hidden_dim"] == 64
    assert architecture["descriptor_dim"] == base.shape[-1]

    with pytest.raises(ValueError, match=r"\[\.\.\., 18\]"):
        model(base, scalars[:, :-1])
    with pytest.raises(ValueError, match="ood_mask"):
        model(base, scalars, ood_mask=torch.zeros(3))
    with pytest.raises(ValueError, match="scalar_robust_scale"):
        SurfaceRegionAcceptedV2FullScalarResidualV1(
            descriptor_dim=base.shape[-1],
            scalar_robust_scale=torch.zeros(18),
        )
    with pytest.raises(ValueError, match="unit L2"):
        model(base * 2.0, scalars)


def test_full_scalar_supports_single_unbatched_row() -> None:
    base, scalars = _inputs()
    model = SurfaceRegionAcceptedV2FullScalarResidualV1(
        descriptor_dim=base.shape[-1]
    )

    output = model(base[0], scalars[0], ood_mask=torch.tensor(True))

    assert output.shape == base[0].shape
    assert torch.equal(output, base[0])
