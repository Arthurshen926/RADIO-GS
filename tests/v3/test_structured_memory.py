import torch

from radio_gs.v3.memory.structured_memory import (
    ExtraInstanceCodeOracle,
    LowRankPrivateBranchMemory,
    OrthogonalProductMemory,
    SharedPrivateLayout,
    StructuredMemoryHeads,
    StructuredSharedPrivateMemory,
)
from radio_gs.v3.training.phases import LowRankLatentResidual


def test_heads_derive_views_without_adding_gaussian_state():
    heads = StructuredMemoryHeads()
    latent = torch.randn(7, 512)
    assert heads.visual_view(latent).shape == (7, 256)
    assert heads.instance_view(latent, 0.25).shape == (7, 32)
    assert heads.boundary_view(latent).shape == (7, 16)
    assert heads.orthogonality_loss().ndim == 0


def test_oracle_is_explicitly_not_deployment_eligible():
    oracle = ExtraInstanceCodeOracle(11)
    assert oracle().shape == (11, 16)
    assert oracle.deployment_eligible is False


def test_low_rank_residual_starts_as_identity():
    residual = LowRankLatentResidual(rank=4)
    latent = torch.randn(5, 512)
    torch.testing.assert_close(residual(latent), latent)


def test_shared_private_is_one_d512_with_expected_product_space():
    model = StructuredSharedPrivateMemory(torch.randn(7, 512))
    assert tuple(model.state_dict()) == (
        "memory",
        "visual_to_instance.weight",
        "context_to_boundary.weight",
        "scale_adapter.weight",
        "scale_adapter.bias",
    )
    assert model.visual_view().shape == (7, 448)
    assert model.semantic_view().shape == (7, 128)
    assert model.instance_view(0.25).shape == (7, 48)
    assert model.boundary_view().shape == (7, 16)
    assert model.layout.slices["instance"] == slice(448, 496)


def test_instance_and_boundary_losses_cannot_rewrite_visual_columns():
    model = StructuredSharedPrivateMemory(torch.randn(7, 512))
    weights = torch.linspace(-1, 1, 48)
    instance_loss = (model.instance_view(0.3) * weights).sum()
    instance_loss.backward()
    gradient = model.memory.grad
    assert torch.count_nonzero(gradient[:, :448]) == 0
    assert torch.count_nonzero(gradient[:, 448:496]) > 0
    assert torch.count_nonzero(gradient[:, 496:]) == 0
    model.zero_grad(set_to_none=True)
    model.boundary_view().square().sum().backward()
    gradient = model.memory.grad
    assert torch.count_nonzero(gradient[:, :496]) == 0
    assert torch.count_nonzero(gradient[:, 496:]) > 0


def test_layout_rejects_a_hidden_sidecar_or_non_d512_partition():
    try:
        SharedPrivateLayout(shared=320, semantic=128, instance=48, boundary=15)
    except ValueError as error:
        assert "sum to 512" in str(error)
    else:
        raise AssertionError("non-D512 layout was accepted")


def test_s1_visual_instance_boundary_layout_is_a_single_d512():
    layout = SharedPrivateLayout(shared=448, semantic=0, instance=48, boundary=16)
    model = StructuredSharedPrivateMemory(torch.randn(5, 512), layout=layout)
    assert model.visual_view().shape == (5, 448)
    assert model.semantic_view().shape == (5, 0)
    assert model.instance_view().shape == (5, 48)


def test_learned_product_basis_is_exactly_orthogonal_and_identity_initialized():
    model = OrthogonalProductMemory(torch.randn(5, 512))
    value = torch.randn(4, 512)
    torch.testing.assert_close(model.rotate(value), value)
    with torch.no_grad():
        model.basis_angles.uniform_(-0.5, 0.5)
    rotated = model.rotate(value)
    torch.testing.assert_close(
        rotated @ rotated.T, value @ value.T, atol=2e-5, rtol=2e-5
    )
    assert model.shared_capability_view(torch.randn(4, 320)).shape == (4, 512)


def test_low_rank_private_branches_are_zero_output_and_cannot_rewrite_shared():
    initial = torch.randn(7, 512)
    model = LowRankPrivateBranchMemory(initial)
    hard = StructuredSharedPrivateMemory(initial)
    hard.scale_adapter.load_state_dict(model.scale_adapter.state_dict())
    torch.testing.assert_close(model.instance_view(0.3), hard.instance_view(0.3))
    torch.testing.assert_close(model.boundary_view(), hard.boundary_view())
    loss = (model.instance_view(0.3) * torch.linspace(-1, 1, 48)).sum()
    loss.backward()
    assert torch.count_nonzero(model.memory.grad[:, :448]) == 0
    assert model.instance_up.weight.grad is not None
    assert torch.count_nonzero(model.instance_up.weight.grad) > 0
