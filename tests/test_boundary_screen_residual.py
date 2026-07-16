import torch

from radio_gs.field import BoundaryConditionedScreenResidual


def test_boundary_residual_initializes_as_exact_identity_readout():
    module = BoundaryConditionedScreenResidual(feature_dim=8, rank=2, hidden_dim=4)
    rgb = torch.rand(1, 3, 5, 6)
    depth = torch.rand(1, 1, 5, 6) + 1
    alpha = torch.rand(1, 1, 5, 6)
    torch.testing.assert_close(module(rgb, depth, alpha), torch.zeros(1, 8, 5, 6))


def test_boundary_conditions_respond_to_observable_discontinuities():
    module = BoundaryConditionedScreenResidual(feature_dim=8, rank=2, hidden_dim=4)
    rgb = torch.zeros(1, 3, 4, 4)
    depth = torch.ones(1, 1, 4, 4)
    alpha = torch.ones(1, 1, 4, 4)
    rgb[:, :, :, 2:] = 1
    depth[:, :, 2:, :] = 2
    alpha[:, :, :, 2:] = 0
    condition = module.conditions(rgb, depth, alpha)
    assert condition[:, 0].max() > 0
    assert condition[:, 1].max() > 0
    assert condition[:, 2].max() > 0
    assert condition.shape == (1, 3, 4, 4)
