import torch

from radio_gs.field import ZeroMeanViewResidual


def _module(mean, *, rank=2):
    module = ZeroMeanViewResidual(
        num_gaussians=mean.shape[0],
        coefficient_dim=4,
        rank=rank,
        mean_view_direction=mean,
        row_gate=torch.ones(mean.shape[0]),
        residual_scale=0.5,
    )
    with torch.no_grad():
        module.local_codes.fill_(0.7)
    return module


def test_view_residual_is_zero_at_per_primitive_training_mean():
    mean = torch.tensor([[0.2, 0.3, 0.4], [-0.1, 0.5, 0.2]])
    module = _module(mean)

    delta = module.delta_from_directions(mean)

    torch.testing.assert_close(delta, torch.zeros_like(delta))


def test_view_residual_has_exact_weighted_zero_mean():
    directions = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ]
    )
    weights = torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
    mean = (directions * weights[..., None]).sum(dim=0) / weights.sum(dim=0)[:, None]
    module = _module(mean)
    deltas = torch.stack(
        [module.delta_from_directions(directions[index]) for index in range(3)]
    )

    weighted_mean = (deltas * weights[..., None]).sum(dim=0) / weights.sum(dim=0)[:, None]

    torch.testing.assert_close(weighted_mean, torch.zeros_like(weighted_mean), atol=1e-6, rtol=1e-6)


def test_zero_initial_local_codes_preserve_canonical_render_coefficients():
    mean = torch.zeros(3, 3)
    module = ZeroMeanViewResidual(3, 4, 2, mean)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    viewmat = torch.eye(4)
    viewmat[2, 3] = -2.0

    delta = module(positions, viewmat)

    torch.testing.assert_close(delta, torch.zeros_like(delta))


def test_rank_eight_residual_is_low_capacity_relative_to_dense_coefficients():
    mean = torch.zeros(11, 3)
    module = ZeroMeanViewResidual(11, 256, 8, mean)

    assert module.rank / module.coefficient_dim == 0.03125
    assert module.local_codes.numel() == 11 * 8
