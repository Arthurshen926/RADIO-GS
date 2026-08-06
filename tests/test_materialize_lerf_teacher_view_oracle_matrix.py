import itertools

import torch
import torch.nn.functional as F

from radio_gs.scripts.materialize_lerf_teacher_view_oracle_matrix import (
    fit_two_spherical_modes,
    geodesic_project,
    normalized_logsumexp_response,
)


def test_geodesic_projection_enforces_closed_form_angle_cap() -> None:
    base = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    teacher = torch.tensor([[0.0, 1.0, 0.0], [0.999, 0.04, 0.0]])
    projected = geodesic_project(base, teacher, 0.15)
    angles = torch.acos((F.normalize(base, dim=-1) * projected).sum(dim=-1))
    target_angles = torch.acos(
        (F.normalize(base, dim=-1) * F.normalize(teacher, dim=-1)).sum(dim=-1)
    )
    assert torch.allclose(angles, target_angles.clamp(max=0.15), atol=1e-5)


def test_two_mode_fit_is_permutation_invariant() -> None:
    views = F.normalize(
        torch.tensor(
            [[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-1.0, 0.0, 0.0], [-0.9, 0.1, 0.0]]]
        ),
        dim=-1,
    )
    mask = torch.ones(1, 4, dtype=torch.bool)
    frames = torch.tensor([[7, 3, 11, 5]])
    reference_modes, reference_weights = fit_two_spherical_modes(views, mask, frames)
    for permutation in itertools.permutations(range(4)):
        order = torch.tensor(permutation)
        modes, weights = fit_two_spherical_modes(
            views[:, order], mask[:, order], frames[:, order]
        )
        assert torch.allclose(modes, reference_modes, atol=1e-6)
        assert torch.allclose(weights, reference_weights)


def test_normalized_lse_uses_weights_and_is_not_unconditional_max() -> None:
    descriptors = F.normalize(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), dim=-1
    )
    query = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    response = normalized_logsumexp_response(
        descriptors, torch.tensor([[0.5, 0.5]]), query, beta=10.0
    )
    assert 0.0 < float(response) < 1.0
    expected = torch.log(torch.tensor(0.5 * torch.exp(torch.tensor(10.0)) + 0.5)) / 10.0
    assert torch.allclose(response.squeeze(), expected, atol=1e-6)
