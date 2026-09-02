from __future__ import annotations

import torch

from radio_gs.v4.completion.spatial_slots import TokenSpatialSupportSlots


def _inputs():
    unary = torch.tensor(
        [[1.0, 0.0], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]],
        dtype=torch.float32,
    )
    centres = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    token_features = torch.randn(1, 6)
    token_centres = torch.zeros(1, 3)
    token_frames = torch.eye(3)[None]
    token_scales = torch.ones(1, 3)
    clamp_mask = torch.tensor([True, False, False, False])
    clamp = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    return (
        unary,
        centres,
        token_features,
        token_centres,
        token_frames,
        token_scales,
        clamp_mask,
        clamp,
    )


def test_spatial_slots_preserve_clamp_simplex_and_train():
    model = TokenSpatialSupportSlots(input_dimension=6, hidden_dimension=12)
    values = _inputs()
    output = model(*values)

    assert torch.equal(output.probabilities[values[-2]], values[-1][values[-2]])
    torch.testing.assert_close(
        output.probabilities.sum(-1), torch.ones(4), atol=1e-6, rtol=0
    )
    assert output.slot_centres_local.shape == (1, 7, 3)
    assert output.slot_scales_local.shape == (1, 7, 3)
    loss = -output.probabilities[1:, 0].clamp_min(1e-8).log().mean()
    loss.backward()
    assert model.token_network[-1].weight.grad is not None
    assert float(model.token_network[-1].weight.grad.abs().sum()) > 0
    assert model.fusion_parameter.grad is not None
    assert float(model.fusion_parameter.grad.abs()) > 0


def test_spatial_slots_are_equivariant_to_joint_rigid_rotation():
    model = TokenSpatialSupportSlots(input_dimension=6, hidden_dimension=12)
    values = list(_inputs())
    original = model(*values).probabilities
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    values[1] = values[1] @ rotation.T
    values[3] = values[3] @ rotation.T
    values[4] = rotation[None] @ values[4]
    rotated = model(*values).probabilities

    torch.testing.assert_close(rotated, original, atol=1e-6, rtol=0)


def test_spatial_only_slots_have_exactly_zero_token_bias():
    model = TokenSpatialSupportSlots(
        input_dimension=6, hidden_dimension=12, use_token_bias=False
    )
    with torch.no_grad():
        model.token_network[-1].bias[-1] = 3.0

    output = model(*_inputs())

    assert torch.equal(output.token_bias, torch.zeros_like(output.token_bias))
