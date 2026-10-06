import torch
from radio_gs.training.source_region_readback import region_render_weights


def test_readback_matches_forward_render_with_observed_normalization():
    masks = torch.tensor([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=torch.bool)
    pid = torch.tensor([0, 0, 1, 1, 2])
    gid = torch.tensor([0, 1, 1, 2, 2])
    values = torch.tensor([0.9, 0.1, 0.8, 0.2, 1.0])
    observed = torch.tensor([True, True, False])
    weights, supported, valid = region_render_weights(masks, pid, gid, values, observed)
    scores = torch.tensor([1., -1., 100.])
    # Pixel 0 = .8; pixel 1 = -1 after dividing out unobserved mass.
    torch.testing.assert_close(weights @ scores, torch.tensor([-.1, -1., 0.]))
    assert supported.tolist() == [True, True, False]
    assert valid.tolist() == [True, True, False]


def test_conditional_membership_pool_is_not_render_pool():
    masks = torch.ones((1, 1), dtype=torch.bool)
    weights, _, _ = region_render_weights(masks, torch.tensor([0, 0]),
        torch.tensor([0, 1]), torch.tensor([.99, .01]), torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(weights, torch.tensor([[.99, .01]]))
    # P(mask|element) is 1 for both, incorrectly yielding .5/.5.
    assert not torch.allclose(weights, torch.full_like(weights, .5))
