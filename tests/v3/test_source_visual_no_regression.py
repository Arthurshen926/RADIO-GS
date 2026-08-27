import torch

from radio_gs.field.basis_decoder import AffineBasisDecoder
from radio_gs.v3.evaluation.source_visual_no_regression import render_decoded_field


def test_render_decoded_field_matches_exact_weighted_compositor():
    decoder = AffineBasisDecoder(
        feature_dim=2,
        coefficient_dim=2,
        mean=torch.zeros(2),
        scale=torch.ones(2),
        basis=torch.eye(2),
        trainable_basis=False,
    )
    latent = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    value, alpha = render_decoded_field(
        latent,
        decoder,
        torch.tensor([0, 1, 1]),
        torch.tensor([0, 0, 1]),
        torch.tensor([0.25, 0.5, 1.0]),
        num_pixels=2,
        device=torch.device("cpu"),
        chunk_size=1,
    )
    assert torch.allclose(value, torch.tensor([[0.25, 1.0], [0.0, 2.0]]))
    assert torch.allclose(alpha, torch.tensor([0.75, 1.0]))
