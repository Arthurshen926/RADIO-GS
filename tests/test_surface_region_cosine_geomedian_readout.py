import torch
import torch.nn.functional as F

from radio_gs.scripts.train_surface_region_cosine_geomedian_readout import (
    robust_targets,
    streaming_cosine_geometric_median,
)


def test_streaming_cosine_geomedian_matches_released_tangent_update():
    observations = F.normalize(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-0.25, 1.0]]]), dim=-1
    )
    result = streaming_cosine_geometric_median(
        observations, torch.ones(1, 3, dtype=torch.bool)
    )
    expected = observations[:, 0]
    cumulative = torch.ones(1, 1)
    for index in (1, 2):
        cumulative = cumulative + 1.0
        current = observations[:, index]
        tangent = current - (current * expected).sum(-1, keepdim=True) * expected
        expected = F.normalize(expected + tangent / cumulative, dim=-1)
    assert torch.allclose(result, expected, atol=1e-7, rtol=0)


def test_streaming_cosine_geomedian_ignores_padded_views_and_supports_weights():
    observations = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]]
    )
    mask = torch.tensor([[True, True, False, False]])
    weighted = streaming_cosine_geometric_median(
        observations, mask, weights=torch.tensor([[2.0, 1.0, 99.0, 99.0]])
    )
    expected = F.normalize(torch.tensor([[1.0, 1.0 / 3.0]]), dim=-1)
    assert torch.allclose(weighted, expected, atol=1e-7, rtol=0)
    changed_padding = observations.clone()
    changed_padding[:, 2:] = torch.tensor([[[0.3, 0.7], [0.8, 0.2]]])
    assert torch.equal(
        weighted,
        streaming_cosine_geometric_median(
            changed_padding,
            mask,
            weights=torch.tensor([[2.0, 1.0, 5.0, 7.0]]),
        ),
    )


def test_robust_targets_select_observed_token_nearest_robust_descriptor():
    descriptors = F.normalize(
        torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ]
        ),
        dim=-1,
    )
    tokens = torch.tensor(
        [[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]]
    )
    data = {
        "official_summary_tokens": tokens,
        "official_crop_summaries": descriptors,
        "teacher_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    token, robust, mean, all_descriptors, mask = robust_targets(
        data, torch.tensor([0])
    )
    similarities = torch.einsum("bvd,bd->bv", descriptors, robust)
    expected_index = int(similarities[0].argmax())
    assert torch.equal(token, tokens[:, expected_index])
    assert torch.allclose(robust.norm(dim=-1), torch.ones(1), atol=1e-7)
    assert torch.allclose(mean.norm(dim=-1), torch.ones(1), atol=1e-7)
    assert torch.equal(all_descriptors, descriptors)
    assert bool(mask.all())
