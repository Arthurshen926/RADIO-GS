import torch

from radio_gs.interfaces.source_heldout_direct_pair_selector import (
    DIRECT_FEATURE_NAMES,
    SELECTED_INDICES,
    direct_pair_ranker_probability,
    fit_direct_pair_monotone_ranker,
    oriented_direct_features,
)


def test_direct_pair_selector_is_low_capacity_and_monotone():
    generator = torch.Generator().manual_seed(9)
    features = torch.rand(96, len(DIRECT_FEATURE_NAMES), generator=generator)
    features[:, 1] *= 6.0
    oriented = oriented_direct_features(features)
    labels = (oriented[:, 0] + oriented[:, 1] + oriented[:, -2]) > 1.8
    scenes = torch.arange(96) % 3
    queries = torch.arange(96) % 12
    targets = torch.arange(96) // 3
    model = fit_direct_pair_monotone_ranker(
        features,
        labels,
        scenes,
        queries,
        targets,
        maximum_iterations=40,
    )
    probability = direct_pair_ranker_probability(model, features)
    assert oriented.shape == (96, len(SELECTED_INDICES))
    assert probability.shape == labels.shape
    assert torch.isfinite(probability).all()
    assert bool((model.positive_weights >= 0.0).all())
    assert probability[labels].mean() > probability[~labels].mean()

