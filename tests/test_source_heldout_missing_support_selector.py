import torch

from radio_gs.interfaces.source_heldout_missing_support_selector import (
    fit_monotone_ranker,
    oriented_features,
    ranker_probability,
    scene_query_target_balanced_weights,
)


def _features(rows: int = 72) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    values = torch.rand(rows, 12, generator=generator)
    values[:, 1] *= 8.0
    values[:, 9] *= 5.0
    labels = (
        values[:, 0]
        + 0.1 * values[:, 1]
        + values[:, 10]
        + values[:, 11]
        - values[:, 5]
        - values[:, 6]
        - values[:, 7]
        - values[:, 8]
    ) > 0.0
    return values, labels


def test_oriented_features_transform_count_and_deficit():
    features, _ = _features(4)
    oriented = oriented_features(features)
    assert oriented.shape == (4, 9)
    assert torch.allclose(oriented[:, 1], torch.log1p(features[:, 1].double()))
    assert torch.allclose(oriented[:, 3], -features[:, 5].double())
    assert torch.allclose(oriented[:, 6], -features[:, 8].double())
    assert torch.allclose(oriented[:, -1], features[:, 11].double())


def test_balance_equalizes_scene_and_complete_query_target_groups():
    scenes = torch.tensor([0, 0, 0, 0, 1, 1])
    queries = torch.tensor([0, 0, 0, 1, 0, 0])
    targets = torch.tensor([2, 2, 3, 4, 5, 5])
    weight = scene_query_target_balanced_weights(scenes, queries, targets)
    assert torch.isclose(weight[scenes == 0].sum(), weight[scenes == 1].sum())
    assert torch.isclose(weight[0] + weight[1], weight[2])


def test_monotone_ranker_fits_signal_and_has_nonnegative_weights():
    features, labels = _features()
    scenes = torch.arange(features.shape[0]) % 3
    queries = torch.arange(features.shape[0]) % 9
    targets = torch.arange(features.shape[0]) // 3
    model = fit_monotone_ranker(
        features, labels, scenes, queries, targets, maximum_iterations=40
    )
    probability = ranker_probability(model, features)
    assert probability.shape == labels.shape
    assert torch.isfinite(probability).all()
    assert (model.positive_weights >= 0.0).all()
    assert probability[labels].mean() > probability[~labels].mean()
