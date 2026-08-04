import torch

from radio_gs.field.directional_distribution import (
    directional_prototype_coverage,
    directional_prototype_observation_cosines,
    directional_set_ranking_loss,
    directional_set_rms_loss,
    fit_two_direction_prototypes,
)


def test_two_prototypes_recover_bimodal_directions_lost_by_center():
    observations = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.98, 0.2]],
            [[0.0, 1.0]],
            [[0.2, 0.98]],
        ]
    )
    valid = torch.ones(4, 1, dtype=torch.bool)

    result = fit_two_direction_prototypes(observations, valid)
    coverage = directional_prototype_coverage(
        result.prototypes, observations, valid
    )

    assert result.valid.tolist() == [True]
    assert result.mixture_weight.sum(dim=-1).allclose(torch.ones(1))
    assert coverage["prototype_weighted_mean_cosine"] > (
        coverage["center_weighted_mean_cosine"] + 0.2
    )
    assert coverage["prototype_p05_cosine"] > coverage["center_p05_cosine"]
    center, prototype, mass = directional_prototype_observation_cosines(
        result.prototypes, observations, valid
    )
    assert center.shape == prototype.shape == mass.shape == (4,)


def test_two_prototypes_collapse_safely_for_unimodal_rows_and_mask_empty_rows():
    observations = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.99, 0.01], [0.0, 0.0]],
            [[1.0, -0.01], [0.0, 0.0]],
        ]
    )
    valid = torch.tensor([[True, False], [True, False], [True, False]])

    result = fit_two_direction_prototypes(observations, valid)

    assert result.valid.tolist() == [True, False]
    assert torch.cosine_similarity(
        result.prototypes[0, 0], result.prototypes[0, 1], dim=0
    ) > 0.999
    assert torch.equal(result.prototypes[1], torch.zeros_like(result.prototypes[1]))


def test_directional_losses_preserve_tail_and_positive_negative_order():
    predicted = torch.tensor([[1.0, 0.0]], requires_grad=True)
    positives = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    negatives = torch.tensor([[[-1.0, 0.0]], [[0.8, 0.2]]])
    valid = torch.ones(2, 1, dtype=torch.bool)

    rms = directional_set_rms_loss(predicted, positives, valid)
    ranking = directional_set_ranking_loss(
        predicted, positives, valid, negatives, valid
    )
    (rms + ranking).backward()

    assert rms > 0
    assert ranking > 0
    assert predicted.grad is not None
    assert bool(torch.isfinite(predicted.grad).all())
