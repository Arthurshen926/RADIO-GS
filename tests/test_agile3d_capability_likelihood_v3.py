from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_ratio_training_dataset import (
    legal_quantized_object_ids,
    stable_class_sample_indices,
)
from radio_gs.scripts.train_capability_likelihood_ratio_head import (
    prevalence_weighted_posterior_ranking_loss,
)


def test_all_legal_objects_are_selected_without_subselection() -> None:
    labels = torch.tensor([0, 7, 3, 7, 11, 0, 3]).numpy()
    assert legal_quantized_object_ids(labels) == [3, 7, 11]


def test_training_row_sampling_is_stable_and_class_stratified() -> None:
    target = torch.tensor([False, True, False, True, False, True, False])
    first = stable_class_sample_indices(
        target, scene_id="scene0000_00", object_id=3, click_count=2, per_class=2
    )
    second = stable_class_sample_indices(
        target, scene_id="scene0000_00", object_id=3, click_count=2, per_class=2
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert bool(target[first[0].long()].all())
    assert not bool(target[first[1].long()].any())


def test_prevalence_weighted_strata_equal_ordinary_population_bce() -> None:
    positive = torch.tensor([1.2, -0.3])
    negative = torch.tensor([-0.8, 0.4, -1.1, 0.2, -0.5, -0.1])
    prevalence = 0.25
    loss, detail = prevalence_weighted_posterior_ranking_loss(
        positive, negative, prevalence=prevalence, ranking_weight=0.0
    )
    prior_logit = math.log(prevalence / (1.0 - prevalence))
    population = torch.cat((positive, negative)) + prior_logit
    target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected = F.binary_cross_entropy_with_logits(population, target)
    assert torch.allclose(loss, expected)
    assert torch.allclose(detail["posterior_bce"], expected)


def test_prior_correction_removes_foreground_prevalence_intercept() -> None:
    ell = torch.tensor([-1.0, 0.0, 1.0])
    prevalence = 0.02
    posterior = torch.sigmoid(ell + math.log(prevalence / (1 - prevalence)))
    recovered = torch.logit(posterior) - math.log(prevalence / (1 - prevalence))
    assert torch.allclose(recovered, ell, atol=1e-5)
