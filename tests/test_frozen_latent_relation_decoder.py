import torch

from radio_gs.models.frozen_latent_relation_decoder import symmetric_pair_features
from radio_gs.scripts.train_evaluate_frozen_latent_relation_decoder import (
    balanced_relation_loss,
    conditional_same_score,
    soft_support_iou,
    support_iou,
    visibility_normalized_track_posterior,
)


def test_pair_features_are_invariant_to_pair_order() -> None:
    embedding = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.0, 4.0]])
    centroids = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-2.0, 1.0, 0.0]])
    areas = torch.tensor([0.1, 0.2, 0.4])
    extent = torch.tensor([4.0, 4.0, 4.0])
    forward = symmetric_pair_features(
        embedding, torch.tensor([0, 1]), torch.tensor([1, 2]), centroids, areas, extent
    )
    reverse = symmetric_pair_features(
        embedding, torch.tensor([1, 2]), torch.tensor([0, 1]), centroids, areas, extent
    )
    assert torch.equal(forward, reverse)


def test_support_iou_has_exact_set_semantics() -> None:
    assert support_iou({1, 2, 3}, {2, 3, 4}) == 0.5
    assert support_iou(set(), set()) == 1.0


def test_ternary_relation_score_distinguishes_unknown_from_known() -> None:
    logits = torch.tensor(
        [[5.0, 0.0, -2.0], [0.0, 5.0, -2.0], [-2.0, 0.0, 5.0]]
    )
    labels = torch.tensor([0, 1, -1], dtype=torch.int8)
    assert balanced_relation_loss(logits, labels) < 0.02
    score = conditional_same_score(logits[:2])
    assert score[0] < 0.01 and score[1] > 0.99


def test_visibility_normalized_track_posterior_does_not_count_occlusion_negative() -> None:
    # Two proposals are equally related. Gaussian 0 appears whenever visible;
    # proposal/view 1 cannot see it and therefore must not halve its posterior.
    relation = torch.tensor([[1.0, 1.0]])
    support = torch.tensor([[True, False], [False, True]])
    visibility = torch.tensor([[True, False], [False, True]])

    posterior = visibility_normalized_track_posterior(
        relation, support, visibility
    )

    assert torch.allclose(posterior, torch.ones_like(posterior))
    iou = soft_support_iou(
        posterior,
        torch.tensor([[True, True]]),
        torch.tensor([[True, True]]),
    )
    assert torch.allclose(iou, torch.ones_like(iou))
