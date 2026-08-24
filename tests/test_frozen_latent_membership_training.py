import pytest
import torch

from radio_gs.models.frozen_latent_membership_decoder import (
    FrozenLatentMembershipDecoder,
)
from radio_gs.scripts.train_evaluate_frozen_latent_membership_decoder import (
    _calibrate_threshold,
    _encode_gaussian_table,
    _scores_for_proposal,
    _similarity_scores_for_proposal,
    build_training_pairs,
    compose_membership_query_features,
    track_augmented_training_targets,
    visible_membership_target,
)


def test_cached_membership_scores_match_dense_forward():
    torch.manual_seed(7)
    model = FrozenLatentMembershipDecoder(latent_dim=5, query_dim=7, hidden_dim=3)
    model.eval()
    latent = torch.randn(11, 5)
    query = torch.randn(7)
    rows = torch.tensor([9, 1, 5, 0, 10])
    expected = torch.sigmoid(
        model(latent[rows], query[None].expand(rows.numel(), -1))
    )
    encoded = _encode_gaussian_table(
        model, latent, device=torch.device("cpu"), chunk_size=4
    )
    actual = _scores_for_proposal(
        model,
        encoded,
        query,
        rows,
        device=torch.device("cpu"),
        chunk_size=2,
    )
    torch.testing.assert_close(actual, expected)


def test_membership_query_features_add_native_appearance_without_scale_bias():
    semantic = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    appearance = torch.tensor([[0.0, 5.0, 0.0], [6.0, 0.0, 0.0]])
    combined = compose_membership_query_features(semantic, appearance)
    assert combined.shape == (2, 5)
    assert torch.allclose(combined.norm(dim=-1), torch.ones(2))
    assert torch.allclose(combined[:, :2].norm(dim=-1), torch.full((2,), 2**-0.5))


def test_membership_query_features_reject_misaligned_appearance():
    with pytest.raises(ValueError, match="proposal axis"):
        compose_membership_query_features(torch.randn(2, 3), torch.randn(3, 4))


def test_similarity_control_matches_dense_cosine_on_selected_rows():
    features = torch.randn(9, 7)
    query = torch.randn(7)
    rows = torch.tensor([8, 2, 5, 1])
    expected = torch.nn.functional.normalize(features[rows], dim=-1) @ torch.nn.functional.normalize(query, dim=-1)
    actual = _similarity_scores_for_proposal(
        features, query, rows, device=torch.device("cpu"), chunk_size=2
    )
    assert torch.allclose(actual, expected, atol=1e-6)
    normalized = torch.nn.functional.normalize(features, dim=-1)
    cached = _similarity_scores_for_proposal(
        normalized,
        query,
        rows,
        device=torch.device("cpu"),
        chunk_size=2,
        features_are_normalized=True,
    )
    assert torch.allclose(cached, expected, atol=1e-6)


def test_training_pairs_treat_invisible_rows_as_unknown():
    supports = [torch.tensor([0, 2]), torch.tensor([3])]
    proposal_views = torch.tensor([0, 1])
    visible = torch.tensor(
        [[True, True, True, False, False], [False, False, False, True, True]]
    )
    proposal, row, label, weight = build_training_pairs(
        supports=supports,
        proposal_views=proposal_views,
        view_observed=visible,
        selected=torch.tensor([True, True]),
        positive_cap=8,
        negative_cap=8,
        seed=1,
    )
    pairs = set(zip(proposal.tolist(), row.tolist(), label.tolist()))
    assert pairs == {(0, 0, 1.0), (0, 2, 1.0), (0, 1, 0.0), (1, 3, 1.0), (1, 4, 0.0)}
    for item in (0, 1):
        selected = proposal == item
        assert torch.isclose(weight[selected & (label == 1)].sum(), torch.tensor(0.5))
        assert torch.isclose(weight[selected & (label == 0)].sum(), torch.tensor(0.5))


def test_training_pairs_preserve_soft_exact_mpr_targets():
    proposal, row, label, _weight = build_training_pairs(
        supports=[torch.tensor([0, 2])],
        proposal_views=torch.tensor([0]),
        view_observed=torch.tensor([[True, True, True]]),
        selected=torch.tensor([True]),
        positive_cap=8,
        negative_cap=8,
        seed=1,
        support_probabilities=[torch.tensor([0.25, 0.8])],
    )
    labels = {(int(p), int(r)): float(y) for p, r, y in zip(proposal, row, label)}
    assert labels[(0, 0)] == 0.25
    assert labels[(0, 2)] == pytest.approx(0.8)
    assert labels[(0, 1)] == 0.0


def test_threshold_calibration_maximizes_macro_iou():
    values = [
        (torch.tensor([0.9, 0.8, 0.1]), torch.tensor([True, True, False])),
        (torch.tensor([0.7, 0.2]), torch.tensor([True, False])),
    ]
    threshold, iou = _calibrate_threshold(values, 32)
    assert 0.2 < threshold <= 0.7
    assert iou == 1.0


def test_visible_membership_target_uses_the_dense_carrier_domain():
    visible = torch.tensor([7, 1, 9, 4, 2])
    support = torch.tensor([2, 7, 8])
    assert torch.equal(
        visible_membership_target(visible, support, num_rows=10),
        torch.tensor([True, False, False, False, True]),
    )


def test_track_augmentation_uses_training_views_only():
    rows = torch.tensor([0, 1, 0, 1, 2])
    proposals = torch.tensor([0, 0, 1, 1, 2])
    supports = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2])]
    augmented, observed, stats = track_augmented_training_targets(
        rows=rows,
        proposals=proposals,
        weights=torch.ones(rows.numel()),
        supports=supports,
        proposal_views=torch.tensor([0, 1, 2]),
        view_observed=torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
                [False, False, True, True],
            ]
        ),
        training=torch.tensor([True, True, False]),
        minimum_soft_cosine=0.5,
    )
    assert stats["num_tracks"] == 1
    assert torch.equal(augmented[0], torch.tensor([0, 1]))
    assert torch.equal(augmented[1], torch.tensor([0, 1]))
    assert torch.equal(augmented[2], torch.tensor([2]))
    assert torch.equal(observed[0], torch.tensor([True, True, True, False]))
    assert torch.equal(observed[2], torch.tensor([False, False, True, True]))
