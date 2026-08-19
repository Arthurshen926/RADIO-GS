import torch

from radio_gs.querying.sam_categorical_instance_posterior import (
    propagate_categorical_identity_over_proposals,
)


def test_exact_mpr_proposals_extend_markers_without_overwriting_strong_identity():
    scores = torch.tensor(
        [
            [0.90, 0.10],  # immutable class 0 marker
            [0.50, 0.51],  # ambiguous member becomes class 0
            [0.10, 0.90],  # immutable class 1 marker
            [0.51, 0.50],  # ambiguous member becomes class 1
        ]
    )
    rows = torch.tensor([0, 1, 2, 3])
    proposals = torch.tensor([0, 0, 1, 1])
    weights = torch.ones(4)
    result, stats = propagate_categorical_identity_over_proposals(
        scores,
        rows,
        proposals,
        weights,
        num_proposals=2,
        semantic_tolerance=0.02,
        minimum_supporting_proposals=1,
    )
    assert result.argmax(dim=1).tolist() == [0, 0, 1, 1]
    assert stats["changed_rows"] == 2
    torch.testing.assert_close(result[[0, 2]], scores[[0, 2]])


def test_conflicting_instance_markers_stop_at_watershed():
    scores = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.50, 0.50]])
    rows = torch.tensor([0, 1, 2])
    proposals = torch.zeros(3, dtype=torch.long)
    result, stats = propagate_categorical_identity_over_proposals(
        scores,
        rows,
        proposals,
        torch.ones(3),
        num_proposals=1,
        consensus_threshold=0.7,
    )
    torch.testing.assert_close(result, scores)
    assert stats["changed_rows"] == 0


def test_independent_source_views_are_required_for_multiview_consensus():
    scores = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.50, 0.51]])
    rows = torch.tensor([0, 2, 1, 2])
    proposals = torch.tensor([0, 0, 1, 1])
    weights = torch.ones(4)
    same_view, same_stats = propagate_categorical_identity_over_proposals(
        scores,
        rows,
        proposals,
        weights,
        num_proposals=2,
        proposal_view_indices=torch.tensor([4, 4]),
        semantic_tolerance=0.02,
        minimum_supporting_proposals=2,
        minimum_supporting_views=2,
    )
    torch.testing.assert_close(same_view, scores)
    assert same_stats["changed_rows"] == 0

    cross_view, cross_stats = propagate_categorical_identity_over_proposals(
        scores,
        rows,
        proposals,
        weights,
        num_proposals=2,
        proposal_view_indices=torch.tensor([4, 9]),
        semantic_tolerance=0.02,
        minimum_supporting_proposals=2,
        minimum_supporting_views=2,
    )
    assert cross_view.argmax(dim=1).tolist() == [0, 0, 0]
    assert cross_stats["changed_rows"] == 1
