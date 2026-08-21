import torch

from radio_gs.querying.source_multiview_object_tracks import (
    build_source_multiview_object_tracks,
)


def test_reciprocal_cross_view_overlap_builds_track_and_rejects_singleton():
    # proposals 0/1 see the same object from independent views; proposal 2 is
    # an unrelated singleton in a third view.
    result = build_source_multiview_object_tracks(
        torch.tensor([0, 1, 0, 1, 3]),
        torch.tensor([0, 0, 1, 1, 2]),
        torch.ones(5),
        torch.tensor([4, 8, 9]),
        num_rows=4,
        num_proposals=3,
    )
    assert result.num_tracks == 1
    assert result.proposal_track_indices.tolist() == [0, 0, -1]
    assert result.track_view_counts.tolist() == [2]
    assert set(result.row_indices.tolist()) == {0, 1}


def test_same_view_proposals_never_associate():
    result = build_source_multiview_object_tracks(
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 0, 1, 1]),
        torch.ones(4),
        torch.tensor([3, 3]),
        num_rows=2,
        num_proposals=2,
    )
    assert result.num_tracks == 0


def test_proposal_permutation_preserves_membership_partition():
    rows = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3])
    proposals = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    views = torch.tensor([0, 1, 0, 1])
    original = build_source_multiview_object_tracks(
        rows, proposals, torch.ones(8), views, num_rows=4, num_proposals=4
    )
    order = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(4)
    permuted = build_source_multiview_object_tracks(
        rows,
        inverse[proposals],
        torch.ones(8),
        views[order],
        num_rows=4,
        num_proposals=4,
    )
    original_sets = sorted(
        tuple(sorted(original.row_indices[original.track_indices == track].tolist()))
        for track in range(original.num_tracks)
    )
    permuted_sets = sorted(
        tuple(sorted(permuted.row_indices[permuted.track_indices == track].tolist()))
        for track in range(permuted.num_tracks)
    )
    assert original_sets == permuted_sets
