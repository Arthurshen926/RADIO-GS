import torch

from radio_gs.querying.object_track_extent_authority import (
    compile_object_track_extent_authority,
)


def test_track_authority_aggregates_views_but_preserves_unknown_rows() -> None:
    tracks = compile_object_track_extent_authority(
        torch.tensor([4, 4]), torch.tensor([0, 1]), torch.tensor([1, 0]),
        [torch.tensor([2, 3]), torch.tensor([2, 5])],
        [torch.tensor([0.5, 0.8]), torch.tensor([0.6, 0.7])],
        [torch.tensor([8, 9]), torch.tensor([9, 10])],
    )
    track = tracks[4]
    assert torch.equal(track.proposal_rows, torch.tensor([0, 1]))
    assert torch.equal(track.positive_rows, torch.tensor([2, 3, 5]))
    assert torch.allclose(track.positive_probability, torch.tensor([0.8, 0.8, 0.7]))
    assert torch.equal(track.positive_view_count, torch.tensor([2, 1, 1]))
    assert torch.equal(track.explicit_negative_rows, torch.tensor([8, 9, 10]))
    assert 7 not in track.explicit_negative_rows.tolist()


def test_track_positive_evidence_overrides_conflicting_negative() -> None:
    track = compile_object_track_extent_authority(
        torch.tensor([0]), torch.tensor([0]), torch.tensor([0]),
        [torch.tensor([1, 2])], [torch.tensor([0.9, 0.5])],
        [torch.tensor([2, 3])],
    )[0]
    assert torch.equal(track.explicit_negative_rows, torch.tensor([3]))
