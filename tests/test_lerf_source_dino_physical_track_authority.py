import torch

from radio_gs.scripts.build_lerf_source_dino_physical_track_authority import (
    best_proposal_by_minimum_overlap,
    transport_mask,
)


def test_transport_mask_abstains_for_missing_correspondence() -> None:
    mask = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
    mapping = torch.tensor([2, -1, 0, 1])
    transported = transport_mask(mask, mapping)
    assert torch.equal(transported, torch.tensor([[0, 0], [1, 0]], dtype=torch.bool))


def test_minimum_overlap_preserves_partial_as_nonzero_conflict() -> None:
    transported = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
    candidates = torch.tensor([[[1, 0], [0, 0]], [[0, 0], [1, 1]]], dtype=torch.bool)
    index, score, intersections = best_proposal_by_minimum_overlap(transported, candidates)
    assert index == 0
    assert score == 1.0
    assert intersections.tolist() == [1, 0]
