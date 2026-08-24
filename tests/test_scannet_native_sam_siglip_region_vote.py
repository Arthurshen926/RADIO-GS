import torch

from radio_gs.scripts.evaluate_scannet_native_sam_siglip_region_vote import (
    _region_per_view,
)


def test_region_vote_gives_each_source_view_one_authority() -> None:
    # Row zero has two nested proposals in view zero and one proposal in view
    # one.  The two views, not the three masks, must receive equal authority.
    region, count, agreement = _region_per_view(
        row_indices=torch.tensor([0, 0, 0]),
        proposal_indices=torch.tensor([0, 1, 2]),
        weights=torch.ones(3),
        proposal_views=torch.tensor([0, 0, 1]),
        proposal_scores=torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        num_rows=2,
    )
    assert torch.allclose(region[0], torch.tensor([0.5, 0.5]))
    assert count.tolist() == [2, 0]
    assert agreement.tolist() == [0.5, 0.0]


def test_region_vote_normalizes_membership_inside_each_view() -> None:
    region, count, agreement = _region_per_view(
        row_indices=torch.tensor([0, 0]),
        proposal_indices=torch.tensor([0, 1]),
        weights=torch.tensor([1.0, 3.0]),
        proposal_views=torch.tensor([0, 0]),
        proposal_scores=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        num_rows=1,
    )
    assert torch.allclose(region[0], torch.tensor([0.25, 0.75]))
    assert count.item() == 1
    assert agreement.item() == 1.0
