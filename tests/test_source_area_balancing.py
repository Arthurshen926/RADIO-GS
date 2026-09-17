import torch
from radio_gs.training.source_area_balancing import area_balanced_pixel_votes


def test_parent_and_part_octaves_each_get_half_vote_despite_duplicates():
    masks=torch.tensor([[1,1,1,1],[1,0,0,0],[1,0,0,0]],dtype=torch.bool)
    votes=area_balanced_pixel_votes(masks)
    torch.testing.assert_close(votes[:,0],torch.tensor([.5,.25,.25]))
    torch.testing.assert_close(votes.sum(0),torch.ones(4))
    perm=torch.tensor([2,0,1]);torch.testing.assert_close(area_balanced_pixel_votes(masks[perm]),votes[perm])


def test_empty_regions_and_uncovered_pixels_never_gain_evidence():
    masks=torch.tensor([[0,0,0],[1,0,0]],dtype=torch.bool)
    votes=area_balanced_pixel_votes(masks)
    torch.testing.assert_close(votes, masks.float())
