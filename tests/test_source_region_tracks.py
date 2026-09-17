import pytest
import torch
from radio_gs.training.source_region_tracks import consistent_region_matches


def test_nontransitive_pairwise_matches_do_not_merge_disjoint_scopes():
    masks = torch.ones(4, 4, 4, dtype=torch.bool)
    masks[0, 2:] = False
    masks[3, :2] = False
    links = [(0, 1, .9), (1, 2, .8), (2, 3, .7), (3, 2, .7)]
    retained, report = consistent_region_matches(links, masks, torch.tensor([0, 1, 2, 0]))
    assert retained == links[:2]
    assert report['rejected'][0]['same_view_iou'] == 0
    assert (0, 2, .8) not in retained  # do not manufacture transitive positives
    assert report['rejected_links_are_unknown']


def test_same_view_duplicates_are_allowed_and_order_is_deterministic():
    masks = torch.ones(4, 4, 4, dtype=torch.bool)
    links = [(0, 1, .9), (1, 2, .8), (2, 3, .7)]
    _, report = consistent_region_matches(links, masks, torch.tensor([0, 1, 2, 0]))
    _, reverse = consistent_region_matches(links[::-1], masks, torch.tensor([0, 1, 2, 0]))
    assert report == reverse
    assert report['accepted_edges'] == 3


def test_duplicate_similarity_is_checked_for_all_pairs_not_transitively():
    masks = torch.ones(6, 1, 100, dtype=torch.bool)
    masks[0, :, :10] = False
    masks[2, :, :5] = False
    masks[2, :, -5:] = False
    masks[4, :, -10:] = False
    links = [(0, 1, .9), (1, 2, .8), (2, 3, .7), (3, 4, .6), (4, 5, .5)]
    _, report = consistent_region_matches(links, masks, torch.tensor([0, 1, 0, 2, 0, 3]), duplicate_iou=.89)
    assert len(report['rejected']) == 1
    assert report['rejected'][0]['regions'] == [3, 4]


def test_invalid_same_view_edge_rejected():
    with pytest.raises(ValueError):
        consistent_region_matches([(0, 1, .8)], torch.ones(2, 4, 4, dtype=torch.bool), torch.zeros(2))
