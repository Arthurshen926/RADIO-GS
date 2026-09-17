import pytest
import torch
from radio_gs.training.source_region_hierarchy import source_region_hierarchy, exclusive_scope_masks


def test_nested_levels_preserve_union_without_mixing_scales():
    masks = torch.zeros(3, 6, 6, dtype=torch.bool)
    masks[0] = True
    masks[1, 1:5, 1:5] = True
    masks[2, 2:4, 2:4] = True
    rel = source_region_hierarchy(masks, torch.zeros(3))
    assert rel.sum() == 3
    for scope in ['fine', 'coarse']:
        result = exclusive_scope_masks(masks, rel, scope=scope)
        assert torch.equal(result.any(0), masks.any(0))
        assert result.sum(0).max() == 1
    assert exclusive_scope_masks(masks, rel, scope='fine')[2].sum() == 4
    assert exclusive_scope_masks(masks, rel, scope='coarse')[0].sum() == 36


def test_duplicates_partial_overlap_and_other_views_are_not_containment():
    masks = torch.zeros(5, 6, 6, dtype=torch.bool)
    masks[:2, :4, :4] = True
    masks[2, 2:, 2:] = True
    masks[3, :2, :2] = True
    # Empty masks must never become children.
    rel = source_region_hierarchy(masks, torch.tensor([0, 0, 0, 1, 0]))
    assert not rel.any()
    assert torch.equal(exclusive_scope_masks(masks, rel, scope='fine'), masks)


def test_approximate_containment_keeps_child_pixels_outside_parent():
    masks = torch.zeros(2, 10, 10, dtype=torch.bool)
    masks[1, :8] = True
    masks[0, :2] = True
    masks[0, 9, 0] = True
    rel = source_region_hierarchy(masks, torch.zeros(2))
    assert rel[0, 1]
    for scope in ['fine', 'coarse']:
        result = exclusive_scope_masks(masks, rel, scope=scope)
        assert result[0, 9, 0]
        assert torch.equal(result.any(0), masks.any(0))


def test_invalid_inputs():
    with pytest.raises(ValueError):
        source_region_hierarchy(torch.zeros(2, 4, 4), torch.zeros(2))
    with pytest.raises(ValueError):
        source_region_hierarchy(torch.zeros(2, 4, 4, dtype=torch.bool), torch.zeros(2), max_area_ratio=1)
