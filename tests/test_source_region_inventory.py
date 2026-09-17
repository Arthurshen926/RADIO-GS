import torch
from radio_gs.training.source_region_inventory import augment_region_inventory


def test_grounded_augmentation_preserves_query_free_parts_and_validation():
    parent=torch.tensor([[[1,0],[0,0]],[[0,0],[0,1]]],dtype=torch.bool)
    views=torch.tensor([0,1]);desc=torch.eye(2)
    extra=torch.cat([parent[:1],torch.ones(1,2,2,dtype=torch.bool)])
    m,v,d,ids=augment_region_inventory(parent,views,desc,extra,torch.tensor([0,0]),torch.ones(2,2))
    assert ids.tolist()==[1]
    assert torch.equal(m[:2],parent) and torch.equal(d[:2],desc) and torch.equal(v[:2],views)
    assert len(m)==3  # Preserve nested part + whole, not just grounded whole.


def test_same_mask_in_different_views_is_not_duplicate():
    mask=torch.ones(1,2,2,dtype=torch.bool)
    m,v,d,ids=augment_region_inventory(mask,torch.tensor([0]),torch.ones(1,2),mask,torch.tensor([1]),torch.ones(1,2))
    assert ids.tolist()==[0] and len(m)==2
