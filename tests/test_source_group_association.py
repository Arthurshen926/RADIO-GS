import torch
import pytest
from radio_gs.training.source_group_association import associate_source_groups


def test_merges_compatible_views_but_not_same_view_or_different_semantics():
    masks=torch.tensor([[1,1,1,0],[1,1,0,1],[1,1,1,0]],dtype=torch.bool)
    descriptors=torch.tensor([[1.,0],[1.,0],[0.,1]])
    assert associate_source_groups(masks,descriptors,torch.tensor([0,1,2]))==[[0,1]]
    assert associate_source_groups(masks[:2],descriptors[:2],torch.tensor([0,0]))==[]


def test_rejects_part_whole_and_transitive_bridges():
    masks=torch.zeros(3,12,dtype=torch.bool);masks[0,:6]=True;masks[1,3:9]=True;masks[2,6:]=True
    groups=associate_source_groups(masks,torch.ones(3,2),torch.arange(3))
    assert groups==[[0,1]]  # A overlaps B, B overlaps C, A and C disagree.
    whole=torch.ones(2,12,dtype=torch.bool);whole[0,2:]=False
    assert associate_source_groups(whole,torch.ones(2,2),torch.arange(2))==[]


def test_empty_membership_fails_closed():
    with pytest.raises(ValueError,match='nonempty'):
        associate_source_groups(torch.zeros(2,4,dtype=torch.bool),torch.ones(2,2),torch.arange(2))


def test_occluded_surfaces_are_unknown_not_negative():
    masks=torch.zeros(2,10,dtype=torch.bool)
    masks[0,:6]=True;masks[1,4:]=True
    # Only the overlapping two surface elements are visible in both views.
    assert associate_source_groups(masks,torch.ones(2,2),torch.arange(2))==[]
    assert associate_source_groups(masks,torch.ones(2,2),torch.arange(2),visibility=masks)==[[0,1]]
    assert associate_source_groups(masks,torch.ones(2,2),torch.arange(2),visibility=masks,minimum_shared_elements=3)==[]
    with pytest.raises(ValueError,match='visibility'):
        associate_source_groups(masks,torch.ones(2,2),torch.arange(2),visibility=torch.zeros_like(masks))
