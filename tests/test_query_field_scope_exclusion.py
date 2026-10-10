import pytest
import torch
from radio_gs.query_field.observations import partial_mask_target
from radio_gs.query_field.scope_exclusion import union_exclusion


def fixture():
    masks=torch.zeros(5,9,27,dtype=torch.bool)
    for i in range(4):masks[i,2:7,1+6*i:6+6*i]=True
    masks[4]=masks[0]|masks[2]  # A spanning region cannot exclude an accepted part.
    return masks,torch.tensor([1.,4.,1.,4.,1.])


def test_single_region_replays_existing_permission():
    masks,scales=fixture();visible=torch.ones_like(masks[0]);visible[4,15]=False
    for i in range(len(masks)):
        _,old=partial_mask_target(masks,i,scales);old &= visible
        accepted=torch.arange(len(masks))==i
        new,_=union_exclusion(masks,scales,accepted,visible,old)
        assert torch.equal(new,old)


def test_mixed_scale_union_recovers_disjoint_interiors_but_not_background():
    masks,scales=fixture();accepted=torch.tensor([True,True,False,False,False]);visible=torch.ones_like(masks[0])
    old=torch.stack([partial_mask_target(masks,i,scales)[1] for i in [0,1]]).all(0)
    new,permitted=union_exclusion(masks,scales,accepted,visible,old)
    assert permitted.tolist()==[False,False,True,True,False]
    assert new[4,15] and new[4,21] and not old[4,15] and not old[4,21]
    assert not bool((new&masks[accepted].any(0)).any()) and not new[0,0]
    order=torch.tensor([4,3,2,1,0])
    other,_=union_exclusion(masks[order],scales[order],accepted[order],visible,old)
    assert torch.equal(other,new)


def test_empty_scope_and_unobserved_pixels_do_not_become_negative():
    masks,scales=fixture();visible=torch.zeros_like(masks[0]);old=torch.zeros_like(visible)
    empty=torch.zeros(len(masks),dtype=torch.bool)
    new,_=union_exclusion(masks,scales,empty,visible,old);assert not new.any()
    empty[0]=True;new,_=union_exclusion(masks,scales,empty,visible,old);assert not new.any()
    with pytest.raises(ValueError,match='conflicts'):
        union_exclusion(masks,scales,empty,visible,torch.ones_like(old))
