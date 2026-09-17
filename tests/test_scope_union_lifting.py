import pytest
import torch
from radio_gs.training.scope_union_lifting import lift_mask_union


def test_union_precedes_projection_for_complementary_parts():
    mapping=torch.tensor([[1.,0.],[1.,0.]]).to_sparse()
    parts=torch.eye(2,dtype=torch.bool)
    combined,seen=lift_mask_union(mapping,parts)
    separate=torch.stack([lift_mask_union(mapping,p[None])[0] for p in parts]).amax(0)
    assert combined.tolist()==[1.,0.] and separate.tolist()==[.5,0.]
    assert seen.tolist()==[True,False]


def test_duplicates_and_nested_masks_do_not_double_count():
    mapping=torch.eye(3).to_sparse();masks=torch.tensor([[1,1,0],[1,0,0],[1,1,0]])
    value,_=lift_mask_union(mapping,masks)
    assert value.tolist()==[1.,1.,0.]
    with pytest.raises(ValueError):lift_mask_union(mapping,masks[:0])
