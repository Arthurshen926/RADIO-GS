import pytest
import torch
from radio_gs.querying.signed_reference_affinity import patch_authority,signed_reference_probability


def test_unknown_and_mixed_patches_have_no_authority():
    p=torch.zeros(4,4,dtype=torch.bool);n=p.clone();p[0,0]=True;n[1,1]=True;p[3,3]=True
    labels=patch_authority(p,n,(2,2))
    assert labels.tolist()==[[0,0],[0,1]]
    seed=torch.ones_like(p);control=patch_authority(p,n,(2,2),seed)
    assert control.tolist()==[[0,1],[1,1]]
    with pytest.raises(ValueError):patch_authority(p,p,(2,2))


def test_affinity_transfers_to_disconnected_target_without_unknown_negative():
    r=torch.tensor([[1.,0.],[.99,.01],[0.,1.],[.01,.99],[1.,0.]])
    labels=torch.tensor([1,1,-1,-1,0]);t=torch.tensor([[1.,0.],[0.,1.],[1.,0.]])
    prob,record=signed_reference_probability(r,t,labels)
    assert prob[0]>.89 and prob[1]<.11 and prob[2]==prob[0]
    assert record['source_loo_accuracy']==1 and record['unknown_patches']==1
    # Unknown features can change arbitrarily without changing the query.
    r[-1]=torch.tensor([.7,.7]);other,_=signed_reference_probability(r,t,labels)
    torch.testing.assert_close(prob,other)


def test_unseparable_or_incomplete_reference_is_neutral():
    r=torch.ones(4,2);labels=torch.tensor([1,1,-1,-1])
    p,d=signed_reference_probability(r,r,labels)
    assert (p==.5).all() and d['fallback']=='nonpositive_source_margin'
    p,d=signed_reference_probability(r,r,torch.tensor([1,0,0,0]))
    assert (p==.5).all() and d['fallback']=='insufficient_signed_support'
