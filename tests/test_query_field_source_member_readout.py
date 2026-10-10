import pytest
import torch
from radio_gs.query_field.source_member_readout import read_source_members
from radio_gs.query_field.part_conditioning import read_parts


def test_weak_known_evidence_changes_member_admission_but_unknown_keeps_prior():
    prior=torch.tensor([.1,.9,.8,.9]);available=torch.tensor([True,True,True,False])
    p=torch.tensor([[.01],[0.],[0.],[.01]]);n=torch.tensor([[0.],[.01],[0.],[0.]])
    prepared=(p,n,{})
    finite,_=read_parts(prior,available,prepared);source,details=read_source_members(prior,available,prepared)
    assert finite[0]<.5 and finite[1]>.5
    torch.testing.assert_close(source,torch.tensor([1.,0.,.8,0.]));assert details['prior_only_on_unknown']


def test_fraction_is_scale_invariant_and_protection_cannot_remove_members():
    prior=torch.tensor([.2,.8]);available=torch.ones(2,dtype=torch.bool)
    p=torch.tensor([[.2,.3],[.1,.0]]);n=torch.tensor([[.4,.1],[.2,.3]])
    a,_=read_source_members(prior,available,(p,n,{}))
    b,_=read_source_members(prior,available,(p*.001,n*.001,{}))
    c,_=read_source_members(prior,available,(p,n,{}),protect=True)
    torch.testing.assert_close(a,b);assert bool((c>=a).all())
    with pytest.raises(ValueError):read_source_members(prior,available,(p*float('nan'),n,{}))
