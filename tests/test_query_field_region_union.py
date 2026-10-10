import pytest
import torch
from radio_gs.query_field.region_union import prepare_union_parts
from radio_gs.query_field.part_conditioning import prepare_parts,read_parts

class Parts:
    count=4;view_count=1;views=torch.tensor([0,0])
    def matrix(self,key):
        return {'positive':torch.tensor([[1.,0.],[0.,1.],[0.,0.],[0.,0.]]),
                'negative':torch.tensor([[0.,1.],[1.,0.],[1.,1.],[0.,1.]])}[key].to_sparse()


def test_union_does_not_turn_another_correct_part_into_background():
    prior=torch.tensor([.4,.4,.7,.6]);available=torch.ones(4,dtype=torch.bool);regions=torch.ones(2)
    mixture=prepare_parts(Parts(),prior,available,region_probability=regions)
    union=prepare_union_parts(Parts(),prior,available,regions)
    a,_=read_parts(prior,available,mixture);b,details=read_parts(prior,available,union)
    assert bool((a[:2]<.5).all()) and bool((b[:2]>.5).all())
    assert b[2]<.5 and b[3]==prior[3]  # Unknown for one accepted part is not reliable union background.
    assert details['admitted_regions']==2
    torch.testing.assert_close(union[0][:,0],torch.tensor([1.,1.,0.,0.]))
    torch.testing.assert_close(union[1][:,0],torch.tensor([0.,0.,1.,0.]))


def test_single_region_matches_mixture_and_empty_acceptance_abstains():
    prior=torch.tensor([.4,.4,.7,.6]);available=torch.ones(4,dtype=torch.bool)
    for regions in [torch.tensor([.8,.1]),torch.tensor([.5,.1])]:
        mixture=prepare_parts(Parts(),prior,available,region_probability=regions)
        union=prepare_union_parts(Parts(),prior,available,regions)
        for a,b in zip(mixture[:2],union[:2]):torch.testing.assert_close(a,b)
    with pytest.raises(ValueError):prepare_union_parts(Parts(),prior,available,torch.ones(3))
