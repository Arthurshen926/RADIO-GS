import pytest
import torch
from radio_gs.query_field.part_conditioning import integrate_views,condition_parts


def test_leave_one_view_out_never_self_protects_and_unknown_abstains():
    prior=torch.tensor([.7,.2]);p=torch.tensor([[.4],[0.]]);n=torch.tensor([[.6],[0.]])
    a,_=integrate_views(prior,p,n);b,protected=integrate_views(prior,p,n,protect=True)
    torch.testing.assert_close(a,b);torch.testing.assert_close(n,protected)
    assert b[1]==prior[1]


def test_independent_positive_protection_matches_literal_view_loop():
    p=torch.tensor([[.8,0.,.2],[0.,0.,0.],[.2,.3,0.]])
    n=torch.tensor([[0.,.9,.4],[0.,0.,.5],[.5,.2,.1]])
    prior=torch.tensor([.3,.7,.5]);expected=n.clone()
    for i in range(3):
        for v in range(3):
            ids=[j for j in range(3) if j!=v]
            denominator=(p[i,ids]+n[i,ids]).sum()
            expected[i,v]*=1-(p[i,ids].sum()/denominator if denominator>0 else 0)
    scores,actual=integrate_views(prior,p,n,protect=True)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(scores,(prior+p.sum(1))/(1+p.sum(1)+expected.sum(1)))
    control,_=integrate_views(prior,p,n)
    assert (scores>=control).all()  # Protection alone can also increase false members.


class Evidence:
    count=4;view_count=2;views=torch.tensor([0,0,1])
    def matrix(self,key):
        return {'positive':torch.tensor([[1.,0.,1.],[1.,0.,1.],[0.,1.,0.],[0.,0.,0.]]),
                'negative':torch.tensor([[0.,1.,0.],[0.,1.,0.],[1.,0.,1.],[0.,0.,0.]])}[key].to_sparse()


def test_conditioning_can_extend_a_supported_part_without_promoting_unknown():
    prior=torch.tensor([.9,.4,.1,.7]);available=torch.ones(4,dtype=torch.bool)
    score,d=condition_parts(Evidence(),prior,available,protect=True)
    assert score[1]>.5 and score[2]<.1 and score[3]==prior[3]
    assert d['admitted_regions']==2 and d['newly_selected']==1
    # A tiny positive margin must remain weak rather than becoming a full view.
    weak,_=condition_parts(Evidence(),torch.tensor([.50001,.50001,.1,.7]),available)
    assert (weak[:2]-.50001).abs().max()<1e-4
    with pytest.raises(ValueError):condition_parts(Evidence(),prior*2,available)
