import torch
from radio_gs.training.scope_view_relation import classify_visible_scope,exclude_view


def test_support_part_whole_and_disjoint_are_distinct():
    v=torch.ones(100,dtype=torch.bool);a=torch.arange(100)<30;b=torch.arange(100)<70;c=torch.arange(100)>=70
    assert classify_visible_scope(a,a,v)[0]=='support'
    assert classify_visible_scope(a,b,v)[0]=='current_part'
    assert classify_visible_scope(b,a,v)[0]=='current_whole'
    assert classify_visible_scope(a,c,v)[0]=='disjoint'


def test_unseen_foreground_is_not_a_negative_correspondence():
    target=torch.arange(100)<50;projection=torch.arange(100)>50;visible=torch.arange(100)>40
    assert classify_visible_scope(target,projection,visible)[0]=='weak_visibility'
    assert classify_visible_scope(target,torch.zeros_like(target),torch.ones_like(target))[0]=='weak_visibility'


def test_single_view_cannot_validate_itself_with_roundoff():
    n=torch.tensor([.5000001,.75]);d=torch.tensor([1.0000001,2.]);count=torch.tensor([1,2],dtype=torch.int16)
    mean,known=exclude_view(n,d,count,torch.tensor([.5,.5]),torch.tensor([1.,1.]),torch.tensor([True,True]))
    assert known.tolist()==[False,True]
    assert mean[0]==0 and mean[1]==.25
