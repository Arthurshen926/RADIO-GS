import pytest
import torch
from radio_gs.querying.local_response_credit import LocalResponseCredit, huber, matched_uniform_edit


def test_exact_huber_credit_with_cancellation_and_multiple_queries():
    op = torch.tensor([[.5,.5,0.],[0.,.25,.75],[1.,0.,0.]],dtype=torch.float64).to_sparse()
    old = torch.tensor([[.1,.9],[.7,.2],[.2,.6]],dtype=torch.float64)
    new = torch.tensor([[.9,.1],[.1,.8],[.8,.3]],dtype=torch.float64)
    y = torch.tensor([[.8,.1],[.2,.6],[1.,0.]],dtype=torch.float64)
    w = torch.tensor([[1.,.2],[.6,1.],[0.,.8]],dtype=torch.float64)
    credit=LocalResponseCredit(old,new);credit.add(op,y,w)
    expected=(w*(huber(y-op.to_dense()@old,.25)-huber(y-op.to_dense()@new,.25))).sum()
    assert torch.allclose(credit.net.sum(),expected,atol=1e-12)
    assert torch.all(credit.gross >= credit.net.abs()-1e-12)
    assert abs(credit.report()['conservation_error']) < 1e-12


def test_harm_unknown_and_opposite_edits_receive_no_permission():
    old=torch.tensor([[.1],[.1],[.1],[.8]])
    new=torch.tensor([[.9],[.9],[.9],[.2]])
    op=torch.eye(4).to_sparse();y=torch.tensor([[1.],[0.],[1.],[0.]])
    credit=LocalResponseCredit(old,new);credit.add(op,y,torch.tensor([[1.],[1.],[0.],[1.]]))
    assert torch.equal(credit.gate(new),torch.tensor([[1.],[0.],[0.],[1.]]))
    opposite=old.clone();opposite[0]=0
    assert credit.gate(opposite).sum()==0


def test_conflicting_views_shrink_update_and_matched_control_preserves_signed_mass():
    old=torch.tensor([[.1,.8],[.7,.2],[.3,.5]],dtype=torch.float64)
    new=torch.tensor([[.9,.4],[.1,.9],[.8,.1]],dtype=torch.float64)
    credit=LocalResponseCredit(old,new)
    op=torch.eye(3,dtype=torch.float64).to_sparse()
    credit.add(op,new,torch.ones_like(old));credit.add(op,old,torch.full_like(old,.5))
    gate=credit.gate(new)
    assert torch.allclose(gate,torch.full_like(old,1/3),atol=1e-12)
    # Nonuniform gate makes the matched control test more than an identity.
    gate[0,0]=0;gate[1,1]=.8
    matched,report=matched_uniform_edit(old,new,gate)
    for sign in (1,-1):
        expected=((sign*(new-old)).clamp_min(0)*gate).sum(0)
        assert torch.allclose((sign*(matched-old)).clamp_min(0).sum(0),expected,atol=1e-12)


def test_invalid_inputs_rejected():
    old=torch.zeros(2,1)
    with pytest.raises(ValueError):LocalResponseCredit(old,old+float('nan'))
    c=LocalResponseCredit(old,old+1)
    with pytest.raises(ValueError):c.add(torch.eye(2).to_sparse(),old,-torch.ones_like(old))


def test_credit_conservation_does_not_imply_gated_loss_improvement():
    # Opposite element edits cancel in projection. Keeping only the positively
    # credited edit breaks that cancellation and may overshoot the source target.
    old=torch.tensor([[.5],[.5]],dtype=torch.float64)
    new=torch.tensor([[1.],[.1]],dtype=torch.float64)
    op=torch.tensor([[.5,.5]],dtype=torch.float64).to_sparse()
    y=torch.tensor([[.55]],dtype=torch.float64)
    c=LocalResponseCredit(old,new);c.add(op,y,torch.ones_like(y))
    gated=old+c.gate(new)*(new-old)
    loss=lambda value:huber(y-torch.sparse.mm(op,value),.25).sum()
    assert loss(new)<loss(old)<loss(gated)
    assert abs(c.report()['conservation_error'])<1e-12


def test_element_credits_match_independent_numerical_path_integral():
    matrix=torch.tensor([[.7,.3,0.],[.2,.3,.5]],dtype=torch.float64)
    old=torch.tensor([[.1],[.8],[.2]],dtype=torch.float64)
    new=torch.tensor([[.9],[.1],[.6]],dtype=torch.float64)
    y=torch.tensor([[.95],[.1]],dtype=torch.float64)
    weight=torch.tensor([[.3],[.9]],dtype=torch.float64)
    c=LocalResponseCredit(old,new);c.add(matrix.to_sparse(),y,weight)
    steps=torch.linspace(0,1,2001,dtype=torch.float64)
    path=old[None]+steps[:,None,None]*(new-old)[None]
    gradient=matrix.T@(weight*(y-matrix@path).clamp(-.25,.25))
    expected=(new-old)*torch.trapezoid(gradient,steps,dim=0)
    assert torch.allclose(c.net,expected,atol=1e-8)
