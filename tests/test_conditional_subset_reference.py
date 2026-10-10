import torch
from scripts.research.conditional_subset_reference import silhouette_loss,solve


def test_unknown_pixels_do_not_enter_silhouette_loss_or_gradient():
    p=torch.tensor([True,False,False]);n=torch.tensor([False,True,False])
    a=torch.tensor([.2,.02,.8],requires_grad=True);b=torch.tensor([.2,.02,.01])
    loss=silhouette_loss(a,p,n);torch.testing.assert_close(loss,silhouette_loss(b,p,n))
    loss.backward();assert a.grad[2]==0 and a.grad[0]<0 and a.grad[1]>0


def test_empty_source_reference_does_not_fit_or_change_initializer():
    initial=torch.tensor([.2,.8]);result,details=solve(None,initial,[],1.)
    assert torch.equal(result,initial) and details['steps']==0
