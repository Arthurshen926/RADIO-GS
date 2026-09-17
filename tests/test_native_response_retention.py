import torch
from torch.nn import functional as F
from radio_gs.training.native_response_retention import region_element_weights,native_response_loss


def test_sparse_region_weights_match_pixel_then_region_average_with_unknowns():
    matrix=torch.tensor([[.3,.5,.2],[.2,.1,.7],[0.,0.,1.]]).to_sparse();observed=torch.tensor([True,True,False]);support=matrix@observed.float();mask=torch.ones(3,dtype=torch.bool)
    ids,w=region_element_weights(matrix,support,mask,observed)
    assert ids.tolist()==[0,1] and torch.allclose(w,torch.tensor([.375,.625]))
    assert region_element_weights(matrix,support,torch.tensor([False,False,True]),observed) is None


def test_response_loss_matches_actual_mixture_not_normalized_mean_and_detaches_teacher():
    z=torch.tensor([[10.,0.],[0.,1.]],requires_grad=True);head=torch.nn.Linear(2,2)
    with torch.no_grad():head.weight.copy_(torch.eye(2));head.bias.zero_()
    q=torch.eye(2,requires_grad=True);teacher=torch.ones(2,requires_grad=True);weight=torch.tensor([.5,.5]);t=F.normalize(teacher.detach(),dim=0);actual=torch.tensor([.5,.5]);loss=native_response_loss(z,head,q,teacher,weight)
    expected=F.mse_loss(actual,t)+.07**2*F.kl_div(F.log_softmax(actual/.07,dim=0),F.softmax(t/.07,dim=0),reduction='sum')
    assert torch.allclose(loss,expected,atol=1e-6)
    loss.backward();assert z.grad.norm()>0 and head.weight.grad.norm()>0
    assert q.grad is None and teacher.grad is None
