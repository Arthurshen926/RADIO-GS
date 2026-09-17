import torch
from scripts.audits.train_rendered_extent import balanced_mask_loss


def test_lift_and_reproject_is_not_identity_even_with_correct_visibility():
    render=torch.tensor([[1.,0.],[.5,.5],[0.,1.]])
    target=torch.tensor([[1.],[0.],[0.]])
    lifted=(render.T@target)/render.sum(0)[:,None]
    rerendered=render@lifted
    assert not torch.allclose(rerendered,target)
    assert rerendered[1,0]>0 and rerendered[0,0]<1


def test_render_loss_respects_unknown_pixels_and_backpropagates_to_elements():
    matrix=torch.tensor([[1.,0.],[.5,.5],[0.,1.]]).to_sparse()
    elements=torch.tensor([[.8],[.2]],requires_grad=True)
    prediction=torch.sparse.mm(matrix,elements)
    valid=torch.tensor([True,False,True]);target=torch.tensor([[1.],[0.],[0.]])
    a=balanced_mask_loss(prediction,target,valid)
    target[1]=1
    b=balanced_mask_loss(prediction,target,valid)
    assert torch.equal(a,b)
    a.backward()
    assert elements.grad[0]<0 and elements.grad[1]>0
