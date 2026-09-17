import torch
from scripts.audits.refine_rendered_partial_extents import masked_loss


def test_ignored_negative_cannot_push_shared_element_response_down():
    # A two-pixel renderer shares one element across a positive and an
    # unverified negative pixel. Abstention removes that negative gradient.
    element = torch.tensor([[.3]], requires_grad=True)
    renderer = torch.tensor([[1.], [1.]]).to_sparse()
    prediction = torch.sparse.mm(renderer, element)
    truth = torch.tensor([[1.], [0.]])
    loss = masked_loss(prediction, truth, torch.tensor([[True], [False]]))
    loss.backward()
    assert element.grad.item() < 0
    p = torch.tensor([[.3], [.3]], requires_grad=True)
    masked_loss(p, truth, torch.tensor([[True], [False]])).backward()
    assert p.grad[0].item() < 0 and p.grad[1].item() == 0


def test_empty_supervision_has_zero_finite_gradient():
    p = torch.tensor([[.2], [.8]], requires_grad=True)
    loss = masked_loss(p, torch.tensor([[1.], [0.]]), torch.zeros_like(p, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and loss.item() == 0 and not p.grad.any()
