import pytest
import torch
from radio_gs.models.query_scope_correction import QueryScopeCorrection


def test_zero_initialization_is_exact_and_trainable():
    torch.manual_seed(4)
    model=QueryScopeCorrection(6,4)
    p=torch.tensor([[-9.,9.],[-2.,2.]])
    n=-p
    z=torch.randn(2,4);q=torch.randn(2,6);active=torch.ones(2,dtype=torch.bool)
    value=model(p,n,z,q,active)
    assert torch.equal(value,p)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(value,torch.tensor([[1.,0.],[1.,0.]]))
    loss.backward()
    assert model.query_coefficients[-1].bias.grad.abs().sum()>0


def test_fallback_and_row_batch_invariance_and_native_limit():
    torch.manual_seed(5)
    model=QueryScopeCorrection(6,4)
    with torch.no_grad():model.query_coefficients[-1].bias[-1]=.5
    p=torch.randn(7,3);n=torch.randn(7,3);z=torch.randn(7,4);q=torch.randn(3,6);active=torch.tensor([True,False,True])
    value=model(p,n,z,q,active)
    assert torch.equal(value[:,1],p[:,1])
    assert torch.allclose(value[:2],model(p[:2],n[:2],z[:2],q,active))
    assert torch.allclose(value[:,0:1],model(p[:,0:1],n[:,0:1],z,q[:1],active[:1]))
    assert torch.allclose(value[:,0].sigmoid(),(p[:,0].sigmoid()+n[:,0].sigmoid())/2)
    with torch.no_grad():model.query_coefficients[-1].bias[-1]=2
    assert torch.allclose(model(p,n,z,q,active)[:,0],n[:,0],atol=2e-6)
    with pytest.raises(ValueError):model(p,n[:1],z,q,active)


def test_sigmoid_control_preserves_parent_and_recovers_from_negative_coefficients():
    model=QueryScopeCorrection(6,4,mode='sigmoid_v1')
    p=torch.full((3,1),-2.);n=-p;z=torch.randn(3,4);q=torch.randn(1,6);active=torch.ones(1,dtype=torch.bool)
    assert torch.equal(model(p,n,z,q,active),p)
    model.enabled.fill_(True)
    with torch.no_grad():model.query_coefficients[-1].bias[-1]=-10.
    value=model(p,n,z,q,active)
    torch.nn.functional.binary_cross_entropy_with_logits(value,torch.ones_like(value)).backward()
    assert model.query_coefficients[-1].bias.grad[-1]<0
