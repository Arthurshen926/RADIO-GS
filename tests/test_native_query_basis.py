import torch
import pytest
from radio_gs.training.native_query_basis import native_query_basis


def test_full_basis_preserves_native_cosines_without_origin_shift():
    q=torch.tensor([[1.,2.,0.],[1.,0.,3.],[0.,2.,1.],[2.,1.,3.]])
    basis,energy=native_query_basis(q,3)
    native=torch.nn.functional.normalize(q,dim=-1)
    projected=torch.nn.functional.normalize(q@basis.T,dim=-1)
    assert torch.allclose(projected@projected.T,native@native.T,atol=1e-6)
    assert energy==pytest.approx(1)


def test_basis_is_detached_and_rejects_unsupported_dimension():
    q=torch.eye(3,requires_grad=True)
    assert not native_query_basis(q,2)[0].requires_grad
    with pytest.raises(ValueError):native_query_basis(q,4)
