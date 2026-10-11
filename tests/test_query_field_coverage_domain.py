"""An observed-only parity check must not conceal semantic-only member loss."""
from types import SimpleNamespace
import pytest
import torch
from radio_gs.query_field.geometry import member_domain_mask,relaxed_subset_alpha,selected_alpha
from radio_gs.query_field.coverage_factors import compile_coverage_view,CoverageView


def test_explicit_domain_is_validated_without_changing_legacy_default():
    field=SimpleNamespace(xyz=torch.zeros(3,3),observed=torch.tensor([True,False,False]))
    assert member_domain_mask(field) is field.observed
    available=torch.tensor([True,True,False])
    assert torch.equal(member_domain_mask(field,available),available)
    for invalid in [available.float(),available[:2],available[:,None]]:
        with pytest.raises(ValueError):member_domain_mask(field,invalid)
    assert field.observed.tolist()==[True,False,False]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='gsplat requires CUDA')
def test_semantic_only_hidden_member_matches_actual_subset_and_gradient():
    pytest.importorskip('gsplat')
    xyz=torch.tensor([[0.,0.,2.],[0.,0.,3.],[0.,0.,4.]],device='cuda')
    field=SimpleNamespace(xyz=xyz,quaternions=xyz.new_tensor([[1.,0.,0.,0.]]).repeat(3,1),
        scales=torch.full_like(xyz,.1),opacities=xyz.new_full((3,1),.9999),
        observed=torch.tensor([True,False,False],device='cuda'),world_units_per_metre=1.)
    available=torch.tensor([True,True,False],device='cuda')
    camera=SimpleNamespace(intrinsic=torch.tensor([[30.,0.,16.5],[0.,30.,16.5],[0.,0.,1.]]),
        world_to_camera=torch.eye(4),height=32,width=32)
    pixels=torch.tensor([16*32+16]);atoms=torch.tensor([0])
    old=CoverageView(compile_coverage_view(field,camera,32,32,pixels,atoms),3,1).cuda()
    new=CoverageView(compile_coverage_view(field,camera,32,32,pixels,atoms,member_domain=available),3,1).cuda()
    assert old.index.tolist()==[0] and new.index.tolist()==[0,1]
    member=xyz.new_tensor([0.,1.,0.])
    assert float(old(member))==0 and float(new(member))>.99
    for values in [[0.,1.,0.],[1.,1.,1.],[0.,0.,1.],[0.,0.,0.]]:
        member=xyz.new_tensor(values)
        reference=selected_alpha(field,(member>.5)&available,camera,32,32)[pixels]
        torch.testing.assert_close(new(member),reference,atol=1e-4,rtol=0)
    member=xyz.new_tensor([0.,.6,.9],requires_grad=True)
    prediction=new(member)
    reference=relaxed_subset_alpha(field,member,camera,32,32,member_domain=available)[pixels]
    torch.testing.assert_close(prediction,reference,atol=1e-4,rtol=0)
    actual_gradient=torch.autograd.grad(prediction.sum(),member)[0]
    expected_gradient=torch.autograd.grad(reference.sum(),member)[0]
    torch.testing.assert_close(actual_gradient,expected_gradient,atol=2e-4,rtol=0)
    assert actual_gradient[1]>0 and actual_gradient[2]==0
    assert field.observed.tolist()==[True,False,False]
