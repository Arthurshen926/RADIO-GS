"""The regression is visibility changing after Gaussian selection, not feature math."""
from types import SimpleNamespace

import pytest
import torch

from radio_gs.query_field import CapabilityField
from radio_gs.query_field.geometry import projection, relaxed_subset_alpha, selected_alpha


@pytest.mark.skipif(not torch.cuda.is_available(),reason="actual gsplat visibility requires CUDA")
def test_subset_surrogate_exposes_hidden_members_and_preserves_carrier():
    pytest.importorskip("gsplat")
    xyz=torch.tensor([[0.,0.,2.],[0.,0.,2.1],[0.,0.,3.]],device='cuda')
    geometry=dict(xyz=xyz,normals=torch.tensor([[0.,0.,1.]],device='cuda').repeat(3,1),
        quaternions=torch.tensor([[1.,0.,0.,0.]],device='cuda').repeat(3,1),
        scales=torch.full((3,3),.1,device='cuda'),opacities=torch.full((3,1),.9999,device='cuda'))
    field=CapabilityField(geometry,torch.ones(3,4,device='cuda'),torch.ones(3,3,device='cuda'),
                          torch.randn(3,2,device='cuda'),torch.ones(3,dtype=torch.bool,device='cuda')).cuda()
    camera=SimpleNamespace(intrinsic=torch.tensor([[30.,0.,16.5],[0.,30.,16.5],[0.,0.,1.]]),
                           world_to_camera=torch.eye(4),height=32,width=32)
    numerical=SimpleNamespace(xyz=field.xyz,get_xyz=lambda:field.xyz,get_rotation=lambda:field.quaternions,
                               get_scaling=lambda:field.scales,get_opacity=lambda:field.opacities)
    raw,_,_,_=projection(numerical,camera,32,32,1.)
    centre=16*32+16
    raw=raw.coalesce()
    hidden=(raw.indices()[0]==centre)&(raw.indices()[1]==2)
    assert not hidden.any() or float(raw.values()[hidden].sum())<1e-4
    membership=torch.tensor([0.,0.,1.],device='cuda',requires_grad=True)
    before={k:v.clone() for k,v in field.named_buffers()}
    soft=relaxed_subset_alpha(field,membership,camera,32,32)
    hard=selected_alpha(field,membership.detach()>.5,camera,32,32)
    torch.testing.assert_close(soft,hard,atol=2e-4,rtol=2e-4)
    assert soft[centre]>.99
    soft.sum().backward()
    assert membership.grad[2]>0
    assert torch.isfinite(membership.grad).all()
    for key,value in field.named_buffers():
        assert torch.equal(before[key],value),key
