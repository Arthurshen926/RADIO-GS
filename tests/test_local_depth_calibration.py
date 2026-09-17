import pytest
import torch
from radio_gs.geometry.local_depth_calibration import local_depth_scales


def test_local_regions_do_not_mix_across_depth_boundary():
    xy=torch.zeros(8,2);depth=torch.tensor([1.]*4+[3.]*4)
    ref=torch.tensor([1.1]*4+[2.7]*4)
    scale,known=local_depth_scales(xy,depth,ref,torch.zeros(2,2),torch.tensor([1.,3.]),fallback_scale=1.)
    assert known.all()
    assert torch.allclose(scale,torch.tensor([1.1,.9],dtype=torch.float64))


def test_unsupported_locations_fall_back_and_outlier_does_not_dominate():
    xy=torch.zeros(5,2);d=torch.ones(5);r=torch.tensor([1.05]*4+[8.])
    scale,known=local_depth_scales(xy,d,r,torch.tensor([[0.,0.],[1.,1.]]),torch.ones(2),fallback_scale=1.)
    assert known.tolist()==[True,False]
    assert scale.tolist()==pytest.approx([1.05,1.])


def test_bound_and_nonfinite_query():
    scale,known=local_depth_scales(torch.zeros(4,2),torch.ones(4),torch.ones(4)*4,
        torch.tensor([[0.,0.],[float('nan'),0.]]),torch.ones(2),fallback_scale=1.)
    assert scale.tolist()==pytest.approx([1.2,1.])
    assert known.tolist()==[True,False]


def test_normal_warp_matches_analytic_surface_tangents():
    from radio_gs.geometry.local_depth_calibration import warp_normals_for_image_scale
    y,x=torch.meshgrid(torch.arange(5,dtype=torch.float64),torch.arange(5,dtype=torch.float64),indexing='ij')
    k=torch.tensor([[10.,0,2],[0,10.,2],[0,0,1.]],dtype=torch.float64)
    p=torch.stack([(x-2)/10,(y-2)/10,torch.ones_like(x)],-1)
    n=torch.zeros_like(p);n[...,2]=1
    s=torch.exp(.02*x+.03*y)
    result=warp_normals_for_image_scale(p,n,s,k)
    tangent_u=torch.tensor([.1,0.,0.])+p*.02
    tangent_v=torch.tensor([0.,.1,0.])+p*.03
    expected=torch.nn.functional.normalize(torch.linalg.cross(tangent_u,tangent_v),dim=-1)
    assert torch.allclose(result,expected,atol=1e-7)
    assert torch.allclose(warp_normals_for_image_scale(p,n,torch.ones_like(s)*1.2,k),n)
