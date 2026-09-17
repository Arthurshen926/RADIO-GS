import pytest
import torch
from radio_gs.geometry.source_scale_calibration import fit_reprojection_scale


def fixture():
    n = 30
    ray = torch.stack([torch.linspace(-.3,.3,n), torch.zeros(n), torch.ones(n)], 1)
    pose = torch.eye(4).repeat(n,1,1); pose[:,0,3] = -.5
    k = torch.tensor([[100.,0,64],[0,100.,48],[0,0,1.]]).repeat(n,1,1)
    xyz = ray * 2 + pose[:,:3,3]
    uv = torch.einsum('nij,nj->ni',k,xyz); uv = uv[:,:2]/uv[:,2:]
    return ray, torch.zeros(3), pose, k, uv


def test_recovers_scale_from_parallax():
    result = fit_reprojection_scale(*fixture(), initial_scale=1.5)
    assert abs(result['scale']-2) < .01
    assert result['training_loss'] < result['initial_training_loss']
    assert result['requires_independent_validation']


def test_no_parallax_does_not_invent_scale():
    ray, origin, pose, k, _ = fixture(); pose[:,0,3] = 0
    uv = torch.einsum('nij,nj->ni',k,ray); uv=uv[:,:2]/uv[:,2:]
    result=fit_reprojection_scale(ray,origin,pose,k,uv,initial_scale=1.5)
    assert abs(result['scale']-1.5)<.01
    assert not result['fitted']
    assert result['reason']=='unobservable_scale_without_parallax'


def test_insufficient_and_invalid():
    values=fixture()
    assert not fit_reprojection_scale(*values,initial_scale=1.5,minimum_samples=31)['fitted']
    with pytest.raises(ValueError):fit_reprojection_scale(*values,initial_scale=-1)
