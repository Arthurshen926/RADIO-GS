from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree
import torch

from radio_gs.query_field.surface_point import SurfacePointAdapter
from radio_gs.querying.registered_observation_query import local_surface,geometric_anchor


def test_gaussian_surface_adapter_preserves_v10_input_function_without_clamping():
    x,y=np.meshgrid(np.linspace(-1,1,15),np.linspace(-1,1,15))
    xyz=np.concatenate([np.c_[x.ravel(),y.ravel(),np.full(x.size,z)] for z in [1.,2.]]).astype(np.float32)
    field=SimpleNamespace(xyz=torch.from_numpy(xyz));adapter=SurfacePointAdapter(field)
    tree=cKDTree(xyz);surface=local_surface(xyz,tree);k=np.array([[100.,0,100],[0,100,100],[0,0,1.]])
    for pixel in [[100,100],[121,128],[3,4],[280,270]]:
        query=dict(modality='point_2d',positive_pixel_uv=pixel,camera_to_world=np.eye(4).tolist(),camera_intrinsics=k.tolist())
        anchor,_=geometric_anchor(xyz,query,tree,surface)
        packet,detail=adapter.packet(pixel,k,np.eye(4))
        assert packet.positive.indices.item()==anchor
        assert not packet.exact_positive and not detail['hard_positive_constraint']
        packet.validate(len(xyz),1,1)


def test_surface_registration_rejects_nonfinite_camera_and_empty_front_domain():
    xyz=torch.tensor([[0.,0.,-1.],[1.,0.,-1.],[0.,1.,-1.],[1.,1.,-1.]])
    adapter=SurfacePointAdapter(SimpleNamespace(xyz=xyz))
    with pytest.raises(ValueError,match='no carrier in front'):
        adapter.packet([0,0],np.eye(3),np.eye(4))
    k=np.eye(3);k[0,0]=np.nan
    with pytest.raises(ValueError,match='invalid registered'):
        adapter.packet([0,0],k,np.eye(4))
