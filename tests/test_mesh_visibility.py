import pytest
import torch

from radio_gs.geometry.mesh_visibility import rasterize_registered_mesh_depth


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA rasterizer required')
def test_registered_mesh_depth_orientation_and_occlusion():
    pytest.importorskip('nvdiffrast.torch')
    device = 'cuda:0'
    k = torch.tensor([[16.,0,8.],[0,16.,8.],[0,0,1]], device=device)
    # Top-left triangle at depth 2, same angular triangle behind it at depth 4.
    uv = torch.tensor([[1.,1.],[7.,1.],[1.,7.]], device=device)
    xy = (uv-8)/16
    front = torch.cat((xy*2, torch.full((3,1),2.,device=device)), 1)
    back = front*2
    vertices = torch.cat((back, front))
    faces = torch.tensor([[0,1,2],[3,4,5]], device=device)
    depth = rasterize_registered_mesh_depth(vertices, faces, k, 16, 16)
    assert depth[2,2].item() == pytest.approx(2., abs=1e-4)
    assert torch.isinf(depth[13,2])  # Catches a vertical flip mistake.
    assert torch.isinf(depth[2,13])
    rear = rasterize_registered_mesh_depth(back, faces[:1], k, 16, 16)
    assert rear[2,2].item() == pytest.approx(4., abs=1e-4)
