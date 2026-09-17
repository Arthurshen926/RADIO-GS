import pytest
import torch
from radio_gs.geometry.affine_depth_normals import affine_depth_normals


def test_warped_normal_matches_finite_difference_surface_tangents():
    # A tilted plane. Check the actual warped neighboring 3D points, not a
    # second implementation of the normal formula.
    point = torch.tensor([[.4, -.2, 2.]], dtype=torch.float64)
    normal = torch.tensor([[.6, .3, .8]], dtype=torch.float64)
    normal /= normal.norm(dim=-1, keepdim=True)
    first = torch.cross(normal[0], torch.tensor([0., 1., 0.], dtype=torch.float64), dim=0)
    second = torch.cross(normal[0], first, dim=0)
    def warp(x):
        return x * (1.3 + .7 / x[..., 2:])
    result = affine_depth_normals(point, normal, 1.3, .7)[0]
    for tangent in [first, second]:
        derivative = (warp(point+1e-5*tangent)-warp(point-1e-5*tangent))[0]/2e-5
        assert abs(float(result @ derivative)) < 1e-9
    assert not torch.allclose(result, normal[0], atol=1e-3)


def test_scale_only_and_world_units_invariance():
    point = torch.tensor([[.4, -.2, 2.]])
    normal = torch.nn.functional.normalize(torch.tensor([[.6, .3, .8]]), dim=-1)
    torch.testing.assert_close(affine_depth_normals(point, normal, 1.3, 0.), normal)
    torch.testing.assert_close(affine_depth_normals(point, normal, 1.3, .7),
                               affine_depth_normals(point*10, normal, 1.3, 7.))
    with pytest.raises(ValueError):
        affine_depth_normals(point, normal, 1., -3.)


def test_geometry_loader_transports_normals_without_changing_depth(tmp_path):
    from types import SimpleNamespace
    from radio_gs.scripts.repair_moge_geometry import load_views
    from radio_gs.utils.immutable_artifacts import sha256_file
    h, w = 4, 6
    normal = torch.nn.functional.normalize(torch.tensor([.4, .2, 1.]), dim=0)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    rays = torch.stack([(x+.5)/w, (y+.5)/h, torch.ones_like(x)], -1).float()
    depth = 2. / (rays @ normal)
    path = tmp_path / 'prediction.pt'
    torch.save(dict(model_checkpoint_sha256='fixture', point_map=rays*depth[..., None],
                    normals=normal.expand(h, w, 3).clone(), validity=torch.ones(h, w, dtype=torch.bool),
                    intrinsics=torch.eye(3)), path)
    camera = SimpleNamespace(intrinsic=torch.diag(torch.tensor([float(w), float(h), 1.])),
                             width=w, height=h, camera_to_world=torch.eye(4))
    prior = SimpleNamespace(records={1: {'prediction': str(path), 'prediction_sha256': sha256_file(path)}},
                            manifest={'model_checkpoint_sha256': 'fixture'}, scale=1.,
                            calibration={1: {'accepted': True, 'scale': 1.3, 'offset': .7}}, cameras={1: camera})
    corrected = load_views(prior, w, 'cpu')[0]
    legacy = load_views(prior, w, 'cpu', legacy_affine_normals=True)[0]
    assert torch.equal(corrected['depth'], legacy['depth'])
    torch.testing.assert_close(corrected['normals'], affine_depth_normals(rays*depth[..., None], normal.expand(h,w,3), 1.3, .7))
    assert not torch.allclose(corrected['normals'], legacy['normals'])
