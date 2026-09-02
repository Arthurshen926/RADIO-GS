import pytest
import torch

from radio_gs.v4.carrier import Camera, GaussianCarrier, MeshCarrier, SurfaceVoxelCarrier
from radio_gs.v4.registration.surface_projection import projection_entropy


def camera(height=4, width=4):
    intrinsic = torch.tensor([[2.0, 0, width / 2], [0, 2.0, height / 2], [0, 0, 1.0]])
    return Camera("source", intrinsic, torch.eye(4), height, width)


def test_gaussian_carrier_preserves_unknown_as_distinct_evidence(tmp_path):
    payload = {
        "gaussian_ids": torch.tensor([0, 1, 0]),
        "pixel_ids": torch.tensor([0, 1, 2]),
        "base_weights": torch.tensor([1.0, 1.0, 0.5]),
        "num_gaussians": 2,
        "num_pixels": 4,
    }
    path = tmp_path / "projection.pt"
    torch.save(payload, path)
    carrier = GaussianCarrier(2, {"source": path})
    signal = torch.tensor([[1.0, 0.0], [0.0, float("nan")]])
    state = torch.tensor([[1, 0], [-1, -1]])
    evidence = carrier.lift(signal, camera(2, 2), state=state)
    assert torch.allclose(evidence.positive_weight, torch.tensor([1.0, 0.0]))
    assert torch.allclose(evidence.negative_weight, torch.tensor([0.0, 1.0]))
    assert torch.allclose(evidence.unknown_weight, torch.tensor([0.5, 0.0]))
    assert torch.allclose(carrier.render_posterior(torch.tensor([0.4, 0.8]), camera(2, 2)), torch.tensor([[0.4, 0.8], [0.2, 0.0]]))


def test_mesh_raycast_projects_visible_surface():
    pytest.importorskip("open3d")
    vertices = torch.tensor([[-3.0, -3.0, 2.0], [3.0, -3.0, 2.0], [3.0, 3.0, 2.0], [-3.0, 3.0, 2.0]])
    triangles = torch.tensor([[0, 1, 2], [0, 2, 3]])
    mesh = MeshCarrier(vertices, triangles)
    projection = mesh.project(camera())
    assert int((projection.pixel_weight_sum() > 0).sum()) == 16
    assert torch.allclose(projection.pixel_weight_sum(), torch.ones(4, 4), atol=1e-5)
    assert mesh.neighbors().edge_index.shape[1] == 10
    entropy = projection_entropy(projection)
    assert entropy["effective_contributors"] <= 3


def test_sparse_voxel_projects_visible_surface():
    vertices = torch.tensor([
        [-3.0, -3.0, 2.0], [3.0, -3.0, 2.0],
        [3.0, 3.0, 2.0], [-3.0, 3.0, 2.0],
    ])
    voxel_points = torch.cat([vertices, torch.tensor([[0.0, 0.0, 2.0]])])
    voxels = SurfaceVoxelCarrier.from_points(
        voxel_points,
        0.5,
        maximum_splat_radius=2,
        surface_band_voxels=1.5,
        maximum_contributors_per_pixel=8,
    )
    voxel_projection = voxels.project(camera())
    assert voxel_projection.element_ids.numel() > 0
    counts = torch.bincount(voxel_projection.pixel_ids)
    assert int(counts.max()) <= 8


def test_sparse_voxel_rejects_invalid_geometry_and_does_not_reuse_stale_camera_key():
    kwargs = {
        "maximum_splat_radius": 1,
        "surface_band_voxels": 1.5,
        "maximum_contributors_per_pixel": 8,
    }
    try:
        SurfaceVoxelCarrier(torch.tensor([[float("nan"), 0.0, 1.0]]), 0.1, **kwargs)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite centres were accepted")
    try:
        SurfaceVoxelCarrier(
            torch.tensor([[0.0, 0.0, 1.0]]), 0.1,
            confidence=torch.tensor([-1.0]), **kwargs,
        )
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative confidence was accepted")

    carrier = SurfaceVoxelCarrier(torch.tensor([[0.0, 0.0, 2.0]]), 0.1, **kwargs)
    first = camera()
    shifted_pose = torch.eye(4)
    shifted_pose[0, 3] = 10.0
    second = Camera(first.key, first.intrinsic, shifted_pose, first.height, first.width)
    assert carrier.project(first).element_ids.numel() > 0
    assert carrier.project(second).element_ids.numel() == 0


@pytest.mark.parametrize(
    "override",
    [
        {"maximum_splat_radius": 1.5},
        {"maximum_splat_radius": True},
        {"maximum_contributors_per_pixel": 8.5},
        {"maximum_contributors_per_pixel": False},
    ],
)
def test_sparse_voxel_rejects_non_exact_integer_projection_configuration(override):
    kwargs = {
        "maximum_splat_radius": 1,
        "surface_band_voxels": 1.5,
        "maximum_contributors_per_pixel": 8,
        **override,
    }
    with pytest.raises(ValueError, match="integer"):
        SurfaceVoxelCarrier(torch.tensor([[0.0, 0.0, 2.0]]), 0.1, **kwargs)
