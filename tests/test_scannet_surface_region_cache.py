import torch

from radio_gs.scripts.build_scannet_surface_region_cache import (
    _excluded_spaces,
    _lift_observation,
    _physical_space,
    _project_region_box,
    _scene_names,
    _teacher_medoid,
    _voxel_fuse,
)


def test_voxel_fusion_is_deterministic() -> None:
    xyz = torch.tensor([[0.00, 0, 0], [0.01, 0, 0], [0.05, 0, 0], [0.10, 0, 0]])
    features = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    fused_xyz, fused_features, footprint, count = _voxel_fuse(
        xyz, features, torch.full((4,), 0.02), 0.04
    )
    assert len(fused_xyz) == 3
    assert count.max() == 2
    assert torch.isfinite(fused_features).all() and torch.isfinite(footprint).all()


def test_teacher_medoid_selects_consensus_view() -> None:
    tokens = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    assert _teacher_medoid(tokens) in {0, 1}


def test_lifting_samples_pixel_centres_with_align_corners_false() -> None:
    depth = torch.ones(2, 3)
    intrinsic = torch.eye(4)
    spatial = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    _xyz, sampled, _footprint = _lift_observation(
        depth, intrinsic, intrinsic, torch.eye(4), spatial,
        stride=1, color_size=(3, 2),
    )
    torch.testing.assert_close(sampled[:, 0], spatial.flatten(), atol=1e-6, rtol=0)


def test_singleton_surface_region_gets_a_valid_teacher_crop() -> None:
    intrinsic = torch.eye(4)
    intrinsic[0, 0] = intrinsic[1, 1] = 100.0
    intrinsic[0, 2] = intrinsic[1, 2] = 50.0
    box = _project_region_box(
        torch.tensor([[0.0, 0.0, 1.0]]), torch.ones(100, 100),
        intrinsic, intrinsic, torch.eye(4), (100, 100), min_visible=1,
        context_pad=0.0,
    )
    assert box is not None
    top, left, bottom, right = box
    assert bottom - top == 24 and right - left == 24


def test_exclusion_is_applied_to_every_rescan_of_a_physical_space(tmp_path) -> None:
    root = tmp_path / "frames"
    for scene in ("scene0012_00", "scene0012_02", "scene0013_00"):
        (root / scene).mkdir(parents=True)
    split = tmp_path / "split.txt"
    split.write_text("scene0012_00\nscene0012_02\nscene0013_00\n")
    exclusion = tmp_path / "pfir_dev.txt"
    exclusion.write_text("# comment\nscene0012_02\n")

    spaces, records = _excluded_spaces(str(exclusion), "")
    assert spaces == {"scene0012"}
    assert records[0]["sha256"]
    assert _physical_space("scene0012_02") == "scene0012"
    assert _scene_names(
        split,
        root,
        excluded_physical_spaces=spaces,
    ) == ["scene0013_00"]
