import json
from pathlib import Path

import numpy as np
from PIL import Image

from radio_gs.benchmarks.scannet_pfir.preparation.build_bbox_crops import (
    build_bbox_crop,
)
from radio_gs.benchmarks.scannet_pfir.protocol import (
    FramePaths,
    audit_manifest,
    canonical_json_sha256,
    exclusion_frame_ids,
    freeze_manifest,
    instance_surface_coverage,
    load_matrix,
    mask_sha256,
    resolve_frame_observations,
    sha256_file,
)
from radio_gs.benchmarks.scannet_pfir.split.select_scene_subset import (
    select_scene_subset,
)


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix)


def _frame(tmp_path: Path, frame_id: str, translation: float) -> FramePaths:
    paths = {}
    for name, suffix in (
        ("rgb", ".jpg"),
        ("depth", ".png"),
        ("instance", ".png"),
        ("label", ".png"),
    ):
        path = tmp_path / name / f"{frame_id}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)
        paths[name] = path
    pose = np.eye(4)
    pose[0, 3] = translation
    pose_path = tmp_path / "pose" / f"{frame_id}.txt"
    _write_matrix(pose_path, pose)
    return FramePaths(frame_id=frame_id, pose=pose_path, **paths)


def test_no_frame_leakage_uses_temporal_and_pose_union(tmp_path: Path) -> None:
    frames = [_frame(tmp_path, f"{index:06d}", index * 0.20) for index in range(13)]
    # Frame 12 is pose-near frame 6 even though it is temporally distant.
    near_pose = load_matrix(frames[6].pose)
    near_pose[0, 3] += 0.01
    _write_matrix(frames[12].pose, near_pose)
    excluded = exclusion_frame_ids(
        frames,
        [frames[6].frame_id],
        temporal_radius=2,
        translation_m=0.10,
        rotation_deg=8.0,
    )
    assert set(excluded) == {
        frames[index].frame_id for index in (4, 5, 6, 7, 8, 12)
    }


def test_2d_3d_instance_resolution_uses_depth_pose_mesh(tmp_path: Path) -> None:
    frame = _frame(tmp_path, "000000", 0.0)
    Image.fromarray(np.full((2, 2), 1000, dtype=np.uint16)).save(frame.depth)
    Image.fromarray(np.full((2, 2), 3001, dtype=np.uint16)).save(frame.instance)
    Image.fromarray(np.full((2, 2), 3, dtype=np.uint16)).save(frame.label)
    intrinsic = np.eye(4)
    mesh_xyz = np.array(
        [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float32
    )
    observations = resolve_frame_observations(
        frame,
        mesh_xyz,
        np.full(4, 7, dtype=np.int32),
        intrinsic,
        intrinsic,
        depth_stride=1,
        maximum_mesh_distance_m=0.01,
    )
    assert observations[3001].instance_id_3d == 7
    assert observations[3001].nyu40_class_id == 3
    assert observations[3001].resolution_purity == 1.0
    assert observations[3001].valid_depth_votes == 4


def test_query_crop_reproducibility_and_fixed_padding() -> None:
    image = Image.fromarray(np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3))
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 3:8] = True
    first, box_a = build_bbox_crop(image, mask, padding=0.10)
    second, box_b = build_bbox_crop(image, mask, padding=0.10)
    assert box_a == box_b == (2, 1, 9, 7)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert mask_sha256(mask) == mask_sha256(mask.copy())


def test_instance_coverage_is_surface_distance_not_sample_count() -> None:
    target = np.stack(
        [np.linspace(0, 1, 11), np.zeros(11), np.zeros(11)], axis=1
    )
    sparse_observation = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0]])
    assert instance_surface_coverage(
        target, [sparse_observation], distance_m=0.26
    ) == 1.0
    assert instance_surface_coverage(
        target, [sparse_observation[:1]], distance_m=0.26
    ) < 0.3


def test_manifest_hashes_and_method_manifest_hide_gt(tmp_path: Path) -> None:
    scene = tmp_path / "frames" / "scene0000_00"
    for directory in ("color", "instance"):
        (scene / directory).mkdir(parents=True)
    rgb = scene / "color" / "000000.jpg"
    Image.fromarray(np.full((10, 10, 3), 127, dtype=np.uint8)).save(rgb)
    instances = np.zeros((10, 10), dtype=np.uint16)
    instances[2:8, 2:8] = 3001
    Image.fromarray(instances).save(scene / "instance" / "000000.png")
    field_ids = ["000600", "000700"]
    record = {
        "benchmark_version": "scannet-pfir-small-v1",
        "query_id": "q0",
        "scene_id": "scene0000_00",
        "space_id": "scene0000",
        "query_frame_id": "000000",
        "query_rgb_path": str(rgb),
        "query_rgb_sha256": sha256_file(rgb),
        "instance_id_3d": 7,
        "encoded_instance_id_2d": 3001,
        "nyu40_class_id": 3,
        "instance_label": "cabinet",
        "bbox_xyxy": [1, 1, 9, 9],
        "bbox_padding": 0.10,
        "mask_sha256": mask_sha256(instances == 3001),
        "query_type": "bbox",
        "difficulty": "easy_medium",
        "field_frame_manifest_sha256": canonical_json_sha256(field_ids),
        "field_frame_ids": field_ids,
        "query_exclusion_frames": ["000000", "000100"],
        "nearest_field_pose_translation": 0.2,
        "nearest_field_pose_rotation": 10.0,
        "instance_field_visibility_count": 5,
        "instance_surface_coverage": 0.8,
        "instance_mesh_vertex_count": 700,
        "resolution_purity": 0.99,
        "same_category_distractor_count": 1,
        "candidate_instance_ids_3d": [7, 9],
        "candidate_instance_class_ids": {"7": 3, "9": 3},
        "method_visible_query_fields": ["crop_rgb", "scene_id"],
        "query_pose_used_by_method": False,
        "query_depth_used_by_method": False,
        "query_mask_used_by_method": False,
    }
    release = freeze_manifest([record], tmp_path / "benchmark", split_role="dev")
    assert release["audit"]["valid"]
    method = json.loads(
        (tmp_path / "benchmark" / "manifest.method.json").read_text()
    )
    assert set(method["queries"][0]["available_method_inputs"]) == {
        "scene_id",
        "crop_rgb",
    }
    serialized = json.dumps(method)
    assert "instance_id_3d" not in serialized
    assert "nyu40_class_id" not in serialized
    for name, expected in release["manifest_sha256"].items():
        assert sha256_file(tmp_path / "benchmark" / name) == expected


def test_public_manifest_audit_rejects_leaked_query_frame() -> None:
    field_ids = ["000000", "000100"]
    payload = {
        "benchmark_version": "scannet-pfir-small-v1",
        "visibility": "public",
        "queries": [
            {
                "query_id": "q",
                "scene_id": "scene0000_00",
                "query_frame_id": "000000",
                "field_frame_ids": field_ids,
                "query_exclusion_frames": ["000000"],
                "field_frame_manifest_sha256": canonical_json_sha256(field_ids),
                "query_type": "bbox",
                "bbox_padding": 0.10,
            }
        ],
    }
    report = audit_manifest(payload)
    assert not report["valid"]
    assert any("leakage" in error for error in report["errors"])


def test_scene_subset_keeps_physical_spaces_disjoint() -> None:
    selected = select_scene_subset(
        ["scene0001_00", "scene0001_01", "scene0002_00", "scene0003_00"],
        count=3,
        seed=7,
    )
    assert len(selected) == 3
    assert len({scene.split("_")[0] for scene in selected}) == 3
