from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from radio_gs.benchmarks.scannet_pfpr.prepare_field_contract import (
    AGILE_DENSE_FIELD_CONTRACT_VERSION,
    FIELD_SOURCE_CONTRACT_FILENAME,
    FIELD_CONTRACT_VERSION,
    SCANNET_FULL_OBSERVATION_FIELD_CONTRACT_VERSION,
    SCANNET_FULL_OBSERVATION_PFPR_FIELD_CONTRACT_VERSION,
    materialize_agile_dense_field_contract,
    materialize_pfpr_field_contract,
    materialize_scannet_full_observation_field_contract,
    materialize_scannet_full_observation_pfpr_field_contract,
    select_depth_coverage_frame_indices,
    select_pose_diverse_frame_indices,
)
from radio_gs.benchmarks.scannet_pfpr.protocol import query_frame_exclusion_digest
from radio_gs.scripts.build_geometry_render_contract import (
    validate_field_source_contract,
)


def _write_frame_scene(root: Path, scene: str) -> None:
    scene_root = root / scene
    for modality, suffix in (("color", ".jpg"), ("depth", ".png"), ("pose", ".txt")):
        directory = scene_root / modality
        directory.mkdir(parents=True)
        for frame in (0, 20, 40):
            path = directory / f"{frame:06d}{suffix}"
            if modality == "pose":
                path.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")
            else:
                path.write_bytes(b"frame")
    for name in (
        "intrinsics_color.txt",
        "intrinsics_depth.txt",
        "extrinsics_color.txt",
        "extrinsics_depth.txt",
    ):
        (scene_root / name).write_text("1\n", encoding="utf-8")


def test_pfpr_field_contract_excludes_only_exact_query_source_frames(tmp_path: Path) -> None:
    dense = tmp_path / "dense"
    _write_frame_scene(dense, "scene0001_00")
    method = {
        "benchmark_version": "scannet-pfpr-small-v1",
        "queries": [{"scene_id": "scene0001_00"}],
    }
    evaluator = {
        "benchmark_version": "scannet-pfpr-small-v1",
        "queries": [
            {
                "scene_id": "scene0001_00",
                "source_frame_id": "000020",
                "anchor_world_xyz": [1.0, 2.0, 3.0],
                "source_depth_pixel_uv": [10, 20],
            },
            {
                "scene_id": "scene0001_00",
                "source_frame_id": "000020",
                "anchor_world_xyz": [4.0, 5.0, 6.0],
                "source_depth_pixel_uv": [30, 40],
            },
        ],
    }
    method_path = tmp_path / "method.json"
    evaluator_path = tmp_path / "evaluator.json"
    method_path.write_text(json.dumps(method), encoding="utf-8")
    evaluator_path.write_text(json.dumps(evaluator), encoding="utf-8")
    # Geometry training already rejects this pose; the field source must use
    # the exact same validity rule instead of letting MPR fail later.
    (dense / "scene0001_00" / "pose" / "000040.txt").write_text(
        "nan 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8"
    )

    output = tmp_path / "field"
    report = materialize_pfpr_field_contract(
        method_path, evaluator_path, dense, output, mode="symlink"
    )

    scene = report["scenes"][0]
    assert report["field_contract_version"] == FIELD_CONTRACT_VERSION
    assert report["uses_private_anchor"] is False
    assert report["uses_private_depth_pixel"] is False
    assert scene["field_frame_count"] == 1
    assert scene["excluded_query_source_frame_count"] == 1
    assert scene["invalid_or_nonfinite_pose_frame_count"] == 1
    assert not (output / "scene0001_00" / "color" / "000020.jpg").exists()
    assert (output / "scene0001_00" / "color" / "000000.jpg").is_symlink()
    assert not (output / "scene0001_00" / "color" / "000040.jpg").exists()
    saved = json.loads((output / "scene0001_00" / "pfpr_field_contract.json").read_text())
    assert "anchor_world_xyz" not in json.dumps(saved)
    assert "source_depth_pixel_uv" not in json.dumps(saved)


def test_agile_dense_field_contract_uses_all_valid_frames_without_labels(tmp_path: Path) -> None:
    dense = tmp_path / "dense"
    _write_frame_scene(dense, "scene0001_00")
    # The source may have evaluator-only annotations, but the materialized
    # scene must never inherit them into geometry/MPR input.
    (dense / "scene0001_00" / "instance").mkdir()
    (dense / "scene0001_00" / "label").mkdir()
    (dense / "scene0001_00" / "pose" / "000040.txt").write_text(
        "nan 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8"
    )

    output = tmp_path / "agile_field"
    report = materialize_agile_dense_field_contract(
        dense, output, mode="symlink", scenes=("scene0001_00",)
    )

    scene = report["scenes"][0]
    assert report["field_contract_version"] == AGILE_DENSE_FIELD_CONTRACT_VERSION
    assert scene["source_policy"] == "all_valid_dense_rgbd_observations"
    assert scene["field_frame_count"] == 2
    assert scene["excluded_query_source_frame_count"] == 0
    assert not (output / "scene0001_00" / "instance").exists()
    assert not (output / "scene0001_00" / "label").exists()
    saved = json.loads(
        (output / "scene0001_00" / FIELD_SOURCE_CONTRACT_FILENAME).read_text()
    )
    assert saved["uses_private_anchor"] is False


def test_pose_diverse_selection_is_deterministic_and_ignores_invalid_poses() -> None:
    poses = {
        0: np.eye(4),
        1: np.full((4, 4), np.nan),
        2: np.array(
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        ),
        3: np.array(
            [[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        ),
    }

    selection = select_pose_diverse_frame_indices(poses, max_frames=2)

    assert selection == [0, 3]


def test_depth_coverage_selection_prefers_new_surface_over_near_duplicate() -> None:
    class FakeFrame:
        def __init__(self) -> None:
            self.camera_to_world = np.eye(4)

        def decompress_depth(self, _compression: str) -> np.ndarray:
            return np.full(16, 1000, dtype=np.uint16)

    frames = [FakeFrame(), FakeFrame(), FakeFrame()]
    poses = {
        0: np.eye(4),
        1: np.array(
            [[1, 0, 0, 0.01], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        ),
        2: np.array(
            [[1, 0, 0, 2.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        ),
    }

    selected, order, covered = select_depth_coverage_frame_indices(
        frames,
        poses,
        intrinsic_depth=np.eye(4),
        depth_width=4,
        depth_height=4,
        depth_compression_type="zlib_ushort",
        max_frames=2,
        voxel_size_m=0.05,
        depth_stride=1,
    )

    assert selected == [0, 2]
    assert order == [0, 2]
    assert covered > 0


def test_full_observation_field_contract_decodes_only_rgbd_pose(tmp_path: Path) -> None:
    sens_root = tmp_path / "raw"
    scene = "scene0001_00"
    sens_path = sens_root / scene / f"{scene}.sens"
    sens_path.parent.mkdir(parents=True)
    sens_path.write_bytes(b"synthetic-sens")

    class FakeFrame:
        def __init__(self, pose: np.ndarray, value: int) -> None:
            self.camera_to_world = pose
            self._value = value

        def decompress_color(self, _compression: str) -> Image.Image:
            return Image.fromarray(
                np.full((6, 8, 3), self._value, dtype=np.uint8)
            )

        def decompress_depth(self, _compression: str) -> np.ndarray:
            return np.full(12, 1000, dtype=np.uint16)

    class FakeSensor:
        color_compression_type = "jpeg"
        depth_compression_type = "zlib_ushort"
        depth_width = 4
        depth_height = 3
        intrinsic_color = np.eye(4)
        intrinsic_depth = np.eye(4)
        extrinsic_color = np.eye(4)
        extrinsic_depth = np.eye(4)

        def __init__(self, path: Path) -> None:
            assert path == sens_path
            self.frames = [
                FakeFrame(np.eye(4), 10),
                FakeFrame(np.full((4, 4), np.nan), 20),
                FakeFrame(
                    np.array(
                            [
                            [1, 0, 0, 0.01],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float64,
                    ),
                    30,
                ),
                FakeFrame(
                    np.array(
                        [
                            [1, 0, 0, 5],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float64,
                    ),
                    40,
                ),
            ]

    output = tmp_path / "field"
    report = materialize_scannet_full_observation_field_contract(
        sens_root,
        output,
        scenes=(scene,),
        max_frames=2,
        sensor_factory=FakeSensor,
    )

    record = report["scenes"][0]
    assert report["field_contract_version"] == SCANNET_FULL_OBSERVATION_FIELD_CONTRACT_VERSION
    assert record["source_policy"] == "full_sens_greedy_depth_voxel_coverage_query_free_frames"
    assert record["full_sens_frame_count"] == 4
    assert record["all_valid_pose_frame_count"] == 3
    assert record["selected_frame_indices"] == [0, 3]
    assert record["frame_selection_policy"] == "depth_voxel_coverage"
    assert record["coverage_voxel_count"] > 0
    assert record["uses_instances_or_semantic_labels"] is False
    assert (output / scene / "color" / "000000.jpg").is_file()
    assert (output / scene / "color" / "000003.jpg").is_file()
    assert Image.open(output / scene / "color" / "000000.jpg").size == (4, 3)
    assert not (output / scene / "color" / "000001.jpg").exists()
    assert not (output / scene / "instance").exists()
    assert not (output / scene / "label").exists()
    saved = json.loads((output / scene / FIELD_SOURCE_CONTRACT_FILENAME).read_text())
    assert saved["field_contract_version"] == SCANNET_FULL_OBSERVATION_FIELD_CONTRACT_VERSION
    assert saved["source_color_size"] == [8, 6]
    assert "anchor_world_xyz" not in json.dumps(saved)
    assert "source_depth_pixel_uv" not in json.dumps(saved)


def test_full_observation_pfpr_contract_excludes_private_query_source_frames(
    tmp_path: Path,
) -> None:
    """A full `.sens` field may use every non-query frame, but no query frame."""

    sens_root = tmp_path / "raw"
    scene = "scene0001_00"
    sens_path = sens_root / scene / f"{scene}.sens"
    sens_path.parent.mkdir(parents=True)
    sens_path.write_bytes(b"synthetic-sens")
    method_path = tmp_path / "method.json"
    evaluator_path = tmp_path / "evaluator.json"
    method_path.write_text(
        json.dumps(
            {
                "benchmark_version": "scannet-pfpr-small-v1",
                "queries": [{"scene_id": scene}],
            }
        ),
        encoding="utf-8",
    )
    evaluator_path.write_text(
        json.dumps(
            {
                "benchmark_version": "scannet-pfpr-small-v1",
                "queries": [
                    {
                        "scene_id": scene,
                        "source_frame_id": "000002",
                        "anchor_world_xyz": [1.0, 2.0, 3.0],
                        "source_depth_pixel_uv": [10, 20],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeFrame:
        def __init__(self, pose: np.ndarray, value: int) -> None:
            self.camera_to_world = pose
            self._value = value

        def decompress_color(self, _compression: str) -> Image.Image:
            return Image.fromarray(np.full((3, 4, 3), self._value, dtype=np.uint8))

        def decompress_depth(self, _compression: str) -> np.ndarray:
            return np.full(12, 1000, dtype=np.uint16)

    class FakeSensor:
        color_compression_type = "jpeg"
        depth_compression_type = "zlib_ushort"
        depth_width = 4
        depth_height = 3
        intrinsic_color = np.eye(4)
        intrinsic_depth = np.eye(4)
        extrinsic_color = np.eye(4)
        extrinsic_depth = np.eye(4)

        def __init__(self, path: Path) -> None:
            assert path == sens_path
            self.frames = [
                FakeFrame(np.eye(4), 10),
                FakeFrame(np.full((4, 4), np.nan), 20),
                FakeFrame(np.eye(4), 30),
                FakeFrame(
                    np.array(
                        [[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        dtype=np.float64,
                    ),
                    40,
                ),
            ]

    output = tmp_path / "field"
    report = materialize_scannet_full_observation_pfpr_field_contract(
        method_path,
        evaluator_path,
        sens_root,
        output,
        scenes=(scene,),
        max_frames=0,
        sensor_factory=FakeSensor,
    )

    record = report["scenes"][0]
    assert report["field_contract_version"] == (
        SCANNET_FULL_OBSERVATION_PFPR_FIELD_CONTRACT_VERSION
    )
    assert record["full_sens_frame_count"] == 4
    assert record["field_frame_count"] == 2
    assert record["excluded_query_source_frame_count"] == 1
    assert not (output / scene / "color" / "000002.jpg").exists()
    assert (output / scene / "color" / "000000.jpg").is_file()
    assert (output / scene / "color" / "000003.jpg").is_file()
    saved = json.loads((output / scene / FIELD_SOURCE_CONTRACT_FILENAME).read_text())
    encoded = json.dumps(saved)
    assert "anchor_world_xyz" not in encoded
    assert "source_depth_pixel_uv" not in encoded
    assert "000002" not in encoded
    assert "anchor_world_xyz" not in json.dumps(report)
    assert "source_depth_pixel_uv" not in json.dumps(report)
    assert "000002" not in json.dumps(report)


def test_full_observation_pfpr_v2_rejects_a_mismatched_public_frame_commitment(
    tmp_path: Path,
) -> None:
    """A v2 field may never be decoded from a silently different split."""

    scene = "scene0001_00"
    method_path = tmp_path / "method.json"
    evaluator_path = tmp_path / "evaluator.json"
    method_path.write_text(
        json.dumps(
            {
                "benchmark_version": "scannet-pfpr-small-v2",
                "queries": [{"scene_id": scene}],
            }
        ),
        encoding="utf-8",
    )
    evaluator_path.write_text(
        json.dumps(
            {
                "benchmark_version": "scannet-pfpr-small-v2",
                "scene_domains": [
                    {
                        "scene_id": scene,
                        "excluded_query_source_frame_ids_sha256": query_frame_exclusion_digest(
                            [7]
                        ),
                    }
                ],
                "queries": [
                    {
                        "scene_id": scene,
                        "source_frame_id": "000002",
                        "anchor_world_xyz": [1.0, 2.0, 3.0],
                        "source_depth_pixel_uv": [10, 20],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="public commitment"):
        materialize_scannet_full_observation_pfpr_field_contract(
            method_path,
            evaluator_path,
            tmp_path / "missing_raw",
            tmp_path / "field",
            scenes=(scene,),
        )


def test_render_contract_requires_matching_full_observation_provenance() -> None:
    with pytest.raises(ValueError, match="matching full source contract"):
        validate_field_source_contract(
            "scannet_full_observation_v1",
            {"field_contract_version": AGILE_DENSE_FIELD_CONTRACT_VERSION},
            source_root="/tmp/field/scene0001_00",
        )

    validate_field_source_contract(
        "scannet_full_observation_v1",
        {
            "field_contract_version": SCANNET_FULL_OBSERVATION_FIELD_CONTRACT_VERSION,
            "source_sens_sha256": "a" * 64,
            "full_sens_frame_count": 100,
            "field_frame_manifest_sha256": "b" * 64,
        },
        source_root="/tmp/field/scene0001_00",
    )


def test_render_contract_accepts_full_sens_pfpr_source_only_with_heldout_provenance() -> None:
    source = {
        "field_contract_version": (
            SCANNET_FULL_OBSERVATION_PFPR_FIELD_CONTRACT_VERSION
        ),
        "source_sens_sha256": "a" * 64,
        "full_sens_frame_count": 100,
        "field_frame_manifest_sha256": "b" * 64,
        "excluded_query_source_frame_count": 3,
        "excluded_query_source_frame_ids_sha256": "c" * 64,
    }
    validate_field_source_contract(
        "scannet_full_observation_pfpr_queryheldout_v1",
        source,
        source_root="/tmp/field/scene0001_00",
    )
    with pytest.raises(ValueError, match="matching full source contract"):
        validate_field_source_contract(
            "scannet_full_observation_pfpr_queryheldout_v1",
            {**source, "field_contract_version": SCANNET_FULL_OBSERVATION_FIELD_CONTRACT_VERSION},
            source_root="/tmp/field/scene0001_00",
        )
