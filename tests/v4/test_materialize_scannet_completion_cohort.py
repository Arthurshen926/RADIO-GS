import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from radio_gs.v4.training.materialize_scannet_completion_cohort import (
    FROZEN_RADIO_GRID_HW,
    GL_TO_CV,
    _radio_grid_hw,
    materialize_scene,
    run,
)


def _fixture(tmp_path: Path):
    scene_id = "scene0001_00"
    field_root = tmp_path / "field"
    field = field_root / scene_id
    (field / "color").mkdir(parents=True)
    (field / "pose").mkdir()
    (field / "intrinsic").mkdir()
    frame_ids = list(range(10, 18))
    for frame_id in frame_ids:
        Image.new("RGB", (4, 3), color=(frame_id, 0, 0)).save(
            field / "color" / f"{frame_id:06d}.jpg"
        )
        pose = np.eye(4)
        pose[0, 3] = frame_id
        np.savetxt(field / "pose" / f"{frame_id:06d}.txt", pose)
    intrinsic = np.eye(4)
    intrinsic[0, 0] = 100
    intrinsic[1, 1] = 101
    intrinsic[0, 2] = 648
    intrinsic[1, 2] = 484
    np.savetxt(field / "intrinsic" / "intrinsic_color.txt", intrinsic)
    (field / "field_source_contract.json").write_text(json.dumps({
        "scene_id": scene_id,
        "frame_selection_policy": "depth_voxel_coverage",
        "selection_order_frame_indices": frame_ids,
        "source_color_size": [1296, 968],
        "uses_instances_or_semantic_labels": False,
        "contains_instance_or_label_directories": False,
    }))
    annotation_root = tmp_path / "annotations"
    annotation = annotation_root / scene_id
    annotation.mkdir(parents=True)
    (annotation / f"{scene_id}_vh_clean_2.ply").write_bytes(b"ply\n")
    (annotation / f"{scene_id}.segs.json").write_text("{}")
    (annotation / f"{scene_id}.aggregation.json").write_text("{}")
    return scene_id, field_root, annotation_root


def _add_auto_cohort_scene(
    *, field_root: Path, annotation_root: Path, scene_id: str, frame_base: int
) -> None:
    field = field_root / scene_id
    (field / "color").mkdir(parents=True)
    (field / "pose").mkdir()
    (field / "intrinsic").mkdir()
    frame_ids = list(range(frame_base, frame_base + 8))
    source_positions = {0, 2, 5, 7}
    for position, frame_id in enumerate(frame_ids):
        if position in source_positions:
            Image.new("RGB", (4, 3), color=(frame_id % 256, 0, 0)).save(
                field / "color" / f"{frame_id:06d}.jpg"
            )
        pose = np.eye(4)
        pose[0, 3] = frame_id
        np.savetxt(field / "pose" / f"{frame_id:06d}.txt", pose)
    intrinsic = np.eye(4)
    intrinsic[0, 0] = 100
    intrinsic[1, 1] = 101
    intrinsic[0, 2] = 648
    intrinsic[1, 2] = 484
    np.savetxt(field / "intrinsic" / "intrinsic_color.txt", intrinsic)
    (field / "field_source_contract.json").write_text(json.dumps({
        "scene_id": scene_id,
        "frame_selection_policy": "depth_voxel_coverage",
        "selection_order_frame_indices": frame_ids,
        "source_color_size": [1296, 968],
        "uses_instances_or_semantic_labels": False,
        "contains_instance_or_label_directories": False,
    }))
    annotation = annotation_root / scene_id
    annotation.mkdir(parents=True)
    (annotation / f"{scene_id}_vh_clean_2.ply").write_bytes(b"ply\n")
    (annotation / f"{scene_id}.segs.json").write_text("{}")
    (annotation / f"{scene_id}.aggregation.json").write_text("{}")


def _auto_cohort_args(
    *, field_roots: list[Path], annotation_root: Path, output_root: Path,
    scene_limit: int, validation_per_field_shard: int,
) -> Namespace:
    return Namespace(
        field_root=[str(path) for path in field_roots],
        annotation_root=str(annotation_root),
        output_root=str(output_root),
        scene_id=[],
        scene_limit=scene_limit,
        validation_per_field_shard=validation_per_field_shard,
        total_view_count=8,
        observation_view_count=4,
    )


def test_materializer_binds_query_free_source_and_geometry_only_heldout(tmp_path):
    scene_id, field_root, annotation_root = _fixture(tmp_path)
    for frame_id in (11, 13, 14, 16):
        (field_root / scene_id / "color" / f"{frame_id:06d}.jpg").unlink()
    output_root = tmp_path / "output"
    receipt = materialize_scene(
        scene_id=scene_id,
        field_roots=[field_root],
        annotation_root=annotation_root,
        output_root=output_root,
    )
    scene = output_root / scene_id
    assert receipt["observation_positions"] == [0, 2, 5, 7]
    assert receipt["heldout_positions"] == [1, 3, 4, 6]
    assert [row["role"] for row in receipt["frames"]] == [
        "source_observation",
        "heldout_geometry_only",
        "source_observation",
        "heldout_geometry_only",
        "heldout_geometry_only",
        "source_observation",
        "heldout_geometry_only",
        "source_observation",
    ]
    assert sorted(path.name for path in (scene / "source_rgb").iterdir()) == [
        "000010.jpg", "000012.jpg", "000015.jpg", "000017.jpg"
    ]
    with Image.open(scene / "color" / "000010.jpg") as image:
        assert image.size == (1296, 968)
    assert not (scene / "color" / "000011.jpg").exists()
    assert receipt["source_rgb_resize"]["heldout_rgb_path_resolved"] is False
    assert receipt["source_rgb_resize"]["heldout_rgb_materialized_in_scene_root"] is False
    assert receipt["source_rgb_resize"]["heldout_rgb_decoded_or_opened"] is False
    assert receipt["source_rgb_resize"]["heldout_rgb_content_hashed"] is False
    for row in receipt["frames"]:
        if row["role"] == "heldout_geometry_only":
            assert row["staged_rgb_present"] is False
            assert "source_rgb_path" not in row
            assert "staged_rgb_path" not in row
            assert row["rgb_content_opened_or_decoded"] is False
            assert row["rgb_content_hashed"] is False
            assert "source_rgb_sha256" not in row
            assert "staged_rgb_sha256" not in row


def test_negative_validation_count_fails_before_materialization():
    with pytest.raises(ValueError, match="cannot be negative"):
        run(Namespace(scene_limit=0, validation_per_field_shard=-1))


def test_automatic_cohort_is_balanced_family_disjoint_and_order_invariant(tmp_path):
    annotation_root = tmp_path / "annotations"
    shard_a = tmp_path / "field_a"
    shard_b = tmp_path / "field_b"
    scenes_by_shard = {
        shard_a: ["scene0001_00", "scene0002_00", "scene0003_00"],
        # scene0001_01 aliases the same physical scan as scene0001_00.
        shard_b: [
            "scene0001_01", "scene0004_00", "scene0005_00", "scene0006_00"
        ],
    }
    frame_base = 100
    for field_root, scene_ids in scenes_by_shard.items():
        for scene_id in scene_ids:
            _add_auto_cohort_scene(
                field_root=field_root,
                annotation_root=annotation_root,
                scene_id=scene_id,
                frame_base=frame_base,
            )
            frame_base += 10

    forward = run(_auto_cohort_args(
        field_roots=[shard_a, shard_b],
        annotation_root=annotation_root,
        output_root=tmp_path / "forward",
        scene_limit=4,
        validation_per_field_shard=1,
    ))
    reversed_roots = run(_auto_cohort_args(
        field_roots=[shard_b, shard_a],
        annotation_root=annotation_root,
        output_root=tmp_path / "reversed",
        scene_limit=4,
        validation_per_field_shard=1,
    ))

    assert forward["scene_ids"] == reversed_roots["scene_ids"]
    assert forward["split"]["training_scene_ids"] == (
        reversed_roots["split"]["training_scene_ids"]
    )
    assert forward["split"]["validation_scene_ids"] == (
        reversed_roots["split"]["validation_scene_ids"]
    )
    assert forward["selection"]["policy"].endswith(
        "balanced_per_field_shard_v1"
    )
    assert forward["selection"]["candidate_scene_count"] == 7
    assert forward["scene_count"] == 4
    families = [scene_id.split("_")[0] for scene_id in forward["scene_ids"]]
    assert len(families) == len(set(families))
    assert sorted(
        counts["total"]
        for counts in forward["split"]["field_shard_counts"].values()
    ) == [2, 2]
    assert sorted(
        counts["validation"]
        for counts in forward["split"]["field_shard_counts"].values()
    ) == [1, 1]

    unlimited = run(_auto_cohort_args(
        field_roots=[shard_b, shard_a],
        annotation_root=annotation_root,
        output_root=tmp_path / "unlimited",
        scene_limit=0,
        validation_per_field_shard=0,
    ))
    assert unlimited["selection"]["policy"].endswith(
        "all_eligible_per_field_shard_v1"
    )
    assert "balanced" not in unlimited["selection"]["policy"]
    unlimited_families = [
        scene_id.split("_")[0] for scene_id in unlimited["scene_ids"]
    ]
    assert len(unlimited_families) == len(set(unlimited_families))


def test_materialized_transforms_round_trip_to_raw_scannet_opencv_pose(tmp_path):
    scene_id, field_root, annotation_root = _fixture(tmp_path)
    output_root = tmp_path / "output"
    materialize_scene(
        scene_id=scene_id,
        field_roots=[field_root],
        annotation_root=annotation_root,
        output_root=output_root,
    )
    transforms = json.loads(
        (output_root / scene_id / "transforms.json").read_text()
    )
    assert {key: transforms[key] for key in ("w", "h", "fl_x", "fl_y", "cx", "cy")} == {
        "w": 1296, "h": 968, "fl_x": 100.0, "fl_y": 101.0,
        "cx": 648.0, "cy": 484.0
    }
    staged = np.asarray(transforms["frames"][0]["transform_matrix"])
    raw = np.loadtxt(field_root / scene_id / "pose" / "000010.txt")
    np.testing.assert_allclose(staged @ GL_TO_CV, raw)
    assert (output_root / scene_id / "points3d.ply").resolve() == (
        annotation_root / scene_id / f"{scene_id}_vh_clean_2.ply"
    ).resolve()


def test_frozen_radio_grid_rejects_native_low_resolution_source(tmp_path):
    scene_id, field_root, annotation_root = _fixture(tmp_path)
    contract_path = field_root / scene_id / "field_source_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["source_color_size"] = [640, 480]
    contract_path.write_text(json.dumps(contract))
    assert _radio_grid_hw((1296, 968)) == FROZEN_RADIO_GRID_HW
    assert _radio_grid_hw((640, 480)) == (30, 40)
    try:
        materialize_scene(
            scene_id=scene_id,
            field_roots=[field_root],
            annotation_root=annotation_root,
            output_root=tmp_path / "output",
        )
    except ValueError as error:
        assert "not the frozen" in str(error)
    else:
        raise AssertionError("low-resolution source contract must fail closed")
