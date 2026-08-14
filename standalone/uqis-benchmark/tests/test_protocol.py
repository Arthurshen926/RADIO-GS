import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from uqis_benchmark.construction import (
    REFERIT3D_VIEW_DEPENDENCE_RULE,
    backproject_world_point,
    align_depth_to_color_raster,
    build_image_query_crop,
    derive_paired_point_prompt,
    recompute_union_frame_exclusion,
    select_interior_pixel,
    select_query_frame_cover,
    select_profiled_expression,
    select_view_independent_expression,
)
from uqis_benchmark.protocol import (
    PREREGISTERED_TEST_SCENES,
    QUERY_MANIFEST_NAMES,
    QueryModality,
    UQISProtocolConfig,
    audit_release,
    freeze_release,
)
from uqis_benchmark.workspace import stage_query_workspace


def _scene(tmp_path: Path) -> dict:
    mesh = tmp_path / "scene" / "mesh_xyz.npy"
    instances = tmp_path / "private" / "mesh_instances.npy"
    mesh.parent.mkdir(parents=True)
    instances.parent.mkdir(parents=True)
    np.save(mesh, np.array([[0, 0, 1], [0.1, 0, 1], [1, 0, 1]], np.float32))
    np.save(instances, np.array([1, 1, 2], np.int32))
    return {
        "scene_id": "scene0000_00",
        "mesh_xyz_path": str(mesh),
        "mesh_instance_ids_path": str(instances),
        "query_frame_ids": ["000020"],
        "withheld_frame_ids": ["000000", "000020", "000040"],
        "field_frame_ids": ["000100", "000120"],
        "max_query_frames": 3,
    }


def _target(tmp_path: Path) -> dict:
    crop = tmp_path / "crop.png"
    Image.new("RGB", (224, 224), (127, 10, 20)).save(crop)
    intrinsic = np.array([[100.0, 0, 5], [0, 100.0, 5], [0, 0, 1]])
    pose = np.eye(4)
    point = backproject_world_point((5, 5), 1.0, intrinsic, pose)
    return {
        "scene_id": "scene0000_00",
        "instance_id": 1,
        "nyu40_class_id": 5,
        "mesh_vertex_count": 501,
        "size_bucket": "medium",
        "same_class_distractor_instance_ids": [2],
        "query_frame_id": "000020",
        "expression": "the left red chair",
        "expression_annotation_id": "7",
        "expression_source": "nr3d",
        "expression_view_independent": True,
        "crop_rgb_path": str(crop),
        "camera_to_world": pose.tolist(),
        "camera_intrinsics": intrinsic.tolist(),
        "raster_size": [10, 10],
        "positive_pixel_uv": [5, 5],
        "click_depth_m": 1.0,
        "point_world_xyz": point.tolist(),
        "projection_pixels": 1000,
        "projection_fraction": 0.01,
        "projection_purity": 0.95,
        "field_surface_coverage": 0.75,
        "field_visibility_count": 5,
    }


def _pilot_config() -> UQISProtocolConfig:
    return UQISProtocolConfig(
        min_targets_per_scene=1,
        min_same_class_targets_per_scene=1,
        min_semantic_categories_per_scene=1,
    )


def test_freeze_keeps_cross_modality_pairing_evaluator_only(tmp_path: Path) -> None:
    release = freeze_release(
        [_scene(tmp_path)],
        [_target(tmp_path)],
        tmp_path / "release",
        split_role="pilot",
        query_id_salt=b"0123456789abcdef",
        config=_pilot_config(),
        allow_incomplete_pilot=True,
    )
    assert release["audit"]["valid"]
    method_ids = []
    for modality in QueryModality:
        payload = json.loads(
            (tmp_path / "release" / QUERY_MANIFEST_NAMES[modality.value]).read_text()
        )
        row = payload["queries"][0]
        method_ids.append(row["query_id"])
        serialized = json.dumps(payload).lower()
        assert "target_id" not in serialized
        assert "instance_id" not in serialized
        assert "nyu40" not in serialized
        if modality is QueryModality.POINT_2D:
            assert not any("rgb" in key.lower() for key in row)
            assert set(row["available_method_inputs"]) == {
                "scene_id",
                "camera_to_world",
                "camera_intrinsics",
                "raster_size",
                "positive_pixel_uv",
            }
    assert len(set(method_ids)) == 4
    evaluator = json.loads(
        (tmp_path / "release" / "target_manifest.evaluator.json").read_text()
    )
    assert set(evaluator["targets"][0]["queries"].values()) == set(method_ids)
    assert audit_release(tmp_path / "release")["valid"]


def test_audit_rejects_rgb_added_to_strict_2d_manifest(tmp_path: Path) -> None:
    freeze_release(
        [_scene(tmp_path)],
        [_target(tmp_path)],
        tmp_path / "release",
        split_role="pilot",
        query_id_salt=b"0123456789abcdef",
        config=_pilot_config(),
        allow_incomplete_pilot=True,
    )
    path = tmp_path / "release" / QUERY_MANIFEST_NAMES["point_2d"]
    payload = json.loads(path.read_text())
    payload["queries"][0]["rendered_rgb_path"] = "/private/render.png"
    path.write_text(json.dumps(payload))
    report = audit_release(tmp_path / "release", check_files=False)
    assert not report["valid"]
    assert any("method-visible fields" in error for error in report["errors"])


def test_freeze_rejects_nonpaired_2d_3d_surface_points(tmp_path: Path) -> None:
    target = _target(tmp_path)
    target["point_world_xyz"] = [9.0, 9.0, 9.0]
    with pytest.raises(ValueError, match="same surface point"):
        freeze_release(
            [_scene(tmp_path)],
            [target],
            tmp_path / "release",
            split_role="pilot",
            query_id_salt=b"0123456789abcdef",
            config=_pilot_config(),
            allow_incomplete_pilot=True,
        )


def test_query_frame_cover_is_minimal_and_deterministic() -> None:
    cover = select_query_frame_cover(
        {
            1: {"000010": 1.0, "000020": 0.5},
            2: {"000020": 1.0, "000030": 0.8},
            3: {"000030": 1.0},
        },
        maximum_frames=3,
    )
    assert cover.frame_ids == ("000020", "000030")
    assert cover.target_to_frame == {1: "000020", 2: "000020", 3: "000030"}


def test_interior_click_and_backprojection_are_exactly_paired() -> None:
    mask = np.zeros((9, 9), bool)
    mask[1:8, 1:8] = True
    depth = np.full((9, 9), 2000, np.uint16)
    correspondence = np.ones_like(mask)
    correspondence[4, 4] = False
    prompt = derive_paired_point_prompt(
        mask,
        depth,
        np.array([[100.0, 0, 4], [0, 100.0, 4], [0, 0, 1]]),
        np.eye(4),
        valid_correspondence=correspondence,
    )
    assert prompt["positive_pixel_uv"] == [3, 3]
    np.testing.assert_allclose(prompt["point_world_xyz"], [-0.02, -0.02, 2.0])
    assert select_interior_pixel(mask)[0] == (4, 4)


def test_image_crop_uses_fixed_padding_canvas_and_resize() -> None:
    image = Image.new("RGB", (10, 10), (255, 0, 0))
    mask = np.zeros((10, 10), bool)
    mask[0:4, 0:4] = True
    crop, box = build_image_query_crop(
        image, mask, padding_fraction=0.25, output_size=20, fill_rgb=(0, 0, 0)
    )
    assert box == (-1, -1, 5, 5)
    assert crop.size == (20, 20)
    assert np.asarray(crop)[0, 0].max() == 0


def test_nr3d_selection_uses_object_id_and_view_independent_sort() -> None:
    rows = [
        {
            "scan_id": "scene0000_00",
            "target_id": 4,
            "ann_id": "20",
            "utterance": "the right chair",
            "correct_guess": True,
            "mentions_target_class": True,
            "dep_or_indep": "indep",
        },
        {
            "scan_id": "scene0000_00",
            "target_id": 4,
            "ann_id": "3",
            "utterance": "the red chair",
            "correct_guess": True,
            "mentions_target_class": True,
            "dep_or_indep": "indep",
        },
        {
            "scan_id": "scene0000_00",
            "target_id": 4,
            "ann_id": "1",
            "utterance": "the chair on your left",
            "correct_guess": True,
            "mentions_target_class": True,
            "dep_or_indep": "dependent",
        },
    ]
    selected = select_view_independent_expression(
        rows, scene_id="scene0000_00", official_instance_id=5
    )
    assert selected["annotation_id"] == "3"
    assert selected["expression"] == "the red chair"


def test_raw_nr3d_tokens_apply_official_referit3d_view_rule() -> None:
    rows = [
        {
            "scan_id": "scene0000_00",
            "target_id": "4",
            "assignmentid": "dependent",
            "utterance": "the chair on the left",
            "tokens": "['the', 'chair', 'on', 'the', 'left']",
            "correct_guess": "True",
            "mentions_target_class": "True",
            "dataset": "nr3d",
        },
        {
            "scan_id": "scene0000_00",
            "target_id": "4",
            "assignmentid": "independent",
            "utterance": "the red wooden chair",
            "tokens": "['the', 'red', 'wooden', 'chair']",
            "correct_guess": "True",
            "mentions_target_class": "True",
            "dataset": "nr3d",
        },
    ]

    selected = select_view_independent_expression(
        rows, scene_id="scene0000_00", official_instance_id=5
    )

    assert selected["annotation_id"] == "independent"
    assert selected["view_dependence_rule"] == REFERIT3D_VIEW_DEPENDENCE_RULE


def test_v2_text_profile_prefers_nonspatial_expression_over_annotation_order() -> None:
    rows = [
        {
            "scan_id": "scene0000_00",
            "target_id": "4",
            "assignmentid": "1",
            "utterance": "the chair closest to the window",
            "tokens": "['the', 'chair', 'closest', 'to', 'the', 'window']",
            "correct_guess": "True",
            "mentions_target_class": "True",
            "uses_spatial_lang": "True",
            "dataset": "nr3d",
        },
        {
            "scan_id": "scene0000_00",
            "target_id": "4",
            "assignmentid": "20",
            "utterance": "the red wooden chair",
            "tokens": "['the', 'red', 'wooden', 'chair']",
            "correct_guess": "True",
            "mentions_target_class": "True",
            "uses_spatial_lang": "False",
            "dataset": "nr3d",
        },
    ]

    selected = select_profiled_expression(
        rows, scene_id="scene0000_00", official_instance_id=5
    )

    assert selected["annotation_id"] == "20"
    assert selected["evaluation_tier"] == "unified_core"
    assert selected["relational_language_required"] is False


def test_v2_text_profile_marks_target_relational_when_no_nonspatial_row_exists() -> None:
    selected = select_profiled_expression(
        [
            {
                "scan_id": "scene0000_00",
                "target_id": "4",
                "assignmentid": "1",
                "utterance": "the chair beside the desk",
                "tokens": "['the', 'chair', 'beside', 'the', 'desk']",
                "correct_guess": "True",
                "mentions_target_class": "True",
                "uses_spatial_lang": "True",
                "dataset": "nr3d",
            }
        ],
        scene_id="scene0000_00",
        official_instance_id=5,
    )

    assert selected["evaluation_tier"] == "relational_text_challenge"
    assert selected["relational_language_required"] is True


def test_nr3d_missing_or_malformed_dependence_evidence_fails_closed() -> None:
    common = {
        "scan_id": "scene0000_00",
        "target_id": "4",
        "utterance": "the red wooden chair",
        "correct_guess": "True",
        "mentions_target_class": "True",
    }
    for row in (common, {**common, "tokens": "not a Python literal"}):
        with pytest.raises(ValueError, match="no valid view-independent expression"):
            select_view_independent_expression(
                [row], scene_id="scene0000_00", official_instance_id=5
            )


def test_union_exclusion_uses_full_sensor_order_and_all_query_poses() -> None:
    frame_ids = tuple(f"{index:06d}" for index in range(12))
    poses = {}
    for index, frame_id in enumerate(frame_ids):
        pose = np.eye(4)
        pose[0, 3] = index * 0.2
        poses[frame_id] = pose
    # Frame 11 is temporally distant but intentionally pose-near query frame 2.
    poses["000011"] = poses["000002"].copy()

    excluded = recompute_union_frame_exclusion(
        frame_ids,
        poses,
        ["000002", "000008"],
        temporal_radius=1,
        translation_m=0.1,
        rotation_deg=8.0,
    )

    assert excluded == (
        "000001",
        "000002",
        "000003",
        "000007",
        "000008",
        "000009",
        "000011",
    )


def test_union_exclusion_rejects_sparse_pose_inventory() -> None:
    with pytest.raises(ValueError, match="exactly cover"):
        recompute_union_frame_exclusion(
            ["000000", "000001"], {"000000": np.eye(4)}, ["000000"]
        )


def test_union_exclusion_preserves_nonfinite_frame_in_temporal_order() -> None:
    frame_ids = ["000000", "000001", "000002"]
    poses = {frame_id: np.eye(4) for frame_id in frame_ids}
    poses["000001"] = np.full((4, 4), np.nan)
    poses["000002"][0, 3] = 1.0
    excluded = recompute_union_frame_exclusion(
        frame_ids, poses, ["000000"], temporal_radius=1
    )
    assert excluded == ("000000", "000001")
    with pytest.raises(ValueError, match="query camera pose is not finite"):
        recompute_union_frame_exclusion(
            frame_ids, poses, ["000001"], temporal_radius=1
        )


def test_scanrefer_supplement_uses_bound_target_and_class_mention() -> None:
    rows = [
        {
            "scene_id": "scene0000_00",
            "object_id": 4,
            "object_name": "trash_can",
            "ann_id": 7,
            "description": "the blue trash can beside the desk",
            "token": ["the", "blue", "trash", "can", "beside", "the", "desk"],
            "tokens": ["the", "blue", "trash", "can", "beside", "the", "desk"],
            "dataset": "scanrefer",
        }
    ]
    selected = select_view_independent_expression(
        rows, scene_id="scene0000_00", official_instance_id=5
    )
    assert selected["source"] == "scanrefer"
    assert selected["qualification_rule"] == (
        "scanrefer_bound_target_and_object_name_mention"
    )


def test_scanrefer_supplement_rejects_missing_class_mention() -> None:
    row = {
        "scene_id": "scene0000_00",
        "object_id": 4,
        "object_name": "trash_can",
        "ann_id": 7,
        "description": "the blue thing beside the desk",
        "tokens": ["the", "blue", "thing", "beside", "the", "desk"],
        "dataset": "scanrefer",
    }
    with pytest.raises(ValueError, match="no valid view-independent expression"):
        select_view_independent_expression(
            [row], scene_id="scene0000_00", official_instance_id=5
        )


def test_depth_alignment_z_buffers_into_color_raster() -> None:
    depth = np.asarray([[1000, 2000], [0, 1000]], dtype=np.uint16)
    intrinsic = np.eye(3)
    aligned = align_depth_to_color_raster(
        depth, intrinsic, intrinsic, (2, 2), depth_scale=1000.0
    )
    assert aligned.dtype == np.float32
    assert np.allclose(aligned, [[1.0, 2.0], [0.0, 1.0]])


def test_formal_freeze_is_fail_closed_until_official_constructor_exists() -> None:
    scenes = [{"scene_id": scene_id} for scene_id in PREREGISTERED_TEST_SCENES]
    with pytest.raises(RuntimeError, match="formal UQIS freezing is disabled"):
        freeze_release(
            scenes,
            [],
            "/not/created",
            split_role="test",
            query_id_salt=b"0123456789abcdef",
        )


def test_audit_rejects_top_level_method_leak_and_hash_skipping(tmp_path: Path) -> None:
    freeze_release(
        [_scene(tmp_path)],
        [_target(tmp_path)],
        tmp_path / "release",
        split_role="pilot",
        query_id_salt=b"0123456789abcdef",
        config=_pilot_config(),
        allow_incomplete_pilot=True,
    )
    path = tmp_path / "release" / QUERY_MANIFEST_NAMES["text"]
    payload = json.loads(path.read_text())
    payload["target_pairing"] = {"leak": True}
    path.write_text(json.dumps(payload))

    report = audit_release(tmp_path / "release", check_files=False)

    assert not report["valid"]
    assert any("top-level fields changed" in error for error in report["errors"])
    assert any("asset hashes were skipped" in error for error in report["errors"])


def test_audit_rejects_release_config_drift(tmp_path: Path) -> None:
    freeze_release(
        [_scene(tmp_path)],
        [_target(tmp_path)],
        tmp_path / "release",
        split_role="pilot",
        query_id_salt=b"0123456789abcdef",
        config=_pilot_config(),
        allow_incomplete_pilot=True,
    )
    release_path = tmp_path / "release" / "release.json"
    release = json.loads(release_path.read_text())
    release["protocol_config"]["bootstrap_seed"] += 1
    from uqis_benchmark.protocol import canonical_json_sha256

    release["protocol_config_sha256"] = canonical_json_sha256(
        release["protocol_config"]
    )
    release_path.write_text(json.dumps(release))

    report = audit_release(tmp_path / "release")

    assert not report["valid"]
    assert any("release/common" in error for error in report["errors"])


def test_query_workspace_contains_only_one_sanitized_method_grant(tmp_path: Path) -> None:
    source_crop = _target(tmp_path)["crop_rgb_path"]
    freeze_release(
        [_scene(tmp_path)],
        [_target(tmp_path)],
        tmp_path / "release",
        split_role="pilot",
        query_id_salt=b"0123456789abcdef",
        config=_pilot_config(),
        allow_incomplete_pilot=True,
    )
    image_manifest = json.loads(
        (tmp_path / "release" / QUERY_MANIFEST_NAMES["image"]).read_text()
    )
    query_id = image_manifest["queries"][0]["query_id"]

    receipt = stage_query_workspace(
        tmp_path / "release",
        modality="image",
        query_id=query_id,
        workspace_dir=tmp_path / "workspace",
    )

    workspace_manifest = json.loads(
        (tmp_path / "workspace" / "query_manifest.json").read_text()
    )
    assert receipt["query_count"] == 1
    assert receipt["evaluator_private_files_staged"] is False
    assert len(workspace_manifest["queries"]) == 1
    assert Path(workspace_manifest["queries"][0]["crop_rgb_path"]).name == "query.png"
    assert str(source_crop) not in json.dumps(workspace_manifest)
    assert not (tmp_path / "workspace" / "target_manifest.evaluator.json").exists()
