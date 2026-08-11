import json

import numpy as np
import pytest
import torch

from radio_gs.benchmarks.agile3d_scannet40.build_likelihood_training_dataset import (
    _authorized_source_scene,
    build_scene_split,
    build_training_payload,
    iter_head_training_examples,
    validate_scene_split,
)
from radio_gs.querying.query_likelihood_head import MonotoneQueryLikelihoodHead


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_scene_split_is_stable_disjoint_and_holds_test_labels_closed(tmp_path):
    source = {
        f"scene{index:04d}_00_obj_{index + 1}": {}
        for index in range(8)
    }
    test = {
        "scene1000_00_obj_1": {"obj": {"1": 5}},
        "scene1001_00_obj_1": {"obj": {"1": 7}},
    }
    source_path = _write_json(tmp_path / "train_list.json", source)
    test_path = _write_json(tmp_path / "val_list.json", test)

    first = build_scene_split(
        source_train_list=source_path,
        test_list=test_path,
        development_count=2,
    )
    second = build_scene_split(
        source_train_list=source_path,
        test_list=test_path,
        development_count=2,
    )

    assert first == second
    split = validate_scene_split(first)
    assert split["counts"] == {
        "official_source_train": 8,
        "fit": 6,
        "development_validation": 2,
        "test": 2,
    }
    assert split["safety"]["test_ply_geometry_opened"] is False
    assert split["safety"]["test_ply_labels_opened"] is False


def test_test_scene_is_rejected_before_training_label_access(tmp_path):
    source_path = _write_json(
        tmp_path / "train_list.json",
        {
            "scene0000_00_obj_1": {},
            "scene0001_00_obj_2": {},
            "scene0002_00_obj_3": {},
        },
    )
    test_path = _write_json(
        tmp_path / "val_list.json", {"scene1000_00_obj_1": {}}
    )
    split = build_scene_split(
        source_train_list=source_path,
        test_list=test_path,
        development_count=1,
    )

    with pytest.raises(PermissionError, match="test scene labels are forbidden"):
        _authorized_source_scene(split, "scene1000_00", "fit")


def test_synthetic_shard_has_both_click_signs_and_feeds_generic_head():
    points = np.stack(
        [np.arange(12, dtype=np.float32) * 0.05, np.zeros(12), np.zeros(12)],
        axis=1,
    )
    target = np.zeros(12, dtype=bool)
    target[3:7] = True
    primitive_xyz = torch.from_numpy(points)
    covariance = torch.eye(3)[None].repeat(12, 1, 1) * 0.05**2
    payload = build_training_payload(
        scene_id="scene0000_00",
        object_id=1,
        partition="fit",
        primitive_xyz=primitive_xyz,
        primitive_covariance=covariance,
        prior_probability=torch.full((12,), 0.5),
        coverage=torch.ones(12),
        reliability=torch.linspace(0.5, 1.0, 12),
        primitive_to_point_index=torch.arange(12),
        point_xyz=points,
        point_target=target,
        max_clicks=4,
        affinity_candidate_k=6,
        click_workers=1,
        adapter="synthetic",
    )

    assert any(click["is_positive"] for click in payload["clicks"])
    assert any(not click["is_positive"] for click in payload["clicks"])
    assert payload["click_affinity"].shape[0] == 12
    head = MonotoneQueryLikelihoodHead()
    examples = list(iter_head_training_examples(payload))
    assert len(examples) == len(payload["clicks"])
    for observations, primitive_target, step in examples:
        evidence = head(observations, source="world_click")
        assert evidence.values.shape == primitive_target.shape == (12,)
        assert evidence.confidence.shape == (12,)
        assert step["click_count"] >= 1


def test_primitive_targets_are_gathered_only_through_declared_mapping():
    points = np.stack(
        [np.arange(6, dtype=np.float32), np.zeros(6), np.zeros(6)], axis=1
    )
    target = np.asarray([False, True, False, True, False, False])
    mapping = torch.tensor([1, 3, 4])
    payload = build_training_payload(
        scene_id="scene0000_00",
        object_id=1,
        partition="fit",
        primitive_xyz=torch.from_numpy(points[[1, 3, 4]]),
        primitive_covariance=torch.eye(3)[None].repeat(3, 1, 1),
        prior_probability=torch.full((3,), 0.5),
        coverage=torch.ones(3),
        reliability=torch.ones(3),
        primitive_to_point_index=mapping,
        point_xyz=points,
        point_target=target,
        max_clicks=2,
        affinity_candidate_k=3,
        click_workers=1,
        adapter="synthetic_mapping",
    )

    torch.testing.assert_close(
        payload["primitive_target"], torch.tensor([1.0, 1.0, 0.0])
    )
    assert payload["safety"]["target_used_by_affinity_registration"] is False
