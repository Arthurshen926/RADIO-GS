import json

import numpy as np
import pytest

from radio_gs.benchmarks.agile3d_scannet40.merge_canonical_results import merge


def test_canonical_merge_requires_fixed_scene_set_and_common_protocol(tmp_path) -> None:
    root = tmp_path / "agile"
    (root / "single").mkdir(parents=True)
    np.save(root / "single" / "object_ids.npy", np.asarray([["scene_a", "1"], ["scene_b", "2"]]))
    (root / "single" / "object_classes.txt").write_text("chair\ntable\n", encoding="utf-8")
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({"scene_coverage": [{"scene_id": "scene_a"}, {"scene_id": "scene_b"}]}))
    protocol = {
        "observation_contract": "dense_overlap_pilot",
        "voxel_size_m": 0.05,
        "max_clicks": 20,
        "click_policy": "center_of_largest_FP_or_FN_error_by_inradius",
        "clicked_labels_forced": True,
        "test_set_calibration": False,
        "world_query": "compile_world_3d_query",
        "observation_lift": "none",
        "official_point_readout": "continuous_opacity_weighted_gaussian",
        "evaluation_voxel_size_m": 0.05,
        "click_seed_kernel": "native_gaussian",
        "unary_edge_contrast": 0.0,
        "world_point_prototype_mode": "per_click_local",
        "world_point_prototype_weighting": "equal_click",
    }

    def shard(scene, object_id, semantic, value):
        return {
            "benchmark": "AGILE3D ScanNet40 single-object",
            "protocol": {**protocol, "objects": 1, "scenes": 1},
            "scene_support": [{"scene_id": scene, "continuous_support_fraction": 0.96}],
            "rows": [{"scene_id": scene, "object_id": object_id, "semantic_class": semantic, "trajectory": {str(step): value for step in range(1, 21)}}],
        }

    one, two = tmp_path / "one.json", tmp_path / "two.json"
    one.write_text(json.dumps(shard("scene_a", 1, "chair", 0.25)))
    two.write_text(json.dumps(shard("scene_b", 2, "table", 0.75)))
    output = tmp_path / "merged.json"
    report = merge(root, expected, [two, one], output)
    assert report["metrics"]["IoU@1"] == 0.5
    assert report["support_summary"]["minimum_continuous_support_fraction"] == 0.96
    assert [row["scene_id"] for row in report["rows"]] == ["scene_a", "scene_b"]
    assert report["protocol"]["seed_candidate_k"] == 64
    assert report["protocol"]["solver_type"] == "confidence_random_walker"


def test_canonical_merge_accepts_an_explicit_named_pilot_scene_set(tmp_path) -> None:
    root = tmp_path / "agile"
    (root / "single").mkdir(parents=True)
    np.save(
        root / "single" / "object_ids.npy",
        np.asarray([["scene_a", "1"], ["scene_other", "2"]]),
    )
    (root / "single" / "object_classes.txt").write_text(
        "chair\ntable\n", encoding="utf-8"
    )
    protocol = {
        "observation_contract": "dense_agile_all_observations_pilot",
        "voxel_size_m": 0.05,
        "max_clicks": 20,
        "click_policy": "center_of_largest_FP_or_FN_error_by_inradius",
        "clicked_labels_forced": True,
        "test_set_calibration": False,
        "world_query": "compile_world_3d_query",
        "observation_lift": "none",
        "official_point_readout": "continuous_opacity_weighted_gaussian",
        "evaluation_voxel_size_m": 0.05,
        "click_seed_kernel": "native_gaussian",
        "unary_edge_contrast": 0.0,
        "world_point_prototype_mode": "per_click_local",
        "world_point_prototype_weighting": "support_mass",
    }
    shard = {
        "benchmark": "AGILE3D ScanNet40 single-object",
        "protocol": {**protocol, "objects": 1, "scenes": 1},
        "scene_support": [{"scene_id": "scene_a", "continuous_support_fraction": 0.97}],
        "rows": [
            {
                "scene_id": "scene_a",
                "object_id": 1,
                "semantic_class": "chair",
                "trajectory": {str(step): 0.6 for step in range(1, 21)},
            }
        ],
    }
    input_path, output = tmp_path / "shard.json", tmp_path / "merged.json"
    input_path.write_text(json.dumps(shard), encoding="utf-8")
    report = merge(root, None, [input_path], output, expected_scenes=("scene_a",))
    assert report["protocol"]["expected_scene_report"] == ""
    assert report["protocol"]["expected_scenes"] == ["scene_a"]


def test_full_observation_merge_fails_if_any_scene_misses_its_support_gate(tmp_path) -> None:
    root = tmp_path / "agile"
    (root / "single").mkdir(parents=True)
    np.save(
        root / "single" / "object_ids.npy",
        np.asarray([["scene_a", "1"], ["scene_other", "2"]]),
    )
    (root / "single" / "object_classes.txt").write_text(
        "chair\ntable\n", encoding="utf-8"
    )
    shard = {
        "benchmark": "AGILE3D ScanNet40 single-object",
        "protocol": {
            "observation_contract": "scannet_full_observation_v1",
            "support_gate_required": True,
            "minimum_support_fraction": 0.95,
            "canonical_mpr_contract": "canonical-full-observation-mpr-v1",
            "canonical_mpr_coverage_ranked": True,
            "voxel_size_m": 0.05,
            "max_clicks": 20,
            "click_policy": "center_of_largest_FP_or_FN_error_by_inradius",
            "clicked_labels_forced": True,
            "test_set_calibration": False,
            "world_query": "compile_world_3d_query",
            "observation_lift": "none",
            "official_point_readout": "continuous_opacity_weighted_gaussian",
            "evaluation_voxel_size_m": 0.05,
            "click_seed_kernel": "native_gaussian",
            "unary_edge_contrast": 0.0,
            "world_point_prototype_mode": "per_click_local",
            "world_point_prototype_weighting": "support_mass",
        },
        "scene_support": [{"scene_id": "scene_a", "continuous_support_fraction": 0.94}],
        "rows": [{
            "scene_id": "scene_a",
            "object_id": 1,
            "semantic_class": "chair",
            "trajectory": {str(step): 0.6 for step in range(1, 21)},
        }],
    }
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(shard), encoding="utf-8")
    with pytest.raises(ValueError, match="below its declared continuous-support gate"):
        merge(root, None, [path], tmp_path / "merged.json", expected_scenes=("scene_a",))

    # The explicit 480-view lifting contract is accepted only after the same
    # fixed support gate succeeds; it is not a way to waive the gate.
    shard["scene_support"][0]["continuous_support_fraction"] = 0.96
    shard["protocol"]["canonical_mpr_contract"] = "canonical-full-observation-mpr-v2"
    path.write_text(json.dumps(shard), encoding="utf-8")
    report = merge(root, None, [path], tmp_path / "merged_v2.json", expected_scenes=("scene_a",))
    assert report["protocol"]["canonical_mpr_contract"] == "canonical-full-observation-mpr-v2"

    shard["protocol"]["canonical_mpr_contract"] = "canonical-full-observation-mpr-v3"
    path.write_text(json.dumps(shard), encoding="utf-8")
    report = merge(root, None, [path], tmp_path / "merged_v3.json", expected_scenes=("scene_a",))
    assert report["protocol"]["canonical_mpr_contract"] == "canonical-full-observation-mpr-v3"


def test_full_observation_merge_rejects_a_legacy_temporal_mpr_contract(tmp_path) -> None:
    root = tmp_path / "agile"
    (root / "single").mkdir(parents=True)
    np.save(
        root / "single" / "object_ids.npy",
        np.asarray([["scene_a", "1"], ["scene_other", "2"]]),
    )
    (root / "single" / "object_classes.txt").write_text(
        "chair\ntable\n", encoding="utf-8"
    )
    shard = {
        "benchmark": "AGILE3D ScanNet40 single-object",
        "protocol": {
            "observation_contract": "scannet_full_observation_v1",
            "support_gate_required": True,
            "minimum_support_fraction": 0.95,
            "canonical_mpr_contract": "canonical-mpr-v1",
            "canonical_mpr_coverage_ranked": False,
            "voxel_size_m": 0.05,
            "max_clicks": 20,
            "click_policy": "center_of_largest_FP_or_FN_error_by_inradius",
            "clicked_labels_forced": True,
            "test_set_calibration": False,
            "world_query": "compile_world_3d_query",
            "observation_lift": "none",
            "official_point_readout": "continuous_opacity_weighted_gaussian",
            "evaluation_voxel_size_m": 0.05,
            "click_seed_kernel": "native_gaussian",
            "unary_edge_contrast": 0.0,
            "world_point_prototype_mode": "per_click_local",
            "world_point_prototype_weighting": "support_mass",
        },
        "scene_support": [{"scene_id": "scene_a", "continuous_support_fraction": 0.96}],
        "rows": [{
            "scene_id": "scene_a",
            "object_id": 1,
            "semantic_class": "chair",
            "trajectory": {str(step): 0.6 for step in range(1, 21)},
        }],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(shard), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage-ranked"):
        merge(root, None, [path], tmp_path / "merged.json", expected_scenes=("scene_a",))


def test_canonical_merge_accepts_identical_support_records_from_object_shards(tmp_path) -> None:
    root = tmp_path / "agile"
    (root / "single").mkdir(parents=True)
    np.save(
        root / "single" / "object_ids.npy",
        np.asarray([["scene_a", "1"], ["scene_a", "2"]]),
    )
    (root / "single" / "object_classes.txt").write_text(
        "chair\ntable\n", encoding="utf-8"
    )
    protocol = {
        "observation_contract": "dense_overlap_pilot",
        "voxel_size_m": 0.05,
        "max_clicks": 20,
        "click_policy": "center_of_largest_FP_or_FN_error_by_inradius",
        "clicked_labels_forced": True,
        "test_set_calibration": False,
        "world_query": "compile_world_3d_query",
        "observation_lift": "none",
        "official_point_readout": "continuous_opacity_weighted_gaussian",
        "evaluation_voxel_size_m": 0.05,
        "click_seed_kernel": "native_gaussian",
        "unary_edge_contrast": 0.0,
        "world_point_prototype_mode": "per_click_local",
        "world_point_prototype_weighting": "support_mass",
    }

    def shard(object_id, semantic, value, index):
        return {
            "benchmark": "AGILE3D ScanNet40 single-object",
            "protocol": {**protocol, "objects": 1, "scenes": 1},
            "scene_support": [
                {
                    "scene_id": "scene_a",
                    "continuous_support_fraction": 0.97,
                    "geometry_cache_reused": bool(index),
                }
            ],
            "rows": [
                {
                    "scene_id": "scene_a",
                    "object_id": object_id,
                    "semantic_class": semantic,
                    "trajectory": {str(step): value for step in range(1, 21)},
                }
            ],
            "shard": {
                "object_shard_index": index,
                "object_shard_count": 2,
                "metrics_are_partial": True,
            },
        }

    one, two = tmp_path / "one.json", tmp_path / "two.json"
    one.write_text(json.dumps(shard(1, "chair", 0.25, 0)), encoding="utf-8")
    two.write_text(json.dumps(shard(2, "table", 0.75, 1)), encoding="utf-8")
    report = merge(root, None, [one, two], tmp_path / "merged.json", expected_scenes=("scene_a",))
    assert report["metrics"]["IoU@1"] == 0.5
    assert len(report["merge"]["object_shards_merged"]) == 2
