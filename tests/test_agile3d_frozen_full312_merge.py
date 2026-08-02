import hashlib
import json

import numpy as np
import pytest

from radio_gs.benchmarks.agile3d_scannet40.frozen_full312_contract import (
    FROZEN_FULL312_OBJECT_COUNT,
    FROZEN_FULL312_SCENE_COUNT,
    bind_frozen_method_contract,
    source_contract_bindings_sha256,
)
from radio_gs.benchmarks.agile3d_scannet40.merge_canonical_frozen_full312 import (
    merge,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def frozen_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("agile_full312")
    (root / "single").mkdir(parents=True)
    scenes = [f"scene{index:04d}_00" for index in range(FROZEN_FULL312_SCENE_COUNT)]
    object_rows = []
    for index in range(FROZEN_FULL312_OBJECT_COUNT):
        object_rows.append((scenes[index % len(scenes)], str(index + 1)))
    np.save(root / "single" / "object_ids.npy", np.asarray(object_rows))
    (root / "single" / "object_classes.txt").write_text(
        "".join("chair\n" for _ in object_rows), encoding="utf-8"
    )

    def protocol(*, objects: int, scene_count: int):
        return {
            "result_status": "formal",
            "formal_comparable": True,
            "diagnostic_no_support_gate": False,
            "field_checkpoint_name": "canonical_mpr_v2.pt",
            "capability_cache_name": "official_dino_sam3_views.pt",
            "support_graph_name": "shared_support_graph_k16.pt",
            "reliability_cache_name": "",
            "canonical_mpr_contract": "canonical-full-observation-mpr-v1",
            "canonical_mpr_coverage_ranked": True,
            "observation_contract": "scannet_full_observation_v1",
            "support_gate_required": True,
            "minimum_support_fraction": 0.95,
            "voxel_size_m": 0.05,
            "objects": objects,
            "scenes": scene_count,
            "max_clicks": 10,
            "click_search_workers": 2,
            "click_policy": "center_of_largest_FP_or_FN_error_by_inradius",
            "clicked_labels_forced": True,
            "test_set_calibration": False,
            "world_query": "compile_world_3d_query",
            "selection_mode": "seeded_component",
            "official_coordinate_contract": (
                "released_shifted_5cm_callback_plus_label_free_scene_origin_to_scannet_world"
            ),
            "observation_lift": "none",
            "official_point_readout": "continuous_opacity_weighted_gaussian",
            "readout_candidate_k": 64,
            "readout_support_threshold": 1e-2,
            "evaluation_voxel_size_m": 0.05,
            "voxel_cell_variance_m2": 0.05**2 / 12.0,
            "click_seed_kernel": "native_gaussian",
            "seed_candidate_k": 64,
            "hard_seed_topk": 0,
            "seed_temperature": 1.0,
            "hard_seed_threshold": 0.2,
            "hard_seed_conflict_policy": "positive_priority",
            "hard_seed_conflict_margin": 0.0,
            "prototype_count": 4,
            "prototype_strategy": "weighted_fps",
            "solver_type": "confidence_random_walker",
            "laplacian_weight": 1.0,
            "cg_iterations": 64,
            "support_threshold": 0.5,
            "unary_edge_contrast": 0.0,
            "world_point_prototype_mode": "per_click_local",
            "world_point_max_prototypes": 0,
            "world_point_prototype_weighting": "support_mass",
            "appearance_unary_weight": 1.0,
            "boundary_unary_weight": 0.35,
            "feature_calibration": "none",
            "background_centroids": 0,
            "background_negative_policy": "pooled_mean",
            "calibration_sample_size": 8192,
            "centroid_iterations": 4,
            "score_calibration": "none",
            "score_chunk_size": 8192,
            "channel_confidence_mode": "none",
            "negative_spatial_mode": "none",
            "negative_spatial_steps": 4,
            "negative_spatial_decay": 0.8,
            "spatial_log_weight": 0.25,
            "spatial_floor": 0.01,
            "point_readout_constraint": "none",
            "requires_official_extracted_capability_teachers": False,
        }

    def support(scene: str):
        return {
            "scene_id": scene,
            "continuous_support_fraction": 0.96,
            "support_gate_passed": True,
            "declared_source_contract": "scannet_full_observation_v1",
            "field_source_contract_version": "scannet_full_observation_v1",
            "mpr_observation_contract": "canonical-full-observation-mpr-v1",
            "mpr_full_observation_coverage_order_applied": True,
            "field_source_contract_sha256": _sha(scene + ":source"),
            "field_source_frame_manifest_sha256": _sha(scene + ":frames"),
            "field_checkpoint_sha256": _sha(scene + ":field"),
            "support_graph_sha256": _sha(scene + ":graph"),
            "mpr_observation_contract_sha256": _sha(scene + ":mpr"),
            "primitive_reliability_cache_sha256": "",
        }

    def shard(selected_scenes):
        selected = set(selected_scenes)
        rows = [
            {
                "scene_id": scene,
                "object_id": int(object_id),
                "semantic_class": "chair",
                "trajectory": {str(step): 0.5 for step in range(1, 11)},
            }
            for scene, object_id in object_rows
            if scene in selected
        ]
        scene_support = [support(scene) for scene in selected_scenes]
        current_protocol = protocol(objects=len(rows), scene_count=len(scene_support))
        method_contract, method_sha = bind_frozen_method_contract(current_protocol)
        source_bindings, source_sha = source_contract_bindings_sha256(scene_support)
        return {
            "benchmark": "AGILE3D ScanNet40 single-object",
            "protocol": current_protocol,
            "method_contract": method_contract,
            "method_contract_sha256": method_sha,
            "source_contract_bindings": source_bindings,
            "source_contract_bindings_sha256": source_sha,
            "scene_support": scene_support,
            "rows": rows,
        }

    return root, scenes, shard


def test_frozen_full312_merge_is_exact_and_fail_closed(frozen_fixture, tmp_path):
    root, scenes, make_shard = frozen_fixture
    one = make_shard(scenes[:156])
    two = make_shard(scenes[156:])

    one_path = tmp_path / "one.json"
    two_path = tmp_path / "two.json"
    one_path.write_text(json.dumps(one), encoding="utf-8")
    two_path.write_text(json.dumps(two), encoding="utf-8")
    report = merge(root, [two_path, one_path], tmp_path / "merged.json")
    assert report["protocol"]["scenes"] == 312
    assert report["protocol"]["objects"] == 10_357
    assert report["protocol"]["click_counts"] == [1, 2, 3, 5, 10]
    assert set(report["metrics"]) == {
        "IoU@1",
        "IoU@2",
        "IoU@3",
        "IoU@5",
        "IoU@10",
    }
    assert report["metrics"]["IoU@10"] == 0.5

    with pytest.raises(ValueError, match="incomplete"):
        merge(root, [one_path], tmp_path / "missing.json")

    duplicate = make_shard([scenes[0]])
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate frozen full312 object"):
        merge(root, [one_path, two_path, duplicate_path], tmp_path / "duplicate_out.json")

    bad_method = make_shard(scenes[156:])
    bad_method["method_contract_sha256"] = "a" * 64
    bad_method_path = tmp_path / "bad_method.json"
    bad_method_path.write_text(json.dumps(bad_method), encoding="utf-8")
    with pytest.raises(ValueError, match="stale or forged method"):
        merge(root, [one_path, bad_method_path], tmp_path / "bad_method_out.json")

    different_method = make_shard(scenes[156:])
    different_method["protocol"]["solver_type"] = "random_walker"
    contract, contract_sha = bind_frozen_method_contract(
        different_method["protocol"]
    )
    different_method["method_contract"] = contract
    different_method["method_contract_sha256"] = contract_sha
    different_method_path = tmp_path / "different_method.json"
    different_method_path.write_text(json.dumps(different_method), encoding="utf-8")
    with pytest.raises(ValueError, match="different method contract hash"):
        merge(
            root,
            [one_path, different_method_path],
            tmp_path / "different_method_out.json",
        )

    bad_source = make_shard(scenes[156:])
    bad_source["scene_support"][0]["field_source_contract_sha256"] = "b" * 64
    bad_source_path = tmp_path / "bad_source.json"
    bad_source_path.write_text(json.dumps(bad_source), encoding="utf-8")
    with pytest.raises(ValueError, match="stale or forged source"):
        merge(root, [one_path, bad_source_path], tmp_path / "bad_source_out.json")

    conflicting_source = make_shard([scenes[0]])
    conflicting_source["scene_support"][0]["field_source_contract_sha256"] = (
        "c" * 64
    )
    bindings, bindings_sha = source_contract_bindings_sha256(
        conflicting_source["scene_support"]
    )
    conflicting_source["source_contract_bindings"] = bindings
    conflicting_source["source_contract_bindings_sha256"] = bindings_sha
    conflicting_source_path = tmp_path / "conflicting_source.json"
    conflicting_source_path.write_text(
        json.dumps(conflicting_source), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="conflicting duplicate source contract"):
        merge(
            root,
            [one_path, two_path, conflicting_source_path],
            tmp_path / "conflicting_source_out.json",
        )

    bad_clicks = make_shard(scenes[156:])
    bad_clicks["rows"][0]["trajectory"]["11"] = 0.5
    bad_clicks_path = tmp_path / "bad_clicks.json"
    bad_clicks_path.write_text(json.dumps(bad_clicks), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly clicks 1..10"):
        merge(root, [one_path, bad_clicks_path], tmp_path / "bad_clicks_out.json")
