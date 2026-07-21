import json

import numpy as np

from radio_gs.benchmarks.agile3d_scannet40.protocol import (
    Click,
    aggregate_official_metrics,
    evaluate_interactive_predictions,
    quantize_scannet_points,
    select_next_click,
)
from radio_gs.benchmarks.agile3d_scannet40.evaluate_feature_cache import (
    CanonicalPointPredictor,
    _load_feature_cache,
)
from radio_gs.benchmarks.agile3d_scannet40.merge_results import merge
from radio_gs.querying.support_solver import SupportGraphConfig, SupportSolverConfig
from radio_gs.scripts.export_canonical_field_to_agile3d_mesh import (
    interpolate_feature_matrix,
)


def test_quantization_reconstructs_full_rows() -> None:
    xyz = np.array([[0.10, 0, 0], [0, 0, 0], [0.01, 0, 0]], dtype=np.float32)
    labels = np.array([0, 1, 1])
    scene = quantize_scannet_points(xyz, np.zeros((3, 3)), labels, voxel_size=0.05)
    assert len(scene.coordinates) == 2
    np.testing.assert_array_equal(scene.unique_map, [0, 1])
    np.testing.assert_array_equal(scene.labels[scene.inverse_map], labels)


def test_first_click_is_deepest_target_point() -> None:
    xyz = np.column_stack([np.arange(7), np.zeros((7, 2))]).astype(np.float32)
    target = np.array([0, 1, 1, 1, 1, 1, 0], dtype=bool)
    click = select_next_click(xyz, np.zeros(7, dtype=bool), target, order=0)
    assert click == Click(point_index=3, is_positive=True, order=0)


def test_corrective_click_uses_larger_fp_or_fn_error_radius() -> None:
    xyz = np.column_stack([np.arange(10), np.zeros((10, 2))]).astype(np.float32)
    target = np.zeros(10, dtype=bool)
    target[2:7] = True
    prediction = np.zeros(10, dtype=bool)
    prediction[2] = True
    prediction[8] = True
    click = select_next_click(xyz, prediction, target, order=1)
    assert click is not None
    assert click.is_positive
    assert click.point_index in {4, 5}


def test_click_search_worker_count_is_scheduling_only() -> None:
    rng = np.random.default_rng(7)
    xyz = rng.normal(size=(128, 3)).astype(np.float32)
    target = np.zeros(128, dtype=bool)
    target[15:91] = True
    prediction = np.zeros(128, dtype=bool)
    prediction[20:30] = True
    prediction[100:111] = True
    assert select_next_click(
        xyz, prediction, target, order=1, workers=1
    ) == select_next_click(xyz, prediction, target, order=1, workers=-1)


def test_interaction_forces_clicked_labels_and_aggregates_metrics() -> None:
    xyz = np.column_stack([np.arange(5), np.zeros((5, 2))]).astype(np.float32)
    target = np.array([0, 1, 1, 1, 0], dtype=bool)

    def predictor(_xyz, previous, _clicks):
        return previous

    result = evaluate_interactive_predictions(
        xyz,
        target,
        target,
        np.arange(5),
        predictor,
        max_clicks=20,
    )
    assert result["trajectory"][1] > 0
    metrics = aggregate_official_metrics([result["trajectory"]])
    assert set(("IoU@1", "IoU@15", "NoC@50", "NoC@90")) <= set(metrics)


def test_canonical_predictor_hard_clamps_positive_and_negative_clicks() -> None:
    xyz = np.column_stack([np.arange(6) * 0.05, np.zeros((6, 2))]).astype(np.float32)
    features = np.array(
        [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
        dtype=np.float32,
    )
    predictor = CanonicalPointPredictor(
        xyz,
        features,
        appearance_features=features,
        boundary_features=features,
        observation_valid=None,
        device="cpu",
        graph_config=SupportGraphConfig(neighbors=2),
        solver_config=SupportSolverConfig(
            solver_type="random_walker",
            cg_iterations=16,
        ),
    )
    prediction = predictor(
        xyz,
        np.zeros(6, dtype=bool),
        (
            Click(point_index=1, is_positive=True, order=0),
            Click(point_index=4, is_positive=False, order=1),
        ),
    )
    assert prediction[1]
    assert not prediction[4]


def test_observed_domain_is_exactly_legacy_when_every_row_is_observed() -> None:
    xyz = np.column_stack([np.arange(6) * 0.05, np.zeros((6, 2))]).astype(np.float32)
    features = np.array(
        [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
        dtype=np.float32,
    )
    common = dict(
        appearance_features=features,
        boundary_features=features,
        observation_valid=np.ones(len(xyz), dtype=bool),
        device="cpu",
        graph_config=SupportGraphConfig(neighbors=2),
        solver_config=SupportSolverConfig(solver_type="random_walker", cg_iterations=16),
    )
    observed = CanonicalPointPredictor(
        xyz, features, observation_lift_mode="observed_domain", **common
    )
    legacy = CanonicalPointPredictor(
        xyz, features, observation_lift_mode="full_cache_legacy", **common
    )
    clicks = (
        Click(point_index=1, is_positive=True, order=0),
        Click(point_index=4, is_positive=False, order=1),
    )
    np.testing.assert_array_equal(
        observed(xyz, np.zeros(len(xyz), dtype=bool), clicks),
        legacy(xyz, np.zeros(len(xyz), dtype=bool), clicks),
    )


def test_canonical_predictor_keeps_only_the_positive_seeded_component() -> None:
    xyz = np.array(
        [
            [0.00, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [4.00, 0.0, 0.0],
            [4.05, 0.0, 0.0],
            [4.10, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Every point has the same visual unary. The instance query must still
    # return only the physical component reached by its registered click.
    features = np.array([[1, 0]] * len(xyz), dtype=np.float32)
    predictor = CanonicalPointPredictor(
        xyz,
        features,
        appearance_features=features,
        boundary_features=features,
        observation_valid=None,
        device="cpu",
        graph_config=SupportGraphConfig(neighbors=2),
        solver_config=SupportSolverConfig(
            solver_type="random_walker", cg_iterations=32
        ),
    )
    prediction = predictor(
        xyz,
        np.zeros(len(xyz), dtype=bool),
        (Click(point_index=1, is_positive=True, order=0),),
    )
    np.testing.assert_array_equal(prediction, [True, True, True, False, False, False])


def test_observed_domain_lifts_nearby_unobserved_click_without_zero_feature_seed() -> None:
    xyz = np.array(
        [[0.00, 0.0, 0.0], [0.05, 0.0, 0.0], [0.10, 0.0, 0.0]],
        dtype=np.float32,
    )
    features = np.array([[1, 0], [1, 0], [0, 0]], dtype=np.float32)
    predictor = CanonicalPointPredictor(
        xyz,
        features,
        appearance_features=features,
        boundary_features=features,
        observation_valid=np.array([True, True, False]),
        device="cpu",
        graph_config=SupportGraphConfig(neighbors=1),
        solver_config=SupportSolverConfig(solver_type="random_walker", cg_iterations=16),
        observation_lift_mode="observed_domain",
        observation_lift_neighbors=2,
        observation_lift_maximum_distance_m=0.10,
    )
    prediction = predictor(
        xyz,
        np.zeros(len(xyz), dtype=bool),
        (Click(point_index=2, is_positive=True, order=0),),
    )
    assert prediction[2]
    report = predictor.observation_lift_report()
    assert report["solver_nodes"] == 2
    assert report["projectable_fraction"] == 1.0


def test_observed_domain_refuses_to_hallucinate_remote_unobserved_click() -> None:
    xyz = np.array(
        [[0.00, 0.0, 0.0], [0.05, 0.0, 0.0], [1.00, 0.0, 0.0]],
        dtype=np.float32,
    )
    features = np.array([[1, 0], [1, 0], [0, 0]], dtype=np.float32)
    predictor = CanonicalPointPredictor(
        xyz,
        features,
        appearance_features=features,
        boundary_features=features,
        observation_valid=np.array([True, True, False]),
        device="cpu",
        graph_config=SupportGraphConfig(neighbors=1),
        solver_config=SupportSolverConfig(solver_type="random_walker", cg_iterations=16),
        observation_lift_mode="observed_domain",
        observation_lift_neighbors=2,
        observation_lift_maximum_distance_m=0.10,
    )
    prediction = predictor(
        xyz,
        np.zeros(len(xyz), dtype=bool),
        (Click(point_index=2, is_positive=True, order=0),),
    )
    assert not prediction.any()


def test_vector_feature_interpolation_is_query_and_label_free() -> None:
    source_xyz = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    source_features = np.eye(2, dtype=np.float32)
    target_xyz = np.array([[0, 0, 0], [0.5, 0, 0], [3, 0, 0]], dtype=np.float32)
    mapped, valid = interpolate_feature_matrix(
        source_xyz,
        source_features,
        target_xyz,
        neighbors=2,
        maximum_distance_m=1.0,
    )
    np.testing.assert_array_equal(valid, [True, True, False])
    np.testing.assert_allclose(mapped[0], [1, 0], atol=2e-4)
    np.testing.assert_allclose(mapped[1], [2 ** -0.5, 2 ** -0.5], atol=1e-4)


def test_quantized_feature_cache_requires_exact_unique_map(tmp_path) -> None:
    xyz = np.arange(18, dtype=np.float32).reshape(6, 3)
    unique = np.array([0, 2, 5], dtype=np.int64)
    path = tmp_path / "scene.npz"
    np.savez(
        path,
        xyz=xyz[unique],
        unique_map=unique,
        radio_features=np.ones((3, 4), dtype=np.float16),
        appearance_features=np.ones((3, 2), dtype=np.float16),
        boundary_features=np.ones((3, 2), dtype=np.float16),
    )
    cache = _load_feature_cache(path, xyz, quantized_unique_map=unique)
    assert bool(cache["_is_quantized"])


def test_feature_cache_rejects_misaligned_observation_validity(tmp_path) -> None:
    xyz = np.arange(18, dtype=np.float32).reshape(6, 3)
    path = tmp_path / "scene.npz"
    np.savez(
        path,
        xyz=xyz,
        radio_features=np.ones((6, 4), dtype=np.float16),
        valid=np.ones(5, dtype=bool),
    )
    try:
        _load_feature_cache(path, xyz, quantized_unique_map=np.arange(6))
    except ValueError as error:
        assert "valid does not align" in str(error)
    else:
        raise AssertionError("misaligned feature validity must fail closed")


def test_merge_requires_exact_official_coverage_and_preserves_trajectories(tmp_path) -> None:
    root = tmp_path / "agile"
    (root / "single").mkdir(parents=True)
    np.save(
        root / "single" / "object_ids.npy",
        np.asarray([["scene_a", "1"], ["scene_b", "2"]]),
    )
    (root / "single" / "object_classes.txt").write_text("chair\ntable\n")
    protocol = {
        "voxel_size_m": 0.05,
        "max_clicks": 20,
        "click_policy": "center_of_largest_FP_or_FN_error_by_inradius",
        "clicked_labels_forced": True,
        "test_set_calibration": False,
        "selection_mode": "seeded_component",
        "unary_mode": "shared_capability",
        "appearance_unary_weight": 1.0,
        "boundary_unary_weight": 0.35,
        "observation_lift_mode": "observed_domain",
        "observation_lift_neighbors": 3,
        "observation_lift_maximum_distance_m": 0.10,
    }

    def shard(scene, object_id, semantic, coverage, value):
        return {
            "benchmark": "AGILE3D ScanNet40 single-object",
            "protocol": protocol,
            "scene_coverage": [
                {
                    "scene_id": scene,
                    "quantized_points": 4,
                    "valid_feature_points": 3,
                    "feature_coverage": coverage,
                    "observation_lift": {
                        "mode": "observed_domain",
                        "solver_nodes": 3,
                        "projectable_points": 4,
                        "projectable_fraction": 1.0,
                        "neighbors": 3,
                        "maximum_distance_m": 0.10,
                    },
                }
            ],
            "rows": [
                {
                    "scene_id": scene,
                    "object_id": object_id,
                    "semantic_class": semantic,
                    "trajectory": {str(step): value for step in range(1, 21)},
                }
            ],
        }

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(shard("scene_a", 1, "chair", 0.75, 0.25)))
    second.write_text(json.dumps(shard("scene_b", 2, "table", 1.0, 0.75)))
    output = tmp_path / "merged.json"
    report = merge(root, [second, first], output)
    assert report["metrics"]["IoU@1"] == 0.5
    assert report["coverage_summary"]["mean_feature_coverage"] == 0.875
    assert report["coverage_summary"]["mean_projectable_fraction"] == 1.0
    assert [row["scene_id"] for row in report["rows"]] == ["scene_a", "scene_b"]
