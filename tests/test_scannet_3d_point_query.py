import json

import numpy as np
from scipy.spatial import cKDTree

from radio_gs.querying.unified_query import SupportPropagationConfig
from radio_gs.scripts.eval_scannet_3d_point_query import (
    _local_multiscale_positive_prototypes,
    evaluate_point_queries,
    load_scannet_instances,
)


def test_local_multiscale_prototypes_are_normalized_and_keep_click() -> None:
    xyz = np.stack([np.arange(10) * 0.01, np.zeros(10), np.zeros(10)], axis=1).astype(
        np.float32
    )
    features = np.stack(
        [np.ones(10), np.linspace(0.0, 0.2, 10)], axis=1
    ).astype(np.float32)
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    prototypes = _local_multiscale_positive_prototypes(
        xyz, normalized, (4,), cKDTree(xyz), neighbors=8
    )
    expected_click = features[4] / np.linalg.norm(features[4])
    np.testing.assert_allclose(prototypes[0], expected_click, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(prototypes, axis=1), 1.0, atol=1e-6)
    assert prototypes.shape[0] >= 2


def test_load_scannet_instances(tmp_path) -> None:
    segmentation = tmp_path / "scene.segs.json"
    aggregation = tmp_path / "scene.aggregation.json"
    segmentation.write_text(json.dumps({"segIndices": [10, 10, 20, 20]}))
    aggregation.write_text(
        json.dumps(
            {
                "segGroups": [
                    {"objectId": 0, "label": "chair", "segments": [10]},
                    {"objectId": 3, "label": "table", "segments": [20]},
                ]
            }
        )
    )
    instance_ids, metadata = load_scannet_instances(aggregation, segmentation)
    np.testing.assert_array_equal(instance_ids, [1, 1, 4, 4])
    assert metadata[4]["label"] == "table"


def test_point_query_separates_two_feature_coherent_instances() -> None:
    cluster_a = np.stack([np.arange(4) * 0.01, np.zeros(4), np.zeros(4)], axis=1)
    cluster_b = cluster_a + np.array([1.0, 0.0, 0.0])
    xyz = np.concatenate([cluster_a, cluster_b]).astype(np.float32)
    features = np.concatenate(
        [np.tile([[1.0, 0.0]], (4, 1)), np.tile([[0.0, 1.0]], (4, 1))]
    ).astype(np.float32)
    instance_ids = np.array([1] * 4 + [2] * 4, dtype=np.int32)
    result = evaluate_point_queries(
        xyz,
        features,
        instance_ids,
        {1: {"label": "a"}, 2: {"label": "b"}},
        random_seed=42,
        min_instance_points=1,
        max_instances=None,
        propagation=SupportPropagationConfig(
            neighbors=2, spatial_sigma=0.05, feature_temperature=0.1, iterations=2
        ),
        threshold=0.0,
        component_radius=0.05,
    )
    assert result["num_queries"] == 2
    assert result["macro_connected_iou"] == 1.0
