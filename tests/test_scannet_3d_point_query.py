import json

import numpy as np
from scipy.spatial import cKDTree

from radio_gs.querying.unified_query import SupportPropagationConfig
from radio_gs.scripts.eval_scannet_3d_point_query import (
    _local_multiscale_positive_prototypes,
    choose_official_sam3_point_mask,
    evaluate_point_queries,
    load_scannet_instances,
    select_depth_visible_views,
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


def test_depth_visible_view_selection_rejects_occlusion_and_ranks_center() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    # Project the same z=2 point to the center in all views.  View 1 is
    # occluded; view 2 has a small depth error and therefore follows view 0.
    depth = np.full((3, 5, 5), 2.0, dtype=np.float32)
    depth[1] = 1.0
    depth[2] = 2.02
    alpha = np.ones_like(depth)
    selected = select_depth_visible_views(
        np.array([0.0, 0.0, 2.0], dtype=np.float32),
        poses,
        np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]),
        depth,
        alpha,
        max_views=3,
        depth_tolerance=0.05,
        relative_depth_tolerance=0.0,
        alpha_threshold=0.1,
    )
    assert [row["view_index"] for row in selected] == [0, 2]


def test_official_sam3_point_mask_uses_predicted_quality() -> None:
    masks = np.zeros((3, 4, 4), dtype=bool)
    masks[1, 1:3, 1:3] = True
    selected, index, score = choose_official_sam3_point_mask(
        masks, np.array([0.1, 0.8, 0.2], dtype=np.float32)
    )
    assert index == 1
    assert score == np.float32(0.8)
    np.testing.assert_array_equal(selected, masks[1])
