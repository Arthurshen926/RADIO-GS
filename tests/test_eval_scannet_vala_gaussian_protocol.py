import numpy as np
import pytest
import torch

from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import (
    assign_vala_pseudo_labels,
    build_official_sam_region_graph,
    load_direct_language_score_cache,
    load_method_v1_external_query_features,
    propagate_categorical_identity_over_instance_topology,
    smooth_categorical_scores_with_region_graph,
    volume_weighted_split_metrics,
)
from radio_gs.utils.immutable_artifacts import sha256_file


def test_vala_pseudo_label_density_vote_uses_anisotropic_distance():
    gaussian_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    gaussian_scales = np.array([[1.0, 0.1, 0.1]], dtype=np.float32)
    gaussian_rotations = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    point_xyz = np.array([[0.5, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32)
    point_labels = np.array([1, 2], dtype=np.int32)

    labels, stats = assign_vala_pseudo_labels(
        gaussian_xyz,
        gaussian_scales,
        gaussian_rotations,
        point_xyz,
        point_labels,
        radius_factor=5.0,
        candidate_k=2,
        fallback_k=0,
        class_balance=True,
        chunk_size=1,
    )

    assert labels.tolist() == [1]
    assert stats["empty_before_fallback"] == 0


def test_vala_pseudo_label_fallback_handles_empty_radius():
    labels, stats = assign_vala_pseudo_labels(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.01, 0.01, 0.01]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([7], dtype=np.int32),
        radius_factor=1.0,
        candidate_k=1,
        fallback_k=1,
        chunk_size=1,
    )

    assert labels.tolist() == [7]
    assert stats["empty_before_fallback"] == 1


def test_volume_weighted_metrics_ignore_non_target_ground_truth():
    metrics = volume_weighted_split_metrics(
        pseudo_gt_raw=np.array([1, 1, 2, 40], dtype=np.int32),
        pred_raw=np.array([1, 2, 2, 1], dtype=np.int32),
        significance=np.array([1.0, 3.0, 2.0, 100.0], dtype=np.float32),
        split_ids=[1, 2],
    )

    # Class 1: intersection=1, union=4. Class 2: intersection=2, union=5.
    assert np.isclose(metrics["miou"], (1.0 / 4.0 + 2.0 / 5.0) / 2.0)
    # Class 1 accuracy=1/4; class 2 accuracy=1.
    assert np.isclose(metrics["macc"], (1.0 / 4.0 + 1.0) / 2.0)
    assert metrics["num_valid_gaussians"] == 3


def test_method_v1_external_query_features_are_sha_and_geometry_bound(tmp_path):
    path = tmp_path / "primitive_query_method_v1.pth"
    xyz = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    torch.save(
        {
            "xyz": xyz,
            "summary_features": torch.tensor(
                [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=torch.float16
            ),
            "valid": torch.ones(2, dtype=torch.bool),
            "metadata": {
                "artifact_type": "radio_gs_method_v1_primitive_query_cache",
                "method_id": "radio-gs-method-v1",
                "query_independent": True,
                "postprocessing": "none",
                "construction": "test",
            },
        },
        path,
    )

    features, record = load_method_v1_external_query_features(
        path,
        expected_sha256=sha256_file(path),
        expected_xyz=xyz,
        expected_dim=3,
    )

    assert torch.allclose(features.norm(dim=-1), torch.ones(2))
    assert record["method_id"] == "radio-gs-method-v1"
    with pytest.raises(ValueError, match="xyz mismatch"):
        load_method_v1_external_query_features(
            path,
            expected_sha256=sha256_file(path),
            expected_xyz=xyz + 1.0,
            expected_dim=3,
        )


def test_direct_language_score_cache_is_explicitly_label_open_and_geometry_bound(
    tmp_path,
):
    path = tmp_path / "direct_language_scores.pt"
    xyz = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    torch.save(
        {
            "schema": "radio_gs.scannet_direct_language_score_cache.v1",
            "xyz": xyz,
            "valid": torch.ones(2, dtype=torch.bool),
            "direct_observed": torch.tensor([True, False]),
            "scores_split_19": torch.zeros(2, 19),
            "scores_split_15": torch.zeros(2, 15),
            "scores_split_10": torch.zeros(2, 10),
            "metadata": {
                "artifact_type": "radio_gs_scannet_direct_language_score_cache",
                "construction": "test",
                "query_independent": False,
                "evaluation_diagnostic_only": True,
                "benchmark_masks_opened": False,
                "benchmark_labels_opened": True,
                "text_queries_opened": True,
                "postprocessing": "none",
            },
        },
        path,
    )

    scores, record = load_direct_language_score_cache(
        path,
        expected_sha256=sha256_file(path),
        expected_xyz=xyz,
        split_names=["19", "15", "10"],
    )

    assert scores["19"].shape == (2, 19)
    assert record["evaluation_diagnostic_only"] is True
    assert record["direct_observed_rows"] == 1
    with pytest.raises(ValueError, match="xyz mismatch"):
        load_direct_language_score_cache(
            path,
            expected_sha256=sha256_file(path),
            expected_xyz=xyz + 1.0,
            split_names=["19", "15", "10"],
        )


def test_sam_region_residual_changes_only_low_margin_rows():
    scores = torch.tensor(
        [[0.51, 0.49], [0.90, 0.10], [0.10, 0.90]], dtype=torch.float32
    )
    neighbors = torch.tensor([[1], [2], [1]])
    weights = torch.ones(3, 1)

    refined, stats = smooth_categorical_scores_with_region_graph(
        scores,
        neighbors,
        weights,
        alpha=1.0,
        margin_threshold=0.05,
        device=torch.device("cpu"),
        chunk_size=2,
    )

    assert torch.allclose(refined[0], scores[1])
    assert torch.allclose(refined[1:], scores[1:])
    assert stats["changed_rows"] == 1


def test_sam_region_graph_rejects_spatial_or_feature_boundary_edges():
    xyz = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]])
    features = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    indices, weights, stats = build_official_sam_region_graph(
        xyz,
        features,
        torch.ones(3, dtype=torch.bool),
        k=2,
        radius=0.1,
        similarity_threshold=0.5,
        device=torch.device("cpu"),
        chunk_size=2,
    )

    assert indices.shape == weights.shape
    assert int((weights > 0).sum()) == 0
    assert stats["edge_count"] == 0


def test_sam_instance_topology_preserves_markers_and_extends_identity():
    scores = torch.tensor(
        [[0.90, 0.10], [0.505, 0.500], [0.10, 0.90]], dtype=torch.float32
    )
    neighbors = torch.tensor([[1], [0], [1]])
    weights = torch.ones(3, 1)

    refined, stats = propagate_categorical_identity_over_instance_topology(
        scores,
        neighbors,
        weights,
        seed_margin_threshold=0.05,
        update_margin_threshold=0.01,
        semantic_tolerance=0.01,
        consensus_threshold=0.9,
        iterations=1,
    )

    assert refined[0].argmax().item() == 0
    assert refined[1].argmax().item() == 0
    assert refined[2].argmax().item() == 1
    assert torch.equal(refined[[0, 2]], scores[[0, 2]])
    assert stats["seed_rows"] == 2


def test_sam_instance_topology_rejects_semantically_implausible_marker():
    scores = torch.tensor([[0.90, 0.10], [0.40, 0.50]], dtype=torch.float32)
    refined, stats = propagate_categorical_identity_over_instance_topology(
        scores,
        torch.tensor([[1], [0]]),
        torch.ones(2, 1),
        seed_margin_threshold=0.05,
        update_margin_threshold=0.2,
        semantic_tolerance=0.01,
        consensus_threshold=0.9,
        iterations=1,
    )

    assert torch.equal(refined, scores)
    assert stats["changed_rows"] == 0
