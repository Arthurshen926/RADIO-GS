import numpy as np
import pytest
import torch

from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import (
    assign_vala_pseudo_labels,
    load_method_v1_external_query_features,
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
