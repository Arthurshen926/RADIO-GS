import numpy as np

from radio_gs.scripts.eval_scannet_vala_gaussian_protocol import (
    assign_vala_pseudo_labels,
    volume_weighted_split_metrics,
)


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
