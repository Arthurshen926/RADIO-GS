import pytest
import torch

from radio_gs.scripts.audit_vpr_cache_alignment import (
    audit_vpr_cache_payload_alignment,
    compute_xyz_alignment_stats,
    xyz_sha256,
)
from radio_gs.scripts.train_feature_field import (
    audit_direct_point_teacher_cache_alignment_for_training,
)


def test_compute_xyz_alignment_stats_reports_l2_and_scale():
    model_xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [6.0, 8.0, 0.0],
        ],
        dtype=torch.float32,
    )
    cache_xyz = model_xyz.clone()
    cache_xyz[1, 2] += 0.001

    stats = compute_xyz_alignment_stats(cache_xyz, model_xyz)

    assert stats["count_match"] is True
    assert stats["cache_count"] == 3
    assert stats["model_count"] == 3
    assert stats["max_l2"] == pytest.approx(0.001, abs=1e-8)
    assert stats["mean_l2"] == pytest.approx(0.001 / 3.0, abs=1e-8)
    assert stats["p95_l2"] == pytest.approx(0.0009, abs=1e-8)
    assert stats["scene_scale"] == pytest.approx(10.0)
    assert stats["normalized_max_l2"] == pytest.approx(0.0001, abs=1e-8)


def test_xyz_sha256_is_stable_for_clone_and_sensitive_to_order():
    xyz = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=torch.float32,
    )
    non_contiguous = xyz.t().t()

    assert xyz_sha256(xyz) == xyz_sha256(xyz.clone())
    assert xyz_sha256(xyz) == xyz_sha256(non_contiguous)
    assert xyz_sha256(xyz) != xyz_sha256(xyz.flip(0))


def test_audit_vpr_cache_payload_alignment_reports_missing_xyz():
    model_xyz = torch.zeros(2, 3)
    payload = {
        "summary_features": torch.zeros(2, 4),
        "valid": torch.tensor([True, False]),
        "view_counts": torch.tensor([1.0, 0.0]),
        "metadata": {"scene": "toy"},
    }

    report = audit_vpr_cache_payload_alignment(
        payload,
        model_xyz,
        fail_max_l2=1e-5,
        cache_path="toy.pt",
    )

    assert report["status"] == "missing_xyz"
    assert report["passed"] is False
    assert report["cache_path"] == "toy.pt"
    assert report["feature_key"] == "summary_features"
    assert "xyz" in report["message"]


def test_audit_vpr_cache_payload_alignment_threshold_pass_and_fail():
    model_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cache_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0e-5]])
    payload = {
        "xyz": cache_xyz,
        "summary_features": torch.zeros(2, 4),
        "valid": torch.tensor([True, True]),
        "view_counts": torch.tensor([1.0, 2.0]),
        "metadata": {"scene": "toy"},
    }

    passed = audit_vpr_cache_payload_alignment(
        payload,
        model_xyz,
        fail_max_l2=3.0e-5,
    )
    failed = audit_vpr_cache_payload_alignment(
        payload,
        model_xyz,
        fail_max_l2=1.0e-5,
    )

    assert passed["status"] == "passed"
    assert passed["passed"] is True
    assert failed["status"] == "failed"
    assert failed["passed"] is False
    assert failed["max_l2"] == pytest.approx(2.0e-5, abs=1e-9)
    assert failed["xyz_sha256_match"] is False


def test_training_direct_point_cache_alignment_raises_for_row_mismatch():
    model_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    payload = {
        "xyz": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0e-5]]),
        "features": torch.zeros(2, 4),
        "valid": torch.tensor([True, True]),
    }

    with pytest.raises(RuntimeError, match="row-alignment audit failed"):
        audit_direct_point_teacher_cache_alignment_for_training(
            payload,
            model_xyz,
            direct_point_source="gaussian",
            direct_point_query_mode="gaussian_index",
            cache_path="toy.pt",
            fail_max_l2=1.0e-5,
        )


def test_training_direct_point_cache_alignment_skips_non_gaussian_pool():
    report = audit_direct_point_teacher_cache_alignment_for_training(
        {"features": torch.zeros(2, 4)},
        torch.zeros(2, 3),
        direct_point_source="label_ply",
        direct_point_query_mode="gaussian_index",
        cache_path="toy.pt",
    )

    assert report["status"] == "skipped"
    assert report["passed"] is True
