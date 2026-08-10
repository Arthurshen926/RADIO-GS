from __future__ import annotations

import torch

from radio_gs.scripts import (
    materialize_lerf_gaussian_valid_domain_knn_candidate as candidate,
)


def test_build_candidate_is_finite_bounded_and_invalid_zero() -> None:
    positive = torch.tensor(
        [
            [[0.8], [0.7], [0.6]],
            [[0.0], [0.0], [0.0]],
            [[0.2], [0.3], [0.4]],
        ]
    )
    negative = torch.zeros(3, 3, 4)
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    valid = torch.tensor([True, False, True])
    result = candidate.build_candidate(
        positive, negative, xyz, valid, chunk_size=1
    )
    assert result.scores.shape == (3, 1)
    assert torch.isfinite(result.scores).all()
    assert bool((result.scores >= 0).all())
    assert bool((result.scores <= 1).all())
    assert torch.equal(result.scores[~valid], torch.zeros(1, 1))


def test_deployment_is_exactly_source_selected_gaussian_and_metric_closed() -> None:
    assert candidate.SELECTED_POLICY_ID == "gaussian"
    audit = candidate.access_audit()
    assert audit["source_only_policy_gate_opened"] is True
    assert audit["reliability_sidecar_opened"] is False
    assert audit["benchmark_masks_or_labels_opened"] is False
    assert audit["target_metrics_computed"] is False
    assert audit["gpu_used"] is False
    assert audit["result_dependent_parameters"] is False
