from __future__ import annotations

import torch

from radio_gs.scripts.train_source_marginal_categorical_posterior_v2 import (
    _fold_gate,
    foreground_targets,
    weighted_foreground_metrics,
)


def test_foreground_targets_ignore_rows_without_evaluation_class_mass() -> None:
    target, mass = foreground_targets(
        {"semantic_class_distribution": torch.tensor([[0.2, 0.3], [0.0, 0.0]])}
    )
    assert torch.allclose(target[0], torch.tensor([0.4, 0.6]))
    assert torch.equal(target[1], torch.zeros(2))
    assert torch.allclose(mass, torch.tensor([0.5, 0.0]))


def test_weighted_foreground_metrics_excludes_zero_weight_rows() -> None:
    result = weighted_foreground_metrics(
        torch.tensor([0, 1, 0]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        torch.tensor([1.0, 1.0, 0.0]),
    )
    assert result["miou"] == 1.0
    assert result["macc"] == 1.0


def test_marginal_categorical_fold_gate_rejects_large_regression() -> None:
    assert _fold_gate({"miou_delta": -0.009, "macc_delta": 0.0})["all_passed"]
    assert not _fold_gate({"miou_delta": -0.011, "macc_delta": 0.0})[
        "all_passed"
    ]
