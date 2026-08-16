from __future__ import annotations

import torch

from radio_gs.scripts.train_source_categorical_posterior_v2 import (
    _fold_gate,
    soft_targets,
    weighted_metrics,
)


def test_source_categorical_soft_targets_add_background_mass() -> None:
    payload = {
        "semantic_class_distribution": torch.tensor(
            [[0.7, 0.0], [0.0, 0.0], [0.2, 0.8]]
        )
    }
    target = soft_targets(payload)
    assert torch.allclose(target.sum(dim=-1), torch.ones(3))
    assert torch.allclose(target[:, -1], torch.tensor([0.3, 1.0, 0.0]))


def test_source_categorical_weighted_metrics_excludes_background_from_macro() -> None:
    target = torch.eye(3)
    result = weighted_metrics(
        torch.tensor([0, 1, -1]), target, torch.ones(3)
    )
    assert result["miou"] == 1.0
    assert result["macc"] == 1.0


def test_source_categorical_fold_gate_rejects_large_regression() -> None:
    assert _fold_gate({"miou_delta": -0.009, "macc_delta": 0.0})["all_passed"]
    assert not _fold_gate({"miou_delta": -0.011, "macc_delta": 0.0})["all_passed"]
