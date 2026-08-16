from __future__ import annotations

from radio_gs.scripts.train_source_post_spatial_text_posterior import (
    _fold_admissible,
    _macro_gate,
)


def test_post_spatial_source_gate_requires_all_macro_improvements() -> None:
    metrics = {
        "hard_iou_delta": 0.02,
        "balanced_bce_delta": -0.03,
        "positive_negative_gap_delta": 0.04,
        "local_relation_loss_delta": 0.00001,
        "base_macro_local_relation_loss": 0.001,
    }
    assert _macro_gate(metrics)["all_passed"] is True
    metrics["hard_iou_delta"] = 0.0
    assert _macro_gate(metrics)["all_passed"] is False


def test_post_spatial_fold_rejects_large_iou_or_relation_regression() -> None:
    metrics = {
        "hard_iou_delta": -0.009,
        "local_relation_loss_delta": 0.00001,
        "base_macro_local_relation_loss": 0.001,
    }
    assert _fold_admissible(metrics)["all_passed"] is True
    metrics["hard_iou_delta"] = -0.011
    assert _fold_admissible(metrics)["all_passed"] is False
