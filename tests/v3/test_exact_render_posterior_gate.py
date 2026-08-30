import torch

from radio_gs.v3.evaluation.aggregate_exact_render_posterior_gate import (
    _gate_decision,
)
from radio_gs.v3.evaluation.evaluate_exact_render_posterior import _mask_metrics


def test_mask_metrics_handles_positive_and_empty_targets():
    positive = _mask_metrics(
        torch.tensor([0.9, 0.2, 0.8]),
        torch.tensor([1, 0, 1]),
        torch.ones(3, dtype=torch.bool),
        threshold=0.5,
    )
    empty = _mask_metrics(
        torch.tensor([0.1, 0.2]),
        torch.zeros(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        threshold=0.5,
    )
    assert positive["mask_iou"] == 1.0
    assert positive["brier"] < 0.05
    assert empty["mask_iou"] == 0.0
    assert empty["foreground_fraction"] == 0.0


def test_gate_is_directional_and_uses_relaxed_iou_tolerance():
    reports = [{
        "scene": name,
        "identity_bitwise_preserved": True,
        "positive": {"calibrated": {}},
        "empty": {"calibrated": {"foreground_probability": 0.2}},
    } for name in ("a", "b", "c", "d")]
    positive = {
        "uncalibrated": {"mask_iou": 0.4, "brier": 0.25},
        "calibrated": {"mask_iou": 0.391, "brier": 0.20},
    }
    empty = {
        "uncalibrated": {"foreground_probability": 0.4, "foreground_fraction": 0.3},
        "calibrated": {"foreground_probability": 0.2, "foreground_fraction": 0.2},
    }
    passed, failures = _gate_decision(reports, positive, empty)
    assert passed
    assert not failures


def test_gate_accepts_one_scene_without_positive_audit_authority():
    reports = [{
        "scene": name,
        "identity_bitwise_preserved": True,
        "positive": {"calibrated": {}},
        "empty": {"calibrated": {"foreground_probability": 0.2}},
    } for name in ("a", "b", "c", "d")]
    reports[-1]["positive"]["calibrated"] = None
    positive = {
        "uncalibrated": {"mask_iou": 0.4, "brier": 0.25},
        "calibrated": {"mask_iou": 0.5, "brier": 0.20},
    }
    empty = {
        "uncalibrated": {"foreground_probability": 0.4, "foreground_fraction": 0.3},
        "calibrated": {"foreground_probability": 0.2, "foreground_fraction": 0.2},
    }
    passed, failures = _gate_decision(reports, positive, empty)
    assert passed
    assert not failures
