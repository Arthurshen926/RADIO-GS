from __future__ import annotations

import torch

from radio_gs.scripts.fit_global_prompt_logit_calibrator_clean_pilot import (
    _ranking_and_probability_metrics,
    _strict_ranking_audit,
)


def test_metrics_use_monotone_logit_for_ranking_and_probability_for_threshold() -> None:
    probability = torch.tensor([0.1, 0.4, 0.6, 0.9])
    label = torch.tensor([False, True, False, True])
    base = _ranking_and_probability_metrics(
        ranking_score=probability,
        probability=probability,
        label=label,
    )
    monotone_logit = torch.logit(probability) / 2.0 - 0.7
    calibrated_probability = torch.sigmoid(monotone_logit)
    calibrated = _ranking_and_probability_metrics(
        ranking_score=monotone_logit,
        probability=calibrated_probability,
        label=label,
    )
    assert calibrated["average_precision"] == base["average_precision"]
    assert calibrated["auroc"] == base["auroc"]
    assert calibrated["oracle_iou"] == base["oracle_iou"]
    assert calibrated["area_ratio"] != base["area_ratio"]


def test_strict_ranking_audit_checks_order_and_tie_partition() -> None:
    raw = torch.tensor([0.0, 0.0, 0.2, 0.7, 1.0], dtype=torch.float64)
    transformed = torch.logit(1e-6 + (1 - 2e-6) * raw) / 2.0 - 0.5
    audit = _strict_ranking_audit(raw, transformed)
    assert audit["stable_argsort_equal"] is True
    assert audit["tie_partition_equal"] is True
