from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.audit_source_only_graph_expected_utility_fix4 import (
    _residual_amplitude_summary,
    edge_eligible_mask,
)


def test_edge_gate_applies_probability_reliability_and_ood_conjunctively() -> None:
    features = torch.zeros((3, 21), dtype=torch.float32)
    features[:, 17:19] = torch.tensor([[0.9, 0.9], [0.8, 0.9], [0.9, 0.9]])
    features[2, 0] = 2.0
    eligible, audit = edge_eligible_mask(
        pair_features=features,
        raw_probability=torch.tensor([0.95, 0.95, 0.95]),
        median=torch.zeros(21),
        robust_scale=torch.ones(21),
        raw_probability_minimum=0.9,
        reliability_minimum=0.85,
        ood_raw_limit=1.0,
    )
    assert torch.equal(eligible, torch.tensor([True, False, False]))
    assert audit["raw_probability_gate_count"] == 3
    assert audit["all_three_edge_gate_count"] == 1


def test_residual_amplitude_is_recovered_from_consumer_gain_units() -> None:
    trace = {
        "gains": torch.tensor([[0.8, 0.2]]),
        "marginal_primitives": torch.tensor([[256, 128]]),
        "labels": torch.tensor([[True, False]]),
    }
    selected = torch.tensor([[True, True]])
    summary = _residual_amplitude_summary(
        trace=trace, selected=selected, epsilon_logit=0.4
    )
    assert summary["selected_true"]["mean"] == pytest.approx(0.32)
    assert summary["selected_false"]["mean"] == pytest.approx(0.16)
    assert summary["selected"]["maximum"] <= 0.4

