from __future__ import annotations

import pytest
import torch

from radio_gs.scripts.calibrate_source_only_graph_confidence_v1 import (
    expit,
    leave_one_scene_out_max_audit,
    logit,
    lower_probability,
    one_sided_wilson_lower,
)


def test_wilson_and_logit_lower_are_finite_and_conservative() -> None:
    lower = one_sided_wilson_lower(90, 100)
    assert 0.0 < lower < 0.9
    epsilon = logit(0.9) - logit(lower)
    assert epsilon > 0.0
    assert expit(logit(0.9) - epsilon) == pytest.approx(lower)
    probabilities = torch.tensor([0.9, 0.99], dtype=torch.float32)
    corrected = lower_probability(probabilities, epsilon)
    assert torch.all(corrected < probabilities)
    assert float(corrected[0]) == pytest.approx(lower, abs=1e-6)


def test_wilson_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="counts"):
        one_sided_wilson_lower(2, 1)
    with pytest.raises(ValueError, match="confidence"):
        one_sided_wilson_lower(1, 1, confidence=0.5)


def test_leave_one_scene_out_max_exposes_unique_worst_scene() -> None:
    rows = {
        "a": {
            "tail_logit_nonconformity": 0.4,
            "one_sided_wilson_lower": expit(logit(0.9) - 0.4),
        },
        "b": {
            "tail_logit_nonconformity": 0.0,
            "one_sided_wilson_lower": 0.95,
        },
        "c": {
            "tail_logit_nonconformity": 0.0,
            "one_sided_wilson_lower": 0.96,
        },
    }
    audit = leave_one_scene_out_max_audit(rows)
    by_scene = {row["heldout_scene_id"]: row for row in audit}
    assert by_scene["a"]["covered"] is False
    assert by_scene["b"]["covered"] is True
    assert by_scene["c"]["covered"] is True
    assert by_scene["b"]["fit_max_epsilon_logit"] == pytest.approx(0.4)


def test_lower_probability_is_monotone() -> None:
    values = torch.linspace(0.01, 0.99, 99)
    corrected = lower_probability(values, 0.4)
    assert bool((corrected[1:] > corrected[:-1]).all())
    assert bool((corrected < values).all())
