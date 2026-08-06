from __future__ import annotations

import math

import pytest
import torch

from radio_gs.field.factorized_radio_contract import (
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES,
    FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256,
    reliability_scalar_names_sha256,
)
from radio_gs.training.factorized_radio_loss import (
    FACTORIZED_RADIO_RELIABILITY_POLICY_LEGACY,
    FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE,
    factorized_radio_reconstruction_loss,
)


def _inputs():
    target = torch.tensor([[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
    log_amplitude = torch.tensor([math.log(3.0), math.log(4.0), 0.0])
    valid = torch.tensor([True, True, False])
    reliability = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.5, 1.0],
            [0.8, 0.2, 0.1, 0.75, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return target, log_amplitude, valid, reliability


def _loss(
    predicted,
    target,
    log_amplitude,
    valid,
    reliability,
    *,
    reliability_policy=FACTORIZED_RADIO_RELIABILITY_POLICY_LEGACY,
):
    return factorized_radio_reconstruction_loss(
        predicted,
        target,
        log_amplitude,
        valid,
        reliability,
        reliability_scalar_names=FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES,
        reliability_scalar_names_digest=(
            FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES_SHA256
        ),
        reliability_policy=reliability_policy,
    )


def test_exact_prediction_has_zero_direction_and_amplitude_loss() -> None:
    target, log_amplitude, valid, reliability = _inputs()
    result = _loss(target.clone(), target, log_amplitude, valid, reliability)
    assert result.valid_rows == 2
    assert result.direction.item() == pytest.approx(0.0, abs=1e-7)
    assert result.log_amplitude.item() == pytest.approx(0.0, abs=1e-7)
    assert result.total.item() == pytest.approx(0.0, abs=1e-7)


def test_direction_and_amplitude_are_decoupled() -> None:
    target, log_amplitude, valid, reliability = _inputs()
    scaled = _loss(target * 2.0, target, log_amplitude, valid, reliability)
    assert scaled.direction.item() == pytest.approx(0.0, abs=1e-7)
    assert scaled.log_amplitude.item() > 0.0

    rotated = target.clone()
    rotated[0] = torch.tensor([0.0, 3.0])
    direction_changed = _loss(rotated, target, log_amplitude, valid, reliability)
    assert direction_changed.direction.item() > 0.0
    assert direction_changed.log_amplitude.item() == pytest.approx(0.0, abs=1e-7)


def test_invalid_rows_do_not_affect_loss() -> None:
    target, log_amplitude, valid, reliability = _inputs()
    baseline = _loss(target.clone(), target, log_amplitude, valid, reliability)
    changed = target.clone()
    changed[2] = torch.tensor([1e6, -1e6])
    result = _loss(changed, target, log_amplitude, valid, reliability)
    torch.testing.assert_close(result.total, baseline.total, atol=0, rtol=0)


def test_legacy_policy_does_not_interpret_purity_sentinel() -> None:
    target, log_amplitude, valid, reliability = _inputs()
    predicted = target.clone()
    predicted[0] = torch.tensor([0.0, 3.0])
    baseline = _loss(predicted, target, log_amplitude, valid, reliability)
    changed = reliability.clone()
    changed[0, 4] = 0.0
    changed[1, 4] = 0.25
    actual = _loss(predicted, target, log_amplitude, valid, changed)
    torch.testing.assert_close(actual.total, baseline.total, atol=0, rtol=0)


def test_exact_marginal_visibility_safe_policy_modulates_but_retains_row() -> None:
    target, log_amplitude, valid, reliability = _inputs()
    predicted = target.clone()
    predicted[0] = torch.tensor([0.0, 3.0])
    reliability[0, 4] = 0.0
    reliability[1, 4] = 1.0
    legacy = _loss(predicted, target, log_amplitude, valid, reliability)
    visibility_safe = _loss(
        predicted,
        target,
        log_amplitude,
        valid,
        reliability,
        reliability_policy=(
            FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE
        ),
    )
    assert 0.0 < float(visibility_safe.direction) < float(legacy.direction)


def test_exact_marginal_visibility_safe_policy_rejects_invalid_purity() -> None:
    target, log_amplitude, valid, reliability = _inputs()
    reliability[0, 4] = 1.01
    with pytest.raises(ValueError, match="uniform-half confidence"):
        _loss(
            target,
            target,
            log_amplitude,
            valid,
            reliability,
            reliability_policy=(
                FACTORIZED_RADIO_RELIABILITY_POLICY_MATCHED_EXACT_MARGINAL_VISIBILITY_SAFE
            ),
        )


def test_zero_prediction_has_finite_loss_and_gradient() -> None:
    target, log_amplitude, valid, reliability = _inputs()
    predicted = torch.zeros_like(target, requires_grad=True)
    result = _loss(predicted, target, log_amplitude, valid, reliability)
    result.total.backward()
    assert bool(torch.isfinite(result.total))
    assert predicted.grad is not None
    assert bool(torch.isfinite(predicted.grad).all())


def test_scalar_schema_mismatch_fails_closed() -> None:
    target, log_amplitude, valid, reliability = _inputs()
    names = list(FACTORIZED_RADIO_RELIABILITY_SCALAR_NAMES)
    names[0] = "agreement"
    with pytest.raises(ValueError, match="schema differs"):
        factorized_radio_reconstruction_loss(
            target,
            target,
            log_amplitude,
            valid,
            reliability,
            reliability_scalar_names=names,
            reliability_scalar_names_digest=reliability_scalar_names_sha256(names),
        )
