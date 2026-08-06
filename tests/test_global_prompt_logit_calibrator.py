from __future__ import annotations

import torch

from radio_gs.querying.global_prompt_logit_calibrator import (
    GlobalPromptLogitCalibratorV2,
)


def test_initialization_is_fixed_interiorization_and_positive_temperature() -> None:
    calibrator = GlobalPromptLogitCalibratorV2()
    values = torch.tensor([0.01, 0.2, 0.5, 0.8, 0.99])
    expected = 1e-6 + (1 - 2e-6) * values
    assert torch.allclose(calibrator(values), expected, atol=2e-7, rtol=0)
    assert float(calibrator.temperature) > 0


def test_transform_is_strictly_monotone_for_arbitrary_parameters() -> None:
    calibrator = GlobalPromptLogitCalibratorV2()
    with torch.no_grad():
        calibrator.raw_temperature.fill_(-2.0)
        calibrator.bias.fill_(-1.3)
    values = torch.linspace(0.001, 0.999, 1000)
    calibrated = calibrator.calibrated_logit(values)
    assert bool((calibrated[1:] > calibrated[:-1]).all())
    assert torch.equal(torch.argsort(calibrated), torch.argsort(values))


def test_fit_parameters_receive_gradients_but_inputs_do_not_need_them() -> None:
    calibrator = GlobalPromptLogitCalibratorV2()
    values = torch.tensor([0.1, 0.4, 0.6, 0.9])
    loss = torch.nn.functional.binary_cross_entropy(
        calibrator(values), torch.tensor([0.0, 0.0, 0.0, 1.0])
    )
    loss.backward()
    assert calibrator.raw_temperature.grad is not None
    assert calibrator.bias.grad is not None


def test_strict_domain_audit_accepts_closed_domain_for_interiorization() -> None:
    calibrator = GlobalPromptLogitCalibratorV2()
    audit = calibrator.strict_domain_audit(torch.tensor([0.0, 1e-7, 0.5, 1.0]))
    assert audit["at_zero"] == 1
    assert audit["at_one"] == 1
    calibrated = calibrator.calibrated_logit(
        torch.tensor([0.0, 1e-7, 0.5, 1.0])
    )
    assert bool(torch.isfinite(calibrated).all())
    assert bool((calibrated[1:] > calibrated[:-1]).all())
