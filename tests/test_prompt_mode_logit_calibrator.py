from __future__ import annotations

import torch

from radio_gs.querying.prompt_mode_logit_calibrator import (
    PROMPT_MODES,
    PromptModeLogitCalibratorV3,
)


def test_v3_has_exactly_four_trainable_scalars() -> None:
    calibrator = PromptModeLogitCalibratorV3()
    assert sum(parameter.numel() for parameter in calibrator.parameters()) == 4
    assert tuple(calibrator.calibrators) == PROMPT_MODES


def test_each_prompt_mode_is_strictly_monotone_and_independent() -> None:
    calibrator = PromptModeLogitCalibratorV3().double()
    with torch.no_grad():
        calibrator.calibrators["full_mask"].raw_temperature.fill_(-0.7)
        calibrator.calibrators["full_mask"].bias.fill_(0.4)
        calibrator.calibrators["scribble"].raw_temperature.fill_(0.8)
        calibrator.calibrators["scribble"].bias.fill_(-1.3)
    values = torch.linspace(0, 1, 1001, dtype=torch.float64)
    full = calibrator.calibrated_logit(values, mode="full_mask")
    scribble = calibrator.calibrated_logit(values, mode="scribble")
    assert bool((full[1:] > full[:-1]).all())
    assert bool((scribble[1:] > scribble[:-1]).all())
    assert not torch.equal(full, scribble)


def test_unknown_prompt_mode_fails_closed() -> None:
    calibrator = PromptModeLogitCalibratorV3()
    try:
        calibrator(torch.tensor([0.5]), mode="point")
    except ValueError as error:
        assert "unsupported prompt mode" in str(error)
    else:
        raise AssertionError("unknown prompt mode was accepted")
