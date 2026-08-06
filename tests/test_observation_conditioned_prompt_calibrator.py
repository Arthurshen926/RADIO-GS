from __future__ import annotations

import torch

from radio_gs.querying.observation_conditioned_prompt_calibrator import (
    PROMPT_MODES,
    ObservationConditionedPromptCalibratorV4,
    prompt_case_control_logistic_loss,
)


def test_v4_is_four_scalar_and_strictly_monotone_per_observation_mode() -> None:
    calibrator = ObservationConditionedPromptCalibratorV4().double()
    assert sum(parameter.numel() for parameter in calibrator.parameters()) == 4
    values = torch.linspace(0, 1, 1001, dtype=torch.float64)
    for mode in PROMPT_MODES:
        transformed = calibrator.calibrated_logit(values, mode=mode)
        assert bool(torch.isfinite(transformed).all())
        assert bool((transformed[1:] > transformed[:-1]).all())


def test_v4_state_has_no_scene_or_target_parameter() -> None:
    names = tuple(ObservationConditionedPromptCalibratorV4().state_dict())
    assert names == (
        "calibrators.full_mask.raw_temperature",
        "calibrators.full_mask.bias",
        "calibrators.scribble.raw_temperature",
        "calibrators.scribble.bias",
    )


def test_case_control_loss_is_equal_class_and_equal_prompt() -> None:
    first_logits = torch.tensor([2.0, -1.0, -3.0], dtype=torch.float64)
    first_labels = torch.tensor([True, False, False])
    second_logits = torch.tensor([0.2, 0.4, -0.7, -0.8], dtype=torch.float64)
    second_labels = torch.tensor([True, True, False, False])
    loss = prompt_case_control_logistic_loss(
        [(first_logits, first_labels), (second_logits, second_labels)]
    )
    first = 0.5 * torch.nn.functional.softplus(-first_logits[:1]).mean()
    first = first + 0.5 * torch.nn.functional.softplus(first_logits[1:]).mean()
    second = 0.5 * torch.nn.functional.softplus(-second_logits[:2]).mean()
    second = second + 0.5 * torch.nn.functional.softplus(second_logits[2:]).mean()
    assert torch.equal(loss, 0.5 * (first + second))
