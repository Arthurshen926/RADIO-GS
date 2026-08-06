"""Calibration conditioned only on the known source observation mechanism."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from radio_gs.querying.prompt_mode_logit_calibrator import (
    PROMPT_MODES,
    PromptModeLogitCalibratorV3,
)


class ObservationConditionedPromptCalibratorV4(PromptModeLogitCalibratorV3):
    """Four-scalar monotone calibrator with observation-aware fitting.

    Architecture is intentionally identical to V3: one positive temperature
    and one bias for each source-known prompt mode.  V4 differs only in the
    registered estimator used to fit the scribble branch; no scene or target
    information enters this module.
    """


def prompt_case_control_logistic_loss(
    logits_and_labels: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Equal-class logistic risk per prompt, then equal-prompt averaging.

    This estimates a class-symmetric foreground likelihood score from
    case-control/censored scribble observations.  Prompt pixel count and
    observed foreground prevalence therefore cannot impose a deployment
    class prior.
    """

    if not logits_and_labels:
        raise ValueError("case-control loss requires at least one prompt")
    prompt_losses: list[torch.Tensor] = []
    for logits, labels in logits_and_labels:
        score = torch.as_tensor(logits).reshape(-1)
        target = torch.as_tensor(labels, device=score.device).reshape(-1).bool()
        if score.shape != target.shape or not bool(torch.isfinite(score).all()):
            raise ValueError("case-control prompt tensors differ")
        if not bool(target.any()) or not bool((~target).any()):
            raise ValueError("case-control prompt must contain both classes")
        positive = F.softplus(-score[target]).mean()
        negative = F.softplus(score[~target]).mean()
        prompt_losses.append(0.5 * positive + 0.5 * negative)
    return torch.stack(prompt_losses).mean()


__all__ = [
    "PROMPT_MODES",
    "ObservationConditionedPromptCalibratorV4",
    "prompt_case_control_logistic_loss",
]
