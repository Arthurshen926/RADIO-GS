"""Prompt-interface-conditioned monotone calibration for frozen V1 scores."""

from __future__ import annotations

import torch
from torch import nn

from radio_gs.querying.global_prompt_logit_calibrator import (
    GlobalPromptLogitCalibratorV2,
)


PROMPT_MODES: tuple[str, str] = ("full_mask", "scribble")


class PromptModeLogitCalibratorV3(nn.Module):
    """Use one globally shared temperature/bias pair per prompt interface.

    Prompt mode is known from the source interaction itself.  No scene ID,
    target view, target image, or target label is an input.  Each branch is a
    strict monotone scalar transformation and therefore preserves the frozen
    V1 ordering within its prompt interface.
    """

    def __init__(self, *, probability_eps: float = 1e-6) -> None:
        super().__init__()
        self.calibrators = nn.ModuleDict(
            {
                mode: GlobalPromptLogitCalibratorV2(
                    probability_eps=probability_eps
                )
                for mode in PROMPT_MODES
            }
        )

    def _branch(self, mode: str) -> GlobalPromptLogitCalibratorV2:
        if mode not in PROMPT_MODES:
            raise ValueError(f"unsupported prompt mode: {mode}")
        return self.calibrators[mode]

    def calibrated_logit(
        self, probability: torch.Tensor, *, mode: str
    ) -> torch.Tensor:
        return self._branch(mode).calibrated_logit(probability)

    def forward(self, probability: torch.Tensor, *, mode: str) -> torch.Tensor:
        return torch.sigmoid(self.calibrated_logit(probability, mode=mode))

    def strict_domain_audit(
        self, probability: torch.Tensor, *, mode: str
    ) -> dict[str, int]:
        return self._branch(mode).strict_domain_audit(probability)

    def temperature(self, mode: str) -> torch.Tensor:
        return self._branch(mode).temperature

    def bias(self, mode: str) -> torch.Tensor:
        return self._branch(mode).bias
