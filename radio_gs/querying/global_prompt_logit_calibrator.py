"""Two-parameter global monotone calibration for prompt probabilities."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class GlobalPromptLogitCalibratorV2(nn.Module):
    """Apply one positive temperature and one bias to scalar logits.

    The transformation is strictly increasing for finite input probabilities,
    so it cannot change AP, AUROC, oracle-IoU ordering, or ties.  It is intended
    for the rendered supported-pixel probability after the frozen graph-off
    primitive unary and exact scalar renderer.
    """

    def __init__(
        self,
        *,
        initial_temperature: float = 1.0,
        initial_bias: float = 0.0,
        minimum_temperature: float = 1e-4,
        probability_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if (
            not math.isfinite(float(initial_temperature))
            or initial_temperature <= minimum_temperature
            or not math.isfinite(float(initial_bias))
            or not 0 < minimum_temperature < 1
            or not 0 < probability_eps < 0.5
        ):
            raise ValueError("invalid global prompt calibration initialization")
        self.minimum_temperature = float(minimum_temperature)
        self.probability_eps = float(probability_eps)
        target = torch.tensor(float(initial_temperature - minimum_temperature))
        raw = torch.log(torch.expm1(target))
        self.raw_temperature = nn.Parameter(raw)
        self.bias = nn.Parameter(torch.tensor(float(initial_bias)))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature) + self.minimum_temperature

    def strict_domain_audit(self, probability: torch.Tensor) -> dict[str, int]:
        """Audit the closed input domain handled by strict interiorization."""

        value = torch.as_tensor(probability)
        if not value.is_floating_point():
            value = value.float()
        if not bool(torch.isfinite(value).all()) or bool(
            ((value < 0) | (value > 1)).any()
        ):
            raise ValueError("prompt probability must be finite in [0,1]")
        return {
            "count": int(value.numel()),
            "at_zero": int((value == 0).sum()),
            "at_one": int((value == 1).sum()),
            "at_or_below_probability_eps": int(
                (value <= self.probability_eps).sum()
            ),
            "at_or_above_one_minus_probability_eps": int(
                (value >= 1.0 - self.probability_eps).sum()
            ),
        }

    def forward(self, probability: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.calibrated_logit(probability))

    def calibrated_logit(self, probability: torch.Tensor) -> torch.Tensor:
        """Return the unsaturated monotone score used for fitting/ranking."""

        value = torch.as_tensor(probability, device=self.raw_temperature.device)
        if not value.is_floating_point():
            value = value.float()
        value = value.to(dtype=self.raw_temperature.dtype)
        if not bool(torch.isfinite(value).all()) or bool(
            ((value < 0) | (value > 1)).any()
        ):
            raise ValueError("prompt probability must be finite in [0,1]")
        interior = self.probability_eps + (
            1.0 - 2.0 * self.probability_eps
        ) * value
        logit = torch.logit(interior)
        return logit / self.temperature + self.bias
