"""Fixed v3 phase controls and the optional low-rank latent residual."""

from __future__ import annotations

from enum import IntEnum

import torch
from torch import nn


class TrainingPhase(IntEnum):
    HEADS_ONLY = 1
    LOW_RANK_RESIDUAL = 2
    ALTERNATING = 3
    CONTROLLED_REFINEMENT = 4


class LowRankLatentResidual(nn.Module):
    """A rank-bounded update inside D512, never an additional deployed field."""

    def __init__(self, latent_dim: int = 512, rank: int = 8) -> None:
        super().__init__()
        if latent_dim != 512 or not 0 < rank < latent_dim:
            raise ValueError("v3 latent residual must be low-rank inside D512")
        self.down = nn.Linear(latent_dim, rank, bias=False)
        self.up = nn.Linear(rank, latent_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(latent).float()
        if value.ndim != 2 or value.shape[1] != 512:
            raise ValueError("low-rank residual requires [N,512]")
        return value + self.up(self.down(value))


def phase_step_order(phase: TrainingPhase) -> tuple[str, ...]:
    if phase in (TrainingPhase.HEADS_ONLY, TrainingPhase.LOW_RANK_RESIDUAL):
        return ("instance", "heldout_validation")
    if phase == TrainingPhase.ALTERNATING:
        return ("visual", "instance", "boundary", "heldout_validation")
    if phase == TrainingPhase.CONTROLLED_REFINEMENT:
        return ("membership", "source_heldout_render", "frozen_sam_refine", "multiview_audit", "instance")
    raise ValueError("unsupported v3 training phase")


__all__ = ["LowRankLatentResidual", "TrainingPhase", "phase_step_order"]
