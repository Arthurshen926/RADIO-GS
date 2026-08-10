"""Independent Stage-II commit policy over the frozen V2.1C projection solver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from radio_gs.optimization import adamw_two_constraint_projection_v21c as frozen


MAX_CANDIDATE_RECONSTRUCTION_ERROR = 2e-6
MINIMUM_PROJECTED_DOT = -2e-7


def commit(
    optimizer: torch.optim.AdamW,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    gradients: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if tuple(gradients) != frozen.GRADIENT_ORDER:
        raise ValueError("V2.1C Stage-II gradient axis differs")
    evidence = frozen.commit_projected_adamw_step(
        optimizer,
        named_parameters,
        gradients["absolute"],
        gradients["pairwise"],
    )
    if (
        evidence["kkt"]["passed"] is not True
        or float(evidence["projected_dot"]["absolute"])
        < MINIMUM_PROJECTED_DOT
        or float(evidence["projected_dot"]["pairwise"])
        < MINIMUM_PROJECTED_DOT
        or float(evidence["adamw_candidate_reconstruction_max_abs_error"])
        > MAX_CANDIDATE_RECONSTRUCTION_ERROR
        or evidence["adamw_moments_advanced_before_projection"] is not True
        or evidence["decoupled_weight_decay_in_candidate"] is not True
    ):
        raise RuntimeError("V2.1C Stage-II update certificate failed closed")
    return evidence


__all__ = [
    "MAX_CANDIDATE_RECONSTRUCTION_ERROR",
    "MINIMUM_PROJECTED_DOT",
    "commit",
]
