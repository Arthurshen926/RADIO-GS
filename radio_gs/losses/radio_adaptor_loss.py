"""Frozen RADIO adaptor consistency losses."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from radio_gs.models.radio_adaptors import project_feature_map_with_adaptor


def compute_radio_adaptor_alignment_loss(
    decoded: torch.Tensor,
    target: torch.Tensor,
    adaptors: Mapping[str, nn.Module],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match decoded and teacher RADIO features in frozen adaptor spaces.

    Returns an unweighted mean cosine-distance loss and per-adaptor scalar
    losses.  Callers are responsible for multiplying the configured weight.
    """
    if not adaptors:
        return decoded.sum() * 0.0, {}

    losses: dict[str, torch.Tensor] = {}
    for name, adaptor in adaptors.items():
        pred = project_feature_map_with_adaptor(decoded, adaptor)
        with torch.no_grad():
            ref = project_feature_map_with_adaptor(target, adaptor)
        losses[name] = 1.0 - (pred * ref).sum(dim=1).mean()

    total = torch.stack(list(losses.values())).mean()
    return total, losses

