"""Low-capacity query-free calibration of canonical primitive relations."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


RELATION_FEATURE_NAMES = (
    "log_geometry_affinity", "log_appearance_affinity", "log_boundary_affinity",
    "normalized_edge_length", "absolute_log_scale_ratio",
)


def edge_relation_features(payload: dict) -> torch.Tensor:
    edge = torch.as_tensor(payload["edge_index"]).long().cpu()
    xyz = torch.as_tensor(payload["xyz"]).float().cpu()
    sigma = torch.as_tensor(payload["local_sigma"]).float().cpu().clamp_min(1e-6)
    channels = payload.get("edge_channels", {})
    values = []
    for name in ("geometry", "appearance", "boundary"):
        if name not in channels:
            raise ValueError(f"relation graph lacks {name!r} edge channel")
        values.append(torch.as_tensor(channels[name]).float().cpu().clamp_min(1e-12).log())
    src, dst = edge
    length = torch.linalg.vector_norm(xyz[src] - xyz[dst], dim=-1)
    pair_scale = (sigma[src] * sigma[dst]).sqrt().clamp_min(1e-6)
    values.append(length / pair_scale)
    values.append((sigma[src].log() - sigma[dst].log()).abs())
    return torch.stack(values, dim=-1)


class MonotonicRelationCalibrator(nn.Module):
    """A five-weight logistic model with declared monotonicity."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_affinity_weights = nn.Parameter(torch.zeros(3))
        self.raw_penalty_weights = nn.Parameter(torch.zeros(2))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        affinity = F.softplus(self.raw_affinity_weights)
        penalty = F.softplus(self.raw_penalty_weights)
        return self.bias + features[..., :3] @ affinity - features[..., 3:] @ penalty

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        return self(features).sigmoid()

    def architecture(self) -> dict:
        return {"type": "monotonic_logistic_edge_calibrator",
                "feature_names": list(RELATION_FEATURE_NAMES), "parameters": 6}

