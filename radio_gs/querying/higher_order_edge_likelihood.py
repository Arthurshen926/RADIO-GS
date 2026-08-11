"""Small scene-shared edge likelihood over symmetric higher-order features."""

from __future__ import annotations

import torch
import torch.nn as nn


class SymmetricHigherOrderEdgeLikelihood(nn.Module):
    """An endpoint-exchange-invariant MLP over pre-symmetrized pair features."""

    schema_version = "symmetric-higher-order-instance-edge-lr-v10"

    def __init__(self, *, feature_count: int, hidden_width: int = 16) -> None:
        super().__init__()
        feature_count = int(feature_count)
        hidden_width = int(hidden_width)
        if feature_count <= 0 or hidden_width <= 0 or hidden_width > 128:
            raise ValueError("edge feature count/hidden width is invalid")
        self.feature_count = feature_count
        self.hidden_width = hidden_width
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width // 2),
            nn.SiLU(),
            nn.Linear(hidden_width // 2, 1),
        )

    def log_likelihood_ratio(self, symmetric_features: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(symmetric_features).float()
        if (
            values.ndim != 2
            or values.shape[1] != self.feature_count
            or not values.shape[0]
            or not bool(torch.isfinite(values).all())
            or bool(((values < 0) | (values > 1)).any())
        ):
            raise ValueError("higher-order edge features must be finite [E,D] in [0,1]")
        return self.network(values).squeeze(-1)
