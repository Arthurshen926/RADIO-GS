"""Bounded query-free correction of a RADIO summary token.

The adapter operates before the frozen official SigLIP summary head.  It can
therefore improve text-response fidelity without introducing a replacement
text projection.  Its output preserves the input norm and is confined to a
declared angular trust region around the original summary-token direction.
"""

from __future__ import annotations

import hashlib
import json
import math

import torch
from torch import nn
import torch.nn.functional as F


class LowRankTangentSummaryAdapter(nn.Module):
    """Apply a low-rank, hard angle-bounded tangent update to summary tokens.

    The down projection is initialized with orthonormal rows and the up
    projection is initialized to exact zero, so construction is initially an
    identity mapping.  For a nonzero update, its component parallel to the
    input direction is removed before a saturating angular radius is applied.
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        rank: int = 32,
        max_angle_degrees: float = 0.1,
        *,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.max_angle_degrees = float(max_angle_degrees)
        self.eps = float(eps)
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.rank <= 0 or self.rank > self.feature_dim:
            raise ValueError("rank must lie in [1, feature_dim]")
        if (
            not math.isfinite(self.max_angle_degrees)
            or not 0.0 < self.max_angle_degrees < 90.0
        ):
            raise ValueError("max_angle_degrees must lie in (0,90)")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be finite and positive")

        self.down = nn.Linear(self.feature_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.feature_dim, bias=False)
        nn.init.orthogonal_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    @property
    def max_angle_radians(self) -> float:
        return math.radians(self.max_angle_degrees)

    def forward(self, summary_tokens: torch.Tensor) -> torch.Tensor:
        if not isinstance(summary_tokens, torch.Tensor):
            raise TypeError("summary_tokens must be a torch.Tensor")
        if (
            not summary_tokens.is_floating_point()
            or summary_tokens.ndim < 1
            or summary_tokens.shape[-1] != self.feature_dim
        ):
            raise ValueError(
                "summary_tokens must be floating point with final dimension "
                f"{self.feature_dim}"
            )
        if not bool(torch.isfinite(summary_tokens).all().item()):
            raise ValueError("summary_tokens must contain only finite values")

        norms = torch.linalg.vector_norm(summary_tokens, dim=-1, keepdim=True)
        if not bool((norms > self.eps).all().item()):
            raise ValueError("summary_tokens must have nonzero finite norms")
        unit = summary_tokens / norms
        raw_update = self.up(self.down(unit))
        tangent = raw_update - (raw_update * unit).sum(
            dim=-1,
            keepdim=True,
        ) * unit
        tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
        max_radius = math.tan(self.max_angle_radians)
        radius = max_radius * torch.tanh(tangent_norm / max_radius)
        # The analytic limit of ``radius / tangent_norm`` at zero is one.
        # Selecting that limit explicitly is important: a zero-initialized up
        # projection must still receive a first-step gradient.
        tangent_scale = torch.where(
            tangent_norm > self.eps,
            radius / tangent_norm.clamp_min(self.eps),
            torch.ones_like(tangent_norm),
        )
        bounded_tangent = tangent * tangent_scale
        adapted_unit = F.normalize(
            unit + bounded_tangent,
            dim=-1,
            eps=self.eps,
        )
        return adapted_unit * norms

    def architecture(self) -> dict[str, int | float | str]:
        payload: dict[str, int | float | str] = {
            "name": "low_rank_tangent_summary_adapter_v1",
            "feature_dim": self.feature_dim,
            "rank": self.rank,
            "max_angle_degrees": self.max_angle_degrees,
            "norm_policy": "preserve_exact_input_l2_norm",
            "update_geometry": "input_direction_tangent_hard_angular_cap",
            "down_initialization": "orthogonal_rows",
            "up_initialization": "zeros",
        }
        payload["digest"] = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return payload
