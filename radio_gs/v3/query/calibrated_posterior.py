"""Constant-size null-aware posterior for the canonical D512+R5 memory."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class NullCalibratedPosterior(nn.Module):
    """Fuse query evidence while using boundary to arbitrate local vs expanded support."""

    def __init__(self) -> None:
        super().__init__()
        positive = (2.0, 2.0, 3.0, 0.5, 0.5, 0.25, 0.5)
        negative = (2.0, 2.0, 2.0)
        self.raw_positive_weight = nn.Parameter(torch.tensor([
            _inverse_softplus(value) for value in positive
        ]))
        self.raw_negative_weight = nn.Parameter(torch.tensor([
            _inverse_softplus(value) for value in negative
        ]))
        self.bias = nn.Parameter(torch.tensor(-2.5))
        self.register_buffer(
            "positive_feature_mask", torch.ones(7), persistent=False
        )
        self.register_buffer(
            "negative_feature_mask", torch.ones(3), persistent=False
        )

    def forward(
        self,
        *,
        identity: torch.Tensor,
        instance: torch.Tensor,
        null: torch.Tensor,
        negative: torch.Tensor,
        unknown: torch.Tensor,
        boundary: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        positive_features, negative_features = self.evidence_features(
            identity=identity,
            instance=instance,
            null=null,
            negative=negative,
            unknown=unknown,
            boundary=boundary,
            reliability=reliability,
        )
        return torch.sigmoid(self.logit_from_features(positive_features, negative_features))

    def evidence_features(
        self,
        *,
        identity: torch.Tensor,
        instance: torch.Tensor,
        null: torch.Tensor,
        negative: torch.Tensor,
        unknown: torch.Tensor,
        boundary: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the fixed ten-dimensional sufficient evidence vector."""

        values = [
            torch.as_tensor(value).float().reshape(-1)
            for value in (identity, instance, null, negative, unknown, boundary)
        ]
        if len({value.shape for value in values}) != 1:
            raise ValueError("calibrated posterior evidence axes differ")
        r = torch.as_tensor(reliability, device=values[0].device).float()
        if r.shape != (values[0].numel(), 5):
            raise ValueError("calibrated posterior requires canonical R5")
        if not all(bool(torch.isfinite(value).all()) for value in (*values, r)):
            raise ValueError("calibrated posterior evidence must be finite")
        identity_value, instance_value, null_value, negative_value, unknown_value, edge = values
        # An unsigned edge cannot pick a side.  It only transfers authority from
        # propagated instance support back to the query-local identity evidence.
        boundary_contrast = edge.clamp(0, 1) * (identity_value - instance_value)
        positive_features = torch.stack((
            identity_value, instance_value, boundary_contrast,
            r[:, 0], r[:, 1], r[:, 3], r[:, 4],
        ), dim=1)
        effective_unknown = torch.maximum(unknown_value, r[:, 2])
        negative_features = torch.stack(
            (null_value, negative_value, effective_unknown), dim=1
        )
        return positive_features, negative_features

    def logit_from_features(
        self, positive_features: torch.Tensor, negative_features: torch.Tensor
    ) -> torch.Tensor:
        """Apply the constrained constant-size calibrator to prepared features."""

        positive_features = torch.as_tensor(positive_features).float()
        negative_features = torch.as_tensor(
            negative_features, device=positive_features.device
        ).float()
        if (
            positive_features.ndim != 2
            or positive_features.shape[1] != 7
            or negative_features.shape != (positive_features.shape[0], 3)
        ):
            raise ValueError("calibrated posterior feature axes differ")
        positive_weight = F.softplus(self.raw_positive_weight)
        negative_weight = F.softplus(self.raw_negative_weight)
        return (
            self.bias
            + (
                positive_features
                * positive_weight
                * self.positive_feature_mask
            ).sum(1)
            - (
                negative_features
                * negative_weight
                * self.negative_feature_mask
            ).sum(1)
        )


def load_null_calibrated_posterior(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> NullCalibratedPosterior:
    """Load a source-only calibrator without accepting benchmark-selected state."""

    payload = torch.load(Path(path).resolve(strict=True), map_location="cpu")
    if payload.get("schema") != "radio_gs.sugm_v3.null_calibrated_posterior.v1":
        raise ValueError("null-calibrated posterior schema differs")
    metadata = payload.get("metadata", {})
    if (
        not metadata.get("source_only")
        or metadata.get("target_rgb_opened")
        or metadata.get("benchmark_metrics_opened")
        or metadata.get("unknown_pairs_used_as_negative")
    ):
        raise ValueError("null-calibrated posterior selection authority differs")
    module = NullCalibratedPosterior()
    disabled = metadata.get("disabled_positive_feature_indices", [])
    if any(int(index) not in range(7) for index in disabled):
        raise ValueError("null-calibrated posterior feature mask differs")
    module.positive_feature_mask[
        torch.as_tensor(disabled, dtype=torch.long)
    ] = 0
    module.load_state_dict(payload["state_dict"], strict=True)
    return module.to(device).eval()


__all__ = ["NullCalibratedPosterior", "load_null_calibrated_posterior"]
