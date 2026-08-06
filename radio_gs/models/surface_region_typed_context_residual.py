"""Bounded typed-context residual over an immutable AcceptedV2 descriptor.

The module has no surface-region selector, readout, RADIO head, or teacher.
Those objects remain external frozen authorities.  Its final projection is
zero initialized, inactive and OOD rows are hard bitwise fallbacks, and every
active update is restricted to the tangent plane of the unit AcceptedV2
descriptor with a fixed angular bound.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.interfaces.surface_region_typed_context import (
    TYPED_CONTEXT_FEATURE_DIM,
    TYPED_CONTEXT_STATISTIC_DIM,
)


@dataclass(frozen=True)
class TypedContextResidualOutput:
    base_descriptor: torch.Tensor
    semantic_descriptor: torch.Tensor
    tangent_update: torch.Tensor
    alpha: torch.Tensor
    active_residual: torch.Tensor
    fallback: torch.Tensor


class SurfaceRegionAcceptedV2TypedContextResidualV1(nn.Module):
    """Small query-free context residual with an exact AcceptedV2 fallback."""

    ARCHITECTURE_NAME = "surface_region_accepted_v2_typed_context_residual_v1"
    DESCRIPTOR_DIM = 1536
    CONTEXT_DIM = TYPED_CONTEXT_FEATURE_DIM
    FULL_SCALAR_DIM = 18
    CONTEXT_STATISTIC_DIM = TYPED_CONTEXT_STATISTIC_DIM
    SCALAR_DIM = FULL_SCALAR_DIM + CONTEXT_STATISTIC_DIM
    HIDDEN_DIM = 64

    def __init__(
        self,
        *,
        scalar_median: torch.Tensor | None = None,
        scalar_robust_scale: torch.Tensor | None = None,
        max_angle_radians: float = 0.15,
        max_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.max_angle_radians = float(max_angle_radians)
        self.max_alpha = float(max_alpha)
        if (
            not math.isfinite(self.max_angle_radians)
            or not 0.0 < self.max_angle_radians < math.pi / 2.0
        ):
            raise ValueError("max_angle_radians must lie strictly in (0, pi/2)")
        if not math.isfinite(self.max_alpha) or not 0.0 < self.max_alpha <= 1.0:
            raise ValueError("max_alpha must lie in (0,1]")
        median = self._normalization_buffer(
            scalar_median, default=0.0, label="scalar_median"
        )
        robust_scale = self._normalization_buffer(
            scalar_robust_scale, default=1.0, label="scalar_robust_scale"
        )
        if bool((robust_scale <= 0).any()):
            raise ValueError("scalar_robust_scale must be strictly positive")
        self.register_buffer("scalar_median", median)
        self.register_buffer("scalar_robust_scale", robust_scale)

        self.descriptor_projection = nn.Linear(
            self.DESCRIPTOR_DIM, self.HIDDEN_DIM, bias=False
        )
        self.context_projection = nn.Linear(
            self.CONTEXT_DIM, self.HIDDEN_DIM, bias=False
        )
        self.scalar_projection = nn.Linear(self.SCALAR_DIM, self.HIDDEN_DIM)
        self.fusion_projection = nn.Linear(5 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.activation = nn.GELU()
        self.residual_projection = nn.Linear(self.HIDDEN_DIM, self.DESCRIPTOR_DIM)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    @classmethod
    def _normalization_buffer(
        cls,
        value: torch.Tensor | None,
        *,
        default: float,
        label: str,
    ) -> torch.Tensor:
        result = (
            torch.full((cls.SCALAR_DIM,), float(default), dtype=torch.float32)
            if value is None
            else torch.as_tensor(value).detach().float().cpu().clone()
        )
        if result.shape != (cls.SCALAR_DIM,) or not bool(torch.isfinite(result).all()):
            raise ValueError(f"{label} must be finite with shape [{cls.SCALAR_DIM}]")
        return result

    def _validate_inputs(
        self,
        base_descriptor: torch.Tensor,
        pooled_context_radio_direction: torch.Tensor,
        full_scalar_summary: torch.Tensor,
        typed_context_statistics: torch.Tensor,
        active_mask: torch.Tensor,
        ood_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base = torch.as_tensor(base_descriptor)
        context = torch.as_tensor(pooled_context_radio_direction)
        full_scalar = torch.as_tensor(full_scalar_summary)
        context_stats = torch.as_tensor(typed_context_statistics)
        active = torch.as_tensor(active_mask, device=base.device)
        prefix = base.shape[:-1]
        if (
            base.ndim < 1
            or base.shape[-1] != self.DESCRIPTOR_DIM
            or not base.is_floating_point()
            or not bool(torch.isfinite(base).all())
        ):
            raise ValueError("base_descriptor must be finite [...,1536]")
        if (
            context.shape != (*prefix, self.CONTEXT_DIM)
            or not context.is_floating_point()
            or not bool(torch.isfinite(context).all())
        ):
            raise ValueError("pooled context must be finite [...,1280]")
        if (
            full_scalar.shape != (*prefix, self.FULL_SCALAR_DIM)
            or not full_scalar.is_floating_point()
            or not bool(torch.isfinite(full_scalar).all())
            or context_stats.shape != (*prefix, self.CONTEXT_STATISTIC_DIM)
            or not context_stats.is_floating_point()
            or not bool(torch.isfinite(context_stats).all())
        ):
            raise ValueError("typed-context scalar carriers differ")
        if active.dtype != torch.bool or active.shape != prefix:
            raise ValueError("active_mask must be boolean and descriptor aligned")
        ood = (
            torch.zeros(prefix, dtype=torch.bool, device=base.device)
            if ood_mask is None
            else torch.as_tensor(ood_mask, device=base.device)
        )
        if ood.dtype != torch.bool or ood.shape != prefix:
            raise ValueError("ood_mask must be boolean and descriptor aligned")
        if base.device != self.scalar_median.device:
            raise ValueError("typed-context model and descriptors must share a device")
        if base.dtype != self.residual_projection.weight.dtype:
            raise ValueError("base descriptor and model dtypes differ")
        base_norm = torch.linalg.vector_norm(base, dim=-1)
        if not torch.allclose(
            base_norm, torch.ones_like(base_norm), rtol=0.0, atol=2e-4
        ):
            raise ValueError("base_descriptor must use the unit L2 gauge")
        if bool(active.any()):
            context_norm = torch.linalg.vector_norm(context.float(), dim=-1)
            if not torch.allclose(
                context_norm[active],
                torch.ones_like(context_norm[active]),
                rtol=0.0,
                atol=1e-3,
            ):
                raise ValueError("active pooled context must use the unit L2 gauge")
        if bool(context[~active].count_nonzero()) or bool(
            context_stats[~active].count_nonzero()
        ):
            raise ValueError("inactive typed-context carriers must be exact zero")
        combined = torch.cat((full_scalar, context_stats), dim=-1).to(
            device=base.device, dtype=base.dtype
        )
        return (
            base,
            context.to(device=base.device, dtype=base.dtype),
            combined,
            active,
            ood,
        )

    def _candidate(
        self,
        base: torch.Tensor,
        context: torch.Tensor,
        scalar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_scalar = (
            scalar - self.scalar_median
        ) / self.scalar_robust_scale
        h0 = self.descriptor_projection(base)
        hc = self.context_projection(context)
        hs = self.scalar_projection(normalized_scalar)
        hidden = self.activation(
            self.fusion_projection(
                torch.cat(
                    (h0, hc, hs, h0 * torch.tanh(hc), h0 * torch.tanh(hs)),
                    dim=-1,
                )
            )
        )
        raw = self.residual_projection(hidden)
        base_norm_sq = base.square().sum(dim=-1, keepdim=True)
        tangent = raw - ((raw * base).sum(dim=-1, keepdim=True) / base_norm_sq) * base
        tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
        angle_scale = math.tan(self.max_angle_radians) / tangent_norm.clamp_min(1e-12)
        alpha = torch.minimum(
            torch.full_like(tangent_norm, self.max_alpha), angle_scale
        )
        update = alpha * tangent
        return F.normalize(base + update, dim=-1), update, alpha

    def forward_with_diagnostics(
        self,
        base_descriptor: torch.Tensor,
        pooled_context_radio_direction: torch.Tensor,
        full_scalar_summary: torch.Tensor,
        typed_context_statistics: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        ood_mask: torch.Tensor | None = None,
    ) -> TypedContextResidualOutput:
        base, context, scalar, declared_active, ood = self._validate_inputs(
            base_descriptor,
            pooled_context_radio_direction,
            full_scalar_summary,
            typed_context_statistics,
            active_mask,
            ood_mask,
        )
        candidate, update, alpha = self._candidate(base, context, scalar)
        zero_update = (update == 0).all(dim=-1, keepdim=True)
        identity_forward = torch.where(zero_update, base, candidate)
        trainable_candidate = candidate + (identity_forward - candidate).detach()
        active = declared_active & ~ood
        semantic = torch.where(active[..., None], trainable_candidate, base)
        audited_update = torch.where(
            active[..., None], update, torch.zeros_like(update)
        )
        effective_alpha = torch.where(
            zero_update, torch.zeros_like(alpha), alpha
        )
        audited_alpha = torch.where(
            active[..., None], effective_alpha, torch.zeros_like(effective_alpha)
        )
        return TypedContextResidualOutput(
            base_descriptor=base,
            semantic_descriptor=semantic,
            tangent_update=audited_update,
            alpha=audited_alpha,
            active_residual=active,
            fallback=~active,
        )

    def forward(
        self,
        base_descriptor: torch.Tensor,
        pooled_context_radio_direction: torch.Tensor,
        full_scalar_summary: torch.Tensor,
        typed_context_statistics: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        ood_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_diagnostics(
            base_descriptor,
            pooled_context_radio_direction,
            full_scalar_summary,
            typed_context_statistics,
            active_mask=active_mask,
            ood_mask=ood_mask,
        ).semantic_descriptor

    def architecture(self) -> dict[str, int | float | str]:
        payload: dict[str, int | float | str] = {
            "name": self.ARCHITECTURE_NAME,
            "base": "immutable_external_accepted_v2_descriptor",
            "descriptor_dim": self.DESCRIPTOR_DIM,
            "context_dim": self.CONTEXT_DIM,
            "full_scalar_dim": self.FULL_SCALAR_DIM,
            "context_statistic_dim": self.CONTEXT_STATISTIC_DIM,
            "hidden_dim": self.HIDDEN_DIM,
            "conditioning": "base_context_scalar_with_two_multiplicative_interactions",
            "residual_gauge": "accepted_v2_descriptor_tangent_plane",
            "max_angle_radians": self.max_angle_radians,
            "max_alpha": self.max_alpha,
            "initialization": "zero_final_projection_exact_base_identity",
            "fallback": "inactive_or_ood_bitwise_accepted_v2_base",
            "trainable_parameter_count": sum(
                value.numel() for value in self.parameters() if value.requires_grad
            ),
        }
        payload["digest"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload


__all__ = [
    "SurfaceRegionAcceptedV2TypedContextResidualV1",
    "TypedContextResidualOutput",
]
