"""Dual-output surface-region descriptor with an exact official baseline.

The module keeps the promoted surface summary readout and RADIO's official
SigLIP2 summary head frozen.  A small context-conditioned FiLM branch may
change only the semantic descriptor; the official summary token and official
descriptor remain available as unmodified control outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.interfaces.surface_region_summary import SurfaceRegionSummaryReadoutV2


@dataclass(frozen=True)
class SurfaceRegionDualDescriptorOutput:
    """Named outputs from :class:`SurfaceRegionDualDescriptor`."""

    official_token: torch.Tensor
    official_descriptor: torch.Tensor
    semantic_descriptor: torch.Tensor


class SurfaceRegionDualDescriptor(nn.Module):
    """Add a context-conditioned residual after the official SigLIP2 head.

    The trainable path is ``LN(128) -> Linear(256) -> GELU`` followed by a
    1536-D gamma/beta FiLM projection and a scalar gate.  With the default
    dimensions it contains exactly 823,041 trainable parameters.  The FiLM
    projection is initialized to zero, so the semantic descriptor is exactly
    the normalized official descriptor at initialization.
    """

    ARCHITECTURE_NAME = "surface_region_dual_descriptor_v1"

    def __init__(
        self,
        summary_readout: SurfaceRegionSummaryReadoutV2,
        official_summary_head: nn.Module,
        *,
        descriptor_dim: int = 1536,
        bottleneck_dim: int = 256,
        initial_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if not isinstance(summary_readout, SurfaceRegionSummaryReadoutV2):
            raise TypeError("summary_readout must be SurfaceRegionSummaryReadoutV2")
        if not isinstance(official_summary_head, nn.Module):
            raise TypeError("official_summary_head must be an nn.Module")
        if int(descriptor_dim) <= 0 or int(bottleneck_dim) <= 0:
            raise ValueError("descriptor_dim and bottleneck_dim must be positive")
        if not 0.0 < float(initial_gate) < 1.0:
            raise ValueError("initial_gate must lie strictly between zero and one")

        self.summary_readout = summary_readout
        self.official_summary_head = official_summary_head
        self.context_dim = int(summary_readout.hidden_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.initial_gate = float(initial_gate)

        self.context_norm = nn.LayerNorm(self.context_dim)
        self.context_projection = nn.Linear(self.context_dim, self.bottleneck_dim)
        self.activation = nn.GELU()
        self.film = nn.Linear(self.bottleneck_dim, 2 * self.descriptor_dim)
        self.gate = nn.Linear(self.bottleneck_dim, 1)

        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        self.summary_readout.requires_grad_(False).eval()
        self.official_summary_head.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> "SurfaceRegionDualDescriptor":
        """Keep the two control modules in evaluation mode while training."""

        super().train(mode)
        self.summary_readout.eval()
        self.official_summary_head.eval()
        return self

    def forward(
        self,
        radio_features: torch.Tensor,
        geometry: torch.Tensor,
        *,
        anchor_index: torch.Tensor | int,
        token_mask: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> SurfaceRegionDualDescriptorOutput:
        official_token, context = self.summary_readout.forward_with_context(
            radio_features,
            geometry,
            anchor_index=anchor_index,
            token_mask=token_mask,
            reliability=reliability,
        )
        # The official RADIO path applies the summary head to a singleton
        # token sequence.  Preserve that call shape even though its linear
        # layers would also accept a rank-two tensor.
        official_raw = self.official_summary_head(official_token.unsqueeze(-2)).squeeze(-2)
        if official_raw.shape[-1] != self.descriptor_dim:
            raise ValueError(
                "official_summary_head output dimension does not match descriptor_dim"
            )
        official_descriptor = F.normalize(official_raw, dim=-1)

        z = self.activation(self.context_projection(self.context_norm(context)))
        gamma, beta = torch.tanh(self.film(z)).chunk(2, dim=-1)
        gate_logit = math.log(self.initial_gate / (1.0 - self.initial_gate))
        alpha = torch.sigmoid(self.gate(z) + gate_logit)
        delta = gamma * official_descriptor + beta / math.sqrt(self.descriptor_dim)
        normalized_semantic = F.normalize(
            official_descriptor + alpha * delta,
            dim=-1,
        )

        # A second floating-point normalization is not generally bitwise
        # idempotent.  At the exact zero-residual initialization, select the
        # already-normalized official value in the forward pass while retaining
        # the normalization branch's gradient for the first optimization step.
        zero_residual = (delta == 0).all(dim=-1, keepdim=True)
        exact_semantic = torch.where(
            zero_residual,
            official_descriptor,
            normalized_semantic,
        )
        semantic_descriptor = normalized_semantic + (
            exact_semantic - normalized_semantic
        ).detach()
        return SurfaceRegionDualDescriptorOutput(
            official_token=official_token,
            official_descriptor=official_descriptor,
            semantic_descriptor=semantic_descriptor,
        )

    def trainable_parameter_count(self) -> int:
        """Return the parameter count of the descriptor-residual branch."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def architecture(self) -> dict[str, int | float | str]:
        """Return a deterministic description of the trainable architecture."""

        payload: dict[str, int | float | str] = {
            "name": self.ARCHITECTURE_NAME,
            "summary_readout": "surface_region_summary_readout_v2_frozen",
            "official_summary_head": "c-radio_v4_heads_siglip2-g_frozen",
            "context_dim": self.context_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "descriptor_dim": self.descriptor_dim,
            "initial_gate": self.initial_gate,
            "film_activation": "tanh",
            "semantic_formula": (
                "normalize(e_off+alpha*(gamma*e_off+beta/sqrt(descriptor_dim)))"
            ),
            "trainable_parameter_count": self.trainable_parameter_count(),
        }
        payload["digest"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload


class SurfaceRegionFullScalarResidualOutput(NamedTuple):
    """Auditable outputs from the accepted-V2 full-scalar residual core."""

    base_descriptor: torch.Tensor
    semantic_descriptor: torch.Tensor
    tangent_update: torch.Tensor
    alpha: torch.Tensor
    ood_fallback: torch.Tensor


class SurfaceRegionAcceptedV2FullScalarResidualV1(nn.Module):
    """Bounded full-scalar residual over an immutable accepted-V2 descriptor.

    The caller owns the accepted-V2 region selection, readout, and official
    summary-head path.  This module receives only its already-normalized
    descriptor ``e0`` and the query-independent 18-D aggregation of the
    factorized primitive state (anchor, weighted mean, and weighted standard
    deviation of the six versioned scalars).

    Frozen source-only median/robust-scale buffers standardize the scalar
    carrier.  The robust scale is the normalization authority's exact
    ``1.4826 * MAD`` value (or its versioned zero-MAD fallback), not the raw
    MAD.  A compact content-conditioned branch embeds both ``e0`` and the 18-D
    carrier into 64 dimensions, includes their multiplicative interaction,
    and predicts one descriptor-space residual.  This is important: scalar
    observability statistics may modulate an existing semantic direction, but
    cannot be asked to generate object identity by themselves.  The residual
    is projected onto the tangent plane of ``e0`` and its step size is bounded
    both by ``max_alpha`` and by ``max_angle_radians``.
    The last projection is exactly zero-initialized, so initialization is a
    bitwise identity while retaining a straight-through first-step gradient.
    Rows marked OOD are a hard, gradient-free bitwise fallback to ``e0``.
    """

    ARCHITECTURE_NAME = "surface_region_accepted_v2_full_scalar_residual_v1"
    SCALAR_DIM = 18
    HIDDEN_DIM = 64

    def __init__(
        self,
        *,
        descriptor_dim: int = 1536,
        scalar_median: torch.Tensor | None = None,
        scalar_robust_scale: torch.Tensor | None = None,
        max_angle_radians: float = 0.15,
        max_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.max_angle_radians = float(max_angle_radians)
        self.max_alpha = float(max_alpha)
        if self.descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")
        if (
            not math.isfinite(self.max_angle_radians)
            or not 0.0 < self.max_angle_radians < math.pi / 2.0
        ):
            raise ValueError("max_angle_radians must lie strictly in (0, pi/2)")
        if not math.isfinite(self.max_alpha) or not 0.0 < self.max_alpha <= 1.0:
            raise ValueError("max_alpha must lie in (0, 1]")

        median = self._normalization_buffer(
            scalar_median,
            default=0.0,
            label="scalar_median",
        )
        robust_scale = self._normalization_buffer(
            scalar_robust_scale,
            default=1.0,
            label="scalar_robust_scale",
        )
        if bool((robust_scale <= 0).any()):
            raise ValueError("scalar_robust_scale must be strictly positive")
        self.register_buffer("scalar_median", median)
        self.register_buffer("scalar_robust_scale", robust_scale)

        self.descriptor_projection = nn.Linear(
            self.descriptor_dim,
            self.HIDDEN_DIM,
            bias=False,
        )
        self.scalar_projection = nn.Linear(self.SCALAR_DIM, self.HIDDEN_DIM)
        self.fusion_projection = nn.Linear(3 * self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.activation = nn.GELU()
        self.residual_projection = nn.Linear(self.HIDDEN_DIM, self.descriptor_dim)
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
        if value is None:
            result = torch.full((cls.SCALAR_DIM,), float(default), dtype=torch.float32)
        else:
            result = torch.as_tensor(value).detach().float().cpu().clone()
        if result.shape != (cls.SCALAR_DIM,) or not bool(torch.isfinite(result).all()):
            raise ValueError(f"{label} must be finite with shape [18]")
        return result

    def _validate_inputs(
        self,
        base_descriptor: torch.Tensor,
        aggregate_scalars: torch.Tensor,
        ood_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = torch.as_tensor(base_descriptor)
        scalars = torch.as_tensor(aggregate_scalars)
        if (
            base.ndim < 1
            or base.shape[-1] != self.descriptor_dim
            or not base.is_floating_point()
            or not bool(torch.isfinite(base).all())
        ):
            raise ValueError(
                "base_descriptor must be finite floating [..., descriptor_dim]"
            )
        if (
            scalars.shape != (*base.shape[:-1], self.SCALAR_DIM)
            or not scalars.is_floating_point()
            or not bool(torch.isfinite(scalars).all())
        ):
            raise ValueError(
                "aggregate_scalars must be finite floating [..., 18] aligned "
                "with base_descriptor"
            )
        if base.device != self.scalar_median.device:
            raise ValueError("base_descriptor and residual module devices differ")
        if base.dtype != self.residual_projection.weight.dtype:
            raise ValueError("base_descriptor and residual module dtypes differ")
        norm = torch.linalg.vector_norm(base, dim=-1)
        if not torch.allclose(
            norm,
            torch.ones_like(norm),
            rtol=0.0,
            atol=2e-4,
        ):
            raise ValueError("base_descriptor must use the unit L2 descriptor gauge")
        if ood_mask is None:
            ood = torch.zeros(base.shape[:-1], dtype=torch.bool, device=base.device)
        else:
            ood = torch.as_tensor(ood_mask, device=base.device)
            if ood.dtype != torch.bool or ood.shape != base.shape[:-1]:
                raise ValueError("ood_mask must be boolean and align with descriptor rows")
        return base, scalars.to(device=base.device, dtype=base.dtype), ood

    def _components(
        self,
        base: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_scalars = (
            scalars - self.scalar_median
        ) / self.scalar_robust_scale
        descriptor_hidden = self.descriptor_projection(base)
        scalar_hidden = self.scalar_projection(normalized_scalars)
        interaction = descriptor_hidden * torch.tanh(scalar_hidden)
        hidden = self.activation(
            self.fusion_projection(
                torch.cat(
                    (descriptor_hidden, scalar_hidden, interaction),
                    dim=-1,
                )
            )
        )
        raw_residual = self.residual_projection(hidden)

        # Only an angular update is allowed: magnitude/radial changes to the
        # already-normalized official descriptor are discarded exactly.
        base_squared_norm = base.square().sum(dim=-1, keepdim=True)
        tangent = raw_residual - (
            (raw_residual * base).sum(dim=-1, keepdim=True)
            / base_squared_norm
        ) * base
        tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
        angle_cap = math.tan(self.max_angle_radians) / tangent_norm.clamp_min(1e-12)
        alpha = torch.minimum(
            torch.full_like(tangent_norm, self.max_alpha),
            angle_cap,
        )
        update = alpha * tangent
        candidate = F.normalize(base + update, dim=-1)
        return candidate, update, alpha

    def forward_with_diagnostics(
        self,
        base_descriptor: torch.Tensor,
        aggregate_scalars: torch.Tensor,
        *,
        ood_mask: torch.Tensor | None = None,
    ) -> SurfaceRegionFullScalarResidualOutput:
        """Return the bounded candidate and the source-only fallback audit."""

        base, scalars, ood = self._validate_inputs(
            base_descriptor,
            aggregate_scalars,
            ood_mask,
        )
        candidate, update, alpha = self._components(base, scalars)

        # A second normalization is not bitwise idempotent.  Select the exact
        # base in the forward pass at the zero initialization while preserving
        # candidate gradients so the final zero-initialized layer can learn.
        zero_residual = (update == 0).all(dim=-1, keepdim=True)
        identity_forward = torch.where(zero_residual, base, candidate)
        trainable_candidate = candidate + (identity_forward - candidate).detach()

        # OOD is a true fail-closed branch: unlike zero initialization, it is
        # not a straight-through estimator and cannot train from excluded rows.
        semantic = torch.where(ood[..., None], base, trainable_candidate)
        audited_update = torch.where(ood[..., None], torch.zeros_like(update), update)
        # ``alpha`` is an audit of the effective step, not merely the cap that
        # would have applied to a nonzero tangent.  Reporting ``max_alpha`` at
        # the exact-identity initialization is therefore misleading.
        effective_alpha = torch.where(zero_residual, torch.zeros_like(alpha), alpha)
        audited_alpha = torch.where(
            ood[..., None], torch.zeros_like(effective_alpha), effective_alpha
        )
        return SurfaceRegionFullScalarResidualOutput(
            base_descriptor=base,
            semantic_descriptor=semantic,
            tangent_update=audited_update,
            alpha=audited_alpha,
            ood_fallback=ood,
        )

    def forward(
        self,
        base_descriptor: torch.Tensor,
        aggregate_scalars: torch.Tensor,
        *,
        ood_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return only the semantic descriptor for downstream composition."""

        return self.forward_with_diagnostics(
            base_descriptor,
            aggregate_scalars,
            ood_mask=ood_mask,
        ).semantic_descriptor

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def architecture(self) -> dict[str, int | float | str]:
        payload: dict[str, int | float | str] = {
            "name": self.ARCHITECTURE_NAME,
            "base": "accepted_surface_region_v2_descriptor_immutable_external",
            "scalar_dim": self.SCALAR_DIM,
            "hidden_dim": self.HIDDEN_DIM,
            "descriptor_dim": self.descriptor_dim,
            "scalar_normalization": (
                "frozen_source_median_over_robust_scale_buffers"
            ),
            "conditioning": (
                "content_scalar_concat_with_multiplicative_interaction"
            ),
            "descriptor_projection_bias": "disabled",
            "residual_gauge": "official_descriptor_tangent_plane",
            "max_angle_radians": self.max_angle_radians,
            "max_alpha": self.max_alpha,
            "ood_policy": "hard_bitwise_base_fallback",
            "initialization": "zero_final_projection_exact_base_identity",
            "trainable_parameter_count": self.trainable_parameter_count(),
        }
        payload["digest"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload
