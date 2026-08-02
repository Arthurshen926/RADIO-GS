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
