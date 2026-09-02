"""Token-specific continuous spatial support slots for v4 completion.

The module is a residual over a frozen ``K + null`` posterior.  Seven slots
are anchored in each token's source-only PCA frame: one centre slot and a
positive/negative pair on every principal axis.  A shared token network may
change slot radius, anisotropic scale, and mixture weight.  There is no hard
support envelope, threshold, connected component, target, or query input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


SPATIAL_SLOT_COUNT = 7


@dataclass(frozen=True)
class SpatialSlotOutput:
    probabilities: torch.Tensor
    token_residual_logits: torch.Tensor
    support_scores: torch.Tensor
    slot_centres_local: torch.Tensor
    slot_scales_local: torch.Tensor
    slot_amplitudes: torch.Tensor
    token_bias: torch.Tensor
    fusion_strength: torch.Tensor


class TokenSpatialSupportSlots(nn.Module):
    """Predict a smooth multi-slot spatial residual for every scene token."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int = 96,
        dropout: float = 0.0,
        *,
        use_token_bias: bool = True,
    ) -> None:
        super().__init__()
        if input_dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("slot input and hidden dimensions must be positive")
        if not 0 <= float(dropout) < 1:
            raise ValueError("slot dropout must be in [0, 1)")
        self.input_dimension = int(input_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.dropout = float(dropout)
        self.use_token_bias = bool(use_token_bias)
        parameter_dimension = SPATIAL_SLOT_COUNT * 5 + 1
        self.token_network = nn.Sequential(
            nn.LayerNorm(self.input_dimension),
            nn.Linear(self.input_dimension, self.hidden_dimension),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dimension, self.hidden_dimension),
            nn.GELU(),
            nn.Linear(self.hidden_dimension, parameter_dimension),
        )
        # Start from the source-observed PCA support, with zero token bias.
        nn.init.zeros_(self.token_network[-1].weight)
        nn.init.zeros_(self.token_network[-1].bias)
        self.fusion_parameter = nn.Parameter(torch.tensor(0.1))
        self.register_buffer(
            "canonical_slot_seeds",
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=torch.float32,
            ),
            persistent=True,
        )

    def architecture_receipt(self) -> dict[str, object]:
        return {
            "schema": "radio_gs.surface_object_memory_v4.token_spatial_support_slots.v1",
            "slot_count": SPATIAL_SLOT_COUNT,
            "slot_layout": "pca_centre_and_symmetric_axis_pairs",
            "input_dimension": self.input_dimension,
            "hidden_dimension": self.hidden_dimension,
            "dropout": self.dropout,
            "use_token_bias": self.use_token_bias,
            "source_geometry_authority": "observed_positive_pca_only",
            "frozen_unary_residual": True,
            "observed_clamp_after_fusion": True,
            "target_membership_input": False,
            "heldout_rgb_input": False,
            "query_input": False,
            "hard_threshold": False,
            "hard_radius_or_envelope": False,
            "connected_components": False,
            "v3_dependency": False,
        }

    @staticmethod
    def _validate_inputs(
        unary: torch.Tensor,
        element_centres: torch.Tensor,
        token_features: torch.Tensor,
        token_centres: torch.Tensor,
        token_frames: torch.Tensor,
        token_scales: torch.Tensor,
        clamp_mask: torch.Tensor,
        clamp_probabilities: torch.Tensor,
    ) -> None:
        if unary.ndim != 2 or unary.shape[1] < 2 or not unary.is_floating_point():
            raise ValueError("unary must have shape [B, K+1]")
        batch_count, category_count = unary.shape
        token_count = category_count - 1
        if element_centres.shape != (batch_count, 3):
            raise ValueError("element_centres must have shape [B, 3]")
        if token_features.ndim != 2 or token_features.shape[0] != token_count:
            raise ValueError("token_features must have shape [K, D]")
        if token_centres.shape != (token_count, 3):
            raise ValueError("token_centres must have shape [K, 3]")
        if token_frames.shape != (token_count, 3, 3):
            raise ValueError("token_frames must have shape [K, 3, 3]")
        if token_scales.shape != (token_count, 3):
            raise ValueError("token_scales must have shape [K, 3]")
        if clamp_mask.dtype != torch.bool or clamp_mask.shape != (batch_count,):
            raise ValueError("clamp_mask must be explicit bool [B]")
        if clamp_probabilities.shape != unary.shape:
            raise ValueError("clamp_probabilities must align with unary")
        floating = (
            unary,
            element_centres,
            token_features,
            token_centres,
            token_frames,
            token_scales,
            clamp_probabilities,
        )
        if not all(torch.isfinite(value).all() for value in floating):
            raise ValueError("spatial slot inputs must be finite")
        if bool((unary < 0).any()) or bool((token_scales <= 0).any()):
            raise ValueError("unary must be non-negative and token scales positive")
        expected = torch.ones(batch_count, device=unary.device, dtype=unary.dtype)
        if not torch.allclose(unary.sum(-1), expected, atol=2e-6, rtol=2e-6):
            raise ValueError("unary must lie on the K-plus-null simplex")
        selected = clamp_probabilities[clamp_mask]
        if selected.numel() and not torch.equal(
            selected.sum(-1), torch.ones_like(selected[:, 0])
        ):
            raise ValueError("every selected clamp must be categorical")
        gram = token_frames.transpose(-1, -2) @ token_frames
        identity = torch.eye(3, device=gram.device, dtype=gram.dtype).expand_as(gram)
        if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
            raise ValueError("token_frames must be orthonormal")

    def forward(
        self,
        unary_probabilities: torch.Tensor,
        element_centres: torch.Tensor,
        token_features: torch.Tensor,
        token_centres: torch.Tensor,
        token_frames: torch.Tensor,
        token_scales: torch.Tensor,
        clamp_mask: torch.Tensor,
        clamp_probabilities: torch.Tensor,
    ) -> SpatialSlotOutput:
        unary_input = torch.as_tensor(unary_probabilities)
        if not unary_input.is_floating_point():
            raise ValueError("unary_probabilities must be floating point")
        device = unary_input.device
        dtype = unary_input.dtype
        unary = unary_input.detach()
        centres = torch.as_tensor(element_centres, device=device, dtype=dtype).detach()
        features = torch.as_tensor(token_features, device=device, dtype=dtype).detach()
        token_centre = torch.as_tensor(token_centres, device=device, dtype=dtype).detach()
        frames = torch.as_tensor(token_frames, device=device, dtype=dtype).detach()
        base_scale = torch.as_tensor(token_scales, device=device, dtype=dtype).detach()
        mask = torch.as_tensor(clamp_mask, device=device)
        clamp = torch.as_tensor(
            clamp_probabilities, device=device, dtype=dtype
        ).detach()
        self._validate_inputs(
            unary,
            centres,
            features,
            token_centre,
            frames,
            base_scale,
            mask,
            clamp,
        )

        token_count = unary.shape[1] - 1
        raw = self.token_network(features)
        slot_raw = raw[:, : SPATIAL_SLOT_COUNT * 5].reshape(
            token_count, SPATIAL_SLOT_COUNT, 5
        )
        radius_multiplier = torch.exp(2.0 * torch.tanh(slot_raw[..., 0] / 2.0))
        scale_multiplier = torch.exp(
            2.0 * torch.tanh(slot_raw[..., 1:4] / 2.0)
        )
        amplitudes = 4.0 * torch.tanh(slot_raw[..., 4] / 4.0)
        token_bias = (
            4.0 * torch.tanh(raw[:, -1] / 4.0)
            if self.use_token_bias
            else raw[:, -1] * 0.0
        )
        seeds = self.canonical_slot_seeds.to(device=device, dtype=dtype)
        slot_centres_local = (
            seeds[None, :, :] * base_scale[:, None, :] * radius_multiplier[..., None]
        )
        slot_scales_local = base_scale[:, None, :] * scale_multiplier

        world_delta = centres[:, None, :] - token_centre[None, :, :]
        local_delta = torch.einsum("bki,kij->bkj", world_delta, frames)
        standardized = (
            local_delta[:, :, None, :] - slot_centres_local[None, :, :, :]
        ) / slot_scales_local[None, :, :, :]
        squared_distance = standardized.square().sum(-1)
        component_logits = amplitudes[None, :, :] - 0.5 * squared_distance
        support_scores = torch.logsumexp(component_logits, dim=-1) - torch.logsumexp(
            amplitudes, dim=-1
        )[None, :]
        fusion_strength = 4.0 * torch.tanh(self.fusion_parameter / 4.0)
        residual = fusion_strength * support_scores + token_bias[None, :]
        epsilon = torch.finfo(dtype).tiny
        categorical_logits = torch.cat(
            (
                unary[:, :token_count].clamp_min(epsilon).log() + residual,
                unary[:, token_count:].clamp_min(epsilon).log(),
            ),
            dim=-1,
        )
        probabilities = torch.softmax(categorical_logits, dim=-1)
        probabilities = torch.where(mask[:, None], clamp, probabilities)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("spatial slot posterior became non-finite")
        if not torch.equal(probabilities[mask], clamp[mask]):
            raise RuntimeError("spatial slots changed an exact observed clamp")
        if not torch.allclose(
            probabilities.sum(-1), torch.ones_like(probabilities[:, 0]), atol=2e-6
        ):
            raise RuntimeError("spatial slot posterior left the categorical simplex")
        return SpatialSlotOutput(
            probabilities=probabilities,
            token_residual_logits=residual,
            support_scores=support_scores,
            slot_centres_local=slot_centres_local,
            slot_scales_local=slot_scales_local,
            slot_amplitudes=amplitudes,
            token_bias=token_bias,
            fusion_strength=fusion_strength,
        )
