"""Reliability-conditioned rank-256 residual over immutable AcceptedV2.

This module is an isolated V2.1B candidate.  It does not own a selector,
teacher, text bank, scene embedding, or AcceptedV2 parameter.  Reliability is
computed deterministically from five already-materialized query-free scalar
channels.  It controls only the maximum geodesic step on the unit descriptor
sphere; semantic identity must still come from AcceptedV2 and typed RADIO
context.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from radio_gs.interfaces.surface_region_full_scalar_contract import (
    SURFACE_REGION_FULL_SCALAR_DIM,
    SURFACE_REGION_FULL_SCALAR_NAMES,
    SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
)
from radio_gs.interfaces.surface_region_typed_context import (
    TYPED_CONTEXT_FEATURE_DIM,
    TYPED_CONTEXT_STATISTIC_DIM,
    TYPED_CONTEXT_STATISTIC_NAMES,
    TYPED_CONTEXT_STATISTIC_NAMES_SHA256,
)
from radio_gs.utils.immutable_artifacts import canonical_json_sha256


_MEAN_DISPERSION_INDEX = SURFACE_REGION_FULL_SCALAR_NAMES.index(
    "legacy_reliability_weighted_mean_directional_dispersion"
)
_MEAN_EVIDENCE_INDEX = SURFACE_REGION_FULL_SCALAR_NAMES.index(
    "legacy_reliability_weighted_mean_observation_evidence"
)
_MEAN_PURITY_INDEX = SURFACE_REGION_FULL_SCALAR_NAMES.index(
    "legacy_reliability_weighted_mean_visibility_purity_value"
)
_MEAN_PURITY_KNOWN_INDEX = SURFACE_REGION_FULL_SCALAR_NAMES.index(
    "legacy_reliability_weighted_mean_visibility_purity_known"
)
_CONTEXT_RELIABILITY_INDEX = TYPED_CONTEXT_STATISTIC_NAMES.index(
    "context_reliability_mean"
)
_CONTEXT_RESULTANT_INDEX = TYPED_CONTEXT_STATISTIC_NAMES.index(
    "context_weighted_directional_resultant_length"
)


V21B_RELIABILITY_COMPONENT_NAMES = (
    "legacy_reliability_weighted_mean_directional_agreement",
    "legacy_reliability_weighted_mean_observation_evidence",
    "legacy_reliability_weighted_mean_visibility_purity_times_known_fraction",
    "typed_context_context_reliability_mean",
    "typed_context_weighted_directional_resultant_length",
)
V21B_RELIABILITY_COMPONENT_NAMES_SHA256 = canonical_json_sha256(
    list(V21B_RELIABILITY_COMPONENT_NAMES)
)


@dataclass(frozen=True)
class ReliabilityConditionedResidualV21BOutput:
    """Auditable outputs of the V2.1B spherical residual."""

    base_descriptor: torch.Tensor
    semantic_descriptor: torch.Tensor
    tangent_update: torch.Tensor
    tangent_gain: torch.Tensor
    reliability_score: torch.Tensor
    angular_budget_radians: torch.Tensor
    active_residual: torch.Tensor
    fallback: torch.Tensor


class SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B(nn.Module):
    """Rank-256 typed-context residual with a fixed reliability angle budget."""

    ARCHITECTURE_NAME = (
        "surface_region_accepted_v2_reliability_conditioned_residual_v21b"
    )
    DESCRIPTOR_DIM = 1536
    CONTEXT_DIM = TYPED_CONTEXT_FEATURE_DIM
    FULL_SCALAR_DIM = SURFACE_REGION_FULL_SCALAR_DIM
    CONTEXT_STATISTIC_DIM = TYPED_CONTEXT_STATISTIC_DIM
    SCALAR_DIM = FULL_SCALAR_DIM + CONTEXT_STATISTIC_DIM
    HIDDEN_RANK = 256
    LOW_RELIABILITY_ANGLE_RADIANS = 0.15
    HIGH_RELIABILITY_ANGLE_RADIANS = 0.75
    MAX_TANGENT_GAIN = 0.25
    _EPS = 1e-12

    def __init__(
        self,
        *,
        scalar_median: torch.Tensor | None = None,
        scalar_robust_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
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

        rank = self.HIDDEN_RANK
        self.descriptor_projection = nn.Linear(
            self.DESCRIPTOR_DIM,
            rank,
            bias=False,
        )
        self.context_projection = nn.Linear(self.CONTEXT_DIM, rank, bias=False)
        self.scalar_projection = nn.Linear(self.SCALAR_DIM, rank)
        self.fusion_projection = nn.Linear(5 * rank, rank)
        self.activation = nn.GELU()
        self.residual_projection = nn.Linear(rank, self.DESCRIPTOR_DIM)

        # This is both a forward identity and a usable first-step
        # initialization: the straight-through identity selection below keeps
        # the candidate branch's gradient at the exact zero projection.
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
        if result.shape != (cls.SCALAR_DIM,) or not bool(
            torch.isfinite(result).all()
        ):
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        base = torch.as_tensor(base_descriptor)
        context = torch.as_tensor(pooled_context_radio_direction)
        full_scalar = torch.as_tensor(full_scalar_summary)
        context_statistics = torch.as_tensor(typed_context_statistics)
        prefix = base.shape[:-1]
        if (
            base.ndim < 1
            or base.shape[-1] != self.DESCRIPTOR_DIM
            or not base.is_floating_point()
            or not bool(torch.isfinite(base).all())
        ):
            raise ValueError("base_descriptor must be finite floating [...,1536]")
        if (
            context.shape != (*prefix, self.CONTEXT_DIM)
            or not context.is_floating_point()
            or not bool(torch.isfinite(context).all())
        ):
            raise ValueError("pooled context must be finite floating [...,1280]")
        if (
            full_scalar.shape != (*prefix, self.FULL_SCALAR_DIM)
            or not full_scalar.is_floating_point()
            or not bool(torch.isfinite(full_scalar).all())
        ):
            raise ValueError("full-scalar summary must be finite floating [...,18]")
        if (
            context_statistics.shape
            != (*prefix, self.CONTEXT_STATISTIC_DIM)
            or not context_statistics.is_floating_point()
            or not bool(torch.isfinite(context_statistics).all())
        ):
            raise ValueError(
                "typed-context statistics must be finite floating [...,12]"
            )
        if base.device != self.scalar_median.device:
            raise ValueError("V2.1B model and descriptors must share a device")
        if base.dtype != self.residual_projection.weight.dtype:
            raise ValueError("V2.1B model and descriptors must share a dtype")
        base_norm = torch.linalg.vector_norm(base, dim=-1)
        if not torch.allclose(
            base_norm,
            torch.ones_like(base_norm),
            rtol=0.0,
            atol=2e-4,
        ):
            raise ValueError("base_descriptor must use the unit L2 gauge")

        active = torch.as_tensor(active_mask, device=base.device)
        if active.dtype != torch.bool or active.shape != prefix:
            raise ValueError("active_mask must be boolean and descriptor aligned")
        ood = (
            torch.zeros(prefix, dtype=torch.bool, device=base.device)
            if ood_mask is None
            else torch.as_tensor(ood_mask, device=base.device)
        )
        if ood.dtype != torch.bool or ood.shape != prefix:
            raise ValueError("ood_mask must be boolean and descriptor aligned")
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
            context_statistics[~active].count_nonzero()
        ):
            raise ValueError("inactive typed-context carriers must be exact zero")
        return (
            base,
            context.to(device=base.device, dtype=base.dtype),
            full_scalar.to(device=base.device, dtype=base.dtype),
            context_statistics.to(device=base.device, dtype=base.dtype),
            active,
            ood,
        )

    @classmethod
    def reliability_score(
        cls,
        full_scalar_summary: torch.Tensor,
        typed_context_statistics: torch.Tensor,
    ) -> torch.Tensor:
        """Return the fixed query-free reliability score in ``[0,1]``."""

        full_scalar = torch.as_tensor(full_scalar_summary)
        context_statistics = torch.as_tensor(typed_context_statistics)
        if (
            full_scalar.ndim < 1
            or full_scalar.shape[-1] != cls.FULL_SCALAR_DIM
            or context_statistics.shape
            != (*full_scalar.shape[:-1], cls.CONTEXT_STATISTIC_DIM)
            or not full_scalar.is_floating_point()
            or not context_statistics.is_floating_point()
            or not bool(torch.isfinite(full_scalar).all())
            or not bool(torch.isfinite(context_statistics).all())
        ):
            raise ValueError("V2.1B reliability carriers differ")
        directional_agreement = 1.0 - full_scalar[
            ..., _MEAN_DISPERSION_INDEX
        ].clamp(0.0, 1.0)
        evidence = full_scalar[..., _MEAN_EVIDENCE_INDEX].clamp(0.0, 1.0)
        purity = full_scalar[..., _MEAN_PURITY_INDEX].clamp(0.0, 1.0)
        purity_known = full_scalar[..., _MEAN_PURITY_KNOWN_INDEX].clamp(0.0, 1.0)
        context_reliability = context_statistics[
            ..., _CONTEXT_RELIABILITY_INDEX
        ].clamp(0.0, 1.0)
        context_resultant = context_statistics[
            ..., _CONTEXT_RESULTANT_INDEX
        ].clamp(0.0, 1.0)
        components = torch.stack(
            (
                directional_agreement,
                evidence,
                purity * purity_known,
                context_reliability,
                context_resultant,
            ),
            dim=-1,
        )
        return components.mean(dim=-1).clamp(0.0, 1.0)

    @classmethod
    def angular_budget(cls, reliability_score: torch.Tensor) -> torch.Tensor:
        score = torch.as_tensor(reliability_score)
        if not score.is_floating_point() or not bool(torch.isfinite(score).all()):
            raise ValueError("V2.1B reliability score must be finite floating")
        score = score.clamp(0.0, 1.0)
        return cls.LOW_RELIABILITY_ANGLE_RADIANS + (
            cls.HIGH_RELIABILITY_ANGLE_RADIANS
            - cls.LOW_RELIABILITY_ANGLE_RADIANS
        ) * score

    def _candidate(
        self,
        base: torch.Tensor,
        context: torch.Tensor,
        full_scalar: torch.Tensor,
        context_statistics: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        combined_scalar = torch.cat((full_scalar, context_statistics), dim=-1)
        normalized_scalar = (
            combined_scalar - self.scalar_median
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
        tangent = raw - (
            (raw * base).sum(dim=-1, keepdim=True) / base_norm_sq
        ) * base
        tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
        score = self.reliability_score(full_scalar, context_statistics)
        budget = self.angular_budget(score).unsqueeze(-1)

        # A smooth cap avoids the zero-gradient plateau of a hard minimum.
        # The vector ``tangent_update`` is the exponential-map tangent vector,
        # whose norm is the intended geodesic step.
        requested_angle = self.MAX_TANGENT_GAIN * tangent_norm
        angle = budget * torch.tanh(requested_angle / budget)
        tangent_gain = torch.where(
            tangent_norm > self._EPS,
            angle / tangent_norm.clamp_min(self._EPS),
            torch.full_like(tangent_norm, self.MAX_TANGENT_GAIN),
        )
        tangent_update = tangent_gain * tangent
        sine_over_norm = torch.where(
            tangent_norm > self._EPS,
            torch.sin(angle) / tangent_norm.clamp_min(self._EPS),
            torch.full_like(tangent_norm, self.MAX_TANGENT_GAIN),
        )
        candidate = F.normalize(
            torch.cos(angle) * base + sine_over_norm * tangent,
            dim=-1,
        )
        return candidate, tangent_update, tangent_gain, score, budget

    def forward_with_diagnostics(
        self,
        base_descriptor: torch.Tensor,
        pooled_context_radio_direction: torch.Tensor,
        full_scalar_summary: torch.Tensor,
        typed_context_statistics: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        ood_mask: torch.Tensor | None = None,
    ) -> ReliabilityConditionedResidualV21BOutput:
        base, context, full_scalar, context_statistics, declared_active, ood = (
            self._validate_inputs(
                base_descriptor,
                pooled_context_radio_direction,
                full_scalar_summary,
                typed_context_statistics,
                active_mask,
                ood_mask,
            )
        )
        candidate, tangent_update, tangent_gain, score, budget = self._candidate(
            base,
            context,
            full_scalar,
            context_statistics,
        )
        zero_update = (tangent_update == 0).all(dim=-1, keepdim=True)
        exact_identity = torch.where(zero_update, base, candidate)
        trainable_candidate = candidate + (exact_identity - candidate).detach()

        active = declared_active & ~ood
        semantic = torch.where(active[..., None], trainable_candidate, base)
        audited_update = torch.where(
            active[..., None], tangent_update, torch.zeros_like(tangent_update)
        )
        audited_gain = torch.where(
            active[..., None] & ~zero_update,
            tangent_gain,
            torch.zeros_like(tangent_gain),
        )
        return ReliabilityConditionedResidualV21BOutput(
            base_descriptor=base,
            semantic_descriptor=semantic,
            tangent_update=audited_update,
            tangent_gain=audited_gain,
            reliability_score=score,
            angular_budget_radians=budget.squeeze(-1),
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

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def architecture(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.ARCHITECTURE_NAME,
            "base": "immutable_external_accepted_v2_descriptor",
            "descriptor_dim": self.DESCRIPTOR_DIM,
            "context_dim": self.CONTEXT_DIM,
            "full_scalar_dim": self.FULL_SCALAR_DIM,
            "context_statistic_dim": self.CONTEXT_STATISTIC_DIM,
            "hidden_rank": self.HIDDEN_RANK,
            "conditioning": (
                "base_context_scalar_with_two_multiplicative_interactions"
            ),
            "scalar_normalization": "frozen_source_only_median_robust_scale",
            "full_scalar_names_sha256": SURFACE_REGION_FULL_SCALAR_NAMES_SHA256,
            "typed_context_statistic_names_sha256": (
                TYPED_CONTEXT_STATISTIC_NAMES_SHA256
            ),
            "reliability_component_names": list(
                V21B_RELIABILITY_COMPONENT_NAMES
            ),
            "reliability_component_names_sha256": (
                V21B_RELIABILITY_COMPONENT_NAMES_SHA256
            ),
            "reliability_aggregation": "fixed_unweighted_mean_clamped_0_1",
            "reliability_learned": False,
            "low_reliability_angle_radians": (
                self.LOW_RELIABILITY_ANGLE_RADIANS
            ),
            "high_reliability_angle_radians": (
                self.HIGH_RELIABILITY_ANGLE_RADIANS
            ),
            "angular_budget": "linear_0.15_plus_0.60_times_reliability",
            "max_tangent_gain": self.MAX_TANGENT_GAIN,
            "residual_gauge": "accepted_v2_descriptor_tangent_plane",
            "candidate_map": "unit_sphere_exponential_map",
            "fallback": "inactive_or_ood_bitwise_accepted_v2_base",
            "initialization": "zero_final_projection_exact_base_identity",
            "scene_parameters": False,
            "per_scene_hyperparameters": False,
            "query_independent": True,
            "runtime_query_strings_consumed": False,
            "trainable_parameter_count": self.trainable_parameter_count(),
        }
        payload["digest"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return payload


__all__ = [
    "ReliabilityConditionedResidualV21BOutput",
    "SurfaceRegionAcceptedV2ReliabilityConditionedResidualV21B",
    "V21B_RELIABILITY_COMPONENT_NAMES",
    "V21B_RELIABILITY_COMPONENT_NAMES_SHA256",
]
