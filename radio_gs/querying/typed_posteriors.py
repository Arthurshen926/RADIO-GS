"""Globally shared typed posterior interfaces over Universal Field v1.

The modules in this file never own scene state.  They consume primitive
semantic evidence plus the five query-independent reliability scalars and
emit calibrated Gaussian-domain posteriors.  Their residual paths are
zero-initialized so an untrained checkpoint is exactly Primitive Readout-v0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from radio_gs.universal_field_v1 import RELIABILITY_NAMES


MAX_POST_SPATIAL_PROBABILITY_RESIDUAL = 0.75


def validate_reliability_state(
    reliability: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and return canonical float/bool deployment reliability rows."""

    values = torch.as_tensor(reliability).float()
    mask = torch.as_tensor(valid, device=values.device).bool().reshape(-1)
    if values.ndim != 2 or values.shape != (mask.numel(), len(RELIABILITY_NAMES)):
        raise ValueError("reliability state must be [N,5]")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("reliability state must be finite")
    if bool((values[~mask] != 0).any()):
        raise ValueError("invalid primitives must have zero reliability")
    # A decoded fallback primitive may be query-valid even when no exact-MPR
    # reliability observation exists.  Its all-zero row means "unknown", not
    # fabricated certainty and not mandatory abstention.
    known = mask & values.ne(0).any(dim=-1)
    active = values[known]
    if active.numel():
        resultant, dispersion, log_amplitude_std, evidence, purity = active.unbind(-1)
        if (
            bool((resultant <= 0).any())
            or bool((resultant > 1).any())
            or bool((dispersion < 0).any())
            or bool((dispersion >= 1).any())
            or bool((log_amplitude_std < 0).any())
            or bool((evidence <= 0).any())
            or bool((evidence >= 1).any())
            or bool((purity < 0).any())
            or bool((purity > 1).any())
        ):
            raise ValueError("reliability values are outside the factorized contract")
        if not torch.allclose(
            dispersion,
            1.0 - resultant,
            atol=2e-6,
            rtol=2e-6,
        ):
            raise ValueError("reliability directional dispersion differs")
    return values, mask


@dataclass(frozen=True)
class TextPosteriorOutput:
    logits: torch.Tensor
    probability: torch.Tensor
    valid: torch.Tensor

    def select(self, *, threshold: float, ensure_nonempty: bool = True) -> torch.Tensor:
        cutoff = float(threshold)
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("text posterior threshold must be in [0,1]")
        selected = (self.probability >= cutoff) & self.valid[:, None]
        if ensure_nonempty:
            missing = ~selected.any(dim=0)
            if bool(missing.any()) and bool(self.valid.any()):
                eligible = self.probability.masked_fill(~self.valid[:, None], -1.0)
                anchors = eligible.argmax(dim=0)
                queries = torch.nonzero(missing, as_tuple=False).flatten()
                selected[anchors[queries], queries] = True
        return selected

    @property
    def zero_selection_count(self) -> int:
        return int((~(self.probability > 0).any(dim=0)).sum().item())


class TextPosteriorV2(nn.Module):
    """Continuous semantic-plus-extent residual over primitive text logits."""

    schema = "radio_gs.text_posterior_v2.v1"

    def __init__(self, *, extent_feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        extent_dim = int(extent_feature_dim)
        hidden = int(hidden_dim)
        if extent_dim < 0 or hidden <= 0:
            raise ValueError("TextPosteriorV2 dimensions are invalid")
        self.extent_feature_dim = extent_dim
        input_dim = 1 + 1 + len(RELIABILITY_NAMES) + extent_dim
        self.extent_residual = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.extent_residual[-1].weight)
        nn.init.zeros_(self.extent_residual[-1].bias)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.semantic_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        semantic_logit: torch.Tensor,
        *,
        reliability: torch.Tensor,
        valid: torch.Tensor,
        region_logit: torch.Tensor | None = None,
        extent_features: torch.Tensor | None = None,
    ) -> TextPosteriorOutput:
        semantic = torch.as_tensor(semantic_logit).float()
        if semantic.ndim != 2 or min(semantic.shape) <= 0:
            raise ValueError("semantic_logit must be non-empty [N,Q]")
        rel, mask = validate_reliability_state(reliability, valid)
        rel = rel.to(device=semantic.device, dtype=semantic.dtype)
        mask = mask.to(semantic.device)
        if rel.shape[0] != semantic.shape[0]:
            raise ValueError("semantic and reliability rows differ")
        if region_logit is None:
            region_delta = torch.zeros_like(semantic)
        else:
            region = torch.as_tensor(
                region_logit, device=semantic.device, dtype=semantic.dtype
            )
            if region.shape != semantic.shape or not bool(torch.isfinite(region).all()):
                raise ValueError("region_logit must align with semantic_logit")
            region_delta = region - semantic
        if extent_features is None:
            extent = semantic.new_zeros(
                semantic.shape[0], semantic.shape[1], self.extent_feature_dim
            )
        else:
            extent = torch.as_tensor(
                extent_features, device=semantic.device, dtype=semantic.dtype
            )
            if extent.shape != (
                semantic.shape[0],
                semantic.shape[1],
                self.extent_feature_dim,
            ) or not bool(torch.isfinite(extent).all()):
                raise ValueError("extent_features must be [N,Q,E]")
        rel_features = rel[:, None, :].expand(-1, semantic.shape[1], -1)
        features = torch.cat(
            (semantic[..., None], region_delta[..., None], rel_features, extent),
            dim=-1,
        )
        residual = self.extent_residual(features).squeeze(-1)
        temperature = self.log_temperature.exp().clamp(0.05, 20.0)
        logits = semantic / temperature + self.semantic_bias + residual
        probability = torch.sigmoid(logits).masked_fill(~mask[:, None], 0.0)
        logits = logits.masked_fill(~mask[:, None], 0.0)
        return TextPosteriorOutput(
            logits=logits.contiguous(),
            probability=probability.contiguous(),
            valid=mask.contiguous(),
        )

    def forward_post_spatial(
        self,
        semantic_probability: torch.Tensor,
        *,
        reliability: torch.Tensor,
        valid: torch.Tensor,
        region_probability: torch.Tensor | None = None,
        extent_features: torch.Tensor | None = None,
        residual_scale: float = 1.0,
    ) -> TextPosteriorOutput:
        """Apply the learned extent residual after a frozen spatial readout.

        ``semantic_probability`` is already the final target-blind spatial
        score (for LERF: kNN10 followed by the per-query scene min/max map).
        A zero residual is restored with ``where`` so initialization is an
        exact end-to-end identity, including values at zero and one.  The
        bounded additive probability residual can nevertheless recover rows
        clipped to zero by the frozen VALA map once source-only training has
        supplied positive evidence.
        """

        scale = float(residual_scale)
        if not 0.0 <= scale <= 1.0:
            raise ValueError("post-spatial residual_scale must be in [0,1]")
        semantic = torch.as_tensor(semantic_probability).float()
        if semantic.ndim != 2 or min(semantic.shape) <= 0:
            raise ValueError("semantic_probability must be non-empty [N,Q]")
        if not bool(torch.isfinite(semantic).all()) or bool(
            ((semantic < 0) | (semantic > 1)).any()
        ):
            raise ValueError("semantic_probability must be finite in [0,1]")
        rel, mask = validate_reliability_state(reliability, valid)
        rel = rel.to(device=semantic.device, dtype=semantic.dtype)
        mask = mask.to(semantic.device)
        if rel.shape[0] != semantic.shape[0]:
            raise ValueError("semantic and reliability rows differ")

        if region_probability is None:
            region_delta = torch.zeros_like(semantic)
        else:
            region = torch.as_tensor(
                region_probability, device=semantic.device, dtype=semantic.dtype
            )
            if (
                region.shape != semantic.shape
                or not bool(torch.isfinite(region).all())
                or bool(((region < 0) | (region > 1)).any())
            ):
                raise ValueError("region_probability must align in [0,1]")
            region_delta = region - semantic

        if extent_features is None:
            extent = semantic.new_zeros(
                semantic.shape[0], semantic.shape[1], self.extent_feature_dim
            )
        else:
            extent = torch.as_tensor(
                extent_features, device=semantic.device, dtype=semantic.dtype
            )
            if extent.shape != (
                semantic.shape[0],
                semantic.shape[1],
                self.extent_feature_dim,
            ) or not bool(torch.isfinite(extent).all()):
                raise ValueError("extent_features must be [N,Q,E]")

        rel_features = rel[:, None, :].expand(-1, semantic.shape[1], -1)
        features = torch.cat(
            (semantic[..., None], region_delta[..., None], rel_features, extent),
            dim=-1,
        )
        learned = self.extent_residual(features).squeeze(-1)
        temperature = self.log_temperature.exp().clamp(0.05, 20.0)
        calibration = semantic / temperature - semantic + self.semantic_bias
        residual = MAX_POST_SPATIAL_PROBABILITY_RESIDUAL * torch.tanh(
            learned + calibration
        )
        residual = scale * residual
        probability = (semantic + residual).clamp(0.0, 1.0)
        probability = probability.masked_fill(~mask[:, None], 0.0).contiguous()
        finite = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        logits = torch.logit(finite)
        logits = logits.masked_fill(~mask[:, None], 0.0).contiguous()
        if not torch.equal(
            probability[(residual == 0) & mask[:, None]],
            semantic[(residual == 0) & mask[:, None]],
        ):
            raise RuntimeError("zero post-spatial residual changed the base score")
        return TextPosteriorOutput(logits=logits, probability=probability, valid=mask.contiguous())


@dataclass(frozen=True)
class CategoricalPosteriorOutput:
    logits: torch.Tensor
    probability: torch.Tensor
    prediction: torch.Tensor


class CategoricalPosteriorV2(nn.Module):
    """Mutually exclusive class posterior with a learned background option."""

    schema = "radio_gs.categorical_posterior_v2.v1"

    def __init__(self, *, num_classes: int) -> None:
        super().__init__()
        count = int(num_classes)
        if count < 2:
            raise ValueError("CategoricalPosteriorV2 needs at least two classes")
        self.num_classes = count
        self.class_log_temperature = nn.Parameter(torch.zeros(count))
        self.class_bias = nn.Parameter(torch.zeros(count))
        self.background_bias = nn.Parameter(torch.tensor(-8.0))
        self.background_reliability = nn.Linear(len(RELIABILITY_NAMES), 1, bias=False)
        self.background_ambiguity = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(self.background_reliability.weight)
        nn.init.zeros_(self.background_ambiguity.weight)

    def forward(
        self,
        semantic_logits: torch.Tensor,
        *,
        reliability: torch.Tensor,
        valid: torch.Tensor,
        active_class_indices: torch.Tensor | None = None,
    ) -> CategoricalPosteriorOutput:
        semantic = torch.as_tensor(semantic_logits).float()
        if semantic.ndim != 2 or semantic.shape[1] != self.num_classes:
            raise ValueError("categorical semantic_logits shape differs")
        rel, mask = validate_reliability_state(reliability, valid)
        rel = rel.to(device=semantic.device, dtype=semantic.dtype)
        mask = mask.to(semantic.device)
        if rel.shape[0] != semantic.shape[0]:
            raise ValueError("categorical semantic and reliability rows differ")
        if active_class_indices is None:
            active = torch.arange(self.num_classes, device=semantic.device)
        else:
            active = torch.as_tensor(
                active_class_indices, device=semantic.device, dtype=torch.long
            ).reshape(-1)
            if (
                active.numel() < 2
                or int(active.min()) < 0
                or int(active.max()) >= self.num_classes
                or active.unique().numel() != active.numel()
            ):
                raise ValueError("active categorical class indices differ")
        temperature = self.class_log_temperature[active].exp().clamp(0.05, 20.0)
        class_logits = (
            semantic[:, active] / temperature[None, :]
            + self.class_bias[active][None, :]
        )
        top = torch.topk(class_logits, k=2, dim=-1).values
        margin = top[:, 0] - top[:, 1]
        ambiguity = torch.stack((top[:, 0], margin), dim=-1)
        background = (
            top[:, :1]
            + self.background_bias
            + self.background_reliability(rel)
            + self.background_ambiguity(ambiguity)
        )
        all_logits = torch.cat((class_logits, background), dim=-1)
        all_logits = all_logits.masked_fill(~mask[:, None], -80.0)
        all_logits[~mask, -1] = 0.0
        probability = torch.softmax(all_logits, dim=-1)
        selected = probability.argmax(dim=-1)
        prediction = torch.where(
            selected == active.numel(),
            torch.full_like(selected, -1),
            active[selected.clamp_max(active.numel() - 1)],
        )
        return CategoricalPosteriorOutput(
            logits=all_logits.contiguous(),
            probability=probability.contiguous(),
            prediction=prediction.contiguous(),
        )


class MarginalCategoricalPosteriorV2(nn.Module):
    """Low-dimensional target-blind calibration of class score marginals.

    The two learned strengths are shared by every class.  Class-specific
    location and scale are computed without labels from all valid rows in the
    current scene, so the same rule remains defined for a class that was not
    labeled in a source fit fold.  Zero initialization is an exact primitive
    argmax identity and the module has no background/abstention class.
    """

    schema = "radio_gs.marginal_categorical_posterior_v2.v1"

    def __init__(self, *, num_classes: int, minimum_scale: float = 1.0e-4) -> None:
        super().__init__()
        count = int(num_classes)
        floor = float(minimum_scale)
        if count < 2:
            raise ValueError("MarginalCategoricalPosteriorV2 needs at least two classes")
        if not 0.0 < floor < 1.0:
            raise ValueError("marginal categorical minimum_scale differs")
        self.num_classes = count
        self.minimum_scale = floor
        self.centering_parameter = nn.Parameter(torch.zeros(()))
        self.scaling_parameter = nn.Parameter(torch.zeros(()))

    def scene_marginals(
        self,
        semantic_logits: torch.Tensor,
        *,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic = torch.as_tensor(semantic_logits).float()
        mask = torch.as_tensor(valid, device=semantic.device).bool().reshape(-1)
        if semantic.ndim != 2 or semantic.shape != (mask.numel(), self.num_classes):
            raise ValueError("marginal categorical semantic_logits shape differs")
        if int(mask.sum()) < 2:
            raise ValueError("marginal categorical needs at least two valid rows")
        selected = semantic[mask]
        location = selected.mean(dim=0)
        scale = selected.std(dim=0, unbiased=False).clamp_min(self.minimum_scale)
        return location.contiguous(), scale.contiguous()

    def forward(
        self,
        semantic_logits: torch.Tensor,
        *,
        valid: torch.Tensor,
        class_location: torch.Tensor | None = None,
        class_scale: torch.Tensor | None = None,
        active_class_indices: torch.Tensor | None = None,
    ) -> CategoricalPosteriorOutput:
        semantic = torch.as_tensor(semantic_logits).float()
        mask = torch.as_tensor(valid, device=semantic.device).bool().reshape(-1)
        if semantic.ndim != 2 or semantic.shape != (mask.numel(), self.num_classes):
            raise ValueError("marginal categorical semantic_logits shape differs")
        if class_location is None or class_scale is None:
            if class_location is not None or class_scale is not None:
                raise ValueError("marginal categorical statistics must be provided together")
            location, scale = self.scene_marginals(semantic, valid=mask)
        else:
            location = torch.as_tensor(
                class_location, device=semantic.device, dtype=semantic.dtype
            ).reshape(-1)
            scale = torch.as_tensor(
                class_scale, device=semantic.device, dtype=semantic.dtype
            ).reshape(-1)
            if location.shape != (self.num_classes,) or scale.shape != (
                self.num_classes,
            ):
                raise ValueError("marginal categorical statistics shape differs")
            if not bool(torch.isfinite(location).all()) or not bool(
                torch.isfinite(scale).all()
            ):
                raise ValueError("marginal categorical statistics must be finite")
            if bool((scale < self.minimum_scale).any()):
                raise ValueError("marginal categorical class scale is too small")
        if active_class_indices is None:
            active = torch.arange(self.num_classes, device=semantic.device)
        else:
            active = torch.as_tensor(
                active_class_indices, device=semantic.device, dtype=torch.long
            ).reshape(-1)
            if (
                active.numel() < 2
                or int(active.min()) < 0
                or int(active.max()) >= self.num_classes
                or active.unique().numel() != active.numel()
            ):
                raise ValueError("active marginal categorical class indices differ")
        location = location[active]
        scale = scale[active]
        selected = semantic[:, active]
        common_location = location.mean()
        reference_scale = scale.log().mean().exp()
        centered = selected - location[None, :] + common_location
        standardized = (
            (selected - location[None, :])
            * (reference_scale / scale)[None, :]
            + common_location
        )
        centering_strength = self.centering_parameter.tanh()
        scaling_strength = self.scaling_parameter.tanh()
        logits = (
            selected
            + centering_strength * (centered - selected)
            + scaling_strength * (standardized - centered)
        )
        logits = logits.masked_fill(~mask[:, None], -80.0)
        probability = torch.softmax(logits, dim=-1)
        prediction = active[probability.argmax(dim=-1)]
        prediction = torch.where(mask, prediction, torch.full_like(prediction, -1))
        return CategoricalPosteriorOutput(
            logits=logits.contiguous(),
            probability=probability.contiguous(),
            prediction=prediction.contiguous(),
        )


__all__ = [
    "CategoricalPosteriorOutput",
    "CategoricalPosteriorV2",
    "MarginalCategoricalPosteriorV2",
    "TextPosteriorOutput",
    "TextPosteriorV2",
    "MAX_POST_SPATIAL_PROBABILITY_RESIDUAL",
    "validate_reliability_state",
]
