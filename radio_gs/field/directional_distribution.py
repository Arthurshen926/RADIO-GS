"""Query-free multi-view direction distributions for canonical primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


DIRECTIONAL_PROTOTYPE_CONTRACT = "weighted_spherical_two_prototype_v1"


@dataclass(frozen=True)
class DirectionalPrototypeSet:
    """Two spherical modes and their observation mass per primitive."""

    prototypes: torch.Tensor
    mixture_weight: torch.Tensor
    valid: torch.Tensor
    observation_count: torch.Tensor
    center_resultant: torch.Tensor


def _validated_observations(
    observations: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(observations).float()
    mask = torch.as_tensor(valid, device=values.device).bool()
    if values.ndim != 3:
        raise ValueError("observations must be [views,primitives,channels]")
    if mask.shape != values.shape[:2]:
        raise ValueError("valid must align with observation views and primitives")
    if weights is None:
        mass = mask.to(values.dtype)
    else:
        mass = torch.as_tensor(weights, device=values.device, dtype=values.dtype)
        if mass.shape != mask.shape:
            raise ValueError("weights must align with valid")
        if bool((mass < 0).any()) or not bool(torch.isfinite(mass).all()):
            raise ValueError("weights must be finite and non-negative")
        mass = mass * mask
    directions = F.normalize(values, dim=-1, eps=1e-8)
    nonzero = values.norm(dim=-1) > 1e-8
    mask = mask & nonzero & (mass > 0)
    mass = mass * mask
    return directions, mask, mass


@torch.no_grad()
def fit_two_direction_prototypes(
    observations: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    iterations: int = 4,
) -> DirectionalPrototypeSet:
    """Fit a deterministic two-mode spherical summary without labels.

    The first seed is the observation nearest the weighted mean direction.
    The second maximizes supported angular novelty.  A few weighted spherical
    k-means updates then summarize dominant and minority directions.  There is
    no scene/query threshold: unimodal rows naturally collapse to one mode.
    """

    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")
    directions, mask, mass = _validated_observations(observations, valid, weights)
    views, primitives, channels = directions.shape
    total_mass = mass.sum(dim=0)
    row_valid = total_mass > 0
    count = mask.sum(dim=0)
    weighted_sum = (directions * mass[..., None]).sum(dim=0)
    resultant = weighted_sum.norm(dim=-1) / total_mass.clamp_min(1e-8)
    center = F.normalize(weighted_sum, dim=-1, eps=1e-8)

    center_similarity = (directions * center[None]).sum(dim=-1)
    seed0_score = center_similarity.masked_fill(~mask, -float("inf"))
    seed0_index = seed0_score.argmax(dim=0)
    primitive_index = torch.arange(primitives, device=directions.device)
    seed0 = directions[seed0_index, primitive_index]
    novelty = mass * (1.0 - (directions * seed0[None]).sum(dim=-1)).clamp_min(0.0)
    novelty = novelty.masked_fill(~mask, -float("inf"))
    seed1_index = novelty.argmax(dim=0)
    seed1 = directions[seed1_index, primitive_index]
    prototypes = torch.stack([seed0, seed1], dim=1)
    prototypes[~row_valid] = 0.0

    cluster_mass = torch.zeros(primitives, 2, device=directions.device)
    for _ in range(int(iterations)):
        similarity = torch.einsum("vnd,nkd->vnk", directions, prototypes)
        assignment = similarity.argmax(dim=-1)
        updated = torch.zeros_like(prototypes)
        cluster_mass.zero_()
        for cluster in range(2):
            selected_mass = mass * (assignment == cluster)
            cluster_mass[:, cluster] = selected_mass.sum(dim=0)
            updated[:, cluster] = (
                directions * selected_mass[..., None]
            ).sum(dim=0)
        nonempty = cluster_mass > 0
        normalized = F.normalize(updated, dim=-1, eps=1e-8)
        prototypes = torch.where(nonempty[..., None], normalized, prototypes)
    mixture = cluster_mass / total_mass[:, None].clamp_min(1e-8)
    mixture[~row_valid] = 0.0
    prototypes[~row_valid] = 0.0
    return DirectionalPrototypeSet(
        prototypes=prototypes,
        mixture_weight=mixture,
        valid=row_valid,
        observation_count=count,
        center_resultant=resultant.masked_fill(~row_valid, 0.0),
    )


def directional_prototype_coverage(
    prototypes: torch.Tensor,
    observations: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compare a two-mode set with the single normalized center on observations."""

    center_values, prototype_values, active_mass = (
        directional_prototype_observation_cosines(
            prototypes, observations, valid, weights
        )
    )
    return {
        "center_weighted_mean_cosine": (
            center_values * active_mass
        ).sum() / active_mass.sum(),
        "prototype_weighted_mean_cosine": (
            prototype_values * active_mass
        ).sum() / active_mass.sum(),
        "center_p05_cosine": torch.quantile(center_values, 0.05),
        "prototype_p05_cosine": torch.quantile(prototype_values, 0.05),
    }


def directional_prototype_observation_cosines(
    prototypes: torch.Tensor,
    observations: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return row-filtered center/prototype similarities for exact aggregation."""

    directions, mask, mass = _validated_observations(observations, valid, weights)
    modes = torch.as_tensor(prototypes, device=directions.device).float()
    if modes.shape != (directions.shape[1], 2, directions.shape[2]):
        raise ValueError("prototypes must be [primitives,2,channels]")
    modes = F.normalize(modes, dim=-1, eps=1e-8)
    total_mass = mass.sum()
    if not bool(total_mass > 0):
        raise ValueError("directional coverage has no valid observation")
    weighted_sum = (directions * mass[..., None]).sum(dim=0)
    center = F.normalize(weighted_sum, dim=-1, eps=1e-8)
    center_similarity = (directions * center[None]).sum(dim=-1)
    prototype_similarity = torch.einsum("vnd,nkd->vnk", directions, modes).amax(dim=-1)
    center_values = center_similarity[mask]
    prototype_values = prototype_similarity[mask]
    active_mass = mass[mask]
    return center_values, prototype_values, active_mass


def directional_set_rms_loss(
    predicted: torch.Tensor,
    observations: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tail-sensitive single-center loss used before prototype readout exists."""

    directions, mask, mass = _validated_observations(observations, valid, weights)
    prediction = F.normalize(torch.as_tensor(predicted).float(), dim=-1, eps=1e-8)
    if prediction.shape != directions.shape[1:]:
        raise ValueError("predicted must be [primitives,channels]")
    error = 1.0 - (directions * prediction[None]).sum(dim=-1)
    active = mass.sum()
    if not bool(active > 0):
        return prediction.sum() * 0.0
    # RMS angular error has no hand-selected tail fraction and penalizes a few
    # destroyed minority views more strongly than another mean target does.
    return ((error.square() * mass).sum() / active).clamp_min(0.0).sqrt()


def directional_set_ranking_loss(
    predicted: torch.Tensor,
    positive_observations: torch.Tensor,
    positive_valid: torch.Tensor,
    negative_observations: torch.Tensor,
    negative_valid: torch.Tensor,
) -> torch.Tensor:
    """Require every supported positive direction to outrank context negatives."""

    prediction = F.normalize(torch.as_tensor(predicted).float(), dim=-1, eps=1e-8)
    positives, positive_mask, _ = _validated_observations(
        positive_observations, positive_valid, None
    )
    negatives, negative_mask, _ = _validated_observations(
        negative_observations, negative_valid, None
    )
    if prediction.shape != positives.shape[1:] or prediction.shape != negatives.shape[1:]:
        raise ValueError("directional ranking rows/channels must align")
    positive_similarity = (positives * prediction[None]).sum(dim=-1)
    negative_similarity = (negatives * prediction[None]).sum(dim=-1)
    positive_floor = positive_similarity.masked_fill(~positive_mask, float("inf")).amin(dim=0)
    negative_ceiling = negative_similarity.masked_fill(
        ~negative_mask, -float("inf")
    ).amax(dim=0)
    active = positive_mask.any(dim=0) & negative_mask.any(dim=0)
    if not bool(active.any()):
        return prediction.sum() * 0.0
    # Zero margin encodes ordering only; it introduces no benchmark- or
    # scene-specific separation constant.
    return F.relu(negative_ceiling[active] - positive_floor[active]).mean()
