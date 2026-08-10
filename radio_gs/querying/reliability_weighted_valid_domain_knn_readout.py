"""Query-free reliability-weighted extension of valid-domain LERF kNN.

The released readout averages the ten spatial neighbours uniformly.  That is
the minimum-variance estimator only when every primitive has the same noise
variance.  The source-view payload already provides a query-independent trust
amplitude

``q_i = sqrt(clamp((n_i - 1) / 3, 0, 1) * a_i)``,

where ``n_i`` is the retained-view count and ``a_i`` is their directional
resultant.  This module treats either ``q`` or ``q**2`` as a precision proxy
and combines it with a scene-scale-free Gaussian kernel.  All weights are
normalized per centre, and an all-zero precision neighbourhood falls back to
the spatial kernel.  No query, scene identifier, label, or metric enters the
operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from radio_gs.querying.valid_domain_knn_readout import (
    ValidDomainMultiscaleReadout,
    canonical_negative_probability,
    valid_domain_minmax_remap_scores,
)


CONTRACT = "radio_gs.lerf_reliability_weighted_valid_domain_knn_readout.v2"
KNN_K = 10
RETAINED_VIEW_CAPACITY = 4


@dataclass(frozen=True)
class WeightPolicy:
    policy_id: str
    gaussian_distance: bool
    reliability_power: int
    exclude_self: bool = False


# Eight candidates, fixed before any source or target result is inspected.
POLICIES = (
    WeightPolicy("uniform", False, 0),
    WeightPolicy("gaussian", True, 0),
    WeightPolicy("reliability_amplitude", False, 1),
    WeightPolicy("reliability_precision", False, 2),
    WeightPolicy("gaussian_reliability_amplitude", True, 1),
    WeightPolicy("gaussian_reliability_precision", True, 2),
    WeightPolicy("uniform_leave_self_out", False, 0, True),
    WeightPolicy("gaussian_reliability_precision_leave_self_out", True, 2, True),
)
POLICY_BY_ID = {policy.policy_id: policy for policy in POLICIES}


def source_view_reliability(
    retained_view_count: torch.Tensor,
    directional_resultant: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the existing LERF query-free source-view trust amplitude."""

    count = torch.as_tensor(retained_view_count).detach().float().cpu().reshape(-1)
    agreement = (
        torch.as_tensor(directional_resultant).detach().float().cpu().reshape(-1)
    )
    if count.shape != agreement.shape or count.numel() == 0:
        raise ValueError("retained count and agreement axes differ")
    if (
        not bool(torch.isfinite(count).all())
        or not bool(torch.isfinite(agreement).all())
        or bool((count < 0).any())
        or bool((count > RETAINED_VIEW_CAPACITY).any())
        or not torch.equal(count, count.round())
        or bool((agreement < 0).any())
        or bool((agreement > 1).any())
    ):
        raise ValueError("source-view reliability inputs differ")
    valid = (
        count > 0
        if valid_mask is None
        else torch.as_tensor(valid_mask).detach().bool().cpu().reshape(-1)
    )
    if valid.shape != count.shape or bool((valid & (count <= 0)).any()):
        raise ValueError("valid source rows must have a retained view")
    sufficiency = ((count - 1.0) / (RETAINED_VIEW_CAPACITY - 1)).clamp(0.0, 1.0)
    result = torch.sqrt((sufficiency * agreement).clamp_min(0.0))
    result[~valid] = 0.0
    return result.contiguous()


def valid_domain_knn_distances_indices(
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int = KNN_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return valid global rows plus valid-only distances and indices."""

    points = torch.as_tensor(xyz).detach().float().cpu().contiguous()
    valid = torch.as_tensor(valid_mask).detach().bool().cpu().reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or valid.shape != (points.shape[0],):
        raise ValueError("xyz/valid_mask axes differ")
    if not bool(torch.isfinite(points).all()) or not bool(valid.any()):
        raise ValueError("xyz must be finite and valid_mask non-empty")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")

    from sklearn.neighbors import NearestNeighbors

    rows = torch.where(valid)[0]
    count = min(k, int(rows.numel()))
    distance, local = NearestNeighbors(n_neighbors=count).fit(
        points[rows].numpy()
    ).kneighbors(points[rows].numpy(), return_distance=True)
    return (
        rows.contiguous(),
        torch.from_numpy(distance).float().contiguous(),
        rows[torch.from_numpy(local).long()].contiguous(),
    )


def normalized_neighbor_weights(
    distances: torch.Tensor,
    neighbor_reliability: torch.Tensor,
    *,
    policy_id: str,
    neighbor_is_self: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct normalized spatial/precision weights for one neighbour map."""

    distance = torch.as_tensor(distances).detach().float().cpu().contiguous()
    reliability = (
        torch.as_tensor(neighbor_reliability).detach().float().cpu().contiguous()
    )
    if distance.ndim != 2 or reliability.shape != distance.shape:
        raise ValueError("distance and neighbour reliability axes differ")
    if distance.shape[1] == 0:
        raise ValueError("at least one neighbour is required")
    if (
        not bool(torch.isfinite(distance).all())
        or not bool(torch.isfinite(reliability).all())
        or bool((distance < 0).any())
        or bool((reliability < 0).any())
        or bool((reliability > 1).any())
    ):
        raise ValueError("neighbour weights require finite in-range inputs")
    try:
        policy = POLICY_BY_ID[policy_id]
    except KeyError as error:
        raise ValueError(f"unknown weight policy: {policy_id}") from error

    self_mask = (
        torch.zeros_like(distance, dtype=torch.bool)
        if neighbor_is_self is None
        else torch.as_tensor(neighbor_is_self).detach().bool().cpu()
    )
    if self_mask.shape != distance.shape:
        raise ValueError("self-neighbour mask axes differ")
    if policy.exclude_self and neighbor_is_self is None:
        raise ValueError("leave-self-out policy requires an explicit self mask")

    if policy.gaussian_distance:
        # The farthest kNN distance is a local bandwidth, so the kernel has no
        # scene-unit or density hyperparameter.  Coincident rows use unit mass.
        bandwidth = distance.amax(dim=1, keepdim=True)
        ratio = torch.where(
            bandwidth > 0,
            distance / bandwidth.clamp_min(torch.finfo(torch.float32).tiny),
            torch.zeros_like(distance),
        )
        spatial = torch.exp(-0.5 * ratio.square())
    else:
        spatial = torch.ones_like(distance)
    precision = (
        torch.ones_like(reliability)
        if policy.reliability_power == 0
        else reliability.pow(policy.reliability_power)
    )
    unnormalized = spatial * precision
    if policy.exclude_self:
        unnormalized = unnormalized.masked_fill(self_mask, 0.0)
    mass = unnormalized.sum(dim=1, keepdim=True)
    # Reliability can be exactly zero for every single-view neighbour.  In
    # that case reliability is uninformative, not evidence for zero scores.
    fallback = spatial.masked_fill(self_mask, 0.0) if policy.exclude_self else spatial
    # A one-row domain has no independent neighbour; retaining its self score
    # is the only defined no-expansion fallback.
    fallback = torch.where(
        fallback.sum(dim=1, keepdim=True) > 0, fallback, spatial
    )
    unnormalized = torch.where(mass > 0, unnormalized, fallback)
    return (unnormalized / unnormalized.sum(dim=1, keepdim=True)).contiguous()


def reliability_weighted_valid_domain_knn_smoothed_scores(
    scores: torch.Tensor,
    xyz: torch.Tensor,
    reliability: torch.Tensor,
    *,
    policy_id: str,
    k: int = KNN_K,
    chunk_size: int = 65536,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend raw scores with a normalized valid-neighbour kernel estimate."""

    values = torch.as_tensor(scores).detach().float().cpu().contiguous()
    points = torch.as_tensor(xyz).detach().float().cpu().contiguous()
    trust = torch.as_tensor(reliability).detach().float().cpu().reshape(-1)
    if values.ndim != 2 or points.shape != (values.shape[0], 3):
        raise ValueError("scores and xyz axes differ")
    if trust.shape != (values.shape[0],):
        raise ValueError("reliability must align with scores")
    valid = (
        torch.ones(values.shape[0], dtype=torch.bool)
        if valid_mask is None
        else torch.as_tensor(valid_mask).detach().bool().cpu().reshape(-1)
    )
    if valid.shape != trust.shape or not bool(valid.any()):
        raise ValueError("valid_mask must keep at least one primitive")
    if (
        not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(points).all())
        or not bool(torch.isfinite(trust).all())
        or bool((trust < 0).any())
        or bool((trust > 1).any())
        or bool((trust[~valid] != 0).any())
    ):
        raise ValueError("weighted readout inputs differ")
    policy = POLICY_BY_ID.get(policy_id)
    if policy is None:
        raise ValueError(f"unknown weight policy: {policy_id}")
    retrieval_k = k + 1 if policy.exclude_self else k
    rows, distance, indices = valid_domain_knn_distances_indices(
        points, valid, k=retrieval_k
    )
    weights = normalized_neighbor_weights(
        distance,
        trust[indices],
        policy_id=policy_id,
        neighbor_is_self=indices == rows[:, None],
    )
    result = torch.zeros_like(values)
    step = max(1, int(chunk_size))
    for start in range(0, int(rows.numel()), step):
        stop = min(start + step, int(rows.numel()))
        # Preserve the promoted v1 control bit-for-bit.  Multiplication by a
        # represented 1/k followed by sum has different rounding than mean.
        neighbor = (
            values[indices[start:stop]].mean(dim=1)
            if policy_id == "uniform"
            else (
                values[indices[start:stop]] * weights[start:stop, :, None]
            ).sum(dim=1)
        )
        result[rows[start:stop]] = 0.5 * (
            values[rows[start:stop]] + neighbor
        )
    return result


def reliability_weighted_valid_domain_multiscale_readout(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    reliability: torch.Tensor,
    *,
    policy_id: str,
    k: int = KNN_K,
    chunk_size: int = 65536,
    logit_scale: float = 10.0,
) -> ValidDomainMultiscaleReadout:
    """Apply canonical probability, weighted kNN, remap, and scale choice."""

    if not math.isfinite(float(logit_scale)) or float(logit_scale) <= 0:
        raise ValueError("logit_scale must be finite and positive")
    probability = canonical_negative_probability(
        positive_scores, negative_scores, logit_scale=logit_scale
    )
    count, scales, queries = probability.shape
    valid = torch.as_tensor(valid_mask).detach().bool().cpu().reshape(-1)
    smoothed = reliability_weighted_valid_domain_knn_smoothed_scores(
        probability.reshape(count, scales * queries),
        xyz,
        reliability,
        policy_id=policy_id,
        k=k,
        chunk_size=chunk_size,
        valid_mask=valid,
    ).reshape(count, scales, queries)
    peaks = smoothed[valid].amax(dim=0)
    selected = peaks.argmax(dim=0).long()
    remapped = valid_domain_minmax_remap_scores(
        smoothed.reshape(count, scales * queries), valid_mask=valid
    ).reshape(count, scales, queries)
    query_axis = torch.arange(queries)
    scores = remapped[:, selected, query_axis].contiguous()
    return ValidDomainMultiscaleReadout(
        scores=scores,
        scores_by_scale=remapped.contiguous(),
        smoothed_probability=smoothed.contiguous(),
        selected_scale_indices=selected.contiguous(),
        raw_smoothed_peaks=peaks.contiguous(),
    )


__all__ = [
    "CONTRACT",
    "KNN_K",
    "POLICIES",
    "POLICY_BY_ID",
    "WeightPolicy",
    "normalized_neighbor_weights",
    "reliability_weighted_valid_domain_knn_smoothed_scores",
    "reliability_weighted_valid_domain_multiscale_readout",
    "source_view_reliability",
    "valid_domain_knn_distances_indices",
]
