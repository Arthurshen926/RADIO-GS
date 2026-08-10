"""Source-trained, query-independent AcceptedV2 region co-membership head."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch
from torch import nn


PAIR_FEATURE_NAMES = (
    "frozen_descriptor_cosine",
    "pooled_typed_context_cosine",
    "typed_context_statistics_absolute_difference_mean",
    "typed_context_statistics_absolute_difference_maximum",
    "core_overlap_over_minimum_core_size",
    "anchor_support_adjacency",
    "anchor_support_raw_affinity",
    "centroid_distance_m",
    "centroid_distance_over_maximum_region_radius",
    "absolute_log_radius_difference",
    "same_scale_index",
    "minimum_core_observation_evidence",
    "minimum_core_visibility_purity",
    "absolute_core_observation_evidence_difference",
    "absolute_core_visibility_purity_difference",
)


class RegionCoMembershipV1(nn.Module):
    """One shared symmetric linear-logistic pair head.

    Pair features are symmetric by construction.  A zero initialized head is
    the explicit epoch-zero identity candidate and emits probability 0.5.
    """

    def __init__(self, median: torch.Tensor, robust_scale: torch.Tensor) -> None:
        super().__init__()
        center = torch.as_tensor(median).detach().float().cpu().contiguous()
        scale = torch.as_tensor(robust_scale).detach().float().cpu().contiguous()
        dimension = len(PAIR_FEATURE_NAMES)
        if (
            center.shape != (dimension,)
            or scale.shape != (dimension,)
            or not bool(torch.isfinite(center).all())
            or not bool(torch.isfinite(scale).all())
            or bool((scale <= 0).any())
        ):
            raise ValueError("co-membership normalization must be finite [15]")
        self.register_buffer("feature_median", center)
        self.register_buffer("feature_robust_scale", scale)
        self.logit = nn.Linear(dimension, 1)
        nn.init.zeros_(self.logit.weight)
        nn.init.zeros_(self.logit.bias)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(pair_features)
        if (
            values.ndim != 2
            or values.shape[1] != len(PAIR_FEATURE_NAMES)
            or not values.is_floating_point()
            or not bool(torch.isfinite(values).all())
            or values.device != self.feature_median.device
            or values.dtype != self.logit.weight.dtype
        ):
            raise ValueError("co-membership pair features must be finite float [P,15]")
        normalized = (values - self.feature_median) / self.feature_robust_scale
        return self.logit(normalized).squeeze(-1)

    def probability(self, pair_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(pair_features))


def seed_connected_instance_filter(
    *,
    region_count: int,
    pair_indices: torch.Tensor,
    pair_probabilities: torch.Tensor,
    seed_region_indices: Sequence[int] | torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return regions connected to at least one text seed by accepted edges."""

    count = int(region_count)
    pairs = torch.as_tensor(pair_indices).detach().long().cpu()
    probability = torch.as_tensor(pair_probabilities).detach().float().cpu()
    seeds = torch.as_tensor(seed_region_indices).detach().long().cpu().reshape(-1)
    cutoff = float(threshold)
    if count <= 0:
        raise ValueError("region_count must be positive")
    if (
        pairs.ndim != 2
        or pairs.shape[0] != 2
        or probability.shape != (pairs.shape[1],)
        or not bool(torch.isfinite(probability).all())
        or bool((probability < 0).any())
        or bool((probability > 1).any())
        or not 0.0 <= cutoff <= 1.0
    ):
        raise ValueError("co-membership pair graph differs")
    if (
        seeds.numel() == 0
        or bool((seeds < 0).any())
        or bool((seeds >= count).any())
        or bool((pairs < 0).any())
        or bool((pairs >= count).any())
        or bool((pairs[0] >= pairs[1]).any())
    ):
        raise ValueError("co-membership seeds or canonical pair order differ")
    selected_edges = pairs[:, probability >= cutoff]
    adjacency: list[list[int]] = [[] for _ in range(count)]
    for left, right in selected_edges.T.tolist():
        adjacency[left].append(right)
        adjacency[right].append(left)
    selected = torch.zeros(count, dtype=torch.bool)
    queue: deque[int] = deque()
    for seed in sorted(set(seeds.tolist())):
        selected[seed] = True
        queue.append(seed)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if not bool(selected[neighbor]):
                selected[neighbor] = True
                queue.append(neighbor)
    return selected


__all__ = [
    "PAIR_FEATURE_NAMES",
    "RegionCoMembershipV1",
    "seed_connected_instance_filter",
]
