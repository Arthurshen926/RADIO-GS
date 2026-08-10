"""Frozen text seed plus bounded canonical-region co-membership expansion."""

from __future__ import annotations

import math

import torch

from radio_gs.querying.bounded_region_comembership_readout import (
    bounded_regions_for_seed,
    bridge_free_component_ids,
    thresholded_adjacency,
)
from radio_gs.querying.region_comembership_readout import (
    RegionCoMembershipReadout,
)


FORMAL_METHODS = (
    "maximum_product",
    "dual_path_widest",
    "multipoint_consistency",
)
FORMAL_MAXIMUM_REGIONS = (1, 2, 4, 8)


def bounded_region_union_from_o0(
    *,
    region_o0_scores: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_probabilities: torch.Tensor,
    probability_threshold: float,
    method: str,
    maximum_regions: int,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    num_primitives: int,
) -> RegionCoMembershipReadout:
    """Apply the globally selected source rule to each frozen O0 query seed."""

    scores = torch.as_tensor(region_o0_scores).detach().float().cpu()
    rows = torch.as_tensor(region_rows).detach().long().cpu()
    mask = torch.as_tensor(token_mask).detach().bool().cpu()
    count = int(rows.shape[0]) if rows.ndim == 2 else -1
    primitive_count = int(num_primitives)
    maximum = int(maximum_regions)
    threshold = float(probability_threshold)
    if (
        scores.ndim != 2
        or scores.shape[0] != count
        or scores.shape[1] <= 0
        or mask.shape != rows.shape
        or primitive_count <= 0
        or method not in FORMAL_METHODS
        or maximum not in FORMAL_MAXIMUM_REGIONS
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
        or not bool(torch.isfinite(scores).all())
        or not bool(mask.any(dim=1).all())
        or bool((rows[mask] < 0).any())
        or bool((rows[mask] >= primitive_count).any())
    ):
        raise ValueError("bounded RegionCoMembership O0 readout inputs differ")
    # torch.argmax implements the frozen lower-canonical-index tie break.
    seeds = torch.argmax(scores, dim=0).long().contiguous()
    selected = torch.zeros(count, scores.shape[1], dtype=torch.bool)
    membership = torch.zeros(primitive_count, scores.shape[1], dtype=torch.float32)
    # Validate the complete inference graph even for K=1; K=1 ignores its
    # edges for selection, but must not let a corrupt authority pass silently.
    adjacency = thresholded_adjacency(
        region_count=count,
        pair_indices=pair_indices,
        pair_probabilities=pair_probabilities,
        threshold=threshold,
    )
    components = (
        bridge_free_component_ids(adjacency)
        if maximum > 1 and method == "dual_path_widest"
        else None
    )
    for query, seed in enumerate(seeds.tolist()):
        chosen = (
            (seed,)
            if maximum == 1
            else bounded_regions_for_seed(
                method=method,
                seed_region_index=seed,
                adjacency=adjacency,
                maximum_regions=maximum,
                bridge_free_components=components,
            )
        )
        chosen_tensor = torch.tensor(chosen, dtype=torch.long)
        selected[chosen_tensor, query] = True
        active_rows = rows[chosen_tensor][mask[chosen_tensor]]
        membership[torch.unique(active_rows), query] = 1.0
    return RegionCoMembershipReadout(
        primitive_membership=membership.contiguous(),
        seed_region_indices=seeds,
        selected_region_masks=selected.contiguous(),
    )


__all__ = [
    "FORMAL_MAXIMUM_REGIONS",
    "FORMAL_METHODS",
    "bounded_region_union_from_o0",
]
