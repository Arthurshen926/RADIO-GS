"""Deterministic text-seed readout over a frozen co-membership graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from radio_gs.models.region_comembership_v1 import seed_connected_instance_filter


@dataclass(frozen=True)
class RegionCoMembershipReadout:
    primitive_membership: torch.Tensor
    seed_region_indices: torch.Tensor
    selected_region_masks: torch.Tensor


def connected_region_union_from_o0(
    *,
    region_o0_scores: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_probabilities: torch.Tensor,
    probability_threshold: float,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    num_primitives: int,
) -> RegionCoMembershipReadout:
    """Select the unique highest-O0 seed and return its connected region union."""

    scores = torch.as_tensor(region_o0_scores).detach().float().cpu()
    rows = torch.as_tensor(region_rows).detach().long().cpu()
    mask = torch.as_tensor(token_mask).detach().bool().cpu()
    count = int(rows.shape[0]) if rows.ndim == 2 else -1
    primitive_count = int(num_primitives)
    if (
        scores.ndim != 2
        or scores.shape[0] != count
        or scores.shape[1] <= 0
        or mask.shape != rows.shape
        or primitive_count <= 0
        or not bool(torch.isfinite(scores).all())
        or bool((rows[mask] < 0).any())
        or bool((rows[mask] >= primitive_count).any())
        or bool((mask & (rows < 0)).any())
        or not bool(mask.any(dim=1).all())
    ):
        raise ValueError("RegionCoMembership O0 readout inputs differ")
    # torch.argmax returns the first maximum, which is the frozen lower
    # canonical-region-index tie break.
    seeds = torch.argmax(scores, dim=0).long().contiguous()
    selected = torch.zeros(count, scores.shape[1], dtype=torch.bool)
    membership = torch.zeros(primitive_count, scores.shape[1], dtype=torch.float32)
    for query, seed in enumerate(seeds.tolist()):
        region_mask = seed_connected_instance_filter(
            region_count=count,
            pair_indices=pair_indices,
            pair_probabilities=pair_probabilities,
            seed_region_indices=[seed],
            threshold=probability_threshold,
        )
        selected[:, query] = region_mask
        active_rows = rows[region_mask][mask[region_mask]]
        membership[torch.unique(active_rows), query] = 1.0
    return RegionCoMembershipReadout(
        primitive_membership=membership.contiguous(),
        seed_region_indices=seeds,
        selected_region_masks=selected.contiguous(),
    )


__all__ = ["RegionCoMembershipReadout", "connected_region_union_from_o0"]
