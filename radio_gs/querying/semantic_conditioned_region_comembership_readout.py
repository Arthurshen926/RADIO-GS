"""Absolute-relevance-conditioned bounded canonical-region expansion.

The semantic descriptor owns the query seed and the fixed positive/negative
decision boundary.  Query-independent co-membership only expands inside that
semantic-positive subgraph.  This keeps the two capabilities separate:

* source-distilled SigLIP2 relevance decides *what* the query refers to;
* source-selected co-membership decides *which additional supports* belong to
  the same object.

There is deliberately no benchmark-specific threshold.  Relevance is the
binary positive-versus-canonical-negative probability used by the frozen
query scorer, so ``0.5`` is its model-defined equal-logit boundary.
"""

from __future__ import annotations

import torch

from radio_gs.querying.bounded_region_comembership_query_readout import (
    FORMAL_MAXIMUM_REGIONS,
    FORMAL_METHODS,
)
from radio_gs.querying.bounded_region_comembership_readout import (
    Adjacency,
    bounded_regions_for_seed,
    bridge_free_component_ids,
    thresholded_adjacency,
)
from radio_gs.querying.region_comembership_readout import (
    RegionCoMembershipReadout,
)


ABSOLUTE_RELEVANCE_BOUNDARY = 0.5


def _semantic_positive_adjacency(
    adjacency: Adjacency,
    positive: torch.Tensor,
    *,
    seed_region_index: int,
) -> Adjacency:
    """Remove edges incident to a semantic-negative non-seed endpoint."""

    keep = torch.as_tensor(positive).detach().bool().cpu().clone()
    seed = int(seed_region_index)
    if keep.shape != (len(adjacency),) or seed < 0 or seed >= len(adjacency):
        raise ValueError("semantic-positive adjacency inputs differ")
    # Every query must retain one well-defined mask even if every candidate is
    # below the equal-logit relevance boundary.
    keep[seed] = True
    return [
        [edge for edge in neighbors if bool(keep[node]) and bool(keep[edge[0]])]
        for node, neighbors in enumerate(adjacency)
    ]


def semantic_conditioned_bounded_region_union(
    *,
    region_absolute_relevance: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_probabilities: torch.Tensor,
    probability_threshold: float,
    method: str,
    maximum_regions: int,
    region_rows: torch.Tensor,
    token_mask: torch.Tensor,
    num_primitives: int,
) -> RegionCoMembershipReadout:
    """Select a V2.1 semantic seed and expand only through positive regions.

    ``region_absolute_relevance`` must be computed with the same canonical
    negative bank and logit scale as V2.1 source supervision.  The method, K,
    and edge threshold remain the globally source-selected co-membership rule.
    """

    relevance = torch.as_tensor(region_absolute_relevance).detach().float().cpu()
    rows = torch.as_tensor(region_rows).detach().long().cpu()
    mask = torch.as_tensor(token_mask).detach().bool().cpu()
    count = int(rows.shape[0]) if rows.ndim == 2 else -1
    primitive_count = int(num_primitives)
    maximum = int(maximum_regions)
    if (
        relevance.ndim != 2
        or relevance.shape[0] != count
        or relevance.shape[1] <= 0
        or not bool(torch.isfinite(relevance).all())
        or bool((relevance < 0).any())
        or bool((relevance > 1).any())
        or mask.shape != rows.shape
        or primitive_count <= 0
        or method not in FORMAL_METHODS
        or maximum not in FORMAL_MAXIMUM_REGIONS
        or not bool(mask.any(dim=1).all())
        or bool((rows[mask] < 0).any())
        or bool((rows[mask] >= primitive_count).any())
    ):
        raise ValueError("semantic-conditioned co-membership inputs differ")

    base_adjacency = thresholded_adjacency(
        region_count=count,
        pair_indices=pair_indices,
        pair_probabilities=pair_probabilities,
        threshold=float(probability_threshold),
    )
    # torch.argmax provides the same frozen lower-row tie break as O0.
    seeds = torch.argmax(relevance, dim=0).long().contiguous()
    selected = torch.zeros(count, relevance.shape[1], dtype=torch.bool)
    membership = torch.zeros(primitive_count, relevance.shape[1], dtype=torch.float32)
    for query, seed in enumerate(seeds.tolist()):
        positive = relevance[:, query] >= ABSOLUTE_RELEVANCE_BOUNDARY
        adjacency = _semantic_positive_adjacency(
            base_adjacency, positive, seed_region_index=seed
        )
        components = (
            bridge_free_component_ids(adjacency)
            if maximum > 1 and method == "dual_path_widest"
            else None
        )
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
    "ABSOLUTE_RELEVANCE_BOUNDARY",
    "semantic_conditioned_bounded_region_union",
]
