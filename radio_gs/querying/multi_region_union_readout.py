"""Target-blind multi-region set readout over registered semantic cores.

The readout consumes final region probabilities and query-independent region
memberships.  It never consumes labels, masks, scene names, coordinates, or
benchmark-specific parameters.  Greedy novelty prevents duplicate overlapping
regions from exhausting the fixed set budget while retaining disconnected
high-confidence object parts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class MultiRegionUnionConfig:
    score_threshold: float = 0.6
    maximum_regions: int = 8
    candidate_chunk_rows: int = 4096

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.score_threshold))
            or not 0.0 <= float(self.score_threshold) <= 1.0
        ):
            raise ValueError("score_threshold must be finite and lie in [0,1]")
        if int(self.maximum_regions) <= 0:
            raise ValueError("maximum_regions must be positive")
        if int(self.candidate_chunk_rows) <= 0:
            raise ValueError("candidate_chunk_rows must be positive")


@dataclass(frozen=True)
class MultiRegionUnionResult:
    primitive_membership: torch.Tensor
    selected_region_indices: tuple[tuple[int, ...], ...]
    selected_region_scores: tuple[tuple[float, ...], ...]
    selected_marginal_core_rows: tuple[tuple[int, ...], ...]


def _validated_inputs(
    region_probability: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    *,
    num_primitives: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probability = torch.as_tensor(region_probability).detach().float().cpu()
    rows = torch.as_tensor(region_rows).detach().cpu()
    core = torch.as_tensor(core_mask).detach().bool().cpu()
    if probability.ndim != 2 or min(probability.shape) <= 0:
        raise ValueError("region_probability must be nonempty [R,Q]")
    if rows.ndim != 2 or rows.shape[0] != probability.shape[0]:
        raise ValueError("region_rows must align as [R,T]")
    if core.shape != rows.shape:
        raise ValueError("core_mask must align with region_rows")
    if rows.dtype not in {torch.int32, torch.int64}:
        raise ValueError("region_rows must use an integer dtype")
    if not isinstance(num_primitives, int) or num_primitives <= 0:
        raise ValueError("num_primitives must be positive")
    valid = rows >= 0
    if bool((rows[valid] >= num_primitives).any()):
        raise ValueError("region_rows contains an out-of-range primitive")
    if bool((core & ~valid).any()) or not bool(core.any(dim=1).all()):
        raise ValueError("every region needs a nonempty valid semantic core")
    if not bool(torch.isfinite(probability).all()) or bool(
        ((probability < 0.0) | (probability > 1.0)).any()
    ):
        raise ValueError("region_probability must be finite and lie in [0,1]")
    # Duplicate core rows would give a region an implementation-dependent
    # novelty denominator.  Fail closed instead of silently deduplicating.
    for index in range(rows.shape[0]):
        active = rows[index, core[index]].long()
        if int(torch.unique(active).numel()) != int(active.numel()):
            raise ValueError("a region semantic core contains duplicate primitives")
    return probability.contiguous(), rows.contiguous(), core.contiguous()


def greedy_novelty_union_readout(
    region_probability: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    *,
    num_primitives: int,
    config: MultiRegionUnionConfig | None = None,
) -> MultiRegionUnionResult:
    """Select a fixed-size target-blind region set independently per query.

    At every iteration, candidate utility is its final probability multiplied
    by the fraction of semantic-core primitives not covered by the current
    union.  ``torch.argmax`` supplies the registered smaller-index tie break.
    The returned primitive field is binary because the evidence gate already
    makes the selection decision and the downstream frozen renderer consumes
    membership probabilities.
    """

    chosen = MultiRegionUnionConfig() if config is None else config
    if not isinstance(chosen, MultiRegionUnionConfig):
        raise TypeError("config must be MultiRegionUnionConfig")
    probability, rows, core = _validated_inputs(
        region_probability,
        region_rows,
        core_mask,
        num_primitives=num_primitives,
    )
    region_count, query_count = probability.shape
    core_count = core.sum(dim=1).float()
    output = torch.zeros(num_primitives, query_count, dtype=torch.float32)
    all_indices: list[tuple[int, ...]] = []
    all_scores: list[tuple[float, ...]] = []
    all_marginals: list[tuple[int, ...]] = []

    for query in range(query_count):
        covered = torch.zeros(num_primitives, dtype=torch.bool)
        selected = torch.zeros(region_count, dtype=torch.bool)
        indices: list[int] = []
        scores: list[float] = []
        marginals: list[int] = []
        gated = probability[:, query] >= float(chosen.score_threshold)
        for _ in range(int(chosen.maximum_regions)):
            best_index = -1
            best_utility = 0.0
            best_marginal = 0
            step = int(chosen.candidate_chunk_rows)
            for start in range(0, region_count, step):
                stop = min(start + step, region_count)
                eligible = gated[start:stop] & ~selected[start:stop]
                if not bool(eligible.any()):
                    continue
                chunk_rows = rows[start:stop]
                chunk_core = core[start:stop]
                safe = chunk_rows.clamp_min(0).long()
                novel = chunk_core & ~covered[safe]
                marginal = novel.sum(dim=1)
                utility = probability[start:stop, query] * (
                    marginal.float() / core_count[start:stop]
                )
                utility = utility.masked_fill(~eligible, -1.0)
                local = int(torch.argmax(utility))
                candidate_utility = float(utility[local])
                candidate_index = start + local
                if candidate_utility > best_utility or (
                    candidate_utility == best_utility
                    and candidate_utility > 0.0
                    and (best_index < 0 or candidate_index < best_index)
                ):
                    best_index = candidate_index
                    best_utility = candidate_utility
                    best_marginal = int(marginal[local])
            if best_index < 0 or best_utility <= 0.0:
                break
            selected[best_index] = True
            active_rows = rows[best_index, core[best_index]].long()
            covered[active_rows] = True
            indices.append(best_index)
            scores.append(float(probability[best_index, query]))
            marginals.append(best_marginal)
        output[covered, query] = 1.0
        all_indices.append(tuple(indices))
        all_scores.append(tuple(scores))
        all_marginals.append(tuple(marginals))

    return MultiRegionUnionResult(
        primitive_membership=output.contiguous(),
        selected_region_indices=tuple(all_indices),
        selected_region_scores=tuple(all_scores),
        selected_marginal_core_rows=tuple(all_marginals),
    )


def greedy_connected_expected_mass_union_readout(
    region_probability: torch.Tensor,
    region_rows: torch.Tensor,
    core_mask: torch.Tensor,
    support_edge_index: torch.Tensor,
    *,
    num_primitives: int,
    config: MultiRegionUnionConfig | None = None,
) -> MultiRegionUnionResult:
    """Select one expected-mass seed, then only support-connected regions.

    The seed maximizes ``probability * core_count``.  Every later candidate
    must overlap the current semantic-core union or touch it by one immutable
    support-graph edge, and maximizes ``probability * novel_core_count``.
    No coordinate distance, graph affinity threshold, or fitted coefficient
    enters this discrete query-time rule.
    """

    chosen = MultiRegionUnionConfig() if config is None else config
    if not isinstance(chosen, MultiRegionUnionConfig):
        raise TypeError("config must be MultiRegionUnionConfig")
    probability, rows, core = _validated_inputs(
        region_probability,
        region_rows,
        core_mask,
        num_primitives=num_primitives,
    )
    edge = torch.as_tensor(support_edge_index).detach().cpu()
    if (
        edge.dtype not in {torch.int32, torch.int64}
        or edge.ndim != 2
        or edge.shape[0] != 2
    ):
        raise ValueError("support_edge_index must be integer [2,E]")
    edge = edge.long().contiguous()
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= num_primitives):
        raise ValueError("support_edge_index contains an out-of-range primitive")

    region_count, query_count = probability.shape
    core_count = core.sum(dim=1).float()
    output = torch.zeros(num_primitives, query_count, dtype=torch.float32)
    all_indices: list[tuple[int, ...]] = []
    all_scores: list[tuple[float, ...]] = []
    all_marginals: list[tuple[int, ...]] = []
    edge_source = edge[0]
    edge_target = edge[1]

    for query in range(query_count):
        covered = torch.zeros(num_primitives, dtype=torch.bool)
        selected = torch.zeros(region_count, dtype=torch.bool)
        indices: list[int] = []
        scores: list[float] = []
        marginals: list[int] = []
        gated = probability[:, query] >= float(chosen.score_threshold)
        for iteration in range(int(chosen.maximum_regions)):
            reachable = covered.clone()
            if iteration > 0 and edge.numel():
                forward = covered[edge_source]
                reverse = covered[edge_target]
                reachable[edge_target[forward]] = True
                reachable[edge_source[reverse]] = True

            best_index = -1
            best_utility = 0.0
            best_marginal = 0
            step = int(chosen.candidate_chunk_rows)
            for start in range(0, region_count, step):
                stop = min(start + step, region_count)
                eligible = gated[start:stop] & ~selected[start:stop]
                if not bool(eligible.any()):
                    continue
                chunk_rows = rows[start:stop]
                chunk_core = core[start:stop]
                safe = chunk_rows.clamp_min(0).long()
                novel = chunk_core & ~covered[safe]
                marginal = novel.sum(dim=1)
                if iteration == 0:
                    utility = probability[start:stop, query] * core_count[start:stop]
                else:
                    connected = (chunk_core & reachable[safe]).any(dim=1)
                    eligible &= connected
                    utility = probability[start:stop, query] * marginal.float()
                utility = utility.masked_fill(~eligible, -1.0)
                local = int(torch.argmax(utility))
                candidate_utility = float(utility[local])
                candidate_index = start + local
                if candidate_utility > best_utility or (
                    candidate_utility == best_utility
                    and candidate_utility > 0.0
                    and (best_index < 0 or candidate_index < best_index)
                ):
                    best_index = candidate_index
                    best_utility = candidate_utility
                    best_marginal = int(marginal[local])
            if best_index < 0 or best_utility <= 0.0:
                break
            selected[best_index] = True
            active_rows = rows[best_index, core[best_index]].long()
            covered[active_rows] = True
            indices.append(best_index)
            scores.append(float(probability[best_index, query]))
            marginals.append(best_marginal)
        output[covered, query] = 1.0
        all_indices.append(tuple(indices))
        all_scores.append(tuple(scores))
        all_marginals.append(tuple(marginals))

    return MultiRegionUnionResult(
        primitive_membership=output.contiguous(),
        selected_region_indices=tuple(all_indices),
        selected_region_scores=tuple(all_scores),
        selected_marginal_core_rows=tuple(all_marginals),
    )


__all__ = [
    "MultiRegionUnionConfig",
    "MultiRegionUnionResult",
    "greedy_connected_expected_mass_union_readout",
    "greedy_novelty_union_readout",
]
