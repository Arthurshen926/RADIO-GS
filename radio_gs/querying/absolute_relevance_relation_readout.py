"""Absolute-unary-preserving bounded relation readout.

The relation graph is allowed to add only calibrated positive completion.  It
cannot replace, min-max normalize, rank-normalize, or lower the incoming
absolute relevance.  Query strings are not consumed; the query axis is opaque.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Literal

import torch

from radio_gs.utils.immutable_artifacts import canonical_json_sha256


READOUT_SCHEMA = "radio_gs.absolute_relevance_relation_readout.v1"
READOUT_SCHEMA_VERSION = 1
PATH_METHODS = ("maximum_product", "widest_path")


def source_access() -> dict[str, bool]:
    return {
        "query_strings_consumed": False,
        "query_axis_is_opaque": True,
        "benchmark_images_opened": False,
        "benchmark_masks_opened": False,
        "benchmark_labels_opened": False,
        "target_metrics_computed": False,
        "per_scene_hyperparameters": False,
    }


def readout_contract() -> dict[str, Any]:
    return {
        "schema": READOUT_SCHEMA,
        "schema_version": READOUT_SCHEMA_VERSION,
        "input_unary": "absolute_probability_in_closed_unit_interval",
        "absolute_boundary": "caller_bound_source_calibrated_probability_boundary",
        "query_gate": "maximum_absolute_unary_strictly_above_boundary",
        "relation_gate": "edge_probability_at_least_source_frozen_threshold",
        "path_methods": list(PATH_METHODS),
        "completion": (
            "u_final=u+(1-u)*positive_seed_excess*normalized_path_excess"
        ),
        "invariants": {
            "final_not_below_absolute_unary": True,
            "failed_query_gate_exact_unary": True,
            "outside_bounded_relation_support_exact_unary": True,
            "seed_exact_unary": True,
            "rank_or_minmax_normalization": False,
        },
        "legacy_readout_default_changed": False,
        "source_access": source_access(),
    }


READOUT_CONTRACT_SHA256 = canonical_json_sha256(readout_contract())


@dataclass(frozen=True)
class AbsoluteRelevanceRelationReadout:
    absolute_relevance: torch.Tensor
    final_relevance: torch.Tensor
    seed_region_indices: torch.Tensor
    query_gate: torch.Tensor
    selected_region_masks: torch.Tensor
    path_support: torch.Tensor


def _validated_graph(
    *,
    region_count: int,
    pair_indices: torch.Tensor,
    pair_probabilities: torch.Tensor,
    relation_threshold: float,
) -> list[list[tuple[int, float]]]:
    count = int(region_count)
    pairs = torch.as_tensor(pair_indices).detach().long().cpu()
    probability = torch.as_tensor(pair_probabilities).detach().float().cpu()
    threshold = float(relation_threshold)
    if (
        count <= 0
        or pairs.ndim != 2
        or pairs.shape[0] != 2
        or probability.shape != (pairs.shape[1],)
        or not bool(torch.isfinite(probability).all())
        or bool((probability < 0).any())
        or bool((probability > 1).any())
        or bool((pairs < 0).any())
        or bool((pairs >= count).any())
        or bool((pairs[0] >= pairs[1]).any())
        or not math.isfinite(threshold)
        or not 0.0 <= threshold < 1.0
    ):
        raise ValueError("absolute relevance relation graph differs")
    keys = pairs[0] * count + pairs[1]
    if keys.numel() > 1 and not bool((keys[1:] > keys[:-1]).all()):
        raise ValueError("absolute relevance relation pairs must be sorted unique")
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    for edge in torch.where(probability >= threshold)[0].tolist():
        left = int(pairs[0, edge])
        right = int(pairs[1, edge])
        value = float(probability[edge])
        adjacency[left].append((right, value))
        adjacency[right].append((left, value))
    for row in adjacency:
        row.sort(key=lambda item: item[0])
    return adjacency


def _bounded_path_support(
    *,
    seed: int,
    adjacency: list[list[tuple[int, float]]],
    maximum_regions: int,
    method: Literal["maximum_product", "widest_path"],
) -> tuple[tuple[int, ...], torch.Tensor]:
    count = len(adjacency)
    best = torch.zeros(count, dtype=torch.float64)
    best[seed] = 1.0
    settled = torch.zeros(count, dtype=torch.bool)
    queue: list[tuple[float, int]] = [(-1.0, seed)]
    selected: list[int] = []
    while queue and len(selected) < int(maximum_regions):
        negative_score, node = heapq.heappop(queue)
        score = -negative_score
        if bool(settled[node]) or score + 1e-15 < float(best[node]):
            continue
        settled[node] = True
        selected.append(node)
        for neighbor, probability in adjacency[node]:
            if bool(settled[neighbor]):
                continue
            candidate = (
                score * probability
                if method == "maximum_product"
                else min(score, probability)
            )
            if candidate > float(best[neighbor]) + 1e-15:
                best[neighbor] = candidate
                heapq.heappush(queue, (-candidate, neighbor))
    support = torch.zeros(count, dtype=torch.float32)
    support[selected] = best[selected].float()
    return tuple(selected), support


def absolute_relevance_relation_readout(
    *,
    region_absolute_relevance: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_probabilities: torch.Tensor,
    absolute_boundary: float,
    relation_threshold: float,
    maximum_regions: int,
    path_method: Literal["maximum_product", "widest_path"] = "widest_path",
) -> AbsoluteRelevanceRelationReadout:
    """Add bounded relation completion while retaining the absolute gauge."""

    unary = torch.as_tensor(region_absolute_relevance).detach().float().cpu()
    boundary = float(absolute_boundary)
    threshold = float(relation_threshold)
    maximum = int(maximum_regions)
    if (
        unary.ndim != 2
        or min(unary.shape) <= 0
        or not bool(torch.isfinite(unary).all())
        or bool((unary < 0).any())
        or bool((unary > 1).any())
        or not math.isfinite(boundary)
        or not 0.0 <= boundary < 1.0
        or maximum <= 0
        or path_method not in PATH_METHODS
    ):
        raise ValueError("absolute relevance readout inputs differ")
    adjacency = _validated_graph(
        region_count=unary.shape[0],
        pair_indices=pair_indices,
        pair_probabilities=pair_probabilities,
        relation_threshold=threshold,
    )
    seeds = torch.argmax(unary, dim=0).long().contiguous()
    gate = unary[seeds, torch.arange(unary.shape[1])] > boundary
    selected = torch.zeros_like(unary, dtype=torch.bool)
    path_support = torch.zeros_like(unary)
    final = unary.clone()
    for query, seed in enumerate(seeds.tolist()):
        if not bool(gate[query]):
            continue
        chosen, support = _bounded_path_support(
            seed=seed,
            adjacency=adjacency,
            maximum_regions=maximum,
            method=path_method,
        )
        chosen_tensor = torch.tensor(chosen, dtype=torch.long)
        selected[chosen_tensor, query] = True
        path_support[:, query] = support
        if len(chosen) <= 1:
            continue
        target = chosen_tensor[chosen_tensor != seed]
        normalized_seed_excess = (unary[seed, query] - boundary) / (1.0 - boundary)
        normalized_path_excess = (
            (support[target] - threshold) / (1.0 - threshold)
        ).clamp(0.0, 1.0)
        completion = normalized_seed_excess.clamp(0.0, 1.0) * normalized_path_excess
        base = unary[target, query]
        final[target, query] = base + (1.0 - base) * completion
    if (
        not bool(torch.isfinite(final).all())
        or bool((final < unary).any())
        or bool((final > 1).any())
        or not torch.equal(final[seeds, torch.arange(unary.shape[1])], unary[
            seeds, torch.arange(unary.shape[1])
        ])
        or not torch.equal(final[:, ~gate], unary[:, ~gate])
        or not torch.equal(final[~selected], unary[~selected])
    ):
        raise RuntimeError("absolute relevance preservation invariant failed")
    return AbsoluteRelevanceRelationReadout(
        absolute_relevance=unary.contiguous(),
        final_relevance=final.contiguous(),
        seed_region_indices=seeds,
        query_gate=gate.contiguous(),
        selected_region_masks=selected.contiguous(),
        path_support=path_support.contiguous(),
    )


__all__ = [
    "PATH_METHODS",
    "READOUT_CONTRACT_SHA256",
    "READOUT_SCHEMA",
    "READOUT_SCHEMA_VERSION",
    "AbsoluteRelevanceRelationReadout",
    "absolute_relevance_relation_readout",
    "readout_contract",
    "source_access",
]
