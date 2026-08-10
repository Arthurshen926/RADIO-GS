"""Bounded seed-conditioned alternatives to unbounded graph closure."""

from __future__ import annotations

import heapq
import math
import sys
from typing import Literal, Sequence

import torch


Adjacency = list[list[tuple[int, float, int]]]


def thresholded_adjacency(
    *,
    region_count: int,
    pair_indices: torch.Tensor,
    pair_probabilities: torch.Tensor,
    threshold: float,
) -> Adjacency:
    count = int(region_count)
    pairs = torch.as_tensor(pair_indices).detach().long().cpu()
    probability = torch.as_tensor(pair_probabilities).detach().float().cpu()
    cutoff = float(threshold)
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
        or not 0 <= cutoff <= 1
    ):
        raise ValueError("bounded co-membership graph inputs differ")
    adjacency: Adjacency = [[] for _ in range(count)]
    selected = torch.nonzero(probability >= cutoff, as_tuple=False).flatten().tolist()
    for edge_id in selected:
        left = int(pairs[0, edge_id])
        right = int(pairs[1, edge_id])
        value = float(probability[edge_id])
        adjacency[left].append((right, value, edge_id))
        adjacency[right].append((left, value, edge_id))
    for neighbors in adjacency:
        neighbors.sort(key=lambda row: row[0])
    return adjacency


def bridge_free_component_ids(adjacency: Adjacency) -> torch.Tensor:
    """Return components after every undirected bridge edge is removed."""

    count = len(adjacency)
    if count <= 0:
        raise ValueError("bounded co-membership adjacency is empty")
    sys.setrecursionlimit(max(sys.getrecursionlimit(), count * 2 + 100))
    discovery = [-1] * count
    low = [0] * count
    bridges: set[int] = set()
    tick = 0

    def visit(node: int, parent_edge: int) -> None:
        nonlocal tick
        discovery[node] = tick
        low[node] = tick
        tick += 1
        for neighbor, _probability, edge_id in adjacency[node]:
            if edge_id == parent_edge:
                continue
            if discovery[neighbor] < 0:
                visit(neighbor, edge_id)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    bridges.add(edge_id)
            else:
                low[node] = min(low[node], discovery[neighbor])

    for node in range(count):
        if discovery[node] < 0:
            visit(node, -1)
    component = torch.full((count,), -1, dtype=torch.int64)
    next_component = 0
    for root in range(count):
        if int(component[root]) >= 0:
            continue
        component[root] = next_component
        queue = [root]
        while queue:
            node = queue.pop()
            for neighbor, _probability, edge_id in adjacency[node]:
                if edge_id in bridges or int(component[neighbor]) >= 0:
                    continue
                component[neighbor] = next_component
                queue.append(neighbor)
        next_component += 1
    return component


def bounded_best_path_regions(
    *,
    seed_region_index: int,
    adjacency: Adjacency,
    maximum_regions: int,
    mode: Literal["maximum_product", "widest_path"],
    allowed_component_ids: torch.Tensor | None = None,
) -> tuple[int, ...]:
    seed = int(seed_region_index)
    count = len(adjacency)
    maximum = int(maximum_regions)
    if (
        seed < 0
        or seed >= count
        or maximum <= 0
        or mode
        not in {
            "maximum_product",
            "widest_path",
        }
    ):
        raise ValueError("bounded best-path readout inputs differ")
    allowed = None
    if allowed_component_ids is not None:
        components = torch.as_tensor(allowed_component_ids).detach().long().cpu()
        if components.shape != (count,) or bool((components < 0).any()):
            raise ValueError("bounded best-path component authority differs")
        seed_component = int(components[seed])
        allowed = components == seed_component
    best = torch.zeros(count, dtype=torch.float64)
    best[seed] = 1.0
    settled = torch.zeros(count, dtype=torch.bool)
    queue: list[tuple[float, int]] = [(-1.0, seed)]
    selected: list[int] = []
    while queue and len(selected) < maximum:
        negative_score, node = heapq.heappop(queue)
        score = -negative_score
        if bool(settled[node]) or score + 1e-15 < float(best[node]):
            continue
        settled[node] = True
        selected.append(node)
        for neighbor, probability, _edge_id in adjacency[node]:
            if bool(settled[neighbor]) or (
                allowed is not None and not bool(allowed[neighbor])
            ):
                continue
            candidate = (
                score * probability
                if mode == "maximum_product"
                else min(score, probability)
            )
            if candidate > float(best[neighbor]) + 1e-15:
                best[neighbor] = candidate
                heapq.heappush(queue, (-candidate, neighbor))
    return tuple(selected)


def bounded_multipoint_regions(
    *,
    seed_region_index: int,
    adjacency: Adjacency,
    maximum_regions: int,
) -> tuple[int, ...]:
    """Require two direct selected-region supports after the first addition."""

    seed = int(seed_region_index)
    maximum = int(maximum_regions)
    count = len(adjacency)
    if seed < 0 or seed >= count or maximum <= 0:
        raise ValueError("bounded multipoint readout inputs differ")
    selected = [seed]
    selected_mask = torch.zeros(count, dtype=torch.bool)
    selected_mask[seed] = True
    if maximum == 1 or not adjacency[seed]:
        return tuple(selected)
    first = max(adjacency[seed], key=lambda row: (row[1], -row[0]))[0]
    selected.append(first)
    selected_mask[first] = True
    while len(selected) < maximum:
        support: dict[int, list[float]] = {}
        for chosen in selected:
            for neighbor, probability, _edge_id in adjacency[chosen]:
                if not bool(selected_mask[neighbor]):
                    support.setdefault(neighbor, []).append(probability)
        candidates = []
        for node, values in support.items():
            if len(values) < 2:
                continue
            strongest = sorted(values, reverse=True)[:2]
            score = math.sqrt(strongest[0] * strongest[1])
            candidates.append((score, -node, node))
        if not candidates:
            break
        node = max(candidates)[2]
        selected.append(node)
        selected_mask[node] = True
    return tuple(selected)


def bounded_regions_for_seed(
    *,
    method: str,
    seed_region_index: int,
    adjacency: Adjacency,
    maximum_regions: int,
    bridge_free_components: torch.Tensor | None = None,
) -> tuple[int, ...]:
    if method == "maximum_product":
        return bounded_best_path_regions(
            seed_region_index=seed_region_index,
            adjacency=adjacency,
            maximum_regions=maximum_regions,
            mode="maximum_product",
        )
    if method == "widest_path":
        return bounded_best_path_regions(
            seed_region_index=seed_region_index,
            adjacency=adjacency,
            maximum_regions=maximum_regions,
            mode="widest_path",
        )
    if method == "dual_path_widest":
        if bridge_free_components is None:
            raise ValueError("dual-path readout requires bridge-free components")
        return bounded_best_path_regions(
            seed_region_index=seed_region_index,
            adjacency=adjacency,
            maximum_regions=maximum_regions,
            mode="widest_path",
            allowed_component_ids=bridge_free_components,
        )
    if method == "multipoint_consistency":
        return bounded_multipoint_regions(
            seed_region_index=seed_region_index,
            adjacency=adjacency,
            maximum_regions=maximum_regions,
        )
    raise ValueError("unknown bounded co-membership readout")


__all__ = [
    "bounded_best_path_regions",
    "bounded_multipoint_regions",
    "bounded_regions_for_seed",
    "bridge_free_component_ids",
    "thresholded_adjacency",
]
