"""Versioned, query-free surface-region construction shared by train and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import hashlib
import json
from typing import Iterable

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import torch

from radio_gs.querying.support_solver import (
    PrimitiveSupportGraph,
    SupportGraphConfig,
    build_primitive_support_graph,
)


@njit(cache=True)
def _bounded_dijkstra_batch(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    anchors: np.ndarray,
    limit: float,
    maximum_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse bounded Dijkstra without a ``batch x num_nodes`` allocation.

    Heap tuples are ``(distance, node)``.  They therefore implement the same
    distance-then-node ordering declared by ``token_subsampling`` and allow us
    to stop exactly when the nearest ``maximum_tokens`` nodes are settled.
    """

    count_nodes = indptr.shape[0] - 1
    rows = np.full((anchors.shape[0], maximum_candidates), -1, dtype=np.int64)
    distances = np.full(
        (anchors.shape[0], maximum_candidates), np.inf, dtype=np.float64
    )
    counts = np.zeros(anchors.shape[0], dtype=np.int64)
    best = np.empty(count_nodes, dtype=np.float64)
    stamp = np.zeros(count_nodes, dtype=np.int64)
    settled = np.zeros(count_nodes, dtype=np.int64)
    for batch_index in range(anchors.shape[0]):
        marker = batch_index + 1
        anchor = anchors[batch_index]
        queue = [(0.0, anchor)]
        stamp[anchor] = marker
        best[anchor] = 0.0
        output_count = 0
        while queue and output_count < maximum_candidates:
            distance, node = heapq.heappop(queue)
            if stamp[node] != marker or distance > best[node] or settled[node] == marker:
                continue
            if distance > limit:
                break
            settled[node] = marker
            rows[batch_index, output_count] = node
            distances[batch_index, output_count] = distance
            output_count += 1
            for edge_offset in range(indptr[node], indptr[node + 1]):
                neighbor = indices[edge_offset]
                candidate = distance + float(data[edge_offset])
                if candidate > limit or settled[neighbor] == marker:
                    continue
                if stamp[neighbor] != marker or candidate < best[neighbor]:
                    stamp[neighbor] = marker
                    best[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        counts[batch_index] = output_count
    return rows, distances, counts


@dataclass(frozen=True)
class SurfaceRegionContractV2:
    """The complete region definition used by Route-B training and inference.

    Region membership is a shortest-path ball on one deterministic symmetric
    DINO/SAM/geometry graph.  Token truncation is deterministic (nearest in
    geodesic distance), so cache sharding and batching cannot alter a region.
    """

    version: str = "surface-region-contract-v2"
    radii_m: tuple[float, ...] = (0.25, 0.45, 0.70)
    context_ratio: float = 1.20
    neighbors: int = 16
    spatial_scale: float = 2.0
    appearance_temperature: float = 0.10
    boundary_temperature: float = 0.10
    minimum_appearance_affinity: float = 1e-4
    minimum_boundary_affinity: float = 1e-3
    topology_mode: str = "symmetric_union"
    maximum_tokens: int = 256
    minimum_tokens: int = 24
    feature_normalization: str = "l2_direction"
    scale_semantics: str = "local_graph_sigma_m"
    reliability_semantics: str = "geometric_mean_observation_agreement"
    opacity_semantics: str = "absent"
    expansion: str = "undirected_dijkstra_physical_edge_length"
    token_subsampling: str = "nearest_geodesic_then_node_index"
    path_cost_mode: str = "euclidean"
    path_affinity_floor: float = 1e-4
    token_candidate_limit: int = 256
    core_token_fraction: float = 0.60

    def __post_init__(self) -> None:
        if self.version != "surface-region-contract-v2":
            raise ValueError("unsupported surface-region contract version")
        if not self.radii_m or min(self.radii_m) <= 0:
            raise ValueError("radii_m must contain positive physical radii")
        if tuple(sorted(self.radii_m)) != tuple(self.radii_m):
            raise ValueError("radii_m must be sorted")
        if self.context_ratio < 1.0:
            raise ValueError("context_ratio cannot be smaller than one")
        if self.minimum_tokens <= 0 or self.maximum_tokens < self.minimum_tokens:
            raise ValueError("invalid token-count bounds")
        if min(self.minimum_appearance_affinity, self.minimum_boundary_affinity) < 0:
            raise ValueError("edge-channel thresholds cannot be negative")
        if self.path_cost_mode not in {
            "euclidean", "appearance_boundary_geometric",
        }:
            raise ValueError("unsupported surface-region path_cost_mode")
        if not 0.0 < self.path_affinity_floor <= 1.0:
            raise ValueError("path_affinity_floor must lie in (0,1]")
        if self.token_subsampling not in {
            "nearest_geodesic_then_node_index",
            "core_context_radial_stratified_v1",
        }:
            raise ValueError("unsupported surface-region token_subsampling")
        if self.token_candidate_limit < self.maximum_tokens:
            raise ValueError("token_candidate_limit cannot be below maximum_tokens")
        if not 0.0 < self.core_token_fraction <= 1.0:
            raise ValueError("core_token_fraction must lie in (0,1]")
        # Reuse the graph config validator as the single graph-construction
        # authority rather than duplicating its numerical contract here.
        self.graph_config()

    def graph_config(self) -> SupportGraphConfig:
        return SupportGraphConfig(
            neighbors=self.neighbors,
            spatial_scale=self.spatial_scale,
            appearance_temperature=self.appearance_temperature,
            boundary_temperature=self.boundary_temperature,
            covisibility_weight=0.0,
            topology_mode=self.topology_mode,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["radii_m"] = list(self.radii_m)
        # Preserve the exact v2 digest for legacy/default contracts.  Extended
        # path and sampling semantics become part of the manifest only when
        # they are actually enabled, so an old frozen readout continues to
        # fail closed on genuine changes without becoming unloadable merely
        # because this implementation learned new optional modes.
        if self.path_cost_mode == "euclidean":
            payload.pop("path_cost_mode")
            if self.path_affinity_floor == 1e-4:
                payload.pop("path_affinity_floor")
        else:
            # ``expansion`` is retained for old manifests, but must describe
            # the effective shortest-path metric once relation evidence is
            # part of the edge cost.  This makes a frozen manifest readable
            # without relying on an implementation-specific default.
            payload["expansion"] = (
                "undirected_dijkstra_relation_weighted_physical_edge_cost"
            )
        if self.token_candidate_limit == self.maximum_tokens:
            payload.pop("token_candidate_limit")
        if self.core_token_fraction == 0.60:
            payload.pop("core_token_fraction")
        return payload

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def assert_compatible(self, metadata: dict) -> None:
        if metadata.get("region_contract_version") != self.version:
            raise ValueError("surface-region contract version mismatch")
        if metadata.get("region_contract_sha256") != self.digest:
            raise ValueError("surface-region contract digest mismatch")

    def build_graph(
        self,
        xyz: torch.Tensor,
        *,
        appearance_features: torch.Tensor,
        boundary_features: torch.Tensor,
    ) -> PrimitiveSupportGraph:
        return build_primitive_support_graph(
            xyz,
            appearance_features=appearance_features,
            boundary_features=boundary_features,
            config=self.graph_config(),
        )

    def _csr(self, graph: PrimitiveSupportGraph, xyz: torch.Tensor) -> csr_matrix:
        points = torch.as_tensor(xyz).detach().float().cpu().numpy()
        if points.shape != (graph.num_nodes, 3):
            raise ValueError("graph and xyz do not align")
        edge = graph.edge_index.detach().cpu().numpy()
        if edge.shape[1] == 0:
            return csr_matrix((graph.num_nodes, graph.num_nodes), dtype=np.float32)
        keep = np.ones(edge.shape[1], dtype=bool)
        channels = graph.edge_channels
        if "appearance" in channels:
            keep &= channels["appearance"].detach().cpu().numpy() >= self.minimum_appearance_affinity
        if "boundary" in channels:
            keep &= channels["boundary"].detach().cpu().numpy() >= self.minimum_boundary_affinity
        edge = edge[:, keep]
        if edge.shape[1] == 0:
            return csr_matrix((graph.num_nodes, graph.num_nodes), dtype=np.float32)
        length = np.linalg.norm(points[edge[0]] - points[edge[1]], axis=1).astype(
            np.float32
        )
        if self.path_cost_mode == "appearance_boundary_geometric":
            if not {"appearance", "boundary"}.issubset(channels):
                raise ValueError(
                    "appearance_boundary_geometric path cost requires both "
                    "appearance and boundary edge channels"
                )
            appearance = channels["appearance"].detach().cpu().numpy()[keep]
            boundary = channels["boundary"].detach().cpu().numpy()[keep]
            # The geometric mean is parameter-free, symmetric in the two
            # official capability views, and preserves metre units.  Weak
            # semantic/boundary transitions therefore consume more physical
            # path budget instead of merely surviving a permissive hard gate.
            relation = np.sqrt(
                np.maximum(appearance, self.path_affinity_floor)
                * np.maximum(boundary, self.path_affinity_floor)
            )
            length = length / relation.astype(np.float32)
        # Duplicate edges are harmless mathematically but canonicalizing them
        # here makes train/inference expansion byte-for-byte deterministic.
        return csr_matrix(
            (length, (edge[0], edge[1])),
            shape=(graph.num_nodes, graph.num_nodes),
        ).minimum(
            csr_matrix(
                (length, (edge[1], edge[0])),
                shape=(graph.num_nodes, graph.num_nodes),
            )
        )

    def expand(
        self,
        graph: PrimitiveSupportGraph,
        xyz: torch.Tensor,
        anchor: int,
        radius_m: float,
        *,
        include_context: bool = True,
        prepared_graph: csr_matrix | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return rows, core mask, and geodesic distances for one anchor."""

        anchor = int(anchor)
        radius = float(radius_m)
        if anchor < 0 or anchor >= graph.num_nodes or radius <= 0:
            raise ValueError("anchor/radius is outside the region contract")
        matrix = (self._csr(graph, xyz) if prepared_graph is None else prepared_graph).tocsr()
        rows, distances, counts = _bounded_dijkstra_batch(
            matrix.indptr.astype(np.int64, copy=False),
            matrix.indices.astype(np.int64, copy=False),
            matrix.data.astype(np.float64, copy=False),
            np.asarray([anchor], dtype=np.int64),
            radius * (self.context_ratio if include_context else 1.0),
            self.token_candidate_limit,
        )
        return self._select_tokens(rows[0, :counts[0]], distances[0, :counts[0]], anchor, radius)

    def prepare_graph(self, graph: PrimitiveSupportGraph, xyz: torch.Tensor) -> csr_matrix:
        """Prepare the immutable shortest-path matrix once for many anchors."""
        return self._csr(graph, xyz)

    def expand_many(
        self,
        graph: PrimitiveSupportGraph,
        xyz: torch.Tensor,
        anchors: Iterable[int],
        radius_m: float,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return [self.expand(graph, xyz, anchor, radius_m) for anchor in anchors]

    def expand_batch(
        self,
        graph: PrimitiveSupportGraph,
        xyz: torch.Tensor,
        anchors: Iterable[int],
        radius_m: float,
        *,
        prepared_graph: csr_matrix | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Vectorized shortest paths for an anchor chunk, with identical rows."""
        anchor_array = np.asarray([int(value) for value in anchors], dtype=np.int64)
        if anchor_array.size == 0:
            return []
        radius = float(radius_m)
        matrix = self._csr(graph, xyz) if prepared_graph is None else prepared_graph
        matrix = matrix.tocsr()
        rows_by_anchor, distances_by_anchor, counts = _bounded_dijkstra_batch(
            matrix.indptr.astype(np.int64, copy=False),
            matrix.indices.astype(np.int64, copy=False),
            matrix.data.astype(np.float64, copy=False),
            anchor_array,
            radius * self.context_ratio,
            self.token_candidate_limit,
        )
        result = []
        for batch_index, anchor in enumerate(anchor_array):
            count = int(counts[batch_index])
            rows = rows_by_anchor[batch_index, :count]
            distances = distances_by_anchor[batch_index, :count]
            if count == 0 or rows[0] != anchor:
                raise RuntimeError("bounded Dijkstra lost its anchor")
            result.append(self._select_tokens(rows, distances, anchor, radius))
        return result

    def _select_tokens(
        self,
        candidate_rows: np.ndarray,
        candidate_distances: np.ndarray,
        anchor: int,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the declared deterministic token-selection contract.

        Candidate rows already arrive ordered by (geodesic distance, node id)
        from bounded Dijkstra.  The stratified variant deliberately allocates a
        fixed core/context budget before filling residual slots, preventing a
        dense anchor neighbourhood from silently erasing the context shell.
        """

        rows = np.asarray(candidate_rows, dtype=np.int64)
        distances = np.asarray(candidate_distances, dtype=np.float64)
        if rows.size == 0 or rows[0] != int(anchor):
            raise RuntimeError("bounded Dijkstra candidate set lost its anchor")
        if self.token_subsampling == "nearest_geodesic_then_node_index":
            selected = np.arange(min(self.maximum_tokens, rows.size), dtype=np.int64)
        else:
            selected = self._stratified_indices(rows, distances, int(anchor), float(radius))
        selected_rows = rows[selected]
        selected_distances = distances[selected].astype(np.float32)
        order = np.lexsort((selected_rows, selected_distances))
        selected_rows = selected_rows[order]
        selected_distances = selected_distances[order]
        return (
            torch.from_numpy(selected_rows.copy()),
            torch.from_numpy(selected_distances <= float(radius) + 1e-7),
            torch.from_numpy(selected_distances.copy()),
        )

    def _stratified_indices(
        self,
        rows: np.ndarray,
        distances: np.ndarray,
        anchor: int,
        radius: float,
    ) -> np.ndarray:
        max_tokens = min(self.maximum_tokens, rows.size)
        if rows.size <= self.maximum_tokens:
            return np.arange(rows.size, dtype=np.int64)
        core = distances <= radius + 1e-7
        anchor_index = int(np.flatnonzero(rows == anchor)[0])
        core_budget = max(1, min(max_tokens, int(round(max_tokens * self.core_token_fraction))))
        context_budget = max_tokens - core_budget
        chosen: set[int] = {anchor_index}

        def pick(indices: np.ndarray, budget: int, *, start: float, stop: float) -> None:
            if budget <= 0 or indices.size == 0:
                return
            indices = indices[~np.isin(indices, np.fromiter(chosen, dtype=np.int64))]
            if indices.size == 0:
                return
            normalized = distances[indices] / max(radius, 1e-12)
            # Four equal radial bands in the core and three in the context
            # shell; empty bands donate their quota deterministically.
            bins = 4 if start == 0.0 else 3
            edges = np.linspace(start, stop, bins + 1)
            assigned = 0
            for bin_index in range(bins):
                lower, upper = edges[bin_index], edges[bin_index + 1]
                in_bin = indices[(normalized >= lower - 1e-7) & (
                    normalized <= upper + 1e-7 if bin_index == bins - 1 else normalized < upper
                )]
                remaining_bins = bins - bin_index
                quota = min(in_bin.size, max(0, (budget - assigned + remaining_bins - 1) // remaining_bins))
                if quota:
                    positions = np.linspace(0, in_bin.size - 1, quota).round().astype(np.int64)
                    chosen.update(int(value) for value in in_bin[positions])
                    assigned += quota
            if assigned < budget:
                remaining = indices[~np.isin(indices, np.fromiter(chosen, dtype=np.int64))]
                chosen.update(int(value) for value in remaining[: budget - assigned])

        pick(np.flatnonzero(core), core_budget - 1, start=0.0, stop=1.0)
        pick(np.flatnonzero(~core), context_budget, start=1.0, stop=self.context_ratio)
        if len(chosen) < max_tokens:
            remaining = np.arange(rows.size, dtype=np.int64)
            remaining = remaining[~np.isin(remaining, np.fromiter(chosen, dtype=np.int64))]
            chosen.update(int(value) for value in remaining[: max_tokens - len(chosen)])
        return np.asarray(sorted(chosen), dtype=np.int64)[:max_tokens]


DEFAULT_SURFACE_REGION_CONTRACT_V2 = SurfaceRegionContractV2()
