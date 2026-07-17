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
    maximum_tokens: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse bounded Dijkstra without a ``batch x num_nodes`` allocation.

    Heap tuples are ``(distance, node)``.  They therefore implement the same
    distance-then-node ordering declared by ``token_subsampling`` and allow us
    to stop exactly when the nearest ``maximum_tokens`` nodes are settled.
    """

    count_nodes = indptr.shape[0] - 1
    rows = np.full((anchors.shape[0], maximum_tokens), -1, dtype=np.int64)
    distances = np.full(
        (anchors.shape[0], maximum_tokens), np.inf, dtype=np.float64
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
        while queue and output_count < maximum_tokens:
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
        limit = radius * (self.context_ratio if include_context else 1.0)
        distance = dijkstra(
            self._csr(graph, xyz) if prepared_graph is None else prepared_graph,
            directed=False, indices=anchor, limit=limit
        )
        rows = np.flatnonzero(np.isfinite(distance))
        if anchor not in rows:
            rows = np.concatenate([rows, np.asarray([anchor], dtype=np.int64)])
            distance[anchor] = 0.0
        # Stable lexicographic ordering is the declared truncation contract.
        order = np.lexsort((rows, distance[rows]))
        rows = rows[order][: self.maximum_tokens]
        distances = distance[rows].astype(np.float32)
        core = distances <= radius + 1e-7
        return (
            torch.from_numpy(rows.astype(np.int64)),
            torch.from_numpy(core),
            torch.from_numpy(distances),
        )

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
            self.maximum_tokens,
        )
        result = []
        for batch_index, anchor in enumerate(anchor_array):
            count = int(counts[batch_index])
            rows = rows_by_anchor[batch_index, :count]
            selected_distance = distances_by_anchor[batch_index, :count].astype(np.float32)
            if count == 0 or rows[0] != anchor:
                raise RuntimeError("bounded Dijkstra lost its anchor")
            result.append((
                torch.from_numpy(rows.copy()),
                torch.from_numpy(selected_distance <= radius + 1e-7),
                torch.from_numpy(selected_distance),
            ))
        return result


DEFAULT_SURFACE_REGION_CONTRACT_V2 = SurfaceRegionContractV2()
