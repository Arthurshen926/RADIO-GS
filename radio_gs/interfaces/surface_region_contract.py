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


@njit(cache=True)
def _bounded_dijkstra_eligible_batch(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    anchors: np.ndarray,
    eligibility: np.ndarray,
    limit: float,
    maximum_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run shortest paths on the eligibility-induced graph.

    The anchor is made eligible by the caller.  Every other ineligible node is
    excluded from both emission and traversal, so an excluded fallback node
    cannot become a semantic shortcut between otherwise isolated primary
    nodes.
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
                # The shared batch eligibility may exclude an anchor.  Each
                # row restores only its own anchor, never another row's.
                if not eligibility[neighbor] and neighbor != anchor:
                    continue
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
        if self.reliability_semantics not in {
            "geometric_mean_observation_agreement",
            "uniform_valid",
        }:
            raise ValueError("unsupported surface-region reliability semantics")
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


class InsufficientRegionSupportError(RuntimeError):
    """The caller's eligibility set cannot satisfy a V3 region contract."""

    def __init__(self, *, anchor: int, available_tokens: int, minimum_tokens: int) -> None:
        self.anchor = int(anchor)
        self.available_tokens = int(available_tokens)
        self.minimum_tokens = int(minimum_tokens)
        super().__init__(
            "selection eligibility provides only "
            f"{self.available_tokens} tokens for anchor {self.anchor}; "
            f"surface-region-contract-v3 requires {self.minimum_tokens}"
        )


def _immutable_csr(matrix: csr_matrix) -> csr_matrix:
    result = matrix.tocsr(copy=True)
    result.sum_duplicates()
    result.sort_indices()
    result.data.setflags(write=False)
    result.indices.setflags(write=False)
    result.indptr.setflags(write=False)
    return result


def _primitive_graph_sha256(graph: PrimitiveSupportGraph) -> str:
    digest = hashlib.sha256()
    digest.update(b"radio_gs.prepared_surface_region_graph_v3\0")
    digest.update(np.asarray([graph.num_nodes], dtype=np.int64).tobytes())
    tensors = {
        "edge_index": graph.edge_index,
        "appearance": graph.edge_channels.get("appearance"),
        "boundary": graph.edge_channels.get("boundary"),
    }
    for name, tensor in tensors.items():
        if tensor is None:
            raise ValueError(f"V3 graph is missing edge channel {name!r}")
        array = np.ascontiguousarray(torch.as_tensor(tensor).detach().cpu().numpy())
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _undirected_min_csr(
    source: np.ndarray,
    target: np.ndarray,
    cost: np.ndarray,
    num_nodes: int,
) -> csr_matrix:
    """Canonical undirected CSR, taking the minimum duplicate edge cost."""

    source = np.asarray(source, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    cost = np.asarray(cost, dtype=np.float32)
    if source.size == 0:
        return csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    rows = np.concatenate((source, target))
    columns = np.concatenate((target, source))
    values = np.concatenate((cost, cost))
    order = np.lexsort((columns, rows))
    rows = rows[order]
    columns = columns[order]
    values = values[order]
    group_start = np.empty(rows.size, dtype=bool)
    group_start[0] = True
    group_start[1:] = (rows[1:] != rows[:-1]) | (columns[1:] != columns[:-1])
    starts = np.flatnonzero(group_start)
    values = np.minimum.reduceat(values, starts)
    rows = rows[starts]
    columns = columns[starts]
    return csr_matrix((values, (rows, columns)), shape=(num_nodes, num_nodes))


@dataclass(frozen=True)
class PreparedSurfaceRegionGraphV3:
    """Immutable pair of strict and recovery graphs bound to canonical xyz."""

    semantic_csr: csr_matrix
    soft_recovery_csr: csr_matrix
    xyz: np.ndarray
    graph_sha256: str
    contract_sha256: str

    def __post_init__(self) -> None:
        semantic = _immutable_csr(self.semantic_csr)
        recovery = _immutable_csr(self.soft_recovery_csr)
        points = np.asarray(self.xyz, dtype=np.float32).copy()
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("prepared V3 xyz must have shape [N,3]")
        if not np.isfinite(points).all():
            raise ValueError("prepared V3 xyz must be finite")
        expected_shape = (points.shape[0], points.shape[0])
        if semantic.shape != expected_shape or recovery.shape != expected_shape:
            raise ValueError("prepared V3 graph and xyz do not align")
        if (
            not np.isfinite(semantic.data).all()
            or not np.isfinite(recovery.data).all()
            or np.any(semantic.data < 0)
            or np.any(recovery.data < 0)
        ):
            raise ValueError("prepared V3 edge costs must be finite and non-negative")
        for name in ("graph_sha256", "contract_sha256"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"prepared V3 {name} must be a lowercase SHA-256")
        points.setflags(write=False)
        object.__setattr__(self, "semantic_csr", semantic)
        object.__setattr__(self, "soft_recovery_csr", recovery)
        object.__setattr__(self, "xyz", points)

    @property
    def num_nodes(self) -> int:
        return int(self.xyz.shape[0])


@dataclass(frozen=True)
class SurfaceRegionExpansionV3:
    """One V3 region with semantic membership and recovery made explicit.

    ``support_fill_mask`` denotes eligible support recovered solely to make the
    readout statistically usable.  It is not region membership and is thus
    disjoint from both ``core_mask`` and ``context_mask``.
    """

    rows: torch.Tensor
    core_mask: torch.Tensor
    context_mask: torch.Tensor
    support_fill_mask: torch.Tensor
    semantic_geodesic_distance: torch.Tensor
    recovery_distance: torch.Tensor
    anchor_index: int

    def __post_init__(self) -> None:
        rows = torch.as_tensor(self.rows).detach().long().cpu().reshape(-1).clone()
        core = torch.as_tensor(self.core_mask).detach().bool().cpu().reshape(-1).clone()
        context = torch.as_tensor(self.context_mask).detach().bool().cpu().reshape(-1).clone()
        support_fill = (
            torch.as_tensor(self.support_fill_mask).detach().bool().cpu().reshape(-1).clone()
        )
        semantic_distance = (
            torch.as_tensor(self.semantic_geodesic_distance)
            .detach().float().cpu().reshape(-1).clone()
        )
        recovery_distance = (
            torch.as_tensor(self.recovery_distance)
            .detach().float().cpu().reshape(-1).clone()
        )
        count = rows.numel()
        if count == 0 or any(value.numel() != count for value in (
            core, context, support_fill, semantic_distance, recovery_distance,
        )):
            raise ValueError("V3 expansion tensors must be non-empty and aligned")
        anchor_index = int(self.anchor_index)
        if anchor_index < 0 or anchor_index >= count:
            raise ValueError("V3 anchor_index is outside rows")
        memberships = (
            core.to(torch.int8) + context.to(torch.int8) + support_fill.to(torch.int8)
        )
        if not bool((memberships == 1).all()):
            raise ValueError(
                "V3 core/context/support-fill masks must be mutually exclusive and exhaustive"
            )
        if bool(support_fill.any()):
            if not bool(torch.isinf(semantic_distance[support_fill]).all()):
                raise ValueError("V3 support fill must have infinite semantic distance")
            if not bool(torch.isfinite(recovery_distance[support_fill]).all()):
                raise ValueError("V3 support fill must have a finite recovery distance")
        semantic = ~support_fill
        if not bool(torch.isfinite(semantic_distance[semantic]).all()):
            raise ValueError("V3 semantic members must have finite semantic distance")
        if bool((semantic_distance[semantic] < 0).any()) or bool(
            (recovery_distance[support_fill] < 0).any()
        ):
            raise ValueError("V3 expansion distances cannot be negative")
        if int(torch.unique(rows).numel()) != count:
            raise ValueError("V3 expansion rows must be unique")
        if not bool(core[anchor_index]) or float(semantic_distance[anchor_index]) != 0.0:
            raise ValueError("V3 anchor must be an explicit zero-distance core member")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "core_mask", core)
        object.__setattr__(self, "context_mask", context)
        object.__setattr__(self, "support_fill_mask", support_fill)
        object.__setattr__(self, "semantic_geodesic_distance", semantic_distance)
        object.__setattr__(self, "recovery_distance", recovery_distance)
        object.__setattr__(self, "anchor_index", anchor_index)


@dataclass(frozen=True)
class SurfaceRegionContractV3(SurfaceRegionContractV2):
    """Eligibility-aware, minimum-support surface-region contract.

    Tier 1 is the only source of semantic membership: a hard-affinity-gated,
    physical shortest-path ball.  If it contains too few selectable nodes,
    Tier 2 recovers eligible nodes through the full relation-weighted graph and
    Tier 3 deterministically fills from eligible Euclidean neighbours.  Both
    recovery tiers are explicit readout padding, never semantic membership.
    """

    version: str = "surface-region-contract-v3"
    feature_normalization: str = "l2_direction_plus_log_raw_norm_v1"
    expansion: str = "tiered_eligible_adaptive_expansion_v1"
    minimum_token_policy: str = "eligible_adaptive_support_v1"
    semantic_membership: str = "hard_gated_radius_ball"
    recovery_policy: str = "soft_relation_then_euclidean_v1"
    eligibility_semantics: str = "caller_provided_boolean_or_anchor"
    support_fill_semantics: str = "readout_support_not_region_membership"

    def __post_init__(self) -> None:
        if self.version != "surface-region-contract-v3":
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
        if self.path_cost_mode != "euclidean":
            raise ValueError("V3 semantic membership requires euclidean path_cost_mode")
        if not 0.0 < self.path_affinity_floor <= 1.0:
            raise ValueError("path_affinity_floor must lie in (0,1]")
        if self.token_subsampling != "nearest_geodesic_then_node_index":
            raise ValueError("V3 requires deterministic nearest-geodesic selection")
        if self.reliability_semantics not in {
            "geometric_mean_observation_agreement", "uniform_valid",
        }:
            raise ValueError("unsupported surface-region reliability semantics")
        if self.token_candidate_limit < self.maximum_tokens:
            raise ValueError("token_candidate_limit cannot be below maximum_tokens")
        if not 0.0 < self.core_token_fraction <= 1.0:
            raise ValueError("core_token_fraction must lie in (0,1]")
        expected = {
            "feature_normalization": "l2_direction_plus_log_raw_norm_v1",
            "minimum_token_policy": "eligible_adaptive_support_v1",
            "semantic_membership": "hard_gated_radius_ball",
            "recovery_policy": "soft_relation_then_euclidean_v1",
            "eligibility_semantics": "caller_provided_boolean_or_anchor",
            "support_fill_semantics": "readout_support_not_region_membership",
            "expansion": "tiered_eligible_adaptive_expansion_v1",
        }
        for field_name, required in expected.items():
            if getattr(self, field_name) != required:
                raise ValueError(f"unsupported V3 {field_name}")
        self.graph_config()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["radii_m"] = list(self.radii_m)
        return payload

    def prepare_graph(
        self,
        graph: PrimitiveSupportGraph,
        xyz: torch.Tensor,
    ) -> PreparedSurfaceRegionGraphV3:
        points = torch.as_tensor(xyz).detach().float().cpu().numpy()
        if points.shape != (graph.num_nodes, 3) or not np.isfinite(points).all():
            raise ValueError("graph and finite xyz do not align")
        if not {"appearance", "boundary"}.issubset(graph.edge_channels):
            raise ValueError("V3 requires appearance and boundary edge channels")
        edge = graph.edge_index.detach().cpu().numpy()
        appearance = graph.edge_channels["appearance"].detach().cpu().numpy()
        boundary = graph.edge_channels["boundary"].detach().cpu().numpy()
        if not np.isfinite(appearance).all() or not np.isfinite(boundary).all():
            raise ValueError("V3 edge channels must be finite")
        if edge.shape[1]:
            physical = np.linalg.norm(
                points[edge[0]] - points[edge[1]], axis=1
            ).astype(np.float32)
        else:
            physical = np.empty(0, dtype=np.float32)
        hard_keep = (
            (appearance >= self.minimum_appearance_affinity)
            & (boundary >= self.minimum_boundary_affinity)
        )
        semantic = _undirected_min_csr(
            edge[0, hard_keep], edge[1, hard_keep], physical[hard_keep], graph.num_nodes,
        )
        relation = np.sqrt(
            np.maximum(appearance, self.path_affinity_floor)
            * np.maximum(boundary, self.path_affinity_floor)
        ).astype(np.float32)
        soft_recovery = _undirected_min_csr(
            edge[0], edge[1], physical / relation, graph.num_nodes,
        )
        return PreparedSurfaceRegionGraphV3(
            semantic,
            soft_recovery,
            points,
            _primitive_graph_sha256(graph),
            self.digest,
        )

    def _validate_prepared(
        self,
        prepared: PreparedSurfaceRegionGraphV3,
        graph: PrimitiveSupportGraph,
        xyz: torch.Tensor,
    ) -> None:
        if not isinstance(prepared, PreparedSurfaceRegionGraphV3):
            raise TypeError("prepared_graph must be PreparedSurfaceRegionGraphV3")
        points = torch.as_tensor(xyz).detach().float().cpu().numpy()
        if graph.num_nodes != prepared.num_nodes or not np.array_equal(points, prepared.xyz):
            raise ValueError("prepared V3 graph is not bound to the supplied graph/xyz")
        if prepared.graph_sha256 != _primitive_graph_sha256(graph):
            raise ValueError("prepared V3 graph fingerprint does not match the supplied graph")
        if prepared.contract_sha256 != self.digest:
            raise ValueError("prepared V3 graph belongs to a different region contract")

    @staticmethod
    def _base_eligibility(
        selection_eligibility: torch.Tensor | np.ndarray | None,
        num_nodes: int,
    ) -> np.ndarray:
        if selection_eligibility is None:
            eligible = np.ones(num_nodes, dtype=np.bool_)
        elif isinstance(selection_eligibility, torch.Tensor):
            if selection_eligibility.dtype != torch.bool:
                raise ValueError("selection_eligibility must be boolean")
            eligible = selection_eligibility.detach().cpu().numpy().reshape(-1).copy()
        else:
            raw = np.asarray(selection_eligibility)
            if raw.dtype != np.bool_:
                raise ValueError("selection_eligibility must be boolean")
            eligible = raw.reshape(-1).copy()
        if eligible.shape != (num_nodes,):
            raise ValueError("selection_eligibility must have shape [num_nodes]")
        return eligible

    @classmethod
    def _eligibility(
        cls,
        selection_eligibility: torch.Tensor | np.ndarray | None,
        num_nodes: int,
        anchor: int,
    ) -> np.ndarray:
        eligible = cls._base_eligibility(selection_eligibility, num_nodes)
        eligible[anchor] = True
        return eligible

    @staticmethod
    def _eligible_dijkstra(
        matrix: csr_matrix,
        anchor: int,
        eligibility: np.ndarray,
        *,
        limit: float,
        maximum_candidates: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, distances, counts = _bounded_dijkstra_eligible_batch(
            matrix.indptr.astype(np.int64, copy=False),
            matrix.indices.astype(np.int64, copy=False),
            matrix.data.astype(np.float64, copy=False),
            np.asarray([anchor], dtype=np.int64),
            eligibility,
            float(limit),
            int(maximum_candidates),
        )
        count = int(counts[0])
        if count == 0 or int(rows[0, 0]) != anchor:
            raise RuntimeError("V3 bounded Dijkstra lost its anchor")
        return rows[0, :count], distances[0, :count]

    def _expand_prepared(
        self,
        prepared: PreparedSurfaceRegionGraphV3,
        anchor: int,
        radius: float,
        eligibility: np.ndarray,
        *,
        include_context: bool,
        recover_minimum: bool,
    ) -> SurfaceRegionExpansionV3:
        limit = radius * (self.context_ratio if include_context else 1.0)
        strict_rows, strict_distances = self._eligible_dijkstra(
            prepared.semantic_csr,
            anchor,
            eligibility,
            limit=limit,
            maximum_candidates=self.maximum_tokens,
        )
        soft_rows = np.empty(0, dtype=np.int64)
        soft_distances = np.empty(0, dtype=np.float64)
        if recover_minimum and strict_rows.size < self.minimum_tokens:
            soft_rows, soft_distances = self._eligible_dijkstra(
                prepared.soft_recovery_csr,
                anchor,
                eligibility,
                limit=float("inf"),
                maximum_candidates=min(self.minimum_tokens, int(eligibility.sum())),
            )
        return self._assemble_expansion(
            prepared,
            anchor,
            radius,
            eligibility,
            strict_rows,
            strict_distances,
            soft_rows,
            soft_distances,
            recover_minimum=recover_minimum,
        )

    def _assemble_expansion(
        self,
        prepared: PreparedSurfaceRegionGraphV3,
        anchor: int,
        radius: float,
        eligibility: np.ndarray,
        strict_rows: np.ndarray,
        strict_distances: np.ndarray,
        soft_rows: np.ndarray,
        soft_distances: np.ndarray,
        *,
        recover_minimum: bool,
    ) -> SurfaceRegionExpansionV3:
        """Deterministically assemble identical single and batched results."""

        selected_rows = [int(value) for value in strict_rows]
        semantic_distances = [float(value) for value in strict_distances]
        recovery_distances = [float("inf")] * len(selected_rows)
        support_fill = [False] * len(selected_rows)
        selected = set(selected_rows)

        if recover_minimum and len(selected_rows) < self.minimum_tokens:
            for row, distance in zip(soft_rows, soft_distances):
                row = int(row)
                if row in selected:
                    continue
                selected.add(row)
                selected_rows.append(row)
                semantic_distances.append(float("inf"))
                recovery_distances.append(float(distance))
                support_fill.append(True)
                if len(selected_rows) >= self.minimum_tokens:
                    break

        if recover_minimum and len(selected_rows) < self.minimum_tokens:
            euclidean = np.linalg.norm(
                prepared.xyz - prepared.xyz[anchor][None, :], axis=1
            )
            node_indices = np.arange(prepared.num_nodes, dtype=np.int64)
            for row in np.lexsort((node_indices, euclidean)):
                row = int(row)
                if (not eligibility[row] and row != anchor) or row in selected:
                    continue
                selected.add(row)
                selected_rows.append(row)
                semantic_distances.append(float("inf"))
                recovery_distances.append(float(euclidean[row]))
                support_fill.append(True)
                if len(selected_rows) >= self.minimum_tokens:
                    break

        if recover_minimum and len(selected_rows) < self.minimum_tokens:
            raise InsufficientRegionSupportError(
                anchor=anchor,
                available_tokens=len(selected_rows),
                minimum_tokens=self.minimum_tokens,
            )
        if not self.minimum_tokens <= len(selected_rows) <= self.maximum_tokens and recover_minimum:
            raise RuntimeError("V3 expansion violated token-count bounds")
        semantic_array = np.asarray(semantic_distances, dtype=np.float32)
        support_fill_array = np.asarray(support_fill, dtype=np.bool_)
        core = (~support_fill_array) & (semantic_array <= radius + 1e-7)
        context = (~support_fill_array) & ~core
        anchor_index = selected_rows.index(anchor)
        return SurfaceRegionExpansionV3(
            rows=torch.tensor(selected_rows, dtype=torch.long),
            core_mask=torch.from_numpy(core),
            context_mask=torch.from_numpy(context),
            support_fill_mask=torch.from_numpy(support_fill_array),
            semantic_geodesic_distance=torch.from_numpy(semantic_array),
            recovery_distance=torch.tensor(recovery_distances, dtype=torch.float32),
            anchor_index=anchor_index,
        )

    def expand(
        self,
        graph: PrimitiveSupportGraph,
        xyz: torch.Tensor,
        anchor: int,
        radius_m: float,
        *,
        include_context: bool = True,
        prepared_graph: PreparedSurfaceRegionGraphV3 | None = None,
        selection_eligibility: torch.Tensor | np.ndarray | None = None,
    ) -> SurfaceRegionExpansionV3:
        anchor = int(anchor)
        radius = float(radius_m)
        if anchor < 0 or anchor >= graph.num_nodes or radius <= 0 or not np.isfinite(radius):
            raise ValueError("anchor/radius is outside the region contract")
        prepared = self.prepare_graph(graph, xyz) if prepared_graph is None else prepared_graph
        self._validate_prepared(prepared, graph, xyz)
        eligibility = self._eligibility(selection_eligibility, graph.num_nodes, anchor)
        if include_context and int(eligibility.sum()) < self.minimum_tokens:
            raise InsufficientRegionSupportError(
                anchor=anchor,
                available_tokens=int(eligibility.sum()),
                minimum_tokens=self.minimum_tokens,
            )
        return self._expand_prepared(
            prepared, anchor, radius, eligibility,
            include_context=include_context,
            recover_minimum=include_context,
        )

    def expand_core(
        self,
        graph: PrimitiveSupportGraph,
        xyz: torch.Tensor,
        anchor: int,
        radius_m: float,
        *,
        prepared_graph: PreparedSurfaceRegionGraphV3 | None = None,
        selection_eligibility: torch.Tensor | np.ndarray | None = None,
    ) -> SurfaceRegionExpansionV3:
        return self.expand(
            graph, xyz, anchor, radius_m,
            include_context=False,
            prepared_graph=prepared_graph,
            selection_eligibility=selection_eligibility,
        )

    def expand_many(
        self,
        graph: PrimitiveSupportGraph,
        xyz: torch.Tensor,
        anchors: Iterable[int],
        radius_m: float,
        *,
        selection_eligibility: torch.Tensor | np.ndarray | None = None,
    ) -> list[SurfaceRegionExpansionV3]:
        return self.expand_batch(
            graph, xyz, anchors, radius_m,
            selection_eligibility=selection_eligibility,
        )

    def expand_batch(
        self,
        graph: PrimitiveSupportGraph,
        xyz: torch.Tensor,
        anchors: Iterable[int],
        radius_m: float,
        *,
        prepared_graph: PreparedSurfaceRegionGraphV3 | None = None,
        selection_eligibility: torch.Tensor | np.ndarray | None = None,
    ) -> list[SurfaceRegionExpansionV3]:
        anchor_array = [int(value) for value in anchors]
        if not anchor_array:
            return []
        radius = float(radius_m)
        if radius <= 0 or not np.isfinite(radius):
            raise ValueError("radius is outside the region contract")
        prepared = self.prepare_graph(graph, xyz) if prepared_graph is None else prepared_graph
        self._validate_prepared(prepared, graph, xyz)
        base_eligibility: np.ndarray | None = None
        base_eligible_count = 0
        for anchor in anchor_array:
            if anchor < 0 or anchor >= graph.num_nodes:
                raise ValueError("anchor is outside the region contract")
            if base_eligibility is None:
                base_eligibility = self._base_eligibility(
                    selection_eligibility, graph.num_nodes,
                )
                base_eligible_count = int(base_eligibility.sum())
            available = base_eligible_count + int(not base_eligibility[anchor])
            if available < self.minimum_tokens:
                raise InsufficientRegionSupportError(
                    anchor=anchor,
                    available_tokens=available,
                    minimum_tokens=self.minimum_tokens,
                )
        assert base_eligibility is not None
        anchors_np = np.asarray(anchor_array, dtype=np.int64)
        semantic = prepared.semantic_csr
        strict_rows, strict_distances, strict_counts = _bounded_dijkstra_eligible_batch(
            semantic.indptr.astype(np.int64, copy=False),
            semantic.indices.astype(np.int64, copy=False),
            semantic.data.astype(np.float64, copy=False),
            anchors_np,
            base_eligibility,
            radius * self.context_ratio,
            self.maximum_tokens,
        )
        needs_recovery = []
        for batch_index, anchor in enumerate(anchor_array):
            count = int(strict_counts[batch_index])
            if count == 0 or int(strict_rows[batch_index, 0]) != anchor:
                raise RuntimeError("V3 bounded Dijkstra lost its anchor")
            if count < self.minimum_tokens:
                needs_recovery.append(batch_index)

        soft_by_batch: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if needs_recovery:
            recovery = prepared.soft_recovery_csr
            recovery_anchors = anchors_np[np.asarray(needs_recovery, dtype=np.int64)]
            soft_rows, soft_distances, soft_counts = _bounded_dijkstra_eligible_batch(
                recovery.indptr.astype(np.int64, copy=False),
                recovery.indices.astype(np.int64, copy=False),
                recovery.data.astype(np.float64, copy=False),
                recovery_anchors,
                base_eligibility,
                float("inf"),
                self.minimum_tokens,
            )
            for recovery_index, batch_index in enumerate(needs_recovery):
                count = int(soft_counts[recovery_index])
                anchor = anchor_array[batch_index]
                if count == 0 or int(soft_rows[recovery_index, 0]) != anchor:
                    raise RuntimeError("V3 bounded Dijkstra lost its anchor")
                soft_by_batch[batch_index] = (
                    soft_rows[recovery_index, :count],
                    soft_distances[recovery_index, :count],
                )

        empty_rows = np.empty(0, dtype=np.int64)
        empty_distances = np.empty(0, dtype=np.float64)
        result = []
        for batch_index, anchor in enumerate(anchor_array):
            strict_count = int(strict_counts[batch_index])
            soft_candidate_rows, soft_candidate_distances = soft_by_batch.get(
                batch_index, (empty_rows, empty_distances),
            )
            result.append(self._assemble_expansion(
                prepared,
                anchor,
                radius,
                base_eligibility,
                strict_rows[batch_index, :strict_count],
                strict_distances[batch_index, :strict_count],
                soft_candidate_rows,
                soft_candidate_distances,
                recover_minimum=True,
            ))
        return result


DEFAULT_SURFACE_REGION_CONTRACT_V3 = SurfaceRegionContractV3()


@dataclass(frozen=True)
class SurfaceRegionExpansionV4(SurfaceRegionExpansionV3):
    """A V4 expansion with the same typed tensor schema as V3.

    The distinct Python type prevents a V4 selection from being silently
    recorded as V3 even though the tensors deliberately retain the proven V3
    core/context/support-fill representation.
    """


@dataclass(frozen=True)
class SurfaceRegionContractV4(SurfaceRegionContractV3):
    """Candidate-complete, independently budgeted semantic expansion.

    V3 settles at most ``maximum_tokens`` strict candidates, so a dense core
    can exhaust the search before any context-shell node is even observed.
    V4 first enumerates up to ``token_candidate_limit`` candidates, then
    reserves deterministic core and context budgets.  Unused capacity is
    backfilled core-first and then context, which keeps the output dense
    without allowing either type's reserved budget to displace the other.

    Recovery remains the explicit V3 soft-relation/Euclidean support tier.  No
    local-scale normalization or distance clipping is introduced here: those
    changes need their own evidence and versioned contract.
    """

    version: str = "surface-region-contract-v4"
    expansion: str = "tiered_eligible_core_context_budgeted_v1"
    token_candidate_limit: int = 1024
    token_subsampling: str = (
        "complete_core_then_typed_context_deterministic_backfill_v1"
    )
    semantic_budget_policy: str = "independent_core_context_fraction_v1"

    def __post_init__(self) -> None:
        # This validation is intentionally independent of V3.__post_init__:
        # accepting the new version/policy there would mutate V3's fail-closed
        # protocol boundary.
        if self.version != "surface-region-contract-v4":
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
        if self.path_cost_mode != "euclidean":
            raise ValueError("V4 semantic membership requires euclidean path_cost_mode")
        if not 0.0 < self.path_affinity_floor <= 1.0:
            raise ValueError("path_affinity_floor must lie in (0,1]")
        if self.token_subsampling != (
            "complete_core_then_typed_context_deterministic_backfill_v1"
        ):
            raise ValueError("V4 requires deterministic typed token selection")
        if self.reliability_semantics not in {
            "geometric_mean_observation_agreement", "uniform_valid",
        }:
            raise ValueError("unsupported surface-region reliability semantics")
        if self.token_candidate_limit < self.maximum_tokens:
            raise ValueError("token_candidate_limit cannot be below maximum_tokens")
        if not 0.0 < self.core_token_fraction <= 1.0:
            raise ValueError("core_token_fraction must lie in (0,1]")
        expected = {
            "feature_normalization": "l2_direction_plus_log_raw_norm_v1",
            "minimum_token_policy": "eligible_adaptive_support_v1",
            "semantic_membership": "hard_gated_radius_ball",
            "recovery_policy": "soft_relation_then_euclidean_v1",
            "eligibility_semantics": "caller_provided_boolean_or_anchor",
            "support_fill_semantics": "readout_support_not_region_membership",
            "expansion": "tiered_eligible_core_context_budgeted_v1",
            "semantic_budget_policy": "independent_core_context_fraction_v1",
        }
        for field_name, required in expected.items():
            if getattr(self, field_name) != required:
                raise ValueError(f"unsupported V4 {field_name}")
        self.graph_config()

    def _select_semantic_candidates(
        self,
        rows: np.ndarray,
        distances: np.ndarray,
        anchor: int,
        radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select typed semantic candidates with independent reserved budgets."""

        rows = np.asarray(rows, dtype=np.int64)
        distances = np.asarray(distances, dtype=np.float64)
        if rows.shape != distances.shape or rows.ndim != 1:
            raise ValueError("V4 candidate rows and distances must be aligned vectors")
        if rows.size == 0 or int(rows[0]) != int(anchor):
            raise RuntimeError("V4 bounded Dijkstra candidate set lost its anchor")

        core_indices = np.flatnonzero(distances <= float(radius) + 1e-7)
        context_indices = np.flatnonzero(distances > float(radius) + 1e-7)
        core_budget = max(
            1,
            min(
                self.maximum_tokens,
                int(round(self.maximum_tokens * self.core_token_fraction)),
            ),
        )
        context_budget = self.maximum_tokens - core_budget

        selected_core = core_indices[:core_budget].tolist()
        selected_context = context_indices[:context_budget].tolist()
        remaining_capacity = self.maximum_tokens - (
            len(selected_core) + len(selected_context)
        )
        if remaining_capacity:
            # Donation is deterministic and cannot invalidate either reserved
            # quota: exhaust remaining core first, then remaining context.
            core_remainder = core_indices[core_budget:]
            take = min(remaining_capacity, int(core_remainder.size))
            selected_core.extend(core_remainder[:take].tolist())
            remaining_capacity -= take
        if remaining_capacity:
            context_remainder = context_indices[context_budget:]
            selected_context.extend(context_remainder[:remaining_capacity].tolist())

        selected_indices = np.asarray(
            selected_core + selected_context, dtype=np.int64,
        )
        return rows[selected_indices], distances[selected_indices]

    def _assemble_expansion(
        self,
        prepared: PreparedSurfaceRegionGraphV3,
        anchor: int,
        radius: float,
        eligibility: np.ndarray,
        strict_rows: np.ndarray,
        strict_distances: np.ndarray,
        soft_rows: np.ndarray,
        soft_distances: np.ndarray,
        *,
        recover_minimum: bool,
    ) -> SurfaceRegionExpansionV4:
        expansion = super()._assemble_expansion(
            prepared,
            anchor,
            radius,
            eligibility,
            strict_rows,
            strict_distances,
            soft_rows,
            soft_distances,
            recover_minimum=recover_minimum,
        )
        return SurfaceRegionExpansionV4(
            rows=expansion.rows,
            core_mask=expansion.core_mask,
            context_mask=expansion.context_mask,
            support_fill_mask=expansion.support_fill_mask,
            semantic_geodesic_distance=expansion.semantic_geodesic_distance,
            recovery_distance=expansion.recovery_distance,
            anchor_index=expansion.anchor_index,
        )

    def _expand_prepared(
        self,
        prepared: PreparedSurfaceRegionGraphV3,
        anchor: int,
        radius: float,
        eligibility: np.ndarray,
        *,
        include_context: bool,
        recover_minimum: bool,
    ) -> SurfaceRegionExpansionV4:
        limit = radius * (self.context_ratio if include_context else 1.0)
        strict_rows, strict_distances = self._eligible_dijkstra(
            prepared.semantic_csr,
            anchor,
            eligibility,
            limit=limit,
            maximum_candidates=self.token_candidate_limit,
        )
        strict_rows, strict_distances = self._select_semantic_candidates(
            strict_rows, strict_distances, anchor, radius,
        )
        soft_rows = np.empty(0, dtype=np.int64)
        soft_distances = np.empty(0, dtype=np.float64)
        if recover_minimum and strict_rows.size < self.minimum_tokens:
            soft_rows, soft_distances = self._eligible_dijkstra(
                prepared.soft_recovery_csr,
                anchor,
                eligibility,
                limit=float("inf"),
                maximum_candidates=min(
                    self.token_candidate_limit, int(eligibility.sum())
                ),
            )
        return self._assemble_expansion(
            prepared,
            anchor,
            radius,
            eligibility,
            strict_rows,
            strict_distances,
            soft_rows,
            soft_distances,
            recover_minimum=recover_minimum,
        )

    def expand_batch(
        self,
        graph: PrimitiveSupportGraph,
        xyz: torch.Tensor,
        anchors: Iterable[int],
        radius_m: float,
        *,
        prepared_graph: PreparedSurfaceRegionGraphV3 | None = None,
        selection_eligibility: torch.Tensor | np.ndarray | None = None,
    ) -> list[SurfaceRegionExpansionV4]:
        """Batch V4 expansion exactly matching repeated single expansion."""

        anchor_array = [int(value) for value in anchors]
        if not anchor_array:
            return []
        radius = float(radius_m)
        if radius <= 0 or not np.isfinite(radius):
            raise ValueError("radius is outside the region contract")
        prepared = (
            self.prepare_graph(graph, xyz)
            if prepared_graph is None
            else prepared_graph
        )
        self._validate_prepared(prepared, graph, xyz)
        base_eligibility = self._base_eligibility(
            selection_eligibility, graph.num_nodes,
        )
        base_eligible_count = int(base_eligibility.sum())
        for anchor in anchor_array:
            if anchor < 0 or anchor >= graph.num_nodes:
                raise ValueError("anchor is outside the region contract")
            available = base_eligible_count + int(not base_eligibility[anchor])
            if available < self.minimum_tokens:
                raise InsufficientRegionSupportError(
                    anchor=anchor,
                    available_tokens=available,
                    minimum_tokens=self.minimum_tokens,
                )

        anchors_np = np.asarray(anchor_array, dtype=np.int64)
        semantic = prepared.semantic_csr
        strict_rows, strict_distances, strict_counts = _bounded_dijkstra_eligible_batch(
            semantic.indptr.astype(np.int64, copy=False),
            semantic.indices.astype(np.int64, copy=False),
            semantic.data.astype(np.float64, copy=False),
            anchors_np,
            base_eligibility,
            radius * self.context_ratio,
            self.token_candidate_limit,
        )

        selected_by_batch: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        needs_recovery: list[int] = []
        for batch_index, anchor in enumerate(anchor_array):
            count = int(strict_counts[batch_index])
            if count == 0 or int(strict_rows[batch_index, 0]) != anchor:
                raise RuntimeError("V4 bounded Dijkstra lost its anchor")
            selected = self._select_semantic_candidates(
                strict_rows[batch_index, :count],
                strict_distances[batch_index, :count],
                anchor,
                radius,
            )
            selected_by_batch[batch_index] = selected
            if selected[0].size < self.minimum_tokens:
                needs_recovery.append(batch_index)

        soft_by_batch: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if needs_recovery:
            recovery = prepared.soft_recovery_csr
            recovery_anchors = anchors_np[np.asarray(needs_recovery, dtype=np.int64)]
            soft_rows, soft_distances, soft_counts = _bounded_dijkstra_eligible_batch(
                recovery.indptr.astype(np.int64, copy=False),
                recovery.indices.astype(np.int64, copy=False),
                recovery.data.astype(np.float64, copy=False),
                recovery_anchors,
                base_eligibility,
                float("inf"),
                min(self.token_candidate_limit, graph.num_nodes),
            )
            for recovery_index, batch_index in enumerate(needs_recovery):
                count = int(soft_counts[recovery_index])
                anchor = anchor_array[batch_index]
                if count == 0 or int(soft_rows[recovery_index, 0]) != anchor:
                    raise RuntimeError("V4 bounded Dijkstra lost its anchor")
                soft_by_batch[batch_index] = (
                    soft_rows[recovery_index, :count],
                    soft_distances[recovery_index, :count],
                )

        empty_rows = np.empty(0, dtype=np.int64)
        empty_distances = np.empty(0, dtype=np.float64)
        result = []
        for batch_index, anchor in enumerate(anchor_array):
            semantic_rows, semantic_distances = selected_by_batch[batch_index]
            soft_candidate_rows, soft_candidate_distances = soft_by_batch.get(
                batch_index, (empty_rows, empty_distances),
            )
            result.append(self._assemble_expansion(
                prepared,
                anchor,
                radius,
                base_eligibility,
                semantic_rows,
                semantic_distances,
                soft_candidate_rows,
                soft_candidate_distances,
                recover_minimum=True,
            ))
        return result


DEFAULT_SURFACE_REGION_CONTRACT_V4 = SurfaceRegionContractV4()
