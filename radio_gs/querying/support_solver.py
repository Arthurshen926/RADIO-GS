"""One symmetric primitive support graph and solver for every query modality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .query_spec import SelectionMode, SoftSeedSet


@dataclass(frozen=True)
class SupportGraphConfig:
    neighbors: int = 16
    spatial_scale: float = 2.0
    appearance_temperature: float = 0.10
    boundary_temperature: float = 0.10
    normal_temperature: float = 0.20
    covisibility_weight: float = 0.25
    minimum_sigma: float = 1e-4
    affinity_chunk_size: int = 8192

    def __post_init__(self) -> None:
        if self.neighbors <= 0 or self.affinity_chunk_size <= 0:
            raise ValueError("neighbors and affinity_chunk_size must be positive")
        if min(
            self.spatial_scale,
            self.appearance_temperature,
            self.boundary_temperature,
            self.normal_temperature,
            self.minimum_sigma,
        ) <= 0:
            raise ValueError("graph scales and temperatures must be positive")
        if self.covisibility_weight < 0:
            raise ValueError("covisibility_weight cannot be negative")


@dataclass(frozen=True)
class PrimitiveSupportGraph:
    """Symmetric graph with row-normalized message-passing weights."""

    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    raw_affinity: torch.Tensor
    local_sigma: torch.Tensor
    num_nodes: int

    def __post_init__(self) -> None:
        edges = torch.as_tensor(self.edge_index).long().cpu()
        weight = torch.as_tensor(self.edge_weight).float().cpu().reshape(-1)
        raw = torch.as_tensor(self.raw_affinity).float().cpu().reshape(-1)
        sigma = torch.as_tensor(self.local_sigma).float().cpu().reshape(-1)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must be [2,E]")
        if weight.shape != (edges.shape[1],) or raw.shape != weight.shape:
            raise ValueError("edge weights must align with edge_index")
        if sigma.shape != (int(self.num_nodes),):
            raise ValueError("local_sigma must be [num_nodes]")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= self.num_nodes):
            raise IndexError("graph edge is outside num_nodes")
        if bool((weight < 0).any()) or bool((raw < 0).any()):
            raise ValueError("graph affinities cannot be negative")
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "edge_weight", weight)
        object.__setattr__(self, "raw_affinity", raw)
        object.__setattr__(self, "local_sigma", sigma)

    def to(self, device: torch.device | str) -> "PrimitiveSupportGraph":
        result = object.__new__(PrimitiveSupportGraph)
        object.__setattr__(result, "edge_index", self.edge_index.to(device))
        object.__setattr__(result, "edge_weight", self.edge_weight.to(device))
        object.__setattr__(result, "raw_affinity", self.raw_affinity.to(device))
        object.__setattr__(result, "local_sigma", self.local_sigma.to(device))
        object.__setattr__(result, "num_nodes", self.num_nodes)
        return result


def _feature_matrix(
    values: torch.Tensor | None,
    count: int,
    *,
    name: str,
) -> torch.Tensor | None:
    if values is None:
        return None
    matrix = torch.as_tensor(values).detach().float().cpu()
    if matrix.ndim != 2 or matrix.shape[0] != count:
        raise ValueError(f"{name} must be [num_nodes,D]")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} contains NaN or infinity")
    return F.normalize(matrix, dim=-1, eps=1e-8)


def build_primitive_support_graph(
    xyz: torch.Tensor,
    *,
    appearance_features: torch.Tensor | None = None,
    boundary_features: torch.Tensor | None = None,
    normals: torch.Tensor | None = None,
    view_observations: torch.Tensor | None = None,
    config: SupportGraphConfig = SupportGraphConfig(),
) -> PrimitiveSupportGraph:
    """Build a symmetric adaptive-surface graph without query or GT access.

    ``view_observations`` is an optional training-view visibility matrix
    ``[N,V]``.  It contributes pairwise Jaccard co-visibility, so MPR and the
    query graph share the same physical correspondence evidence.
    """

    from scipy.spatial import cKDTree

    points = torch.as_tensor(xyz).detach().float().cpu()
    if points.ndim != 2 or points.shape[1] != 3 or not bool(torch.isfinite(points).all()):
        raise ValueError("xyz must be finite [N,3]")
    count = points.shape[0]
    if count <= 0:
        raise ValueError("cannot construct an empty support graph")
    if count == 1:
        return PrimitiveSupportGraph(
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_weight=torch.empty(0),
            raw_affinity=torch.empty(0),
            local_sigma=torch.full((1,), config.minimum_sigma),
            num_nodes=1,
        )

    k = min(config.neighbors + 1, count)
    distances_np, neighbors_np = cKDTree(points.numpy()).query(points.numpy(), k=k)
    distances = torch.from_numpy(np.asarray(distances_np, dtype=np.float32)[:, 1:])
    neighbors = torch.from_numpy(np.asarray(neighbors_np, dtype=np.int64)[:, 1:])
    local_sigma = distances.median(dim=1).values.mul(config.spatial_scale)
    local_sigma.clamp_(min=config.minimum_sigma)

    rows = torch.arange(count, dtype=torch.long)[:, None].expand_as(neighbors).reshape(-1)
    cols = neighbors.reshape(-1)
    undirected = torch.stack(
        [torch.cat([rows, cols]), torch.cat([cols, rows])], dim=1
    ).numpy()
    undirected = np.unique(undirected, axis=0)
    row = torch.from_numpy(undirected[:, 0].copy()).long()
    col = torch.from_numpy(undirected[:, 1].copy()).long()
    edge_index = torch.stack([row, col])

    appearance = _feature_matrix(appearance_features, count, name="appearance_features")
    boundary = _feature_matrix(boundary_features, count, name="boundary_features")
    normal = _feature_matrix(normals, count, name="normals")
    visibility = None
    if view_observations is not None:
        visibility = torch.as_tensor(view_observations).detach().bool().cpu()
        if visibility.ndim != 2 or visibility.shape[0] != count:
            raise ValueError("view_observations must be [num_nodes,num_views]")

    raw_parts: list[torch.Tensor] = []
    for start in range(0, row.numel(), config.affinity_chunk_size):
        stop = min(start + config.affinity_chunk_size, row.numel())
        src, dst = row[start:stop], col[start:stop]
        distance2 = (points[src] - points[dst]).square().sum(dim=-1)
        pair_sigma2 = (local_sigma[src] * local_sigma[dst]).clamp_min(
            config.minimum_sigma**2
        )
        log_affinity = -0.5 * distance2 / pair_sigma2
        if appearance is not None:
            cosine = (appearance[src] * appearance[dst]).sum(dim=-1)
            log_affinity.add_((cosine - 1.0) / config.appearance_temperature)
        if boundary is not None:
            cosine = (boundary[src] * boundary[dst]).sum(dim=-1)
            log_affinity.add_((cosine - 1.0) / config.boundary_temperature)
        if normal is not None:
            cosine = (normal[src] * normal[dst]).sum(dim=-1).clamp(-1.0, 1.0)
            log_affinity.add_((cosine - 1.0) / config.normal_temperature)
        if visibility is not None and visibility.shape[1] > 0:
            shared = (visibility[src] & visibility[dst]).sum(dim=-1).float()
            union = (visibility[src] | visibility[dst]).sum(dim=-1).float()
            jaccard = shared / union.clamp_min(1.0)
            log_affinity.add_(config.covisibility_weight * (jaccard - 1.0))
        raw_parts.append(log_affinity.clamp(min=-60.0, max=0.0).exp())
    raw_affinity = torch.cat(raw_parts)
    row_sum = torch.zeros(count, dtype=torch.float32)
    row_sum.index_add_(0, row, raw_affinity)
    edge_weight = raw_affinity / row_sum[row].clamp_min(1e-12)
    return PrimitiveSupportGraph(
        edge_index=edge_index,
        edge_weight=edge_weight,
        raw_affinity=raw_affinity,
        local_sigma=local_sigma,
        num_nodes=count,
    )


@dataclass(frozen=True)
class SupportSolverConfig:
    iterations: int = 12
    residual: float = 0.30
    unary_temperature: float = 0.10
    support_threshold: float = 0.50
    component_edge_threshold: float = 1e-5
    seeded_component_min_weight: float = 0.20
    top_k_components: int = 3

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.unary_temperature <= 0:
            raise ValueError("iterations must be non-negative and temperature positive")
        if not 0 <= self.residual <= 1:
            raise ValueError("residual must be in [0,1]")
        if not 0 <= self.support_threshold <= 1:
            raise ValueError("support_threshold must be in [0,1]")
        if self.component_edge_threshold < 0 or self.top_k_components <= 0:
            raise ValueError("component parameters are invalid")


def _seed_values(seed: SoftSeedSet | None, count: int, device: torch.device) -> torch.Tensor:
    if seed is None:
        return torch.zeros(count, device=device)
    values = seed.weights.to(device)
    if values.shape != (count,):
        raise ValueError("query seeds do not align with support graph")
    return values


def solve_primitive_support(
    graph: PrimitiveSupportGraph,
    unary: torch.Tensor,
    *,
    positive_seeds: SoftSeedSet | None = None,
    negative_seeds: SoftSeedSet | None = None,
    config: SupportSolverConfig = SupportSolverConfig(),
) -> torch.Tensor:
    """Diffuse unary evidence while softly clamping registered evidence."""

    values = torch.as_tensor(unary).float().reshape(-1)
    if values.shape != (graph.num_nodes,) or not bool(torch.isfinite(values).all()):
        raise ValueError("unary must be a finite [num_nodes] vector")
    device = values.device
    working_graph = graph if graph.edge_index.device == device else graph.to(device)
    positive = _seed_values(positive_seeds, graph.num_nodes, device)
    negative = _seed_values(negative_seeds, graph.num_nodes, device)
    prior = torch.sigmoid(values / config.unary_temperature)
    probability = prior

    row, col = working_graph.edge_index
    for _ in range(config.iterations):
        propagated = torch.zeros_like(probability)
        if row.numel():
            propagated.index_add_(
                0, row, working_graph.edge_weight * probability[col]
            )
        probability = config.residual * prior + (1.0 - config.residual) * propagated
        probability = probability * (1.0 - negative)
        probability = probability * (1.0 - positive) + positive
    return probability.clamp(0.0, 1.0)


def _component_labels(
    graph: PrimitiveSupportGraph,
    active: torch.Tensor,
    edge_threshold: float,
) -> torch.Tensor:
    """Connected components of the active support using SciPy's C routine."""

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    mask = torch.as_tensor(active).bool().cpu().numpy()
    labels = torch.full((graph.num_nodes,), -1, dtype=torch.long)
    if not bool(mask.any()):
        return labels
    edges = graph.edge_index.detach().cpu().numpy()
    affinity = graph.raw_affinity.detach().cpu().numpy()
    keep = (
        (affinity >= float(edge_threshold))
        & mask[edges[0]]
        & mask[edges[1]]
    )
    active_rows = np.flatnonzero(mask)
    if not bool(keep.any()):
        labels[torch.from_numpy(active_rows)] = torch.arange(active_rows.size)
        return labels
    adjacency = coo_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.uint8),
            (edges[0, keep], edges[1, keep]),
        ),
        shape=(graph.num_nodes, graph.num_nodes),
    ).tocsr()
    _count, raw_labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    selected = raw_labels[active_rows]
    _unique, compact = np.unique(selected, return_inverse=True)
    labels[torch.from_numpy(active_rows)] = torch.from_numpy(compact).long()
    return labels


def select_support_components(
    graph: PrimitiveSupportGraph,
    probabilities: torch.Tensor,
    selection_mode: SelectionMode,
    *,
    positive_seeds: SoftSeedSet | None = None,
    config: SupportSolverConfig = SupportSolverConfig(),
) -> torch.Tensor:
    """Apply the query-declared, benchmark-independent component policy."""

    values = torch.as_tensor(probabilities).float().reshape(-1)
    if values.shape != (graph.num_nodes,):
        raise ValueError("probabilities do not align with support graph")
    active = values.detach().cpu() >= config.support_threshold
    mode = SelectionMode(selection_mode)
    if mode is SelectionMode.ALL_COMPONENTS or not bool(active.any()):
        return active.to(values.device)
    labels = _component_labels(graph, active, config.component_edge_threshold)
    component_ids = labels[labels >= 0].unique()
    scores = torch.stack(
        [values.detach().cpu()[labels == component].sum() for component in component_ids]
    )
    if mode is SelectionMode.TOP_COMPONENT:
        keep = component_ids[scores.argmax()][None]
    elif mode is SelectionMode.TOP_K:
        count = min(config.top_k_components, component_ids.numel())
        keep = component_ids[scores.topk(count).indices]
    elif mode is SelectionMode.SEEDED_COMPONENT:
        if positive_seeds is None:
            raise ValueError("seeded-component selection requires positive seeds")
        seeds = positive_seeds.weights.detach().cpu()
        keep_list = []
        for component in component_ids:
            if bool((seeds[labels == component] >= config.seeded_component_min_weight).any()):
                keep_list.append(component)
        if not keep_list:
            seed_node = int(seeds.argmax())
            fallback = labels[seed_node]
            if fallback < 0:
                fallback = component_ids[scores.argmax()]
            keep_list = [fallback]
        keep = torch.stack(keep_list)
    else:  # pragma: no cover - Enum makes this unreachable.
        raise ValueError(f"unsupported selection mode: {mode}")
    selected = torch.zeros(graph.num_nodes, dtype=torch.bool)
    for component in keep:
        selected |= labels == component
    return selected.to(values.device)
