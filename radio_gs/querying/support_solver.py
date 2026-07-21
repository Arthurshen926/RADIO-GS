"""One symmetric primitive support graph and solver for every query modality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .query_spec import QueryIntent, SelectionMode, SoftSeedSet


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
    topology_mode: str = "symmetric_union"

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
        if self.topology_mode not in {"symmetric_union", "mutual_knn"}:
            raise ValueError("topology_mode must be symmetric_union or mutual_knn")


@dataclass(frozen=True)
class PrimitiveSupportGraph:
    """Symmetric graph with row-normalized message-passing weights."""

    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    raw_affinity: torch.Tensor
    local_sigma: torch.Tensor
    num_nodes: int
    edge_channels: Mapping[str, torch.Tensor] = field(default_factory=dict)

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
        channels: dict[str, torch.Tensor] = {}
        for name, values in dict(self.edge_channels).items():
            channel = torch.as_tensor(values).float().cpu().reshape(-1)
            if channel.shape != weight.shape:
                raise ValueError(f"edge channel {name!r} does not align with edges")
            if not bool(torch.isfinite(channel).all()) or bool((channel < 0).any()):
                raise ValueError(f"edge channel {name!r} must be finite and non-negative")
            channels[str(name)] = channel
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "edge_weight", weight)
        object.__setattr__(self, "raw_affinity", raw)
        object.__setattr__(self, "local_sigma", sigma)
        object.__setattr__(self, "edge_channels", channels)

    def to(self, device: torch.device | str) -> "PrimitiveSupportGraph":
        result = object.__new__(PrimitiveSupportGraph)
        object.__setattr__(result, "edge_index", self.edge_index.to(device))
        object.__setattr__(result, "edge_weight", self.edge_weight.to(device))
        object.__setattr__(result, "raw_affinity", self.raw_affinity.to(device))
        object.__setattr__(result, "local_sigma", self.local_sigma.to(device))
        object.__setattr__(result, "num_nodes", self.num_nodes)
        object.__setattr__(
            result,
            "edge_channels",
            {name: values.to(device) for name, values in self.edge_channels.items()},
        )
        return result


def mix_support_graph_channels(
    graph: PrimitiveSupportGraph,
    channel_weights: Mapping[str, float],
    *,
    legacy_residual: float = 0.0,
) -> PrimitiveSupportGraph:
    """Compose typed edge channels without letting one channel veto all edges.

    Message passing uses an arithmetic mixture of independently row-normalized
    channel transitions.  Component connectivity uses the corresponding
    weighted geometric affinity, which still preserves weak boundary evidence
    without reproducing the legacy unweighted product of every capability.
    """

    legacy_residual = float(legacy_residual)
    if not 0.0 <= legacy_residual <= 1.0:
        raise ValueError("legacy_residual must be in [0,1]")
    requested = {
        str(name): float(weight)
        for name, weight in channel_weights.items()
        if float(weight) > 0
    }
    if not requested:
        raise ValueError("at least one positive edge-channel weight is required")
    missing = sorted(set(requested) - set(graph.edge_channels))
    if missing:
        raise ValueError(f"support graph lacks requested edge channels: {missing}")
    total = sum(requested.values())
    normalized = {name: weight / total for name, weight in requested.items()}
    row = graph.edge_index[0]
    transition = torch.zeros_like(graph.edge_weight)
    log_affinity = torch.zeros_like(graph.raw_affinity)
    for name, weight in normalized.items():
        raw = graph.edge_channels[name].to(graph.edge_weight.device)
        row_sum = torch.zeros(
            graph.num_nodes, dtype=raw.dtype, device=raw.device
        )
        row_sum.index_add_(0, row, raw)
        transition.add_(weight * raw / row_sum[row].clamp_min(1e-12))
        log_affinity.add_(weight * raw.clamp_min(1e-12).log())
    if legacy_residual > 0:
        transition.mul_(1.0 - legacy_residual).add_(
            graph.edge_weight, alpha=legacy_residual
        )
        log_affinity.mul_(1.0 - legacy_residual).add_(
            graph.raw_affinity.clamp_min(1e-12).log(), alpha=legacy_residual
        )
    return PrimitiveSupportGraph(
        edge_index=graph.edge_index,
        edge_weight=transition,
        raw_affinity=log_affinity.exp(),
        local_sigma=graph.local_sigma,
        num_nodes=graph.num_nodes,
        edge_channels=graph.edge_channels,
    ).to(graph.edge_index.device)


def graph_for_query_intent(
    graph: PrimitiveSupportGraph,
    intent: QueryIntent,
    *,
    policy: str = "typed_if_available",
    legacy_residual: float = 0.0,
) -> PrimitiveSupportGraph:
    """Resolve one frozen, modality-independent channel policy for a query intent."""

    policy = str(policy)
    if policy == "legacy":
        return graph
    if policy == "typed_if_available":
        required = (
            {"geometry", "boundary"}
            if QueryIntent(intent) is QueryIntent.CATEGORY
            else {"geometry", "appearance", "boundary"}
        )
        if not required.issubset(graph.edge_channels):
            return graph
        policy = "typed"
    if policy == "typed":
        policy = (
            "category_mix"
            if QueryIntent(intent) is QueryIntent.CATEGORY
            else "instance_mix"
        )
    policies = {
        "geometry": {"geometry": 1.0},
        "appearance": {"appearance": 1.0},
        "boundary": {"boundary": 1.0},
        "category_mix": {"geometry": 0.5, "boundary": 0.5},
        "instance_mix": {
            "geometry": 0.2,
            "appearance": 0.4,
            "boundary": 0.4,
        },
    }
    if policy not in policies:
        raise ValueError(f"unknown support graph channel policy: {policy!r}")
    return mix_support_graph_channels(
        graph, policies[policy], legacy_residual=legacy_residual
    )


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
        channel_names = ["geometry"]
        if appearance_features is not None:
            channel_names.append("appearance")
        if boundary_features is not None:
            channel_names.append("boundary")
        if normals is not None:
            channel_names.append("normal")
        if view_observations is not None:
            channel_names.append("covisibility")
        return PrimitiveSupportGraph(
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_weight=torch.empty(0),
            raw_affinity=torch.empty(0),
            local_sigma=torch.full((1,), config.minimum_sigma),
            num_nodes=1,
            edge_channels={name: torch.empty(0) for name in channel_names},
        )

    k = min(config.neighbors + 1, count)
    distances_np, neighbors_np = cKDTree(points.numpy()).query(points.numpy(), k=k)
    distances = torch.from_numpy(np.asarray(distances_np, dtype=np.float32)[:, 1:])
    neighbors = torch.from_numpy(np.asarray(neighbors_np, dtype=np.int64)[:, 1:])
    local_sigma = distances.median(dim=1).values.mul(config.spatial_scale)
    local_sigma.clamp_(min=config.minimum_sigma)

    directed_rows = (
        torch.arange(count, dtype=torch.long)[:, None]
        .expand_as(neighbors)
        .reshape(-1)
        .numpy()
    )
    directed_cols = neighbors.reshape(-1).numpy()
    if config.topology_mode == "symmetric_union":
        undirected = np.stack(
            [
                np.concatenate([directed_rows, directed_cols]),
                np.concatenate([directed_cols, directed_rows]),
            ],
            axis=1,
        )
        undirected = np.unique(undirected, axis=0)
        row = torch.from_numpy(undirected[:, 0].copy()).long()
        col = torch.from_numpy(undirected[:, 1].copy()).long()
    else:
        codes = directed_rows.astype(np.int64) * int(count) + directed_cols
        reverse_codes = directed_cols.astype(np.int64) * int(count) + directed_rows
        sorted_codes = np.sort(codes)
        positions = np.searchsorted(sorted_codes, reverse_codes)
        safe_positions = np.minimum(positions, max(0, sorted_codes.size - 1))
        keep = (positions < sorted_codes.size) & (
            sorted_codes[safe_positions] == reverse_codes
        )
        mutual_codes = np.unique(codes[keep])
        row = torch.from_numpy((mutual_codes // int(count)).copy()).long()
        col = torch.from_numpy((mutual_codes % int(count)).copy()).long()
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
    channel_parts: dict[str, list[torch.Tensor]] = {"geometry": []}
    if appearance is not None:
        channel_parts["appearance"] = []
    if boundary is not None:
        channel_parts["boundary"] = []
    if normal is not None:
        channel_parts["normal"] = []
    if visibility is not None and visibility.shape[1] > 0:
        channel_parts["covisibility"] = []
    for start in range(0, row.numel(), config.affinity_chunk_size):
        stop = min(start + config.affinity_chunk_size, row.numel())
        src, dst = row[start:stop], col[start:stop]
        distance2 = (points[src] - points[dst]).square().sum(dim=-1)
        pair_sigma2 = (local_sigma[src] * local_sigma[dst]).clamp_min(
            config.minimum_sigma**2
        )
        geometry_log = -0.5 * distance2 / pair_sigma2
        log_affinity = geometry_log.clone()
        channel_parts["geometry"].append(
            geometry_log.clamp(min=-60.0, max=0.0).exp()
        )
        if appearance is not None:
            cosine = (appearance[src] * appearance[dst]).sum(dim=-1)
            appearance_log = (cosine - 1.0) / config.appearance_temperature
            log_affinity.add_(appearance_log)
            channel_parts["appearance"].append(
                appearance_log.clamp(min=-60.0, max=0.0).exp()
            )
        if boundary is not None:
            cosine = (boundary[src] * boundary[dst]).sum(dim=-1)
            boundary_log = (cosine - 1.0) / config.boundary_temperature
            log_affinity.add_(boundary_log)
            channel_parts["boundary"].append(
                boundary_log.clamp(min=-60.0, max=0.0).exp()
            )
        if normal is not None:
            cosine = (normal[src] * normal[dst]).sum(dim=-1).clamp(-1.0, 1.0)
            normal_log = (cosine - 1.0) / config.normal_temperature
            log_affinity.add_(normal_log)
            channel_parts["normal"].append(
                normal_log.clamp(min=-60.0, max=0.0).exp()
            )
        if visibility is not None and visibility.shape[1] > 0:
            shared = (visibility[src] & visibility[dst]).sum(dim=-1).float()
            union = (visibility[src] | visibility[dst]).sum(dim=-1).float()
            jaccard = shared / union.clamp_min(1.0)
            covisibility_log = config.covisibility_weight * (jaccard - 1.0)
            log_affinity.add_(covisibility_log)
            channel_parts["covisibility"].append(
                covisibility_log.clamp(min=-60.0, max=0.0).exp()
            )
        raw_parts.append(log_affinity.clamp(min=-60.0, max=0.0).exp())
    raw_affinity = torch.cat(raw_parts)
    edge_channels = {
        name: torch.cat(parts) for name, parts in channel_parts.items()
    }
    row_sum = torch.zeros(count, dtype=torch.float32)
    row_sum.index_add_(0, row, raw_affinity)
    edge_weight = raw_affinity / row_sum[row].clamp_min(1e-12)
    return PrimitiveSupportGraph(
        edge_index=edge_index,
        edge_weight=edge_weight,
        raw_affinity=raw_affinity,
        local_sigma=local_sigma,
        num_nodes=count,
        edge_channels=edge_channels,
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
    solver_type: str = "diffusion"
    laplacian_weight: float = 1.0
    cg_iterations: int = 64
    cg_tolerance: float = 1e-5
    hard_seed_threshold: float = 0.20

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.unary_temperature <= 0:
            raise ValueError("iterations must be non-negative and temperature positive")
        if not 0 <= self.residual <= 1:
            raise ValueError("residual must be in [0,1]")
        if not 0 <= self.support_threshold <= 1:
            raise ValueError("support_threshold must be in [0,1]")
        if self.component_edge_threshold < 0 or self.top_k_components <= 0:
            raise ValueError("component parameters are invalid")
        if self.solver_type not in {
            "diffusion", "random_walker", "confidence_random_walker"
        }:
            raise ValueError(
                "solver_type must be diffusion, random_walker, or confidence_random_walker"
            )
        if self.laplacian_weight < 0 or self.cg_iterations <= 0:
            raise ValueError("random-walker parameters are invalid")
        if self.cg_tolerance <= 0 or not 0 <= self.hard_seed_threshold <= 1:
            raise ValueError("CG tolerance/seed threshold are invalid")


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
    normalized_affinity: torch.Tensor | None = None,
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
    if config.solver_type in {"random_walker", "confidence_random_walker"}:
        automatic_weight = (
            confidence_aware_laplacian_weight(
                prior, positive, negative, base_weight=config.laplacian_weight
            )
            if config.solver_type == "confidence_random_walker"
            else float(config.laplacian_weight)
        )
        return solve_seeded_random_walker(
            working_graph,
            prior,
            positive,
            negative,
            config=config,
            laplacian_weight=automatic_weight,
            normalized_affinity=normalized_affinity,
        )
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


def confidence_aware_laplacian_weight(
    prior: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    base_weight: float,
) -> float:
    """Choose graph regularization from query evidence, never benchmark identity.

    Uncertain unary evidence receives more graph regularization.  Reliable
    positive/negative separation permits that regularization to increase, while
    already confident unary evidence stays close to its direct prediction.
    """

    values = torch.as_tensor(prior).float().clamp(1e-6, 1.0 - 1e-6)
    entropy = -(values * values.log() + (1.0 - values) * (1.0 - values).log())
    uncertainty = float(entropy.mean() / torch.log(values.new_tensor(2.0)))

    def weighted_mean(weights: torch.Tensor) -> torch.Tensor | None:
        weights = torch.as_tensor(weights, device=values.device).float().clamp_min(0)
        mass = weights.sum()
        return (values * weights).sum() / mass if float(mass) > 0 else None

    positive_mean = weighted_mean(positive)
    negative_mean = weighted_mean(negative)
    separation = (
        float((positive_mean - negative_mean).abs().clamp(0.0, 1.0))
        if positive_mean is not None and negative_mean is not None
        else 0.0
    )
    multiplier = max(0.25, min(1.5, 0.25 + uncertainty * (0.5 + separation)))
    return float(base_weight) * multiplier


def solve_seeded_random_walker(
    graph: PrimitiveSupportGraph,
    prior: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    config: SupportSolverConfig,
    laplacian_weight: float | None = None,
    normalized_affinity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Solve a confidence-weighted normalized Laplacian with hard seeds.

    Positive/negative seeds above the fixed solver threshold are eliminated
    from the linear system and are therefore exactly 1/0, not soft penalties.
    The remaining unary fidelity is its label-free Bernoulli confidence.
    """

    device = prior.device
    row, col = graph.edge_index
    if normalized_affinity is None:
        normalized_affinity = normalized_laplacian_affinity(graph)
    else:
        normalized_affinity = torch.as_tensor(
            normalized_affinity, device=device
        ).float().reshape(-1)
        if normalized_affinity.shape != graph.raw_affinity.shape:
            raise ValueError("normalized_affinity must align with graph edges")
        if not bool(torch.isfinite(normalized_affinity).all()) or bool(
            (normalized_affinity < 0).any()
        ):
            raise ValueError("normalized_affinity must be finite and non-negative")
    hard_positive = positive >= float(config.hard_seed_threshold)
    hard_negative = negative >= float(config.hard_seed_threshold)
    hard_negative &= ~hard_positive
    fixed = hard_positive | hard_negative
    free = ~fixed
    fixed_values = hard_positive.to(prior.dtype)
    if not bool(free.any()):
        return fixed_values

    confidence = (2.0 * prior - 1.0).abs().clamp_min(0.05)
    lam = float(
        config.laplacian_weight if laplacian_weight is None else laplacian_weight
    )

    def laplacian(vector: torch.Tensor) -> torch.Tensor:
        message = torch.zeros_like(vector)
        if row.numel():
            message.index_add_(0, row, normalized_affinity * vector[col])
        return vector - message

    def operator(vector: torch.Tensor) -> torch.Tensor:
        masked = vector * free
        return (confidence * masked + lam * laplacian(masked)) * free

    right = (confidence * prior - lam * laplacian(fixed_values)) * free
    solution = prior * free
    residual = right - operator(solution)
    direction = residual.clone()
    residual_norm = torch.dot(residual, residual)
    initial_norm = residual_norm.sqrt().clamp_min(1e-12)
    for _ in range(int(config.cg_iterations)):
        product = operator(direction)
        denominator = torch.dot(direction, product).clamp_min(1e-20)
        step = residual_norm / denominator
        solution = solution + step * direction
        next_residual = residual - step * product
        next_norm = torch.dot(next_residual, next_residual)
        if float(next_norm.sqrt() / initial_norm) <= float(config.cg_tolerance):
            residual = next_residual
            break
        direction = next_residual + (next_norm / residual_norm.clamp_min(1e-20)) * direction
        residual = next_residual
        residual_norm = next_norm
    probability = solution * free + fixed_values
    return probability.clamp(0.0, 1.0)


def normalized_laplacian_affinity(graph: PrimitiveSupportGraph) -> torch.Tensor:
    """Return the symmetric normalized affinity used by the random walker.

    It depends only on the frozen scene graph, so callers evaluating many
    click sequences on one graph may cache it without changing the solver or
    its query inputs.  The arithmetic intentionally matches the former
    in-solver construction exactly.
    """

    row, col = graph.edge_index
    affinity = graph.raw_affinity.float()
    degree = torch.zeros(graph.num_nodes, device=affinity.device)
    if row.numel():
        degree.index_add_(0, row, affinity)
    inverse_sqrt = degree.clamp_min(1e-12).rsqrt()
    return affinity * inverse_sqrt[row] * inverse_sqrt[col]


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


def _seeded_component_mask(
    graph: PrimitiveSupportGraph,
    active: torch.Tensor,
    positive_seeds: SoftSeedSet,
    *,
    edge_threshold: float,
    minimum_seed_weight: float,
) -> torch.Tensor:
    """Return exactly the active components touched by declared soft seeds.

    Unlike full connected-component labelling this only traverses components
    that can affect ``SEEDED_COMPONENT`` output.  The graph and active mask are
    unchanged, so this is an inference acceleration rather than a new solver.
    """

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order

    mask = torch.as_tensor(active).bool().cpu().numpy()
    seed_values = positive_seeds.weights.detach().float().cpu().numpy()
    seed_nodes = np.flatnonzero(
        mask & (seed_values >= float(minimum_seed_weight))
    )
    selected = np.zeros(graph.num_nodes, dtype=bool)
    if seed_nodes.size == 0:
        return torch.from_numpy(selected)
    edges = graph.edge_index.detach().cpu().numpy()
    affinity = graph.raw_affinity.detach().cpu().numpy()
    keep = (
        (affinity >= float(edge_threshold))
        & mask[edges[0]]
        & mask[edges[1]]
    )
    if not bool(keep.any()):
        selected[seed_nodes] = True
        return torch.from_numpy(selected)
    adjacency = coo_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.uint8),
            (edges[0, keep], edges[1, keep]),
        ),
        shape=(graph.num_nodes, graph.num_nodes),
    ).tocsr()
    for seed in seed_nodes:
        if selected[seed]:
            continue
        reached = breadth_first_order(
            adjacency,
            int(seed),
            directed=False,
            return_predecessors=False,
        )
        reached = np.asarray(reached, dtype=np.int64)
        selected[reached[mask[reached]]] = True
    return torch.from_numpy(selected)


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
    if mode is SelectionMode.SEEDED_COMPONENT and positive_seeds is not None:
        selected = _seeded_component_mask(
            graph,
            active,
            positive_seeds,
            edge_threshold=config.component_edge_threshold,
            minimum_seed_weight=config.seeded_component_min_weight,
        )
        if bool(selected.any()):
            return selected.to(values.device)
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
