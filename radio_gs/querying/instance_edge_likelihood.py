"""Query-independent, monotone likelihood ratios for canonical instance edges."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .support_solver import PrimitiveSupportGraph


@dataclass(frozen=True)
class InstanceEdgeFeatures:
    appearance_similarity: torch.Tensor
    boundary_similarity: torch.Tensor
    scaled_distance: torch.Tensor
    endpoint_reliability: torch.Tensor
    endpoint_coverage: torch.Tensor

    def validated(self) -> "InstanceEdgeFeatures":
        tensors = {
            name: torch.as_tensor(value).float().reshape(-1)
            for name, value in vars(self).items()
        }
        lengths = {len(value) for value in tensors.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
            raise ValueError("instance edge features must be aligned non-empty vectors")
        for name, value in tensors.items():
            if not bool(torch.isfinite(value).all()) or bool(
                ((value < 0) | (value > 1)).any()
            ):
                raise ValueError(f"instance edge feature {name} must be in [0,1]")
        return InstanceEdgeFeatures(**tensors)

    def matrix(self) -> torch.Tensor:
        values = self.validated()
        return torch.stack(
            [
                values.appearance_similarity,
                values.boundary_similarity,
                values.scaled_distance,
                values.endpoint_reliability,
                values.endpoint_coverage,
            ],
            dim=-1,
        )


class MonotoneInstanceEdgeLikelihood(nn.Module):
    """Global scene-shared edge LLR with audited derivative signs."""

    schema_version = "monotone-query-independent-instance-edge-lr-v9"

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.raw_appearance_weight = nn.Parameter(torch.zeros(()))
        self.raw_boundary_weight = nn.Parameter(torch.zeros(()))
        self.raw_distance_weight = nn.Parameter(torch.zeros(()))
        self.raw_reliability_weight = nn.Parameter(torch.zeros(()))
        self.raw_coverage_weight = nn.Parameter(torch.zeros(()))

    @property
    def signed_weights(self) -> torch.Tensor:
        return torch.stack(
            [
                F.softplus(self.raw_appearance_weight),
                F.softplus(self.raw_boundary_weight),
                -F.softplus(self.raw_distance_weight),
                F.softplus(self.raw_reliability_weight),
                F.softplus(self.raw_coverage_weight),
            ]
        )

    def log_likelihood_ratio(self, features: InstanceEdgeFeatures) -> torch.Tensor:
        return self.bias + features.matrix() @ self.signed_weights


def instance_edge_features_from_graph(
    graph: PrimitiveSupportGraph,
    *,
    reliability: torch.Tensor,
    coverage: torch.Tensor,
) -> InstanceEdgeFeatures:
    required = {"appearance", "boundary", "geometry"}
    if not required.issubset(graph.edge_channels):
        raise ValueError("instance edge likelihood requires typed graph channels")
    node_reliability = torch.as_tensor(
        reliability, device=graph.edge_index.device
    ).float().reshape(-1)
    node_coverage = torch.as_tensor(
        coverage, device=graph.edge_index.device
    ).float().reshape(-1)
    if node_reliability.shape != (graph.num_nodes,) or node_coverage.shape != (
        graph.num_nodes,
    ):
        raise ValueError("edge endpoint authority must align with graph nodes")
    if (
        not bool(torch.isfinite(node_reliability).all())
        or not bool(torch.isfinite(node_coverage).all())
        or bool(((node_reliability < 0) | (node_reliability > 1)).any())
        or bool(((node_coverage < 0) | (node_coverage > 1)).any())
    ):
        raise ValueError("edge endpoint authority must be in [0,1]")
    row, col = graph.edge_index
    geometry = graph.edge_channels["geometry"].float().clamp(1e-12, 1.0)
    normalized_distance = torch.sqrt((-2.0 * geometry.log()).clamp_min(0.0))
    normalized_distance = normalized_distance / (1.0 + normalized_distance)
    return InstanceEdgeFeatures(
        appearance_similarity=graph.edge_channels["appearance"].float().clamp(0, 1),
        boundary_similarity=graph.edge_channels["boundary"].float().clamp(0, 1),
        scaled_distance=normalized_distance.clamp(0, 1),
        endpoint_reliability=torch.sqrt(node_reliability[row] * node_reliability[col]),
        endpoint_coverage=torch.sqrt(node_coverage[row] * node_coverage[col]),
    )


def gate_graph_by_instance_edge_likelihood(
    graph: PrimitiveSupportGraph,
    features: InstanceEdgeFeatures,
    head: MonotoneInstanceEdgeLikelihood,
    *,
    apply_edge_likelihood: bool = False,
    eps: float = 1e-12,
) -> PrimitiveSupportGraph:
    """Prune negative-LR candidate edges and row-normalize conductance.

    The zero LLR boundary is the equal-prior density-ratio decision; it is not
    a benchmark-tuned threshold.  Default-off returns the exact input object.
    """

    if not apply_edge_likelihood:
        return graph
    if float(eps) <= 0:
        raise ValueError("eps must be positive")
    score = head.log_likelihood_ratio(features)
    if score.shape != graph.edge_weight.shape:
        raise ValueError("edge likelihood does not align with graph edges")
    keep = score >= 0
    conductance = graph.edge_weight.float() * torch.sigmoid(score) * keep
    row = graph.edge_index[0]
    row_sum = torch.zeros(
        graph.num_nodes, device=conductance.device, dtype=conductance.dtype
    )
    if row.numel():
        row_sum.index_add_(0, row, conductance)
    transition = conductance / row_sum[row].clamp_min(float(eps))
    result = object.__new__(PrimitiveSupportGraph)
    object.__setattr__(result, "edge_index", graph.edge_index)
    object.__setattr__(result, "edge_weight", transition)
    object.__setattr__(result, "raw_affinity", graph.raw_affinity * keep)
    object.__setattr__(result, "local_sigma", graph.local_sigma)
    object.__setattr__(result, "num_nodes", graph.num_nodes)
    object.__setattr__(result, "edge_channels", graph.edge_channels)
    return result


def absorbing_component_seed_support(
    graph: PrimitiveSupportGraph,
    hard_positive_seeds: torch.Tensor,
) -> torch.Tensor:
    """Return exact retained components touched by each positive seed set.

    Learned instance edges make positive absorption meaningful to convergence:
    the complete retained component is extent, while every unseeded component
    remains exact abstention.  This operation cannot introduce a new edge.
    """

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    seeds = torch.as_tensor(hard_positive_seeds).bool()
    squeeze = seeds.ndim == 1
    if squeeze:
        seeds = seeds[:, None]
    if seeds.ndim != 2 or seeds.shape[0] != graph.num_nodes or seeds.shape[1] <= 0:
        raise ValueError("component seeds must be [N] or [N,K]")
    row = graph.edge_index[0].detach().cpu().numpy()
    col = graph.edge_index[1].detach().cpu().numpy()
    keep = (graph.edge_weight.detach().cpu().numpy() > 0)
    adjacency = coo_matrix(
        (
            torch.ones(int(keep.sum())).numpy(),
            (row[keep], col[keep]),
        ),
        shape=(graph.num_nodes, graph.num_nodes),
    ).tocsr()
    _count, labels_np = connected_components(
        adjacency, directed=False, return_labels=True
    )
    labels = torch.from_numpy(labels_np).long()
    seeds_cpu = seeds.cpu()
    support = torch.zeros_like(seeds_cpu)
    for column in range(seeds_cpu.shape[1]):
        rows = torch.nonzero(seeds_cpu[:, column], as_tuple=False).flatten()
        if rows.numel():
            support[:, column] = torch.isin(labels, labels[rows].unique())
    support = support.to(seeds.device)
    return support[:, 0] if squeeze else support
