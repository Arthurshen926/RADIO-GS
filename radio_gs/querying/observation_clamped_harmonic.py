"""Dirichlet graph completion for a typed source-view object posterior.

This module is intentionally opt-in and independent of the shared query
engine.  It turns non-zero source-observation confidence into an immutable
Dirichlet boundary and solves only the unknown graph interior.  Existing
query compilers and support solvers therefore remain byte-for-byte unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .support_solver import PrimitiveSupportGraph


@dataclass(frozen=True)
class ObservationClampedHarmonicConfig:
    """Numerical and shared hard-seed policy for the Dirichlet solve."""

    cg_iterations: int = 64
    cg_tolerance: float = 1e-5
    hard_seed_threshold: float = 0.20
    hard_seed_conflict_policy: str = "exclusive_relative"
    hard_seed_conflict_margin: float = 0.0

    def __post_init__(self) -> None:
        if int(self.cg_iterations) <= 0 or float(self.cg_tolerance) <= 0:
            raise ValueError("harmonic CG iterations/tolerance must be positive")
        if not 0 <= float(self.hard_seed_threshold) <= 1:
            raise ValueError("harmonic hard-seed threshold must be in [0,1]")
        if self.hard_seed_conflict_policy not in {
            "positive_priority",
            "exclusive_relative",
        }:
            raise ValueError("unknown harmonic hard-seed conflict policy")
        if float(self.hard_seed_conflict_margin) < 0:
            raise ValueError("harmonic hard-seed conflict margin cannot be negative")


def method_contract() -> dict[str, object]:
    """Return the target-independent versioned scientific contract."""

    return {
        "schema_version": 1,
        "method": "observation_clamped_harmonic_extension_v1",
        "source_boundary": "primitive_source_confidence_strictly_greater_than_zero",
        "boundary_value": "already_fused_source_and_field_unary_probability",
        "unknown_interior": "source_confidence_equals_zero",
        "graph_operator": "unnormalized_symmetric_nonnegative_affinity_laplacian",
        "unobserved_component_policy": "preserve_field_prior_bitwise",
        "source_boundary_rewrite": False,
        "connected_selection": False,
        "learned_or_scene_specific_constants": False,
        "uses_target_rgb_mask_or_metric": False,
    }


def _vector(
    value: torch.Tensor,
    *,
    count: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device).float().reshape(-1)
    if result.shape != (count,) or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be a finite [num_nodes] vector")
    return result


def _hard_seed_masks(
    positive: torch.Tensor,
    negative: torch.Tensor,
    config: ObservationClampedHarmonicConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    threshold = float(config.hard_seed_threshold)
    hard_positive = positive >= threshold
    hard_negative = negative >= threshold
    if config.hard_seed_conflict_policy == "positive_priority":
        return hard_positive, hard_negative & ~hard_positive
    margin = float(config.hard_seed_conflict_margin)
    return (
        hard_positive & (positive > negative + margin),
        hard_negative & (negative > positive + margin),
    )


def _boundary_connected_mask(
    graph: PrimitiveSupportGraph,
    boundary: torch.Tensor,
) -> torch.Tensor:
    """Find positive-affinity components with at least one boundary row."""

    fixed = torch.as_tensor(boundary).bool().reshape(-1)
    if fixed.shape != (graph.num_nodes,):
        raise ValueError("harmonic boundary mask must align with graph nodes")
    if not bool(fixed.any()):
        return torch.zeros_like(fixed)

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = graph.edge_index.detach().cpu().numpy()
    affinity = graph.raw_affinity.detach().float().cpu().numpy()
    positive = affinity > 0
    adjacency = coo_matrix(
        (
            np.ones(int(positive.sum()), dtype=np.uint8),
            (edges[0, positive], edges[1, positive]),
        ),
        shape=(graph.num_nodes, graph.num_nodes),
    ).tocsr()
    _count, labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    boundary_labels = np.unique(labels[fixed.detach().cpu().numpy()])
    return torch.from_numpy(np.isin(labels, boundary_labels)).to(fixed.device)


def solve_observation_clamped_harmonic(
    graph: PrimitiveSupportGraph,
    prior_probability: torch.Tensor,
    source_observation_confidence: torch.Tensor,
    *,
    positive_seed_weight: torch.Tensor | None = None,
    negative_seed_weight: torch.Tensor | None = None,
    config: ObservationClampedHarmonicConfig = (
        ObservationClampedHarmonicConfig()
    ),
) -> torch.Tensor:
    """Extend a source posterior to unknown rows under Dirichlet constraints.

    The returned tensor equals ``prior_probability`` exactly on every row with
    non-zero source confidence.  It also preserves the prior exactly on graph
    components that contain no observation.  Optional signed hard seeds reuse
    the shared 0.20/exclusive-relative policy and override their boundary to
    exact 1/0.
    """

    device = torch.as_tensor(prior_probability).device
    count = int(graph.num_nodes)
    prior = _vector(
        prior_probability, count=count, name="prior_probability", device=device
    )
    confidence = _vector(
        source_observation_confidence,
        count=count,
        name="source_observation_confidence",
        device=device,
    )
    if bool(((prior < 0) | (prior > 1)).any()):
        raise ValueError("prior_probability must be in [0,1]")
    if bool(((confidence < 0) | (confidence > 1)).any()):
        raise ValueError("source_observation_confidence must be in [0,1]")
    positive = (
        torch.zeros_like(prior)
        if positive_seed_weight is None
        else _vector(
            positive_seed_weight,
            count=count,
            name="positive_seed_weight",
            device=device,
        )
    )
    negative = (
        torch.zeros_like(prior)
        if negative_seed_weight is None
        else _vector(
            negative_seed_weight,
            count=count,
            name="negative_seed_weight",
            device=device,
        )
    )
    if bool((positive < 0).any()) or bool((negative < 0).any()):
        raise ValueError("harmonic hard-seed weights cannot be negative")

    hard_positive, hard_negative = _hard_seed_masks(
        positive, negative, config
    )
    observed = confidence > 0
    boundary = observed | hard_positive | hard_negative
    if not bool(boundary.any()):
        return prior

    boundary_values = prior.clone()
    boundary_values[hard_positive] = 1.0
    boundary_values[hard_negative] = 0.0
    working_graph = (
        graph if graph.edge_index.device == device else graph.to(device)
    )
    reachable = _boundary_connected_mask(working_graph, boundary).to(device)
    free = reachable & ~boundary
    if not bool(free.any()):
        output = prior.clone()
        output[boundary] = boundary_values[boundary]
        return output

    row, col = working_graph.edge_index
    affinity = working_graph.raw_affinity.float()
    degree = torch.zeros(count, device=device)
    if row.numel():
        degree.index_add_(0, row, affinity)

    def laplacian(vector: torch.Tensor) -> torch.Tensor:
        message = torch.zeros_like(vector)
        if row.numel():
            message.index_add_(0, row, affinity * vector[col])
        return degree * vector - message

    def operator(vector: torch.Tensor) -> torch.Tensor:
        return laplacian(vector * free) * free

    fixed_values = boundary_values * boundary
    right = -laplacian(fixed_values) * free
    solution = prior * free
    residual = right - operator(solution)
    direction = residual.clone()
    residual_norm = torch.dot(residual, residual)
    initial_norm = residual_norm.sqrt().clamp_min(1e-12)
    for _ in range(int(config.cg_iterations)):
        if float(residual_norm) == 0.0:
            break
        product = operator(direction)
        denominator = torch.dot(direction, product)
        if float(denominator) <= 0.0:
            raise RuntimeError(
                "observation-clamped harmonic system is not positive definite"
            )
        step = residual_norm / denominator
        solution = solution + step * direction
        next_residual = residual - step * product
        next_norm = torch.dot(next_residual, next_residual)
        if float(next_norm.sqrt() / initial_norm) <= float(config.cg_tolerance):
            residual = next_residual
            break
        direction = next_residual + (
            next_norm / residual_norm.clamp_min(1e-30)
        ) * direction
        residual = next_residual
        residual_norm = next_norm

    output = prior.clone()
    output[free] = solution[free].clamp(0.0, 1.0)
    output[boundary] = boundary_values[boundary]
    return output

