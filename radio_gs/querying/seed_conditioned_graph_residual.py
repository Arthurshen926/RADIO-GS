"""Bounded, seed-conditioned residuals on a canonical primitive graph.

This module deliberately separates two kinds of evidence.  A likelihood head
supplies the node-local log likelihood ratio from registered observations.
The graph residual only supplies instance extent from query-independent
primitive relations and declared positive/negative seeds.  No object label or
point-level primitive surrogate is accepted by this interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .query_likelihood_head import QueryLikelihoodInputs
from .query_spec import PrimitiveUnaryEvidence
from .support_solver import PrimitiveSupportGraph


def reliability_weighted_support_graph(
    graph: PrimitiveSupportGraph,
    reliability: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> PrimitiveSupportGraph:
    """Apply a query-independent endpoint reliability conductance.

    The input transition can be a registered arithmetic mixture of typed
    geometry/appearance/boundary channels.  Multiplication by the geometric
    mean of endpoint reliability preserves non-negativity; row normalization
    makes the result a substochastic/row-stochastic propagation operator.
    Rows with no reliable outgoing edge remain exactly zero.
    """

    values = torch.as_tensor(
        reliability, device=graph.edge_index.device
    ).float().reshape(-1)
    if values.shape != (graph.num_nodes,) or not bool(torch.isfinite(values).all()):
        raise ValueError("reliability must be a finite [num_nodes] vector")
    if bool(((values < 0) | (values > 1)).any()):
        raise ValueError("reliability must be in [0,1]")
    if float(eps) <= 0:
        raise ValueError("eps must be positive")
    row, col = graph.edge_index
    conductance = graph.edge_weight.float() * torch.sqrt(values[row] * values[col])
    row_sum = torch.zeros(
        graph.num_nodes, dtype=conductance.dtype, device=conductance.device
    )
    if row.numel():
        row_sum.index_add_(0, row, conductance)
    transition = conductance / row_sum[row].clamp_min(float(eps))
    result = object.__new__(PrimitiveSupportGraph)
    object.__setattr__(result, "edge_index", graph.edge_index)
    object.__setattr__(result, "edge_weight", transition)
    object.__setattr__(result, "raw_affinity", graph.raw_affinity)
    object.__setattr__(result, "local_sigma", graph.local_sigma)
    object.__setattr__(result, "num_nodes", graph.num_nodes)
    object.__setattr__(result, "edge_channels", graph.edge_channels)
    return result


def nonnegative_seed_hop_stack(
    graph: PrimitiveSupportGraph,
    seeds: torch.Tensor,
    *,
    steps: int,
    decay: float,
) -> torch.Tensor:
    """Return ``[N,K,steps+1]`` non-negative seed propagation features."""

    values = torch.as_tensor(seeds).float()
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    if (
        values.ndim != 2
        or values.shape[0] != graph.num_nodes
        or values.shape[1] <= 0
        or not bool(torch.isfinite(values).all())
        or bool(((values < 0) | (values > 1)).any())
    ):
        raise ValueError("seeds must be finite [N] or [N,K] values in [0,1]")
    steps = int(steps)
    decay = float(decay)
    if steps < 0 or not 0.0 <= decay <= 1.0:
        raise ValueError("steps must be non-negative and decay must be in [0,1]")
    working = graph if graph.edge_index.device == values.device else graph.to(values.device)
    row, col = working.edge_index
    frontier = values
    outputs = [frontier]
    for _ in range(steps):
        propagated = torch.zeros_like(frontier)
        if row.numel():
            propagated.index_add_(
                0, row, working.edge_weight[:, None] * frontier[col]
            )
        frontier = decay * propagated
        outputs.append(frontier)
    stack = torch.stack(outputs, dim=-1)
    return stack[:, 0] if squeeze else stack


class SeedConditionedGraphResidualHead(nn.Module):
    """A globally bounded graph residual around a frozen local LLR head.

    Hop weights and the residual gate are non-negative.  Therefore adding a
    negative seed can never increase any node logit.  A convex hop mixture and
    ``max_logit_residual`` bound the graph correction globally.  The default
    call is deliberately disabled and returns the base head object unchanged,
    which makes adoption opt-in and preserves the existing query path exactly.
    """

    schema_version = "seed-conditioned-typed-graph-residual-v7"

    def __init__(
        self,
        base_head: nn.Module,
        *,
        propagation_steps: int = 4,
        propagation_decay: float = 0.85,
        max_logit_residual: float = 4.0,
        hard_seed_threshold: float = 0.20,
    ) -> None:
        super().__init__()
        if not hasattr(base_head, "log_likelihood_ratio"):
            raise TypeError("base_head must expose log_likelihood_ratio")
        if int(propagation_steps) < 0:
            raise ValueError("propagation_steps must be non-negative")
        if not 0.0 <= float(propagation_decay) <= 1.0:
            raise ValueError("propagation_decay must be in [0,1]")
        if float(max_logit_residual) <= 0:
            raise ValueError("max_logit_residual must be positive")
        if not 0.0 <= float(hard_seed_threshold) <= 1.0:
            raise ValueError("hard_seed_threshold must be in [0,1]")
        self.base_head = base_head
        for parameter in self.base_head.parameters():
            parameter.requires_grad_(False)
        self.propagation_steps = int(propagation_steps)
        self.propagation_decay = float(propagation_decay)
        self.max_logit_residual = float(max_logit_residual)
        self.hard_seed_threshold = float(hard_seed_threshold)
        self.raw_residual_gate = nn.Parameter(torch.zeros(()))
        self.raw_hop_weights = nn.Parameter(
            torch.zeros(self.propagation_steps + 1)
        )

    @property
    def residual_gate(self) -> torch.Tensor:
        return self.max_logit_residual * torch.sigmoid(self.raw_residual_gate)

    @property
    def hop_weights(self) -> torch.Tensor:
        return torch.softmax(self.raw_hop_weights, dim=0)

    def structured_log_likelihood_ratio(
        self,
        observations: QueryLikelihoodInputs,
        *,
        graph: PrimitiveSupportGraph,
        positive_seeds: torch.Tensor,
        negative_seeds: torch.Tensor,
    ) -> torch.Tensor:
        inputs = observations.validated()
        base = self.base_head.log_likelihood_ratio(inputs)
        positive = torch.as_tensor(positive_seeds, device=base.device).float().reshape(-1)
        negative = torch.as_tensor(negative_seeds, device=base.device).float().reshape(-1)
        if positive.shape != base.shape or negative.shape != base.shape:
            raise ValueError("positive/negative seeds must align with primitive rows")
        working = graph if graph.edge_index.device == base.device else graph.to(base.device)
        positive_hops = nonnegative_seed_hop_stack(
            working,
            positive,
            steps=self.propagation_steps,
            decay=self.propagation_decay,
        )
        negative_hops = nonnegative_seed_hop_stack(
            working,
            negative,
            steps=self.propagation_steps,
            decay=self.propagation_decay,
        )
        contrast = ((positive_hops - negative_hops) * self.hop_weights).sum(dim=-1)
        return base + self.residual_gate * contrast

    def forward(
        self,
        observations: QueryLikelihoodInputs,
        *,
        graph: PrimitiveSupportGraph,
        positive_seeds: torch.Tensor,
        negative_seeds: torch.Tensor,
        source: str,
        apply_residual: bool = False,
    ) -> PrimitiveUnaryEvidence:
        if not apply_residual:
            return self.base_head(observations, source=source)
        inputs = observations.validated()
        log_likelihood_ratio = self.structured_log_likelihood_ratio(
            inputs,
            graph=graph,
            positive_seeds=positive_seeds,
            negative_seeds=negative_seeds,
        )
        positive = torch.as_tensor(
            positive_seeds, device=log_likelihood_ratio.device
        ).float().reshape(-1)
        negative = torch.as_tensor(
            negative_seeds, device=log_likelihood_ratio.device
        ).float().reshape(-1)
        hard_positive = positive >= self.hard_seed_threshold
        hard_negative = (negative >= self.hard_seed_threshold) & ~hard_positive
        probability = torch.sigmoid(log_likelihood_ratio)
        probability = torch.where(hard_negative, torch.zeros_like(probability), probability)
        probability = torch.where(hard_positive, torch.ones_like(probability), probability)
        confidence = (inputs.coverage * inputs.reliability).clamp(0.0, 1.0)
        # A declared hard seed is itself an observation, even if the
        # query-independent field coverage was previously zero at that row.
        confidence = torch.where(
            hard_positive | hard_negative,
            torch.ones_like(confidence),
            confidence,
        )
        return PrimitiveUnaryEvidence.from_probability(
            probability,
            confidence=confidence,
            source=f"{source}:{self.schema_version}",
        )
