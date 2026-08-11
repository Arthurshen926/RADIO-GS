"""Seed-defined absorbing extent with an explicit final-probability contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .query_likelihood_head import QueryLikelihoodInputs
from .query_spec import PrimitiveUnaryEvidence
from .support_solver import PrimitiveSupportGraph


@dataclass(frozen=True)
class StructuredFinalProbability:
    """A final structured prediction that must bypass the old graph solver.

    ``extent_confidence=0`` is an exact abstention.  Selection/readout uses
    zero probability there rather than interpreting the neutral likelihood
    ``q=0.5`` as foreground under a greater-or-equal threshold.  The typed
    graph has already been consumed to establish extent, so passing this
    object through the legacy diffusion solver would be an explicit contract
    violation and double-count the same graph.
    """

    foreground_probability: torch.Tensor
    extent_confidence: torch.Tensor
    hard_positive: torch.Tensor
    hard_negative: torch.Tensor
    source: str
    solver_policy: str = "bypass_existing_graph_solver_with_anchor_and_extent_contract"

    def __post_init__(self) -> None:
        probability = torch.as_tensor(self.foreground_probability).float().reshape(-1)
        confidence = torch.as_tensor(self.extent_confidence).float().reshape(-1)
        positive = torch.as_tensor(self.hard_positive).bool().reshape(-1)
        negative = torch.as_tensor(self.hard_negative).bool().reshape(-1)
        if (
            not probability.numel()
            or probability.shape != confidence.shape
            or probability.shape != positive.shape
            or probability.shape != negative.shape
            or not bool(torch.isfinite(probability).all())
            or not bool(torch.isfinite(confidence).all())
            or bool(((probability < 0) | (probability > 1)).any())
            or bool(((confidence < 0) | (confidence > 1)).any())
            or bool((positive & negative).any())
        ):
            raise ValueError("structured final probability tensors are invalid")
        object.__setattr__(self, "foreground_probability", probability)
        object.__setattr__(self, "extent_confidence", confidence)
        object.__setattr__(self, "hard_positive", positive)
        object.__setattr__(self, "hard_negative", negative)

    @property
    def selection_probability(self) -> torch.Tensor:
        output = torch.where(
            self.extent_confidence > 0,
            self.foreground_probability,
            torch.zeros_like(self.foreground_probability),
        )
        output = torch.where(self.hard_negative, torch.zeros_like(output), output)
        return torch.where(self.hard_positive, torch.ones_like(output), output)

    def as_diagnostic_unary(self) -> PrimitiveUnaryEvidence:
        """Expose ``q,c`` for logging, never as authorization to diffuse again."""

        return PrimitiveUnaryEvidence.from_probability(
            self.foreground_probability,
            confidence=self.extent_confidence,
            source=f"{self.source}:structured-final-diagnostic-only",
        )


def finite_absorbing_seed_reach(
    graph: PrimitiveSupportGraph,
    hard_seeds: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """Finite absorbing reach using a non-negative registered transition.

    Hard seeds are clamped at every iteration.  Exact zero is preserved for
    nodes outside the finite graph neighbourhood, providing an auditable
    abstention state without a learned threshold.
    """

    seeds = torch.as_tensor(hard_seeds).float()
    squeeze = seeds.ndim == 1
    if squeeze:
        seeds = seeds[:, None]
    if (
        seeds.ndim != 2
        or seeds.shape[0] != graph.num_nodes
        or seeds.shape[1] <= 0
        or not bool(torch.isfinite(seeds).all())
        or bool(((seeds < 0) | (seeds > 1)).any())
    ):
        raise ValueError("hard seeds must be finite [N] or [N,K] values in [0,1]")
    steps = int(steps)
    if steps < 0:
        raise ValueError("absorbing steps must be non-negative")
    working = graph if graph.edge_index.device == seeds.device else graph.to(seeds.device)
    row, col = working.edge_index
    state = seeds
    reach = seeds
    for _ in range(steps):
        propagated = torch.zeros_like(state)
        if row.numel():
            propagated.index_add_(0, row, working.edge_weight[:, None] * state[col])
        state = torch.maximum(seeds, propagated.clamp(0.0, 1.0))
        reach = torch.maximum(reach, state)
    return reach[:, 0] if squeeze else reach


class AbsorbingExtentInteractionHead(nn.Module):
    """Use graph reach for extent, then local likelihood only inside extent."""

    schema_version = "absorbing-seed-extent-structured-final-v8"

    def __init__(
        self,
        base_head: nn.Module,
        *,
        absorbing_steps: int = 12,
        hard_seed_threshold: float = 0.20,
    ) -> None:
        super().__init__()
        if not hasattr(base_head, "log_likelihood_ratio"):
            raise TypeError("base_head must expose log_likelihood_ratio")
        if int(absorbing_steps) < 0:
            raise ValueError("absorbing_steps must be non-negative")
        if not 0.0 <= float(hard_seed_threshold) <= 1.0:
            raise ValueError("hard_seed_threshold must be in [0,1]")
        self.base_head = base_head
        for parameter in self.base_head.parameters():
            parameter.requires_grad_(False)
        self.absorbing_steps = int(absorbing_steps)
        self.hard_seed_threshold = float(hard_seed_threshold)

    def forward(
        self,
        observations: QueryLikelihoodInputs,
        *,
        graph: PrimitiveSupportGraph,
        positive_seeds: torch.Tensor,
        negative_seeds: torch.Tensor,
        source: str,
        apply_extent: bool = False,
    ) -> PrimitiveUnaryEvidence | StructuredFinalProbability:
        if not apply_extent:
            return self.base_head(observations, source=source)
        inputs = observations.validated()
        base_probability = torch.sigmoid(self.base_head.log_likelihood_ratio(inputs))
        positive = torch.as_tensor(
            positive_seeds, device=base_probability.device
        ).float().reshape(-1)
        negative = torch.as_tensor(
            negative_seeds, device=base_probability.device
        ).float().reshape(-1)
        if positive.shape != base_probability.shape or negative.shape != base_probability.shape:
            raise ValueError("extent seeds must align with primitive rows")
        hard_positive = positive >= self.hard_seed_threshold
        hard_negative = (negative >= self.hard_seed_threshold) & ~hard_positive
        working = graph if graph.edge_index.device == base_probability.device else graph.to(base_probability.device)
        positive_reach = finite_absorbing_seed_reach(
            working,
            hard_positive.float(),
            steps=self.absorbing_steps,
        )
        negative_reach = finite_absorbing_seed_reach(
            working,
            hard_negative.float(),
            steps=self.absorbing_steps,
        )
        extent_confidence = (positive_reach > 0).to(base_probability.dtype)
        probability = base_probability * (1.0 - negative_reach.clamp(0.0, 1.0))
        probability = torch.where(
            extent_confidence > 0,
            probability,
            torch.full_like(probability, 0.5),
        )
        probability = torch.where(hard_negative, torch.zeros_like(probability), probability)
        probability = torch.where(hard_positive, torch.ones_like(probability), probability)
        extent_confidence = torch.where(
            hard_positive | hard_negative,
            torch.ones_like(extent_confidence),
            extent_confidence,
        )
        return StructuredFinalProbability(
            foreground_probability=probability,
            extent_confidence=extent_confidence,
            hard_positive=hard_positive,
            hard_negative=hard_negative,
            source=f"{source}:{self.schema_version}",
        )
