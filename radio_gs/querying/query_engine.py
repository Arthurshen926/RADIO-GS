"""Query-independent orchestration from typed evidence to 3-D support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from radio_gs.field.field_signature import FeatureSpaceSignature
from .evidence_scorer import EvidenceScoringConfig, score_query_evidence
from .query_spec import QuerySpec
from .support_solver import (
    PrimitiveSupportGraph,
    SupportSolverConfig,
    select_support_components,
    solve_primitive_support,
)


@dataclass(frozen=True)
class QueryResult:
    unary: torch.Tensor
    evidence_components: Mapping[str, torch.Tensor]
    probabilities: torch.Tensor
    selected_support: torch.Tensor

    @property
    def selected_probabilities(self) -> torch.Tensor:
        return self.probabilities * self.selected_support.to(self.probabilities.dtype)


class CanonicalQueryEngine:
    """The sole query-time path shared by text, image, 2-D, and 3-D input."""

    def __init__(
        self,
        graph: PrimitiveSupportGraph,
        *,
        scoring_config: EvidenceScoringConfig = EvidenceScoringConfig(),
        solver_config: SupportSolverConfig = SupportSolverConfig(),
    ) -> None:
        self.graph = graph
        self.scoring_config = scoring_config
        self.solver_config = solver_config

    def execute(
        self,
        query: QuerySpec,
        feature_banks: Mapping[str, torch.Tensor],
        *,
        feature_signatures: Mapping[str, FeatureSpaceSignature] | None = None,
    ) -> QueryResult:
        counts = {name: values.shape[0] for name, values in feature_banks.items()}
        if counts and any(count != self.graph.num_nodes for count in counts.values()):
            raise ValueError("all feature banks must align with graph primitive rows")
        required = {
            name: evidence
            for name, evidence in (
                ("semantic", query.semantic_evidence),
                ("appearance", query.appearance_evidence),
                ("boundary", query.boundary_evidence),
            )
            if evidence is not None
        }
        if required and feature_signatures is None:
            raise ValueError("prototype queries require fail-closed feature_signatures")
        for name, evidence in required.items():
            assert feature_signatures is not None
            if name not in feature_signatures:
                raise KeyError(f"missing signature for {name} feature bank")
            evidence.signature.assert_comparable(feature_signatures[name])
        unary, components = score_query_evidence(
            query, feature_banks, config=self.scoring_config
        )
        if unary.numel() != self.graph.num_nodes:
            if unary.numel() == 0 and self.graph.num_nodes > 0:
                unary = torch.zeros(self.graph.num_nodes)
            else:
                raise ValueError("query unary does not align with support graph")
        probabilities = solve_primitive_support(
            self.graph,
            unary,
            positive_seeds=query.positive_seeds,
            negative_seeds=query.negative_seeds,
            config=self.solver_config,
        )
        selected = select_support_components(
            self.graph,
            probabilities,
            query.selection_mode,
            positive_seeds=query.positive_seeds,
            config=self.solver_config,
        )
        return QueryResult(
            unary=unary,
            evidence_components=components,
            probabilities=probabilities,
            selected_support=selected,
        )
