"""Query-independent orchestration from typed evidence to 3-D support."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch

from radio_gs.field.field_signature import FeatureSpaceSignature
from .evidence_scorer import (
    EvidenceScoringConfig,
    score_query_evidence,
    shrink_unary_by_reliability,
)
from .score_calibration import (
    SceneSpaceCalibration,
    fit_scene_space_calibration,
)
from .query_spec import QueryModality, QuerySpec
from .support_solver import (
    PrimitiveSupportGraph,
    SupportSolverConfig,
    graph_local_seed_influence,
    graph_for_query_intent,
    select_support_components,
    solve_primitive_support,
)


@dataclass(frozen=True)
class QueryResult:
    unary: torch.Tensor
    evidence_components: Mapping[str, torch.Tensor]
    probabilities: torch.Tensor
    selected_support: torch.Tensor
    score_calibration: str = "none"
    reliability_applied: bool = False
    graph_policy: str = "typed_if_available"
    channel_confidence_mode: str = "none"
    negative_spatial_mode: str = "none"

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
        graph_policy: str = "typed_if_available",
        component_graph_policy: str = "same",
        graph_legacy_residual: float = 0.0,
        channel_confidence_mode: str = "none",
        node_reliability: torch.Tensor | None = None,
        score_calibration_by_modality: Mapping[
            QueryModality | str, str
        ] | None = None,
    ) -> None:
        self.graph = graph
        self.scoring_config = scoring_config
        self.solver_config = solver_config
        self.graph_policy = str(graph_policy)
        self.component_graph_policy = str(component_graph_policy)
        self.graph_legacy_residual = float(graph_legacy_residual)
        if not 0.0 <= self.graph_legacy_residual <= 1.0:
            raise ValueError("graph_legacy_residual must be in [0,1]")
        self.channel_confidence_mode = str(channel_confidence_mode)
        if self.channel_confidence_mode not in {
            "none",
            "affinity_mass",
            "max_affinity",
        }:
            raise ValueError(
                "channel_confidence_mode must be none, affinity_mass, or max_affinity"
            )
        if (
            self.channel_confidence_mode != "none"
            and self.graph_legacy_residual > 0
        ):
            raise ValueError(
                "graph_legacy_residual is incompatible with confidence-gated self loops"
            )
        self.node_reliability: torch.Tensor | None = None
        if node_reliability is not None:
            reliability = torch.as_tensor(node_reliability).detach().float().reshape(-1)
            if reliability.shape != (self.graph.num_nodes,):
                raise ValueError("node_reliability must align with graph primitive rows")
            if not bool(torch.isfinite(reliability).all()):
                raise ValueError("node_reliability contains NaN or infinity")
            if bool((reliability < 0).any()) or bool((reliability > 1).any()):
                raise ValueError("node_reliability must be in [0,1]")
            self.node_reliability = reliability.to(self.graph.edge_index.device)
        self.score_calibration_by_modality: dict[QueryModality, str] = {}
        for modality, calibration in (score_calibration_by_modality or {}).items():
            typed_modality = QueryModality(modality)
            calibration = str(calibration)
            # Reuse the scoring-config contract so an invalid modality override
            # fails at construction rather than silently at query time.
            replace(scoring_config, score_calibration=calibration)
            self.score_calibration_by_modality[typed_modality] = calibration
        self._calibrations: dict[str, SceneSpaceCalibration] = {}
        self._calibration_bank_shapes: dict[str, tuple[int, int]] = {}
        self._query_graphs: dict[
            tuple[str, str, float, str], PrimitiveSupportGraph
        ] = {}

    def scoring_config_for_query(self, query: QuerySpec) -> EvidenceScoringConfig:
        """Resolve an explicit modality override without changing global defaults."""

        calibration = self.score_calibration_by_modality.get(query.modality)
        if calibration is None or calibration == self.scoring_config.score_calibration:
            return self.scoring_config
        return replace(self.scoring_config, score_calibration=calibration)

    def execute(
        self,
        query: QuerySpec,
        feature_banks: Mapping[str, torch.Tensor],
        *,
        feature_signatures: Mapping[str, FeatureSpaceSignature] | None = None,
    ) -> QueryResult:
        scoring_config = self.scoring_config_for_query(query)
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
        if (
            scoring_config.feature_calibration != "none"
            or scoring_config.background_centroids > 0
        ):
            for name in required:
                matrix = feature_banks[name]
                shape = tuple(map(int, matrix.shape))
                if name in self._calibration_bank_shapes:
                    if self._calibration_bank_shapes[name] != shape:
                        raise ValueError("feature bank changed after scene calibration")
                    continue
                self._calibrations[name] = fit_scene_space_calibration(
                    matrix,
                    method=scoring_config.feature_calibration,
                    sample_size=scoring_config.calibration_sample_size,
                    background_centroids=scoring_config.background_centroids,
                    centroid_iterations=scoring_config.centroid_iterations,
                )
                self._calibration_bank_shapes[name] = shape
        explicit_negative_influence = None
        positive_spatial_influence = None
        explicit_negative_spatial = None
        if scoring_config.negative_spatial_mode in {
            "truncated_graph_decay",
            "signed_geodesic",
        }:
            local_graph_key = (
                "geometry",
                str(query.intent.value),
                0.0,
                "none",
            )
            if local_graph_key not in self._query_graphs:
                self._query_graphs[local_graph_key] = graph_for_query_intent(
                    self.graph,
                    query.intent,
                    policy="geometry",
                    legacy_residual=0.0,
                    channel_confidence_mode="none",
                )
            local_graph = self._query_graphs[local_graph_key]
            if (
                scoring_config.negative_spatial_mode
                == "truncated_graph_decay"
                and query.negative_seeds is not None
            ):
                explicit_negative_influence = graph_local_seed_influence(
                    local_graph,
                    query.negative_seeds.weights.to(
                        self.graph.edge_index.device
                    ),
                    steps=scoring_config.negative_spatial_steps,
                    decay=scoring_config.negative_spatial_decay,
                )
            elif scoring_config.negative_spatial_mode == "signed_geodesic":
                if query.positive_seed_groups is None:
                    raise ValueError(
                        "signed_geodesic requires per-query positive seed groups"
                    )
                positive_spatial_influence = graph_local_seed_influence(
                    local_graph,
                    query.positive_seed_groups.weights.to(
                        self.graph.edge_index.device
                    ),
                    steps=scoring_config.negative_spatial_steps,
                    decay=scoring_config.negative_spatial_decay,
                )
                if query.negative_seed_groups is not None:
                    explicit_negative_spatial = graph_local_seed_influence(
                        local_graph,
                        query.negative_seed_groups.weights.to(
                            self.graph.edge_index.device
                        ),
                        steps=scoring_config.negative_spatial_steps,
                        decay=scoring_config.negative_spatial_decay,
                    )
        unary, components = score_query_evidence(
            query,
            feature_banks,
            config=scoring_config,
            calibrations=self._calibrations,
            num_nodes=self.graph.num_nodes,
            explicit_negative_influence=explicit_negative_influence,
            positive_spatial_influence=positive_spatial_influence,
            explicit_negative_spatial=explicit_negative_spatial,
        )
        reliability_applied = self.node_reliability is not None and bool(components)
        if reliability_applied:
            assert self.node_reliability is not None
            unary = shrink_unary_by_reliability(unary, self.node_reliability)
            components = {
                name: shrink_unary_by_reliability(values, self.node_reliability)
                for name, values in components.items()
            }
        if unary.numel() != self.graph.num_nodes:
            if unary.numel() == 0 and self.graph.num_nodes > 0:
                unary = torch.zeros(self.graph.num_nodes)
            else:
                raise ValueError("query unary does not align with support graph")
        graph_key = (
            self.graph_policy,
            str(query.intent.value),
            self.graph_legacy_residual,
            self.channel_confidence_mode,
        )
        if graph_key not in self._query_graphs:
            self._query_graphs[graph_key] = graph_for_query_intent(
                self.graph,
                query.intent,
                policy=self.graph_policy,
                legacy_residual=self.graph_legacy_residual,
                channel_confidence_mode=self.channel_confidence_mode,
            )
        query_graph = self._query_graphs[graph_key]
        if self.component_graph_policy == "same":
            component_graph = query_graph
        else:
            component_key = (
                self.component_graph_policy,
                str(query.intent.value),
                self.graph_legacy_residual,
                self.channel_confidence_mode,
            )
            if component_key not in self._query_graphs:
                self._query_graphs[component_key] = graph_for_query_intent(
                    self.graph,
                    query.intent,
                    policy=self.component_graph_policy,
                    legacy_residual=self.graph_legacy_residual,
                    channel_confidence_mode=self.channel_confidence_mode,
                )
            component_graph = self._query_graphs[component_key]
        probabilities = solve_primitive_support(
            query_graph,
            unary,
            positive_seeds=query.positive_seeds,
            negative_seeds=query.negative_seeds,
            config=self.solver_config,
        )
        selected = select_support_components(
            component_graph,
            probabilities,
            query.selection_mode,
            positive_seeds=query.positive_seeds,
            negative_seeds=query.negative_seeds,
            positive_seed_groups=query.positive_seed_groups,
            negative_seed_groups=query.negative_seed_groups,
            config=self.solver_config,
        )
        return QueryResult(
            unary=unary,
            evidence_components=components,
            probabilities=probabilities,
            selected_support=selected,
            score_calibration=scoring_config.score_calibration,
            reliability_applied=reliability_applied,
            graph_policy=self.graph_policy,
            channel_confidence_mode=self.channel_confidence_mode,
            negative_spatial_mode=scoring_config.negative_spatial_mode,
        )
