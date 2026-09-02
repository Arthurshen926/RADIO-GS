"""Token-conditioned soft extent correction for the v4 surface carrier.

The module is a structured residual over a frozen ``K + null`` pointwise
posterior.  It uses only source-derived local facts and object-token contexts
pooled from immutable observed positives.  In particular, its forward API has
no target membership, integer instance label, held-out image, or user-query
input.

Unlike the earlier scalar-edge residual, every carrier edge receives a
continuous compatibility for every scene token.  A learned per-token soft
full-mass prior supplies the global coverage signal, while two differentiable
dual updates reconcile that signal with the local posterior.  No hard edge
threshold, radius, envelope, connected component, or token cap is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .message_passing import (
    EDGE_FEATURE_DIMENSION,
    F71_FEATURE_DIMENSION,
    build_query_free_edge_features,
)


STRUCTURED_EXTENT_ITERATION_COUNT = 2
STRUCTURED_EXTENT_MODES = (
    "full",
    "shared_edge_plus_mass",
    "token_conditioned_edge_plus_mass_bypass",
)
StructuredExtentMode = Literal[
    "full",
    "shared_edge_plus_mass",
    "token_conditioned_edge_plus_mass_bypass",
]


@dataclass(frozen=True)
class StructuredExtentOutput:
    """Auditable tensors produced by :class:`TokenConditionedStructuredExtent`."""

    probabilities: torch.Tensor
    log_probabilities: torch.Tensor
    token_context: torch.Tensor
    mass_context: torch.Tensor
    node_token_affinity: torch.Tensor
    edge_features: torch.Tensor
    base_edge_logits: torch.Tensor
    token_edge_logits: torch.Tensor | None
    predicted_full_mass: torch.Tensor
    predicted_log_full_mass: torch.Tensor
    predicted_dual_posterior_mass: torch.Tensor
    predicted_completed_membership_mass: torch.Tensor
    realized_full_mass: torch.Tensor
    realized_posterior_mass: torch.Tensor
    step_probabilities: tuple[torch.Tensor, torch.Tensor]
    step_realized_full_mass: tuple[torch.Tensor, torch.Tensor]
    step_realized_posterior_mass: tuple[torch.Tensor, torch.Tensor]
    step_dual_biases: tuple[torch.Tensor, torch.Tensor]
    dual_bias: torch.Tensor
    clamp_max_error: torch.Tensor
    transport_step_strengths: torch.Tensor
    dual_step_strengths: torch.Tensor
    edge_weight_mean: torch.Tensor
    edge_weight_minimum: torch.Tensor
    edge_weight_maximum: torch.Tensor


class TokenConditionedStructuredExtent(nn.Module):
    """Two-iteration token-conditioned soft surface extent residual.

    ``mode`` selects one of three capacity-matched causal ablations.  All modes
    instantiate the exact same parameters:

    ``full``
        Token-conditioned edge compatibility and learned soft mass dual.
    ``shared_edge_plus_mass``
        Broadcast the query-free scalar edge logit to every token while keeping
        the learned mass dual.
    ``token_conditioned_edge_plus_mass_bypass``
        Keep token-conditioned edges but hold the mass dual at exactly zero.
        The mass head is still reported and can receive its explicit auxiliary
        training loss, but cannot affect the completion posterior.
    """

    iteration_count = STRUCTURED_EXTENT_ITERATION_COUNT

    def __init__(
        self,
        feature_dimension: int = F71_FEATURE_DIMENSION,
        embedding_dimension: int = 32,
        edge_hidden_dimension: int = 64,
        dropout: float = 0.0,
        mode: StructuredExtentMode = "full",
        edge_chunk_size: int = 32768,
    ) -> None:
        super().__init__()
        if feature_dimension != F71_FEATURE_DIMENSION:
            raise ValueError("structured extent requires the sealed F71 layout")
        if embedding_dimension <= 0 or edge_hidden_dimension <= 0:
            raise ValueError("structured extent hidden dimensions must be positive")
        if not 0 <= float(dropout) < 1:
            raise ValueError("structured extent dropout must be in [0, 1)")
        if mode not in STRUCTURED_EXTENT_MODES:
            raise ValueError(f"unsupported structured extent mode {mode!r}")
        if int(edge_chunk_size) <= 0:
            raise ValueError("edge_chunk_size must be positive")

        self.feature_dimension = int(feature_dimension)
        self.embedding_dimension = int(embedding_dimension)
        self.edge_hidden_dimension = int(edge_hidden_dimension)
        self.dropout = float(dropout)
        self.mode = str(mode)
        self.edge_chunk_size = int(edge_chunk_size)

        self.node_encoder = nn.Sequential(
            nn.Linear(self.feature_dimension, self.embedding_dimension),
            nn.LayerNorm(self.embedding_dimension),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dimension, self.embedding_dimension),
        )
        # The pooled node embedding, log token scale (three axes), and observed
        # support mass are all source/observation facts.
        self.token_encoder = nn.Sequential(
            nn.Linear(self.embedding_dimension + 4, self.embedding_dimension),
            nn.LayerNorm(self.embedding_dimension),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dimension, self.embedding_dimension),
        )
        # Keep the learned mass prior causally separate from the edge branch.
        # In particular, the explicit mass loss in the mass-bypass control may
        # not update node/token encoders that still affect its posterior.
        self.mass_encoder = nn.Sequential(
            nn.Linear(self.feature_dimension + 4, self.embedding_dimension),
            nn.LayerNorm(self.embedding_dimension),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dimension, self.embedding_dimension),
        )
        self.node_affinity_projection = nn.Linear(
            self.embedding_dimension, self.embedding_dimension, bias=False
        )
        self.token_affinity_projection = nn.Linear(
            self.embedding_dimension, self.embedding_dimension, bias=False
        )
        # softplus(0.5413...) is one, so the initial affinity retains a unit
        # frozen-unary log-odds residual without making the unary trainable.
        self.unary_affinity_log_scale = nn.Parameter(torch.tensor(0.5413248546))

        # Three symmetric edge quantities: base logit, agreement coefficient,
        # and disagreement coefficient.  The last two pass through softplus.
        self.edge_network = nn.Sequential(
            nn.Linear(EDGE_FEATURE_DIMENSION, self.edge_hidden_dimension),
            nn.LayerNorm(self.edge_hidden_dimension),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.edge_hidden_dimension, self.edge_hidden_dimension),
            nn.GELU(),
            nn.Linear(self.edge_hidden_dimension, 3),
        )

        self.mass_head = nn.Sequential(
            nn.Linear(self.embedding_dimension, self.embedding_dimension),
            nn.GELU(),
            nn.Linear(self.embedding_dimension, 1),
        )
        # Begin with one predicted missing element per observed element, i.e.
        # predicted full mass = 2 * observed mass.  This is only an initializer;
        # there is no fixed cap or envelope.
        # A tiny non-zero weight keeps the independent mass encoder trainable
        # from the first optimizer step while preserving the 2x-observed prior
        # to numerical precision.
        nn.init.normal_(self.mass_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.mass_head[-1].bias, 0.5413248546)

        # A rational-square parameterization is bounded in [0, 1), begins near
        # identity, and remains trainable over a 480-step formal schedule.  A
        # saturated sigmoid(-6) would require an implausibly large logit change
        # before the residual became scientifically measurable.
        self.transport_step_parameters = nn.Parameter(
            torch.full((self.iteration_count,), 0.1)
        )
        self.dual_step_parameters = nn.Parameter(
            torch.full((self.iteration_count,), 0.1)
        )

    def architecture_receipt(self) -> dict[str, object]:
        return {
            "schema": (
                "radio_gs.surface_object_memory_v4."
                "token_conditioned_structured_extent.v2"
            ),
            "mode": self.mode,
            "iteration_count": self.iteration_count,
            "feature_dimension": self.feature_dimension,
            "embedding_dimension": self.embedding_dimension,
            "edge_feature_dimension": EDGE_FEATURE_DIMENSION,
            "edge_hidden_dimension": self.edge_hidden_dimension,
            "edge_chunk_size": self.edge_chunk_size,
            "dropout": self.dropout,
            "identity_residual_parameterization": "raw_square_over_one_plus_raw_square",
            "identity_residual_initial_parameter": 0.1,
            "identity_residual_initial_strength": 0.1**2 / (1 + 0.1**2),
            "token_context_authority": "immutable_observed_positive_source_facts",
            "edge_policy": (
                "symmetric_token_conditioned_continuous"
                if self.mode != "shared_edge_plus_mass"
                else "symmetric_query_free_shared_continuous_control"
            ),
            "mass_policy": (
                "learned_soft_full_mass_differentiable_dual"
                if self.mode
                != "token_conditioned_edge_plus_mass_bypass"
                else "learned_mass_reported_dual_causally_bypassed_control"
            ),
            "mass_feature_policy": (
                "independent_observed_positive_pooled_F71_scale_and_support_encoder"
            ),
            "completion_mass_policy": (
                "predict_physical_support_mass_and_match_pre_cap_posterior_mass"
            ),
            "observed_policy": "exact_K_plus_null_clamp_after_every_update",
            "integer_instance_identity_input": False,
            "target_membership_input": False,
            "heldout_rgb_input": False,
            "external_query_input": False,
            "hard_threshold": False,
            "hard_radius_or_envelope": False,
            "connected_components": False,
            "token_or_root_cap": False,
            "v3_dependency": False,
        }

    @staticmethod
    def _validate_probability_simplex(
        value: torch.Tensor, *, name: str, element_count: int | None = None
    ) -> None:
        if value.ndim != 2 or value.shape[1] < 2:
            raise ValueError(f"{name} must have shape [N, K+1] with K >= 1")
        if element_count is not None and value.shape[0] != element_count:
            raise ValueError(f"{name} does not align with carrier elements")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite floating point")
        if bool((value < 0).any()):
            raise ValueError(f"{name} must be non-negative")
        expected = torch.ones(value.shape[0], device=value.device, dtype=value.dtype)
        if not torch.allclose(value.sum(-1), expected, rtol=2e-6, atol=2e-6):
            raise ValueError(f"{name} must lie on the K-plus-null simplex")

    def _validate_and_encode_context(
        self,
        *,
        unary: torch.Tensor,
        centres: torch.Tensor,
        local_features: torch.Tensor,
        observed_positive: torch.Tensor,
        clamp_mask: torch.Tensor,
        clamp_probabilities: torch.Tensor,
        voxel_size: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        element_count = unary.shape[0]
        token_count = unary.shape[1] - 1
        if centres.shape != (element_count, 3):
            raise ValueError("centres must have shape [N, 3]")
        if local_features.shape != (element_count, self.feature_dimension):
            raise ValueError("local_features must use the sealed F71 shape [N, 71]")
        if not torch.isfinite(centres).all() or not torch.isfinite(local_features).all():
            raise ValueError("structured extent source facts must be finite")
        if observed_positive.dtype != torch.bool or observed_positive.shape != (
            element_count,
            token_count,
        ):
            raise ValueError("observed_positive must be explicit bool [N, K]")
        if clamp_mask.dtype != torch.bool or clamp_mask.shape != (element_count,):
            raise ValueError("clamp_mask must be explicit bool [N]")
        if clamp_probabilities.shape != unary.shape or not torch.isfinite(
            clamp_probabilities
        ).all():
            raise ValueError("clamp_probabilities must be finite [N, K+1]")
        selected_clamp = clamp_probabilities[clamp_mask]
        if selected_clamp.numel() and (
            bool(((selected_clamp != 0) & (selected_clamp != 1)).any())
            or not bool((selected_clamp.sum(-1) == 1).all())
        ):
            raise ValueError("every clamped row must be an exact K-plus-null one-hot")
        if bool((observed_positive.sum(-1) > 1).any()):
            raise ValueError("an element cannot observe multiple object tokens")
        observed_rows = observed_positive.any(-1)
        if bool((observed_rows & ~clamp_mask).any()):
            raise ValueError("every observed positive must be exactly clamped")
        if selected_clamp.numel() and not torch.equal(
            clamp_probabilities[clamp_mask, :token_count].to(torch.bool),
            observed_positive[clamp_mask],
        ):
            raise ValueError("observed positives and categorical clamps disagree")
        observed_mass = observed_positive.sum(0).to(dtype=unary.dtype)
        if bool((observed_mass <= 0).any()):
            raise ValueError("every scene token requires an observed positive seed")

        node_embedding = self.node_encoder(local_features)
        positive_weight = observed_positive.to(dtype=unary.dtype)
        pooled_node = positive_weight.T @ node_embedding / observed_mass[:, None]
        centre = positive_weight.T @ centres / observed_mass[:, None]
        second_moment = positive_weight.T @ centres.square() / observed_mass[:, None]
        variance = (second_moment - centre.square()).clamp_min(0)
        scale = variance.sqrt().clamp_min(float(voxel_size))
        token_input = torch.cat(
            (
                pooled_node,
                torch.log(scale / float(voxel_size)),
                torch.log1p(observed_mass)[:, None],
            ),
            dim=-1,
        )
        token_context = self.token_encoder(token_input)
        pooled_local = positive_weight.T @ local_features / observed_mass[:, None]
        mass_input = torch.cat(
            (
                pooled_local,
                torch.log(scale / float(voxel_size)),
                torch.log1p(observed_mass)[:, None],
            ),
            dim=-1,
        )
        mass_context = self.mass_encoder(mass_input)

        node_projected = F.normalize(
            self.node_affinity_projection(node_embedding), dim=-1, eps=1e-12
        )
        token_projected = F.normalize(
            self.token_affinity_projection(token_context), dim=-1, eps=1e-12
        )
        learned_affinity = node_projected @ token_projected.T
        # One-vs-rest logits retain categorical negative evidence.  Relative
        # token-vs-null logits incorrectly map an exact wrong-token clamp
        # (p_token=0, p_null=0) to zero instead of a strong negative value.
        epsilon = torch.finfo(unary.dtype).eps
        unary_token = unary[:, :token_count].clamp(epsilon, 1 - epsilon)
        unary_log_odds = torch.logit(unary_token)
        node_token_affinity = learned_affinity + F.softplus(
            self.unary_affinity_log_scale
        ).to(unary.dtype) * unary_log_odds
        if not torch.isfinite(node_token_affinity).all():
            raise RuntimeError("node-token affinity became non-finite")
        return (
            node_embedding,
            token_context,
            mass_context,
            node_token_affinity,
            observed_mass,
        )

    @staticmethod
    def _bounded_step_strength(parameters: torch.Tensor) -> torch.Tensor:
        squared = parameters.square()
        return squared / (1 + squared)

    def _edge_parameters(self, edge_features: torch.Tensor) -> torch.Tensor:
        if edge_features.ndim != 2 or edge_features.shape[1] != EDGE_FEATURE_DIMENSION:
            raise ValueError(
                f"edge_features must have shape [E, {EDGE_FEATURE_DIMENSION}]"
            )
        if not torch.isfinite(edge_features).all():
            raise ValueError("edge_features must be finite")
        parameters = self.edge_network(edge_features)
        if not torch.isfinite(parameters).all():
            raise RuntimeError("structured extent edge parameters became non-finite")
        return parameters

    def _token_logits_from_edge_parameters(
        self,
        parameters: torch.Tensor,
        edges: torch.Tensor,
        node_token_affinity: torch.Tensor,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        """Return one edge chunk without constructing an ``E x K x D`` tensor."""

        base = parameters[start:stop, 0]
        if self.mode == "shared_edge_plus_mass":
            return base[:, None].expand(stop - start, node_token_affinity.shape[1])
        source, destination = edges
        source_affinity = node_token_affinity[source[start:stop]]
        destination_affinity = node_token_affinity[destination[start:stop]]
        agreement = F.softplus(parameters[start:stop, 1])[:, None]
        disagreement = F.softplus(parameters[start:stop, 2])[:, None]
        return (
            base[:, None]
            + agreement * (source_affinity + destination_affinity) * 0.5
            - disagreement * (source_affinity - destination_affinity).abs()
        )

    def score_token_edges(
        self,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_token_affinity: torch.Tensor,
        *,
        edge_ids: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score all edge/token pairs or an aligned sampled subset.

        Passing neither ``edge_ids`` nor ``token_ids`` returns ``[E]`` base and
        ``[E, K]`` token logits.  Passing both aligned integer vectors returns
        two ``[B]`` vectors.  The scorer itself consumes no supervision; a
        trainer may select pairs using train-only labels outside this API.
        """

        edges_input = torch.as_tensor(edge_index)
        if edges_input.dtype == torch.bool or edges_input.is_floating_point():
            raise ValueError("edge_index must contain explicit integer indices")
        edges = edges_input.to(
            device=node_token_affinity.device, dtype=torch.long
        ).detach()
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if node_token_affinity.ndim != 2 or node_token_affinity.shape[1] < 1:
            raise ValueError("node_token_affinity must have shape [N, K]")
        if edges.numel() and (
            int(edges.min()) < 0 or int(edges.max()) >= node_token_affinity.shape[0]
        ):
            raise ValueError("edge endpoint is outside node-token affinity")
        edge_value = torch.as_tensor(
            edge_features,
            device=node_token_affinity.device,
            dtype=node_token_affinity.dtype,
        )
        if edge_value.shape != (edges.shape[1], EDGE_FEATURE_DIMENSION):
            raise ValueError("edge_features and edge_index do not align")
        if (edge_ids is None) != (token_ids is None):
            raise ValueError("edge_ids and token_ids must be both present or both absent")
        source, destination = edges
        if edge_ids is not None:
            selected_edges_input = torch.as_tensor(edge_ids)
            selected_tokens_input = torch.as_tensor(token_ids)
            if (
                selected_edges_input.dtype == torch.bool
                or selected_edges_input.is_floating_point()
                or selected_tokens_input.dtype == torch.bool
                or selected_tokens_input.is_floating_point()
            ):
                raise ValueError("sampled edge and token ids must be integers")
            selected_edges = selected_edges_input.to(
                device=edges.device, dtype=torch.long
            ).detach()
            selected_tokens = selected_tokens_input.to(
                device=edges.device, dtype=torch.long
            ).detach()
            if selected_edges.ndim != 1 or selected_tokens.shape != selected_edges.shape:
                raise ValueError("sampled edge_ids and token_ids must align as [B]")
            if selected_edges.numel() and (
                int(selected_edges.min()) < 0
                or int(selected_edges.max()) >= edges.shape[1]
                or int(selected_tokens.min()) < 0
                or int(selected_tokens.max()) >= node_token_affinity.shape[1]
            ):
                raise ValueError("sampled edge/token id is outside its domain")
            # Score only the sampled rows.  Re-running the edge MLP over all E
            # edges for a B-pair auxiliary loss doubled the dominant forward
            # and retained an unnecessary full autograd graph.
            selected_parameters = self._edge_parameters(edge_value[selected_edges])
            selected_base = selected_parameters[:, 0]
            if self.mode == "shared_edge_plus_mass":
                return selected_base, selected_base
            source_affinity = node_token_affinity[
                source[selected_edges], selected_tokens
            ]
            destination_affinity = node_token_affinity[
                destination[selected_edges], selected_tokens
            ]
            agreement = F.softplus(selected_parameters[:, 1])
            disagreement = F.softplus(selected_parameters[:, 2])
            token_logit = (
                selected_base
                + agreement * (source_affinity + destination_affinity) * 0.5
                - disagreement * (source_affinity - destination_affinity).abs()
            )
            return selected_base, token_logit

        parameters = self._edge_parameters(edge_value)
        base = parameters[:, 0]
        token_pieces: list[torch.Tensor] = []
        for start in range(0, edges.shape[1], self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, edges.shape[1])
            token_pieces.append(
                self._token_logits_from_edge_parameters(
                    parameters, edges, node_token_affinity, start, stop
                )
            )
        token = (
            torch.cat(token_pieces, dim=0)
            if token_pieces
            else node_token_affinity.new_empty((0, node_token_affinity.shape[1]))
        )
        if token.shape != (edges.shape[1], node_token_affinity.shape[1]):
            raise RuntimeError("token-conditioned edge score shape changed")
        if not torch.isfinite(token).all():
            raise RuntimeError("token-conditioned edge logits became non-finite")
        return base, token

    def _bounded_token_transport(
        self,
        values: torch.Tensor,
        edges: torch.Tensor,
        edge_parameters: torch.Tensor,
        node_token_affinity: torch.Tensor,
        safe_degree: torch.Tensor,
        incoming_weight: torch.Tensor,
        materialized_edge_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        incoming = torch.zeros_like(values)
        source, destination = edges
        for start in range(0, edges.shape[1], self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, edges.shape[1])
            weights = (
                materialized_edge_weights[start:stop]
                if materialized_edge_weights is not None
                else torch.sigmoid(
                    self._token_logits_from_edge_parameters(
                        edge_parameters,
                        edges,
                        node_token_affinity,
                        start,
                        stop,
                    )
                )
            )
            incoming.index_add_(
                0,
                destination[start:stop],
                values[source[start:stop]] * weights,
            )
        retained = (1 - incoming_weight / safe_degree[:, None]).clamp(0, 1)
        return retained * values + incoming / safe_degree[:, None]

    def forward(
        self,
        unary_probabilities: torch.Tensor,
        edge_index: torch.Tensor,
        centres: torch.Tensor,
        normals: torch.Tensor,
        local_features: torch.Tensor,
        source_visible: torch.Tensor,
        observed_positive: torch.Tensor,
        clamp_mask: torch.Tensor,
        clamp_probabilities: torch.Tensor,
        *,
        voxel_size: float,
        completion_confidence_cap: float,
        return_token_edge_logits: bool = True,
    ) -> StructuredExtentOutput:
        unary_input = torch.as_tensor(unary_probabilities)
        if not unary_input.is_floating_point():
            raise ValueError("unary_probabilities must be floating point")
        device = unary_input.device
        dtype = unary_input.dtype
        if dtype not in (torch.float32, torch.float64):
            dtype = torch.float32
        # The v10 pointwise unary is a frozen causal input to this entire stage.
        unary = unary_input.to(dtype=dtype).detach()
        self._validate_probability_simplex(unary, name="unary_probabilities")
        if not math.isfinite(float(voxel_size)) or float(voxel_size) <= 0:
            raise ValueError("voxel_size must be finite and positive")
        if (
            not math.isfinite(float(completion_confidence_cap))
            or not 0 < float(completion_confidence_cap) <= 1
        ):
            raise ValueError("completion_confidence_cap must be in (0, 1]")
        if not isinstance(return_token_edge_logits, bool):
            raise ValueError("return_token_edge_logits must be boolean")

        element_count = unary.shape[0]
        token_count = unary.shape[1] - 1
        centres_value = torch.as_tensor(
            centres, device=device, dtype=dtype
        ).detach()
        normals_value = torch.as_tensor(
            normals, device=device, dtype=dtype
        ).detach()
        local_value = torch.as_tensor(
            local_features, device=device, dtype=dtype
        ).detach()
        visible_input = torch.as_tensor(source_visible)
        if visible_input.dtype != torch.bool:
            raise ValueError("source_visible must be an explicit boolean fact")
        visible = visible_input.to(device=device).detach()
        positive_input = torch.as_tensor(observed_positive)
        if positive_input.dtype != torch.bool:
            raise ValueError("observed_positive must be explicit boolean")
        positive = positive_input.to(device=device).detach()
        mask_input = torch.as_tensor(clamp_mask)
        if mask_input.dtype != torch.bool:
            raise ValueError("clamp_mask must be explicit boolean")
        mask = mask_input.to(device=device).detach()
        clamp = torch.as_tensor(
            clamp_probabilities, device=device, dtype=dtype
        ).detach()

        _, token_context, mass_context, node_token_affinity, observed_mass = (
            self._validate_and_encode_context(
                unary=unary,
                centres=centres_value,
                local_features=local_value,
                observed_positive=positive,
                clamp_mask=mask,
                clamp_probabilities=clamp,
                voxel_size=float(voxel_size),
            )
        )

        edges_input = torch.as_tensor(edge_index)
        if edges_input.dtype == torch.bool or edges_input.is_floating_point():
            raise ValueError("edge_index must contain explicit integer indices")
        edges = edges_input.to(device=device, dtype=torch.long).detach()
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edges.numel() and (
            int(edges.min()) < 0 or int(edges.max()) >= element_count
        ):
            raise ValueError("edge endpoint is outside the carrier domain")
        edge_features = build_query_free_edge_features(
            edges,
            centres_value,
            normals_value,
            local_value,
            visible,
            voxel_size=float(voxel_size),
        ).to(dtype=dtype)
        edge_parameters = self._edge_parameters(edge_features)
        base_edge_logits = edge_parameters[:, 0]
        token_edge_logits: torch.Tensor | None = None
        materialized_edge_weights: torch.Tensor | None = None
        if return_token_edge_logits:
            token_pieces = [
                self._token_logits_from_edge_parameters(
                    edge_parameters,
                    edges,
                    node_token_affinity,
                    start,
                    min(start + self.edge_chunk_size, edges.shape[1]),
                )
                for start in range(0, edges.shape[1], self.edge_chunk_size)
            ]
            token_edge_logits = (
                torch.cat(token_pieces, dim=0)
                if token_pieces
                else unary.new_empty((0, token_count))
            )
            materialized_edge_weights = torch.sigmoid(token_edge_logits)

        mass_ratio = F.softplus(self.mass_head(mass_context).squeeze(-1))
        predicted_full_mass = observed_mass * (1 + mass_ratio)
        if not torch.isfinite(predicted_full_mass).all() or bool(
            (predicted_full_mass < observed_mass).any()
        ):
            raise RuntimeError("predicted full token mass is invalid")
        predicted_log_full_mass = predicted_full_mass.log()
        # ``predicted_full_mass`` estimates physical object support.  The
        # K-plus-null posterior represents that support before the downstream
        # confidence cap, so it is already the dual target.  Dividing the
        # missing mass by the cap would force false-positive support merely to
        # compensate for a deliberate confidence calibration.
        confidence_cap = float(completion_confidence_cap)
        predicted_dual_posterior_mass = predicted_full_mass
        predicted_completed_membership_mass = observed_mass + confidence_cap * (
            predicted_full_mass - observed_mass
        )

        source, destination = edges
        degree = unary.new_zeros(element_count)
        incoming_weight = unary.new_zeros(element_count, token_count)
        edge_weight_sum = unary.new_zeros(())
        edge_weight_minimum = unary.new_full((), float("inf"))
        edge_weight_maximum = unary.new_full((), float("-inf"))
        for start in range(0, edges.shape[1], self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, edges.shape[1])
            if stop > start:
                weights = (
                    materialized_edge_weights[start:stop]
                    if materialized_edge_weights is not None
                    else torch.sigmoid(
                        self._token_logits_from_edge_parameters(
                            edge_parameters,
                            edges,
                            node_token_affinity,
                            start,
                            stop,
                        )
                    )
                )
                if not torch.isfinite(weights).all() or bool(
                    ((weights < 0) | (weights > 1)).any()
                ):
                    raise RuntimeError("structured extent edge weights left [0, 1]")
                degree.index_add_(
                    0,
                    destination[start:stop],
                    torch.ones(stop - start, device=device, dtype=dtype),
                )
                incoming_weight.index_add_(
                    0, destination[start:stop], weights
                )
                edge_weight_sum = edge_weight_sum + weights.detach().sum()
                edge_weight_minimum = torch.minimum(
                    edge_weight_minimum, weights.detach().min()
                )
                edge_weight_maximum = torch.maximum(
                    edge_weight_maximum, weights.detach().max()
                )
        safe_degree = degree.clamp_min(1)

        current = torch.where(mask[:, None], clamp, unary)
        dual_bias = unary.new_zeros(token_count)
        transport_strengths = self._bounded_step_strength(
            self.transport_step_parameters
        ).to(dtype)
        dual_strengths = self._bounded_step_strength(
            self.dual_step_parameters
        ).to(dtype)
        step_probabilities: list[torch.Tensor] = []
        step_posterior_masses: list[torch.Tensor] = []
        step_completed_masses: list[torch.Tensor] = []
        step_duals: list[torch.Tensor] = []
        epsilon = torch.finfo(dtype).tiny
        unary_token = unary[:, :token_count]
        unary_null_logit = unary[:, token_count:].clamp_min(epsilon).log()

        for transport_strength, dual_strength in zip(
            transport_strengths, dual_strengths
        ):
            transported = self._bounded_token_transport(
                current[:, :token_count],
                edges,
                edge_parameters,
                node_token_affinity,
                safe_degree,
                incoming_weight,
                materialized_edge_weights,
            )
            candidate_token = (
                (1 - transport_strength) * unary_token
                + transport_strength * transported
            ).clamp_min(epsilon)
            token_logit = candidate_token.log()

            preliminary = torch.softmax(
                torch.cat(
                    (token_logit + dual_bias[None, :], unary_null_logit), dim=-1
                ),
                dim=-1,
            )
            preliminary = torch.where(mask[:, None], clamp, preliminary)
            preliminary_mass = preliminary[:, :token_count].sum(0)
            if self.mode != "token_conditioned_edge_plus_mass_bypass":
                mass_error = predicted_dual_posterior_mass.log() - (
                    preliminary_mass.clamp_min(epsilon).log()
                )
                dual_bias = dual_bias + dual_strength * mass_error
            else:
                # Keep an exact, inspectable causal bypass rather than merely a
                # very small learned coefficient.
                dual_bias = torch.zeros_like(dual_bias)

            current = torch.softmax(
                torch.cat(
                    (token_logit + dual_bias[None, :], unary_null_logit), dim=-1
                ),
                dim=-1,
            )
            current = torch.where(mask[:, None], clamp, current)
            self._validate_probability_simplex(
                current,
                name="structured extent step posterior",
                element_count=element_count,
            )
            if not torch.equal(current[mask], clamp[mask]):
                raise RuntimeError("structured extent changed an exact clamp")
            step_probabilities.append(current)
            posterior_mass = current[:, :token_count].sum(0)
            completed_mass = observed_mass + confidence_cap * (
                posterior_mass - observed_mass
            )
            step_posterior_masses.append(posterior_mass)
            step_completed_masses.append(completed_mass)
            step_duals.append(dual_bias)

        if len(step_probabilities) != self.iteration_count:
            raise RuntimeError("structured extent iteration count changed")
        realized_posterior_mass = step_posterior_masses[-1]
        realized_full_mass = step_completed_masses[-1]
        clamp_max_error = (
            (current[mask] - clamp[mask]).abs().max()
            if bool(mask.any())
            else current.new_zeros(())
        )
        stable_floor = torch.finfo(dtype).tiny
        edge_weight_count = edges.shape[1] * token_count
        edge_weight_mean = (
            edge_weight_sum / edge_weight_count
            if edge_weight_count
            else unary.new_zeros(())
        )
        if not edge_weight_count:
            edge_weight_minimum = unary.new_zeros(())
            edge_weight_maximum = unary.new_zeros(())
        return StructuredExtentOutput(
            probabilities=current,
            log_probabilities=current.clamp_min(stable_floor).log(),
            token_context=token_context,
            mass_context=mass_context,
            node_token_affinity=node_token_affinity,
            edge_features=edge_features,
            base_edge_logits=base_edge_logits,
            token_edge_logits=token_edge_logits,
            predicted_full_mass=predicted_full_mass,
            predicted_log_full_mass=predicted_log_full_mass,
            predicted_dual_posterior_mass=predicted_dual_posterior_mass,
            predicted_completed_membership_mass=(
                predicted_completed_membership_mass
            ),
            realized_full_mass=realized_full_mass,
            realized_posterior_mass=realized_posterior_mass,
            step_probabilities=(step_probabilities[0], step_probabilities[1]),
            step_realized_full_mass=(
                step_completed_masses[0],
                step_completed_masses[1],
            ),
            step_realized_posterior_mass=(
                step_posterior_masses[0],
                step_posterior_masses[1],
            ),
            step_dual_biases=(step_duals[0], step_duals[1]),
            dual_bias=dual_bias,
            clamp_max_error=clamp_max_error,
            transport_step_strengths=transport_strengths,
            dual_step_strengths=dual_strengths,
            edge_weight_mean=edge_weight_mean,
            edge_weight_minimum=edge_weight_minimum,
            edge_weight_maximum=edge_weight_maximum,
        )


__all__ = [
    "STRUCTURED_EXTENT_ITERATION_COUNT",
    "STRUCTURED_EXTENT_MODES",
    "StructuredExtentMode",
    "StructuredExtentOutput",
    "TokenConditionedStructuredExtent",
]
