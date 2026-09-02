"""Bounded learned propagation over the v4 sparse surface carrier.

This module is deliberately downstream of a frozen pointwise completion unary.
It learns whether *surface-local* edges should transport a token distribution;
it never receives a token identity, target membership, query, or benchmark
threshold.  Observed and ineligible elements are supplied as exact categorical
clamps and therefore cannot be changed by propagation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


F71_FEATURE_DIMENSION = 71
F71_RGB_SLICE = slice(0, 3)
F71_AVAILABILITY_INDEX = 3
F71_RADIO_SLICE = slice(4, 68)
F71_NORMAL_SLICE = slice(68, 71)

# Every entry is query-free and symmetric under reversal of a directed carrier
# edge.  Keeping this layout named makes the learned edge receipt auditable.
EDGE_FEATURE_LAYOUT = (
    "absolute_relative_x_over_voxel",
    "absolute_relative_y_over_voxel",
    "absolute_relative_z_over_voxel",
    "relative_distance_over_voxel",
    "normal_cosine",
    "absolute_normal_delta_x",
    "absolute_normal_delta_y",
    "absolute_normal_delta_z",
    "minimum_absolute_normal_tangency",
    "maximum_absolute_normal_tangency",
    "source_available_count_fraction",
    "both_source_available",
    "exactly_one_source_available",
    "neither_source_available",
    "absolute_rgb_delta_r",
    "absolute_rgb_delta_g",
    "absolute_rgb_delta_b",
    "rgb_cosine_when_both_available",
    "radio_cosine_when_both_available",
)
EDGE_FEATURE_DIMENSION = len(EDGE_FEATURE_LAYOUT)
EXTENT_GATE_INITIAL_LOGIT = -3.0


@dataclass(frozen=True)
class SurfaceMessagePassingOutput:
    """Differentiable outputs consumed by categorical, edge, and render losses."""

    probabilities: torch.Tensor
    log_probabilities: torch.Tensor
    edge_logits: torch.Tensor
    edge_weights: torch.Tensor
    edge_features: torch.Tensor
    step_probabilities: tuple[torch.Tensor, ...]
    step_strengths: torch.Tensor
    seed_reachability: torch.Tensor
    extent_logits: torch.Tensor
    extent_weights: torch.Tensor
    step_seed_reachabilities: tuple[torch.Tensor, ...]
    step_extent_weights: tuple[torch.Tensor, ...]
    extent_gate_strengths: torch.Tensor


def _finite_tensor(
    value: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device, dtype=dtype).detach()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    numerator = (first * second).sum(-1)
    denominator = torch.linalg.vector_norm(first, dim=-1) * torch.linalg.vector_norm(
        second, dim=-1
    )
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(first.dtype).tiny),
        torch.zeros_like(numerator),
    ).clamp(-1, 1)


def _validate_f71_facts(
    local_features: torch.Tensor,
    normals: torch.Tensor,
    source_visible: torch.Tensor,
) -> None:
    if local_features.ndim != 2 or local_features.shape[1] != F71_FEATURE_DIMENSION:
        raise ValueError("local_features must use the sealed F71 layout [N, 71]")
    if normals.shape != (local_features.shape[0], 3):
        raise ValueError("normals must have shape [N, 3]")
    if source_visible.shape != (local_features.shape[0],):
        raise ValueError("source_visible must have shape [N]")
    availability = local_features[:, F71_AVAILABILITY_INDEX]
    if not torch.equal(availability, source_visible.to(availability.dtype)):
        raise ValueError("F71 availability must exactly agree with source_visible")
    if not torch.equal(local_features[:, F71_NORMAL_SLICE], normals):
        raise ValueError("F71 normal channels must exactly agree with carrier normals")
    unavailable = ~source_visible
    if bool((local_features[unavailable, F71_RGB_SLICE] != 0).any()) or bool(
        (local_features[unavailable, F71_RADIO_SLICE] != 0).any()
    ):
        raise ValueError("unavailable F71 RGB and RADIO facts must be exactly zero")


def validate_surface_voxel_adjacency(
    edge_index: torch.Tensor,
    centres: torch.Tensor,
    *,
    voxel_size: float,
) -> None:
    """Validate the fixed bidirectional six-neighbour carrier adjacency.

    The v4 surface carrier emits both directions exactly once for every voxel
    face adjacency.  Rejecting duplicates, missing reverse edges, and diagonal
    voxel contacts here prevents learned propagation from silently changing the
    carrier topology or double-counting a hand-built edge list.
    """

    if not math.isfinite(float(voxel_size)) or float(voxel_size) <= 0:
        raise ValueError("voxel_size must be finite and positive")
    # ``SurfaceVoxelCarrier.neighbors`` defines voxel keys with CPU float32
    # ``floor(centres / voxel_size)``.  Repeating that operation on CUDA can
    # round an exact voxel-boundary quotient differently and reject an edge
    # produced by the authority itself.  Validation therefore deliberately
    # replays the carrier's exact device/dtype convention.
    centres_input = torch.as_tensor(centres)
    edges_input = torch.as_tensor(edge_index)
    centres_value = centres_input.detach().to(device="cpu", dtype=torch.float32)
    edges = edges_input.detach().to(device="cpu")
    if centres_value.ndim != 2 or centres_value.shape[1] != 3:
        raise ValueError("centres must have shape [N, 3]")
    if not torch.isfinite(centres_value).all():
        raise ValueError("centres must be finite")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, A]")
    if edges_input.dtype == torch.bool or edges_input.is_floating_point():
        raise ValueError("edge_index must contain explicit integer indices")
    edges = edges.long()
    element_count = centres_value.shape[0]
    if edges.numel() and (
        int(edges.min()) < 0 or int(edges.max()) >= element_count
    ):
        raise ValueError("edge_index endpoint is outside the carrier domain")
    source, destination = edges
    if bool((source == destination).any()):
        raise ValueError("surface adjacency must not contain self edges")
    if not edges.shape[1]:
        return

    directed_codes = source * element_count + destination
    sorted_codes = directed_codes.sort().values
    if bool((sorted_codes[1:] == sorted_codes[:-1]).any()):
        raise ValueError("surface adjacency must not contain duplicate directed edges")
    reverse_codes = destination * element_count + source
    if not torch.equal(sorted_codes, reverse_codes.sort().values):
        raise ValueError("every surface adjacency edge must have exactly one reverse edge")

    voxel_keys = torch.floor(centres_value / float(voxel_size)).to(torch.int64)
    key_delta = (voxel_keys[source] - voxel_keys[destination]).abs()
    is_face_neighbour = (key_delta.sum(-1) == 1) & (key_delta.max(-1).values == 1)
    if not bool(is_face_neighbour.all()):
        raise ValueError("surface adjacency must contain only six-neighbour voxel faces")


def build_query_free_edge_features(
    edge_index: torch.Tensor,
    centres: torch.Tensor,
    normals: torch.Tensor,
    local_features: torch.Tensor,
    source_visible: torch.Tensor,
    *,
    voxel_size: float,
) -> torch.Tensor:
    """Build source-only compatibility facts for directed six-neighbour edges.

    The result has shape ``[A, 19]`` for ``A`` directed edges.  Reverse carrier
    edges receive identical features by construction.  The function accepts no
    token/query/label input, which prevents edge compatibility from becoming an
    object-identity side channel.
    """

    if not math.isfinite(float(voxel_size)) or float(voxel_size) <= 0:
        raise ValueError("voxel_size must be finite and positive")
    centres_input = torch.as_tensor(centres)
    if not centres_input.is_floating_point():
        raise ValueError("centres must be floating point")
    device = centres_input.device
    dtype = centres_input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    centres_value = _finite_tensor(
        centres_input, name="centres", device=device, dtype=dtype
    )
    normals_value = _finite_tensor(
        normals, name="normals", device=device, dtype=dtype
    )
    features_value = _finite_tensor(
        local_features, name="local_features", device=device, dtype=dtype
    )
    visible_input = torch.as_tensor(source_visible)
    if visible_input.dtype != torch.bool:
        raise ValueError("source_visible must be an explicit boolean fact")
    visible = visible_input.to(device=device).detach()
    if centres_value.ndim != 2 or centres_value.shape[1] != 3:
        raise ValueError("centres must have shape [N, 3]")
    _validate_f71_facts(features_value, normals_value, visible)
    if features_value.shape[0] != centres_value.shape[0]:
        raise ValueError("F71 facts and centres must describe the same carrier elements")

    edges_input = torch.as_tensor(edge_index)
    if edges_input.dtype == torch.bool or edges_input.is_floating_point():
        raise ValueError("edge_index must contain explicit integer indices")
    edges = edges_input.to(device=device, dtype=torch.long).detach()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, A]")
    if edges.numel() and (
        int(edges.min()) < 0 or int(edges.max()) >= centres_value.shape[0]
    ):
        raise ValueError("edge_index endpoint is outside the carrier domain")
    source, destination = edges
    if bool((source == destination).any()):
        raise ValueError("surface adjacency must not contain self edges")
    if edges.shape[1] == 0:
        return centres_value.new_empty((0, EDGE_FEATURE_DIMENSION))

    validate_surface_voxel_adjacency(
        edges, centres_value, voxel_size=float(voxel_size)
    )

    displacement = centres_value[destination] - centres_value[source]
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    unit_direction = displacement / distance.clamp_min(torch.finfo(dtype).tiny)[:, None]
    relative = displacement.abs() / float(voxel_size)
    normalized_distance = distance[:, None] / float(voxel_size)

    source_normal = F.normalize(normals_value[source], dim=-1, eps=1e-12)
    destination_normal = F.normalize(normals_value[destination], dim=-1, eps=1e-12)
    normal_cosine = _cosine(source_normal, destination_normal)[:, None]
    normal_delta = (source_normal - destination_normal).abs()
    tangency = torch.stack(
        (
            (source_normal * unit_direction).sum(-1).abs(),
            (destination_normal * unit_direction).sum(-1).abs(),
        ),
        dim=-1,
    ).sort(dim=-1).values

    source_available = visible[source]
    destination_available = visible[destination]
    both_available = source_available & destination_available
    exactly_one_available = source_available ^ destination_available
    neither_available = ~source_available & ~destination_available
    availability = torch.stack(
        (
            (source_available.to(dtype) + destination_available.to(dtype)) / 2,
            both_available.to(dtype),
            exactly_one_available.to(dtype),
            neither_available.to(dtype),
        ),
        dim=-1,
    )
    both = both_available.to(dtype)[:, None]

    source_rgb = features_value[source, F71_RGB_SLICE]
    destination_rgb = features_value[destination, F71_RGB_SLICE]
    rgb_delta = (source_rgb - destination_rgb).abs() * both
    rgb_cosine = (_cosine(source_rgb, destination_rgb)[:, None] * both)
    source_radio = features_value[source, F71_RADIO_SLICE]
    destination_radio = features_value[destination, F71_RADIO_SLICE]
    radio_cosine = (_cosine(source_radio, destination_radio)[:, None] * both)

    result = torch.cat(
        (
            relative,
            normalized_distance,
            normal_cosine,
            normal_delta,
            tangency,
            availability,
            rgb_delta,
            rgb_cosine,
            radio_cosine,
        ),
        dim=-1,
    )
    if result.shape != (edges.shape[1], EDGE_FEATURE_DIMENSION):
        raise RuntimeError("query-free edge feature layout changed unexpectedly")
    if not torch.isfinite(result).all():
        raise RuntimeError("query-free edge features must remain finite")
    return result


class EdgeCompatibilityMLP(nn.Module):
    """Small shared scorer producing one trainable logit per directed edge."""

    def __init__(
        self,
        input_dimension: int = EDGE_FEATURE_DIMENSION,
        hidden_dimension: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dimension != EDGE_FEATURE_DIMENSION:
            raise ValueError(
                f"edge input dimension must match the sealed {EDGE_FEATURE_DIMENSION}-D layout"
            )
        if hidden_dimension <= 0 or not 0 <= float(dropout) < 1:
            raise ValueError("invalid edge MLP hidden dimension/dropout")
        self.input_dimension = int(input_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.dropout = float(dropout)
        self.network = nn.Sequential(
            nn.Linear(self.input_dimension, self.hidden_dimension),
            nn.LayerNorm(self.hidden_dimension),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dimension, self.hidden_dimension),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dimension, 1),
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        if edge_features.ndim != 2 or edge_features.shape[1] != self.input_dimension:
            raise ValueError(
                f"edge_features must have shape [A, {self.input_dimension}]"
            )
        if not torch.isfinite(edge_features).all():
            raise ValueError("edge_features must be finite")
        logits = self.network(edge_features).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise RuntimeError("edge compatibility logits became non-finite")
        return logits


class SurfaceMessagePassing(nn.Module):
    """Two- or three-step local residual with exact observed clamps.

    Incoming edge weights are divided by the fixed directed in-degree, not by
    their learned sum.  Thus the transported neighbours plus retained self mass
    form a convex combination while even a single-neighbour edge keeps a useful
    gradient.  Observed positive clamps also seed a continuous token support
    diffusion.  A learned soft extent gate moves unsupported *unknown* token
    mass back to null without a hard radius, threshold, connected component, or
    target input.

    These 2--3 layers are a local boundary/support residual.  The frozen unary
    remains the global identity and extent prior; this module cannot create a
    separate hard global extent.
    """

    def __init__(
        self,
        step_count: int = 2,
        feature_dimension: int = F71_FEATURE_DIMENSION,
        edge_hidden_dimension: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if step_count not in (2, 3):
            raise ValueError("step_count must be 2 or 3")
        if feature_dimension != F71_FEATURE_DIMENSION:
            raise ValueError("message passing requires the sealed F71 feature layout")
        self.step_count = int(step_count)
        self.feature_dimension = int(feature_dimension)
        self.edge_hidden_dimension = int(edge_hidden_dimension)
        self.dropout = float(dropout)
        self.edge_compatibility = EdgeCompatibilityMLP(
            EDGE_FEATURE_DIMENSION,
            hidden_dimension=self.edge_hidden_dimension,
            dropout=self.dropout,
        )
        self.step_logits = nn.Parameter(torch.zeros(self.step_count))
        # Start as a mild residual: sigmoid(-3) removes only about 4.7% of
        # unsupported token mass per layer.  This prevents the untrained 2-step
        # model from erasing the frozen unary's global coverage.
        self.extent_gate_logits = nn.Parameter(
            torch.full((self.step_count,), EXTENT_GATE_INITIAL_LOGIT)
        )

    def architecture_receipt(self) -> dict[str, object]:
        """Return the immutable causal role/configuration for experiment receipts."""

        return {
            "schema": "radio_gs.surface_object_memory_v4.surface_message_passing.v1",
            "step_count": self.step_count,
            "feature_dimension": self.feature_dimension,
            "edge_feature_dimension": EDGE_FEATURE_DIMENSION,
            "edge_feature_layout": list(EDGE_FEATURE_LAYOUT),
            "edge_hidden_dimension": self.edge_hidden_dimension,
            "dropout": self.dropout,
            "role": "local_surface_boundary_and_observed_seed_support_residual",
            "global_identity_and_extent_authority": "frozen_K_plus_null_unary",
            "adjacency": (
                "fixed_bidirectional_duplicate_free_surface_voxel_six_neighbour"
            ),
            "observed_policy": "exact_K_plus_null_clamp_after_every_step",
            "unknown_policy": "learned_soft_seed_support_gate_mass_returns_to_null",
            "extent_gate_initial_logit": EXTENT_GATE_INITIAL_LOGIT,
            "extent_gate_initial_strength": float(
                torch.sigmoid(torch.tensor(EXTENT_GATE_INITIAL_LOGIT))
            ),
            "target_membership_in_edge_or_extent_features": False,
            "query_in_edge_or_extent_features": False,
            "hard_threshold": False,
            "hard_radius_or_envelope": False,
            "connected_components": False,
        }

    @staticmethod
    def _validate_probabilities(
        probabilities: torch.Tensor,
        *,
        name: str,
        element_count: int | None = None,
    ) -> None:
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError(f"{name} must have shape [N, K+1] with K >= 1")
        if element_count is not None and probabilities.shape[0] != element_count:
            raise ValueError(f"{name} must align with the carrier elements")
        if not torch.isfinite(probabilities).all():
            raise ValueError(f"{name} must be finite")
        if bool((probabilities < 0).any()):
            raise ValueError(f"{name} must be non-negative")
        expected = torch.ones(
            probabilities.shape[0],
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        if not torch.allclose(probabilities.sum(-1), expected, atol=2e-6, rtol=2e-6):
            raise ValueError(f"{name} must lie on the K+null probability simplex")

    @staticmethod
    def _normalize(probabilities: torch.Tensor) -> torch.Tensor:
        result = probabilities.clamp_min(0)
        return result / result.sum(-1, keepdim=True).clamp_min(
            torch.finfo(result.dtype).tiny
        )

    @staticmethod
    def _bounded_transport(
        values: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        edge_weights: torch.Tensor,
        safe_degree: torch.Tensor,
    ) -> torch.Tensor:
        incoming = torch.zeros_like(values)
        incoming_weight = values.new_zeros(values.shape[0])
        if destination.numel():
            incoming.index_add_(
                0, destination, values[source] * edge_weights[:, None]
            )
            incoming_weight.index_add_(0, destination, edge_weights)
        retained_self = (1 - incoming_weight / safe_degree).clamp(min=0, max=1)
        return retained_self[:, None] * values + incoming / safe_degree[:, None]

    def forward(
        self,
        unary_probabilities: torch.Tensor,
        edge_index: torch.Tensor,
        centres: torch.Tensor,
        normals: torch.Tensor,
        local_features: torch.Tensor,
        source_visible: torch.Tensor,
        clamp_mask: torch.Tensor,
        clamp_probabilities: torch.Tensor,
        *,
        voxel_size: float,
    ) -> SurfaceMessagePassingOutput:
        unary_input = torch.as_tensor(unary_probabilities)
        if not unary_input.is_floating_point():
            raise ValueError("unary_probabilities must be floating point")
        device = unary_input.device
        dtype = unary_input.dtype
        if dtype not in (torch.float32, torch.float64):
            dtype = torch.float32
        # The pointwise completion model is a frozen causal input to this stage.
        unary = unary_input.to(dtype=dtype).detach()
        self._validate_probabilities(unary, name="unary_probabilities")
        element_count = unary.shape[0]

        mask_input = torch.as_tensor(clamp_mask)
        if mask_input.dtype != torch.bool:
            raise ValueError("clamp_mask must be an explicit boolean mask")
        mask = mask_input.to(device=device).detach()
        if mask.shape != (element_count,):
            raise ValueError("clamp_mask must have shape [N]")
        clamp = torch.as_tensor(
            clamp_probabilities, device=device, dtype=dtype
        ).detach()
        if clamp.shape != unary.shape or not torch.isfinite(clamp).all():
            raise ValueError("clamp_probabilities must be finite and match unary")
        selected_clamp = clamp[mask]
        if selected_clamp.numel():
            if bool(((selected_clamp != 0) & (selected_clamp != 1)).any()) or not bool(
                (selected_clamp.sum(-1) == 1).all()
            ):
                raise ValueError("every clamped row must be an exact K+null one-hot")

        edges_input = torch.as_tensor(edge_index)
        if edges_input.dtype == torch.bool or edges_input.is_floating_point():
            raise ValueError("edge_index must contain explicit integer indices")
        edges = edges_input.to(device=device, dtype=torch.long).detach()
        centres_input = torch.as_tensor(centres)
        if centres_input.shape != (element_count, 3):
            raise ValueError("centres and unary_probabilities must align")
        edge_features = build_query_free_edge_features(
            edges,
            centres_input.to(device=device),
            torch.as_tensor(normals, device=device),
            torch.as_tensor(local_features, device=device),
            torch.as_tensor(source_visible, device=device),
            voxel_size=voxel_size,
        ).to(dtype=dtype)
        if edges.numel() and int(edges.max()) >= element_count:
            raise ValueError("unary_probabilities and carrier elements do not align")
        edge_logits = self.edge_compatibility(edge_features)
        edge_weights = torch.sigmoid(edge_logits)
        if not torch.isfinite(edge_weights).all() or bool(
            ((edge_weights < 0) | (edge_weights > 1)).any()
        ):
            raise RuntimeError("edge weights must remain finite and bounded")

        source, destination = edges
        in_degree = unary.new_zeros(element_count)
        if destination.numel():
            in_degree.index_add_(0, destination, torch.ones_like(edge_weights))
        safe_degree = in_degree.clamp_min(1)
        current = torch.where(mask[:, None], clamp, unary)
        step_probabilities: list[torch.Tensor] = []
        token_count = unary.shape[1] - 1
        reachability = torch.where(
            mask[:, None], clamp[:, :token_count], unary.new_zeros(element_count, token_count)
        )
        step_seed_reachabilities: list[torch.Tensor] = []
        step_extent_weights: list[torch.Tensor] = []
        step_strengths = torch.sigmoid(self.step_logits).to(dtype=dtype)
        extent_gate_strengths = torch.sigmoid(self.extent_gate_logits).to(dtype=dtype)
        extent_weights = unary.new_ones(element_count, token_count)
        for step_strength, extent_gate_strength in zip(
            step_strengths, extent_gate_strengths
        ):
            reachability = self._bounded_transport(
                reachability, source, destination, edge_weights, safe_degree
            ).clamp(0, 1)
            reachability = torch.where(
                mask[:, None], clamp[:, :token_count], reachability
            )
            extent_weights = 1 - extent_gate_strength * (1 - reachability)

            propagated = self._bounded_transport(
                current, source, destination, edge_weights, safe_degree
            )
            candidate = (1 - step_strength) * unary + step_strength * propagated
            candidate = self._normalize(candidate)
            gated_tokens = candidate[:, :token_count] * extent_weights
            returned_to_null = (candidate[:, :token_count] - gated_tokens).sum(
                -1, keepdim=True
            )
            candidate = torch.cat(
                (gated_tokens, candidate[:, token_count:] + returned_to_null), dim=-1
            )
            candidate = self._normalize(candidate)
            current = torch.where(mask[:, None], clamp, candidate)
            if not torch.isfinite(current).all() or bool((current < 0).any()):
                raise RuntimeError("surface propagation left the finite simplex")
            step_probabilities.append(current)
            step_seed_reachabilities.append(reachability)
            step_extent_weights.append(extent_weights)

        self._validate_probabilities(
            current, name="message-passed probabilities", element_count=element_count
        )
        if selected_clamp.numel() and not torch.equal(current[mask], selected_clamp):
            raise RuntimeError("surface propagation changed an exact observed clamp")
        stable_floor = torch.finfo(current.dtype).tiny
        logit_epsilon = torch.finfo(current.dtype).eps
        stable_extent = extent_weights.clamp(logit_epsilon, 1 - logit_epsilon)
        return SurfaceMessagePassingOutput(
            probabilities=current,
            log_probabilities=current.clamp_min(stable_floor).log(),
            edge_logits=edge_logits,
            edge_weights=edge_weights,
            edge_features=edge_features,
            step_probabilities=tuple(step_probabilities),
            step_strengths=step_strengths,
            seed_reachability=reachability,
            extent_logits=torch.logit(stable_extent),
            extent_weights=extent_weights,
            step_seed_reachabilities=tuple(step_seed_reachabilities),
            step_extent_weights=tuple(step_extent_weights),
            extent_gate_strengths=extent_gate_strengths,
        )


__all__ = [
    "EDGE_FEATURE_DIMENSION",
    "EDGE_FEATURE_LAYOUT",
    "EXTENT_GATE_INITIAL_LOGIT",
    "F71_FEATURE_DIMENSION",
    "EdgeCompatibilityMLP",
    "SurfaceMessagePassing",
    "SurfaceMessagePassingOutput",
    "build_query_free_edge_features",
    "validate_surface_voxel_adjacency",
]
