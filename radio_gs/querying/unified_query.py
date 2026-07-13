"""Typed, benchmark-agnostic queries over one canonical 3-D feature field.

The module intentionally contains no dataset-specific thresholds, labels, or
ground truth access.  A query is converted to positive/negative prototypes,
then every interface uses the same cosine-margin scorer.  Optional support
propagation is a deterministic query-time operation whose parameters must be
fixed before evaluating a test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


class QueryKind(str, Enum):
    TEXT = "text"
    IMAGE_EXEMPLAR = "image_exemplar"
    REGISTERED_2D = "registered_2d"
    POINT_3D = "point_3d"


class QuerySpace(str, Enum):
    SEMANTIC = "semantic"
    REGION = "region"


def _prototype_matrix(value: np.ndarray | Sequence[float], *, role: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError(f"{role} prototypes must have shape [P,D], got {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{role} prototypes contain NaN or infinity")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if bool((norms <= 1e-12).any()):
        raise ValueError(f"{role} prototypes contain a zero vector")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def mean_prototype(features: np.ndarray, indices: np.ndarray | Sequence[int]) -> np.ndarray:
    """Build one normalized prototype from declared prompt/seed indices."""

    values = np.asarray(features, dtype=np.float32)
    selected = np.asarray(indices)
    if values.ndim != 2:
        raise ValueError(f"features must be [N,D], got {values.shape}")
    if selected.dtype == bool:
        if selected.shape != (values.shape[0],):
            raise ValueError("Boolean prototype mask must align with features")
        subset = values[selected]
    else:
        selected = selected.astype(np.int64, copy=False).reshape(-1)
        if selected.size and (int(selected.min()) < 0 or int(selected.max()) >= values.shape[0]):
            raise IndexError("Prototype index is outside the feature array")
        subset = values[selected]
    if subset.shape[0] == 0:
        raise ValueError("Prototype selection is empty")
    return _prototype_matrix(subset.mean(axis=0), role="mean")[0]


@dataclass(frozen=True)
class QuerySpec:
    """A typed query after encoding in the field's readout space."""

    kind: QueryKind
    space: QuerySpace
    positive_prototypes: np.ndarray
    negative_prototypes: np.ndarray | None = None
    positive_seed_indices: tuple[int, ...] = ()
    negative_seed_indices: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive = _prototype_matrix(self.positive_prototypes, role="positive")
        negative = None
        if self.negative_prototypes is not None:
            negative = _prototype_matrix(self.negative_prototypes, role="negative")
            if negative.shape[1] != positive.shape[1]:
                raise ValueError("Positive and negative prototype dimensions differ")
        object.__setattr__(self, "kind", QueryKind(self.kind))
        object.__setattr__(self, "space", QuerySpace(self.space))
        object.__setattr__(self, "positive_prototypes", positive)
        object.__setattr__(self, "negative_prototypes", negative)
        object.__setattr__(
            self, "positive_seed_indices", tuple(int(i) for i in self.positive_seed_indices)
        )
        object.__setattr__(
            self, "negative_seed_indices", tuple(int(i) for i in self.negative_seed_indices)
        )

    @property
    def feature_dim(self) -> int:
        return int(self.positive_prototypes.shape[1])


def _normalized_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError(f"features must have shape [N,D], got {values.shape}")
    if not bool(np.isfinite(values).all()):
        raise ValueError("features contain NaN or infinity")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def score_features(features: np.ndarray, query: QuerySpec) -> np.ndarray:
    """Score ``[N,D]`` features with a max-prototype cosine margin.

    Multiple positives/negatives are useful for prompt ensembles.  With one
    foreground and one background prototype this is exactly
    ``cos(feature, foreground) - cos(feature, background)``.
    """

    normalized = _normalized_features(features)
    if normalized.shape[1] != query.feature_dim:
        raise ValueError(
            f"Feature/query dimensions differ: {normalized.shape[1]} vs {query.feature_dim}"
        )
    positive = (normalized @ query.positive_prototypes.T).max(axis=1)
    if query.negative_prototypes is None:
        scores = positive
    else:
        negative = (normalized @ query.negative_prototypes.T).max(axis=1)
        scores = positive - negative
    return np.ascontiguousarray(scores, dtype=np.float32)


def score_feature_map(features: np.ndarray, query: QuerySpec) -> np.ndarray:
    """Score a ``[C,H,W]`` feature map without benchmark-specific logic."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"features must be [C,H,W], got {values.shape}")
    channels, height, width = values.shape
    flat = np.moveaxis(values, 0, -1).reshape(-1, channels)
    return score_features(flat, query).reshape(height, width)


def _cosine_components_torch(
    features: torch.Tensor,
    positive_prototypes: torch.Tensor,
    negative_prototypes: torch.Tensor,
    *,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or positive_prototypes.ndim != 2 or negative_prototypes.ndim != 2:
        raise ValueError("Torch features/prototypes must all have shape [N,D]")
    if not (
        features.shape[1] == positive_prototypes.shape[1] == negative_prototypes.shape[1]
    ):
        raise ValueError("Torch feature/prototype dimensions differ")
    visual = features.float()
    positive = positive_prototypes.float().to(visual.device)
    negative = negative_prototypes.float().to(visual.device)
    if normalize:
        visual = torch_functional.normalize(visual, dim=-1, eps=1e-8)
        positive = torch_functional.normalize(positive, dim=-1, eps=1e-8)
        negative = torch_functional.normalize(negative, dim=-1, eps=1e-8)
    positive_similarity = visual @ positive.T
    hardest_negative = (visual @ negative.T).max(dim=1, keepdim=True).values
    return positive_similarity, hardest_negative


def cosine_margin_torch(
    features: torch.Tensor,
    positive_prototypes: torch.Tensor,
    negative_prototypes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU form of the shared scorer used by text-query evaluators."""

    positive_similarity, hardest_negative = _cosine_components_torch(
        features, positive_prototypes, negative_prototypes
    )
    return positive_similarity, positive_similarity - hardest_negative


def cosine_bank_torch(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Cosine logits for a semantic query bank, with no dataset postprocess."""

    if features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("Torch features/prototypes must have shape [N,D]")
    if features.shape[1] != prototypes.shape[1]:
        raise ValueError("Torch feature/prototype dimensions differ")
    visual = torch_functional.normalize(features.float(), dim=-1, eps=1e-8)
    bank = torch_functional.normalize(prototypes.float(), dim=-1, eps=1e-8).to(
        visual.device
    )
    return visual @ bank.T


def margin_to_relevancy_torch(margin: torch.Tensor, *, logit_scale: float) -> torch.Tensor:
    """Map a cosine margin to binary relevance with a fixed, GT-free scale."""

    return torch.sigmoid(margin.float() * float(logit_scale))


def cosine_relevancy_torch(
    features: torch.Tensor,
    positive_prototypes: torch.Tensor,
    negative_prototypes: torch.Tensor,
    *,
    logit_scale: float,
    assume_normalized: bool = False,
) -> torch.Tensor:
    """Stable binary softmax preserving the legacy evaluator operation order."""

    positive_similarity, hardest_negative = _cosine_components_torch(
        features,
        positive_prototypes,
        negative_prototypes,
        normalize=not assume_normalized,
    )
    positive_scaled = positive_similarity * float(logit_scale)
    negative_scaled = hardest_negative.expand_as(positive_similarity) * float(logit_scale)
    max_value = torch.maximum(positive_scaled, negative_scaled)
    return torch.exp(positive_scaled - max_value) / (
        torch.exp(positive_scaled - max_value)
        + torch.exp(negative_scaled - max_value)
        + 1e-8
    )


@dataclass(frozen=True)
class SupportPropagationConfig:
    """Fixed parameters for shared spatial/feature graph propagation."""

    neighbors: int = 16
    spatial_sigma: float = 0.08
    feature_temperature: float = 0.10
    iterations: int = 4
    residual: float = 0.35
    clamp_seeds: bool = True
    graph_mode: str = "directed"
    adaptive_spatial: bool = False
    spatial_scale: float = 2.0

    def __post_init__(self) -> None:
        if self.neighbors <= 0 or self.iterations < 0:
            raise ValueError("neighbors must be positive and iterations non-negative")
        if self.spatial_sigma <= 0 or self.feature_temperature <= 0:
            raise ValueError("propagation scales must be positive")
        if not 0.0 <= self.residual <= 1.0:
            raise ValueError("residual must be in [0,1]")
        if self.graph_mode not in {"directed", "symmetric_union"}:
            raise ValueError("graph_mode must be directed or symmetric_union")
        if self.spatial_scale <= 0:
            raise ValueError("spatial_scale must be positive")


@dataclass(frozen=True)
class SupportGraph:
    """Reusable sparse graph; build it once and apply many queries."""

    indices: np.ndarray
    distances: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64)
        distances = np.asarray(self.distances, dtype=np.float32)
        weights = np.asarray(self.weights, dtype=np.float32)
        if indices.ndim != 2 or distances.shape != indices.shape or weights.shape != indices.shape:
            raise ValueError("SupportGraph arrays must share shape [N,K]")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "weights", weights)


def build_support_graph(
    xyz: np.ndarray,
    features: np.ndarray,
    config: SupportPropagationConfig,
) -> SupportGraph:
    """Construct a reusable spatial/feature k-NN graph."""

    points = np.asarray(xyz, dtype=np.float32)
    normalized = _normalized_features(features)
    if points.ndim != 2 or points.shape != (normalized.shape[0], 3):
        raise ValueError("xyz must be [N,3] and align with features")
    if not bool(np.isfinite(points).all()):
        raise ValueError("xyz contains NaN or infinity")
    count = points.shape[0]
    if count <= 1:
        empty = np.empty((count, 0), dtype=np.float32)
        return SupportGraph(empty.astype(np.int64), empty, empty)
    k = min(config.neighbors + 1, count)
    distances, indices = cKDTree(points).query(points, k=k)
    distances = np.asarray(distances, dtype=np.float32)[:, 1:]
    indices = np.asarray(indices, dtype=np.int64)[:, 1:]
    if config.graph_mode == "symmetric_union":
        rows = np.repeat(np.arange(count, dtype=np.int64), indices.shape[1])
        adjacency = csr_matrix(
            (np.ones(rows.shape[0], dtype=np.uint8), (rows, indices.reshape(-1))),
            shape=(count, count),
        )
        symmetric = adjacency.maximum(adjacency.T).tocsr()
        degrees = np.diff(symmetric.indptr)
        width = int(degrees.max(initial=0))
        symmetric_indices = np.repeat(
            np.arange(count, dtype=np.int64)[:, None], width, axis=1
        )
        symmetric_distances = np.full((count, width), np.inf, dtype=np.float32)
        for row in range(count):
            neighbors = symmetric.indices[
                symmetric.indptr[row] : symmetric.indptr[row + 1]
            ]
            degree = neighbors.size
            symmetric_indices[row, :degree] = neighbors
            symmetric_distances[row, :degree] = np.linalg.norm(
                points[neighbors] - points[row], axis=1
            )
        indices = symmetric_indices
        distances = symmetric_distances
    # Avoid materializing [N,K,D], which is several GB for ScanNet SAM3
    # features.  The graph remains exact; only the temporary working set is
    # bounded.
    feature_cosine = np.empty(indices.shape, dtype=np.float32)
    # 4096 keeps the temporary below ~320 MiB for 1280-D, k=16 ScanNet
    # features while avoiding hundreds of costly large allocations.
    affinity_chunk_size = 4096
    for start in range(0, count, affinity_chunk_size):
        stop = min(start + affinity_chunk_size, count)
        feature_cosine[start:stop] = np.einsum(
            "nd,nkd->nk",
            normalized[start:stop],
            normalized[indices[start:stop]],
            optimize=True,
        )
    valid = np.isfinite(distances)
    if config.adaptive_spatial:
        local = np.where(valid, distances, np.nan)
        sigma = np.nanmedian(local, axis=1, keepdims=True) * config.spatial_scale
        sigma = np.where(np.isfinite(sigma), sigma, config.spatial_sigma)
        sigma = np.maximum(sigma, 1e-6)
    else:
        sigma = float(config.spatial_sigma)
    log_weights = (
        -0.5 * (distances / sigma) ** 2
        + (feature_cosine - 1.0) / config.feature_temperature
    )
    log_weights = np.where(valid, log_weights, -np.inf)
    row_max = np.max(log_weights, axis=1, keepdims=True)
    stable = np.exp(log_weights - row_max)
    stable = np.where(valid, stable, 0.0)
    weights = stable / np.maximum(stable.sum(axis=1, keepdims=True), 1e-12)
    return SupportGraph(indices, distances, weights)


def propagate_support(
    xyz: np.ndarray,
    features: np.ndarray,
    scores: np.ndarray,
    query: QuerySpec,
    config: SupportPropagationConfig,
    *,
    graph: SupportGraph | None = None,
) -> np.ndarray:
    """Propagate scores on a sparse spatial/feature k-NN graph.

    This is deliberately label-free: only field geometry/features and declared
    query seed indices are consumed.  It therefore belongs to the method at
    inference time, provided its hyperparameters are not fitted on test GT.
    """

    points = np.asarray(xyz, dtype=np.float32)
    feature_values = np.asarray(features, dtype=np.float32)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"xyz must be [N,3], got {points.shape}")
    if feature_values.ndim != 2 or feature_values.shape[0] != points.shape[0]:
        raise ValueError("features must be [N,D] and align with xyz")
    if values.shape != (points.shape[0],):
        raise ValueError("xyz, features, and scores must share their first dimension")
    if (
        not bool(np.isfinite(points).all())
        or not bool(np.isfinite(feature_values).all())
        or not bool(np.isfinite(values).all())
    ):
        raise ValueError("xyz/features/scores contain NaN or infinity")
    count = points.shape[0]
    if count <= 1 or config.iterations == 0:
        return values.copy()

    if graph is None:
        graph = build_support_graph(points, feature_values, config)
    if graph.indices.shape[0] != count:
        raise ValueError("Support graph does not align with query arrays")
    if graph.indices.shape[1] == 0:
        return values.copy()
    indices = graph.indices
    weights = graph.weights

    original = values.copy()
    propagated = values.copy()
    positive = np.asarray(query.positive_seed_indices, dtype=np.int64)
    negative = np.asarray(query.negative_seed_indices, dtype=np.int64)
    for seed_array in (positive, negative):
        if seed_array.size and (int(seed_array.min()) < 0 or int(seed_array.max()) >= count):
            raise IndexError("Query seed index is outside the graph")
    for _ in range(config.iterations):
        neighbor_mean = np.sum(weights * propagated[indices], axis=1)
        propagated = config.residual * original + (1.0 - config.residual) * neighbor_mean
        if config.clamp_seeds:
            propagated[positive] = np.maximum(original[positive], 1.0)
            propagated[negative] = np.minimum(original[negative], -1.0)
    return np.ascontiguousarray(propagated, dtype=np.float32)


def seed_connected_component(
    candidate_mask: np.ndarray,
    seed_index: int,
    graph: SupportGraph,
    *,
    max_edge_distance: float,
) -> np.ndarray:
    """Keep the candidate component reachable from one declared query seed."""

    candidate = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    seed = int(seed_index)
    if candidate.shape != (graph.indices.shape[0],):
        raise ValueError("candidate mask and support graph do not align")
    if seed < 0 or seed >= candidate.shape[0]:
        raise IndexError("seed index is outside the graph")
    if max_edge_distance <= 0:
        raise ValueError("max_edge_distance must be positive")
    output = np.zeros_like(candidate)
    if not candidate[seed]:
        return output
    output[seed] = True
    stack = [seed]
    while stack:
        row = stack.pop()
        valid = graph.distances[row] <= float(max_edge_distance)
        for neighbor in graph.indices[row, valid]:
            neighbor = int(neighbor)
            if candidate[neighbor] and not output[neighbor]:
                output[neighbor] = True
                stack.append(neighbor)
    return output


def binary_mask(
    scores: np.ndarray, *, threshold: float = 0.0, inclusive: bool = True
) -> np.ndarray:
    """Apply a predeclared threshold; this function performs no calibration."""

    values = np.asarray(scores)
    if not bool(np.isfinite(values).all()):
        raise ValueError("scores contain NaN or infinity")
    return values >= float(threshold) if inclusive else values > float(threshold)
