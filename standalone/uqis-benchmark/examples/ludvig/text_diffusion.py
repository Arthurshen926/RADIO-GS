"""Query-independent DINO topology and label-free LUDVIG text diffusion.

This benchmark-local operator deliberately consumes only persistent CLIP/DINO
field artifacts and a text-derived relevancy vector.  It never consumes RGB,
SAM masks, target labels, evaluator metrics, or test-fitted thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TextDiffusionConfig:
    neighbors: int = 64
    iterations: int = 20
    feature_bandwidth: float = 0.5
    regularizer_bandwidth: float = 2.0
    seed_quantile: float = 0.999
    graph_chunk_size: int = 16_384
    epsilon: float = 1e-8

    def validate(self, node_count: int) -> None:
        if node_count < 2:
            raise ValueError("graph diffusion requires at least two nodes")
        if not 1 <= self.neighbors < node_count:
            raise ValueError("neighbors must be in [1, node_count)")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.feature_bandwidth <= 0 or self.regularizer_bandwidth <= 0:
            raise ValueError("diffusion bandwidths must be positive")
        if not 0 < self.seed_quantile < 1:
            raise ValueError("seed_quantile must be in (0,1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.graph_chunk_size < 1:
            raise ValueError("graph_chunk_size must be positive")


@dataclass(frozen=True)
class DinoGraph:
    neighbor_indices: np.ndarray
    edge_weights: np.ndarray
    normalized_features: np.ndarray


def align_clip_relevancy_to_dino_carrier(
    clip_relevancy: np.ndarray, source_indices: np.ndarray
) -> np.ndarray:
    """Gather a full-carrier CLIP score onto the pruned DINO carrier."""

    relevance = np.asarray(clip_relevancy)
    indices = np.asarray(source_indices)
    if relevance.ndim != 1 or relevance.dtype != np.float32:
        raise ValueError("CLIP relevancy must be a float32 vector")
    if indices.ndim != 1 or indices.dtype != np.int64:
        raise ValueError("DINO source indices must be an int64 vector")
    if len(indices) == 0 or int(indices.min()) < 0 or int(indices.max()) >= len(relevance):
        raise ValueError("DINO source indices escape the CLIP carrier")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("DINO source indices must be unique")
    return relevance[indices].astype(np.float32, copy=False)


def build_dino_graph(
    xyz: np.ndarray,
    dino_features: np.ndarray,
    config: TextDiffusionConfig = TextDiffusionConfig(),
) -> DinoGraph:
    """Build a reusable spatial k-NN graph weighted by DINO similarity."""

    points = np.asarray(xyz, dtype=np.float32)
    features = np.asarray(dino_features, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("DINO carrier xyz must be finite [N,3]")
    if features.ndim != 2 or len(features) != len(points) or not np.isfinite(features).all():
        raise ValueError("DINO features must be finite [N,D] on the same carrier")
    config.validate(len(points))
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if bool((norms <= config.epsilon).any()):
        raise ValueError("DINO graph contains a zero feature")
    normalized = features / norms
    tree = cKDTree(points)
    indices = np.empty((len(points), config.neighbors), dtype=np.int32)
    feature_distance = np.empty(indices.shape, dtype=np.float32)
    for begin in range(0, len(points), config.graph_chunk_size):
        end = min(begin + config.graph_chunk_size, len(points))
        _distance, local_indices = tree.query(
            points[begin:end], k=config.neighbors + 1, workers=-1
        )
        local_indices = np.asarray(local_indices[:, 1:], dtype=np.int32)
        indices[begin:end] = local_indices
        delta = normalized[begin:end, None] - normalized[local_indices]
        feature_distance[begin:end] = np.linalg.norm(delta, axis=2)
    positive = feature_distance[feature_distance > config.epsilon]
    scale = float(np.median(positive)) if positive.size else 1.0
    weights = np.exp(
        -(feature_distance**2)
        / (config.feature_bandwidth * scale**2 + config.epsilon)
    ).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True) + config.epsilon
    return DinoGraph(indices, weights, normalized.astype(np.float32, copy=False))


def _otsu_threshold(values: np.ndarray) -> float:
    samples = np.asarray(values, dtype=np.float64)
    if samples.size < 2 or float(samples.min()) == float(samples.max()):
        return float(samples.min()) if samples.size else 0.0
    counts, edges = np.histogram(
        samples, bins=256, range=(float(samples.min()), float(samples.max()))
    )
    centers = (edges[:-1] + edges[1:]) * 0.5
    total = counts.sum()
    cumulative_count = np.cumsum(counts)
    cumulative_sum = np.cumsum(counts * centers)
    denominator = cumulative_count * (total - cumulative_count)
    between = np.zeros_like(centers)
    valid = denominator > 0
    between[valid] = (
        (total * cumulative_sum[valid] - cumulative_count[valid] * cumulative_sum[-1])
        ** 2
        / denominator[valid]
    )
    return float(centers[int(np.argmax(between))])


def diffuse_clip_relevancy(
    clip_relevancy: np.ndarray,
    graph: DinoGraph,
    config: TextDiffusionConfig = TextDiffusionConfig(),
) -> np.ndarray:
    """Diffuse one text score over a frozen DINO graph without supervision."""

    relevance = np.asarray(clip_relevancy, dtype=np.float32)
    indices = graph.neighbor_indices
    weights = graph.edge_weights
    features = graph.normalized_features
    if relevance.ndim != 1 or len(relevance) != len(indices):
        raise ValueError("CLIP relevancy must match the DINO carrier")
    if not np.isfinite(relevance).all() or bool(((relevance < 0) | (relevance > 1)).any()):
        raise ValueError("CLIP relevancy must be finite in [0,1]")
    config.validate(len(relevance))
    if indices.shape != (len(relevance), config.neighbors) or weights.shape != indices.shape:
        raise ValueError("DINO graph/config mismatch")

    low, high = float(relevance.min()), float(relevance.max())
    normalized = (relevance - low) / (high - low + config.epsilon)
    foreground = normalized[normalized > 0.5]
    mask_threshold = _otsu_threshold(foreground) if foreground.size else 1.0
    mask = normalized > mask_threshold
    seed_values = relevance[mask]
    if seed_values.size:
        high_threshold = _otsu_threshold(seed_values)
        higher = seed_values[seed_values > high_threshold]
        if higher.size:
            high_threshold = _otsu_threshold(higher)
    else:
        high_threshold = float(relevance.max())
    threshold = min(
        float(np.quantile(relevance, config.seed_quantile)), high_threshold
    )
    seed = relevance * (relevance >= threshold)
    if not bool(seed.any()):
        seed[int(np.argmax(relevance))] = float(relevance.max())

    prototype = (features * seed[:, None]).sum(axis=0)
    prototype_norm = float(np.linalg.norm(prototype))
    if prototype_norm <= config.epsilon:
        raise ValueError("text diffusion seed has no DINO support")
    prototype /= prototype_norm
    prototype_distance = np.sqrt(
        np.maximum(2.0 - 2.0 * (features @ prototype), 0.0)
    )
    positive = prototype_distance[prototype_distance > config.epsilon]
    scale = float(np.median(positive)) if positive.size else 1.0
    unary = np.exp(
        -(prototype_distance**2)
        / (config.regularizer_bandwidth * scale**2 + config.epsilon)
    ).astype(np.float32)
    unary_sqrt = np.sqrt(unary)

    value = seed.astype(np.float32, copy=True)
    for _ in range(config.iterations):
        value /= float(np.linalg.norm(value)) + config.epsilon
        propagated = (
            weights * unary_sqrt[indices] * value[indices]
        ).sum(axis=1)
        value = unary_sqrt * propagated
    maximum = float(value.max())
    if not np.isfinite(value).all() or maximum <= config.epsilon:
        raise ValueError("text diffusion collapsed")
    return np.clip(value / maximum, 0.0, 1.0).astype(np.float32)
