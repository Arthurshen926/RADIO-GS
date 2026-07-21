"""Map continuous primitive predictions into ScanNet annotation-mesh space."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def gaussian_scores_to_mesh(
    gaussian_xyz: np.ndarray,
    gaussian_scores: np.ndarray,
    mesh_xyz: np.ndarray,
    *,
    neighbors: int = 3,
    maximum_distance_m: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-distance interpolate scores without reading instance GT."""

    source_xyz = np.asarray(gaussian_xyz, dtype=np.float32)
    scores = np.asarray(gaussian_scores, dtype=np.float32).reshape(-1)
    target_xyz = np.asarray(mesh_xyz, dtype=np.float32)
    if source_xyz.shape != (scores.size, 3) or target_xyz.ndim != 2 or target_xyz.shape[1] != 3:
        raise ValueError("gaussian xyz/scores and mesh xyz do not align")
    if neighbors <= 0 or source_xyz.shape[0] == 0:
        raise ValueError("neighbors/source primitives must be positive")
    distance, index = cKDTree(source_xyz).query(
        target_xyz, k=min(int(neighbors), source_xyz.shape[0])
    )
    distance = np.asarray(distance, dtype=np.float32)
    index = np.asarray(index, dtype=np.int64)
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    valid_neighbors = distance <= float(maximum_distance_m)
    weight = np.where(valid_neighbors, 1.0 / np.maximum(distance, 1e-4), 0.0)
    weight_sum = weight.sum(axis=1)
    mapped = (scores[index] * weight).sum(axis=1) / np.maximum(weight_sum, 1e-8)
    valid = weight_sum > 0
    mapped[~valid] = -np.inf
    return mapped.astype(np.float32), valid


def gaussian_mask_to_mesh(
    gaussian_xyz: np.ndarray,
    gaussian_mask: np.ndarray,
    mesh_xyz: np.ndarray,
    *,
    maximum_distance_m: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    scores, valid = gaussian_scores_to_mesh(
        gaussian_xyz,
        np.asarray(gaussian_mask, dtype=np.float32),
        mesh_xyz,
        neighbors=1,
        maximum_distance_m=maximum_distance_m,
    )
    return (scores >= 0.5) & valid, valid

