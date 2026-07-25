"""Label-free scene calibration for heterogeneous frozen capability spaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SceneSpaceCalibration:
    """Frozen query-independent statistics for one scene capability bank."""

    center: torch.Tensor
    scale: torch.Tensor
    background_centroids: torch.Tensor | None
    method: str
    sample_count: int

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        matrix = torch.as_tensor(values).to(
            device=self.center.device, dtype=torch.float32
        )
        if matrix.shape[-1] != self.center.numel():
            raise ValueError("calibration feature dimension mismatch")
        transformed = (matrix - self.center) / self.scale
        # A primitive exactly at the robust scene centre has a zero centred
        # vector.  Turning that row into a zero feature would create an
        # artificial non-comparable node.  Preserve its original normalized
        # direction instead; this is deterministic and uses no query signal.
        normalized = F.normalize(transformed, dim=-1, eps=1e-8)
        fallback = F.normalize(matrix, dim=-1, eps=1e-8)
        valid = transformed.norm(dim=-1, keepdim=True) > 1e-8
        return torch.where(valid, normalized, fallback)


def deterministic_sample_rows(values: torch.Tensor, count: int) -> torch.Tensor:
    matrix = torch.as_tensor(values)
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or int(count) <= 0:
        raise ValueError("values must be non-empty [N,D] and count positive")
    if matrix.shape[0] <= int(count):
        return matrix.float()
    indices = torch.linspace(
        0,
        matrix.shape[0] - 1,
        int(count),
        device=matrix.device,
    ).round().long()
    return matrix[indices].float()


def _spherical_centroids(
    samples: torch.Tensor,
    count: int,
    iterations: int,
) -> torch.Tensor:
    values = F.normalize(torch.as_tensor(samples).float(), dim=-1, eps=1e-8)
    count = min(max(1, int(count)), values.shape[0])
    # Start from the most typical sample, then cover distinct scene modes by
    # deterministic farthest-first initialization.
    scene_center = F.normalize(values.mean(dim=0), dim=0, eps=1e-8)
    selected = [int((values @ scene_center).argmax())]
    nearest_similarity = values @ values[selected[0]]
    for _ in range(1, count):
        index = int(nearest_similarity.argmin())
        selected.append(index)
        nearest_similarity = torch.maximum(
            nearest_similarity, values @ values[index]
        )
    centers = values[selected]
    for _ in range(max(0, int(iterations))):
        assignment = (values @ centers.T).argmax(dim=1)
        sums = torch.zeros_like(centers)
        masses = torch.zeros(count, device=values.device)
        sums.index_add_(0, assignment, values)
        masses.index_add_(0, assignment, torch.ones_like(assignment, dtype=torch.float32))
        updated = F.normalize(sums, dim=-1, eps=1e-8)
        centers = torch.where((masses > 0)[:, None], updated, centers)
    return centers


def fit_scene_space_calibration(
    features: torch.Tensor,
    *,
    method: str = "diagonal_robust",
    sample_size: int = 8192,
    background_centroids: int = 0,
    centroid_iterations: int = 4,
    minimum_scale: float = 1e-4,
) -> SceneSpaceCalibration:
    """Fit frozen scene statistics without query or label access."""

    matrix = torch.as_tensor(features)
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError("features must be non-empty [N,D]")
    if method not in {"none", "diagonal_robust"}:
        raise ValueError("feature calibration must be none or diagonal_robust")
    if background_centroids < 0 or centroid_iterations < 0 or minimum_scale <= 0:
        raise ValueError("invalid scene calibration parameters")
    sample = deterministic_sample_rows(matrix, int(sample_size))
    if method == "diagonal_robust":
        center = sample.median(dim=0).values
        mad = (sample - center).abs().median(dim=0).values
        robust_scale = 1.4826 * mad
        fallback = sample.std(dim=0, unbiased=False)
        scale = torch.where(
            robust_scale >= float(minimum_scale), robust_scale, fallback
        ).clamp_min(float(minimum_scale))
    else:
        center = torch.zeros(sample.shape[1], device=sample.device)
        scale = torch.ones(sample.shape[1], device=sample.device)
    normalized_sample = F.normalize(
        (sample - center) / scale, dim=-1, eps=1e-8
    )
    centroids = None
    if int(background_centroids) > 0:
        centroids = _spherical_centroids(
            normalized_sample,
            int(background_centroids),
            int(centroid_iterations),
        )
    return SceneSpaceCalibration(
        center=center,
        scale=scale,
        background_centroids=centroids,
        method=method,
        sample_count=int(sample.shape[0]),
    )


def robust_tanh_score_calibration(
    scores: torch.Tensor,
    *,
    tanh_scale: float = 2.0,
    minimum_scale: float = 1e-4,
    preserve_zero: bool = True,
) -> torch.Tensor:
    """Map arbitrary scene score scales to a fixed robust [-1,1] range."""

    values = torch.as_tensor(scores).float().reshape(-1)
    if values.numel() == 0 or tanh_scale <= 0 or minimum_scale <= 0:
        raise ValueError("scores must be non-empty and calibration scales positive")
    center = values.median()
    mad = (values - center).abs().median()
    robust_scale = 1.4826 * mad
    if float(robust_scale) < float(minimum_scale):
        robust_scale = values.std(unbiased=False).clamp_min(float(minimum_scale))
    origin = torch.zeros_like(center) if preserve_zero else center
    return torch.tanh((values - origin) / (float(tanh_scale) * robust_scale))
