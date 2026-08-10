"""Target-blind LERF readout with a valid-primitive kNN domain.

This module is intentionally separate from the frozen evaluator.  It defines
an auditable candidate whose only readout change is that invalid primitives
cannot occupy any of the ``k`` neighbour slots used to smooth valid rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch


CONTRACT = "radio_gs.lerf_valid_domain_knn_readout.v1"


@dataclass(frozen=True)
class ValidDomainMultiscaleReadout:
    scores: torch.Tensor
    scores_by_scale: torch.Tensor
    smoothed_probability: torch.Tensor
    selected_scale_indices: torch.Tensor
    raw_smoothed_peaks: torch.Tensor


@dataclass(frozen=True)
class NeighborDomainAudit:
    total_rows: int
    valid_rows: int
    k: int
    effective_valid_k_min: int
    effective_valid_k_mean: float
    effective_valid_k_max: int
    valid_rows_with_invalid_legacy_neighbor: int

    @property
    def affected_valid_fraction(self) -> float:
        if self.valid_rows == 0:
            return 0.0
        return self.valid_rows_with_invalid_legacy_neighbor / self.valid_rows


def _validate_scores_xyz_valid(
    scores: torch.Tensor,
    xyz: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(scores).detach().float().cpu().contiguous()
    points = torch.as_tensor(xyz).detach().float().cpu().contiguous()
    if values.ndim != 2:
        raise ValueError(f"scores must be [N,C], got {tuple(values.shape)}")
    if points.shape != (int(values.shape[0]), 3):
        raise ValueError(
            f"xyz must be [{int(values.shape[0])},3], got {tuple(points.shape)}"
        )
    count = int(values.shape[0])
    valid = (
        torch.ones(count, dtype=torch.bool)
        if valid_mask is None
        else torch.as_tensor(valid_mask).detach().bool().cpu().reshape(-1)
    )
    if valid.shape != (count,) or not bool(valid.any()):
        raise ValueError("valid_mask must keep at least one primitive")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("scores must be finite")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("xyz must be finite")
    return values, points, valid


def valid_domain_knn_indices(
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global valid-row indices and valid-only neighbours for those rows."""

    points = torch.as_tensor(xyz).detach().float().cpu().contiguous()
    valid = torch.as_tensor(valid_mask).detach().bool().cpu().reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or valid.shape != (points.shape[0],):
        raise ValueError("xyz/valid_mask axes differ")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("xyz must be finite")
    valid_rows = torch.where(valid)[0]
    if not int(valid_rows.numel()):
        raise ValueError("valid_mask must keep at least one primitive")
    if int(k) <= 0:
        raise ValueError("k must be positive")

    from sklearn.neighbors import NearestNeighbors

    valid_points = points[valid_rows]
    neighbors = min(int(k), int(valid_rows.numel()))
    local = NearestNeighbors(n_neighbors=neighbors).fit(
        valid_points.numpy()
    ).kneighbors(valid_points.numpy(), return_distance=False)
    global_neighbors = valid_rows[torch.from_numpy(local).long()]
    return valid_rows, global_neighbors.contiguous()


def valid_domain_knn_smoothed_scores(
    scores: torch.Tensor,
    xyz: torch.Tensor,
    *,
    k: int = 10,
    chunk_size: int = 65536,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Smooth valid rows using exactly ``min(k, valid_count)`` valid neighbours.

    Invalid output rows are exact zero.  They neither contribute to smoothing
    nor participate in the later extrema used for min-max remapping.
    """

    values, points, valid = _validate_scores_xyz_valid(scores, xyz, valid_mask)
    valid_rows, indices = valid_domain_knn_indices(points, valid, k=k)
    result = torch.zeros_like(values)
    step = max(1, int(chunk_size))
    for start in range(0, int(valid_rows.numel()), step):
        stop = min(start + step, int(valid_rows.numel()))
        rows = valid_rows[start:stop]
        neighbour_mean = values[indices[start:stop]].mean(dim=1)
        result[rows] = 0.5 * (values[rows] + neighbour_mean)
    return result


def valid_domain_minmax_remap_scores(
    smoothed_scores: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the released VALA remap, with extrema restricted to valid rows."""

    values = torch.as_tensor(smoothed_scores).detach().float().cpu().contiguous()
    valid = torch.as_tensor(valid_mask).detach().bool().cpu().reshape(-1)
    if values.ndim != 2 or valid.shape != (values.shape[0],) or not bool(valid.any()):
        raise ValueError("smoothed_scores/valid_mask axes differ")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("smoothed scores must be finite")
    low = values[valid].amin(dim=0, keepdim=True)
    high = values[valid].amax(dim=0, keepdim=True)
    span = high - low
    normalized = torch.where(
        span > 1e-9,
        (values - low) / span.clamp_min(1e-9),
        torch.zeros_like(values),
    )
    result = (2.0 * normalized - 1.0).clamp_(0.0, 1.0)
    result[~valid] = 0.0
    return result


def canonical_negative_probability(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    *,
    logit_scale: float = 10.0,
) -> torch.Tensor:
    """Convert independent cosine scores to canonical-negative probability."""

    positive = torch.as_tensor(positive_scores).detach().float().cpu().contiguous()
    negative = torch.as_tensor(negative_scores).detach().float().cpu().contiguous()
    if positive.ndim != 3 or negative.ndim != 3:
        raise ValueError("positive/negative scores must be [N,3,Q]")
    if positive.shape[:2] != negative.shape[:2] or positive.shape[1] != 3:
        raise ValueError("positive/negative primitive and scale axes differ")
    if positive.shape[-1] == 0 or negative.shape[-1] == 0:
        raise ValueError("positive/negative query axes must be non-empty")
    if not bool(torch.isfinite(positive).all()) or not bool(torch.isfinite(negative).all()):
        raise ValueError("positive/negative cosine scores must be finite")
    tolerance = 1e-4
    if bool((positive.abs() > 1.0 + tolerance).any()) or bool(
        (negative.abs() > 1.0 + tolerance).any()
    ):
        raise ValueError("normalized cosine scores must remain in [-1,1]")
    if not math.isfinite(float(logit_scale)) or float(logit_scale) <= 0.0:
        raise ValueError("logit_scale must be finite and positive")
    hardest_negative = negative.amax(dim=-1, keepdim=True)
    return torch.sigmoid((positive - hardest_negative) * float(logit_scale))


def valid_domain_multiscale_readout(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int = 10,
    chunk_size: int = 65536,
    logit_scale: float = 10.0,
) -> ValidDomainMultiscaleReadout:
    """Run canonical probability, valid-domain kNN, remap, and peak scale choice."""

    probability = canonical_negative_probability(
        positive_scores, negative_scores, logit_scale=logit_scale
    )
    count, scales, queries = probability.shape
    valid = torch.as_tensor(valid_mask).detach().bool().cpu().reshape(-1)
    if valid.shape != (count,) or not bool(valid.any()):
        raise ValueError("valid_mask must keep at least one primitive")
    smoothed = valid_domain_knn_smoothed_scores(
        probability.reshape(count, scales * queries),
        xyz,
        k=k,
        chunk_size=chunk_size,
        valid_mask=valid,
    ).reshape(count, scales, queries)
    peaks = smoothed[valid].amax(dim=0)
    selected = peaks.argmax(dim=0).long()
    remapped = valid_domain_minmax_remap_scores(
        smoothed.reshape(count, scales * queries), valid_mask=valid
    ).reshape(count, scales, queries)
    query_axis = torch.arange(queries)
    scores = remapped[:, selected, query_axis].contiguous()
    return ValidDomainMultiscaleReadout(
        scores=scores,
        scores_by_scale=remapped.contiguous(),
        smoothed_probability=smoothed.contiguous(),
        selected_scale_indices=selected.contiguous(),
        raw_smoothed_peaks=peaks.contiguous(),
    )


def audit_legacy_neighbor_domain(
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int = 10,
) -> NeighborDomainAudit:
    """Measure how often legacy all-row kNN loses slots to invalid rows."""

    points = torch.as_tensor(xyz).detach().float().cpu().contiguous()
    valid = torch.as_tensor(valid_mask).detach().bool().cpu().reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or valid.shape != (points.shape[0],):
        raise ValueError("xyz/valid_mask axes differ")
    if not bool(torch.isfinite(points).all()) or not bool(valid.any()):
        raise ValueError("xyz must be finite and valid_mask non-empty")
    if int(k) <= 0:
        raise ValueError("k must be positive")

    from sklearn.neighbors import NearestNeighbors

    neighbors = min(int(k), int(points.shape[0]))
    valid_rows = torch.where(valid)[0]
    indices = NearestNeighbors(n_neighbors=neighbors).fit(points.numpy()).kneighbors(
        points[valid_rows].numpy(), return_distance=False
    )
    effective = valid[torch.from_numpy(indices).long()].sum(dim=1)
    return NeighborDomainAudit(
        total_rows=int(points.shape[0]),
        valid_rows=int(valid_rows.numel()),
        k=neighbors,
        effective_valid_k_min=int(effective.min()),
        effective_valid_k_mean=float(effective.float().mean()),
        effective_valid_k_max=int(effective.max()),
        valid_rows_with_invalid_legacy_neighbor=int((effective < neighbors).sum()),
    )


__all__ = [
    "CONTRACT",
    "NeighborDomainAudit",
    "ValidDomainMultiscaleReadout",
    "audit_legacy_neighbor_domain",
    "canonical_negative_probability",
    "valid_domain_knn_indices",
    "valid_domain_knn_smoothed_scores",
    "valid_domain_minmax_remap_scores",
    "valid_domain_multiscale_readout",
]
