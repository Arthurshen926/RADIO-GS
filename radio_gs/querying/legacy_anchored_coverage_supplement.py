"""One-way, target-blind coverage supplement for frozen LERF scores.

The accepted legacy score cache is the immutable anchor.  All-available source
views may provide scores only for rows that lacked a legacy source teacher and
gained one under the all-available source domain.  New rows never participate
in the legacy kNN, normalization, or scale-selection statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from radio_gs.querying import valid_domain_knn_readout as legacy_readout


CONTRACT = "radio_gs.lerf_legacy_anchored_coverage_supplement.v1"


@dataclass(frozen=True)
class LegacyAnchoredCoverageResult:
    scores: torch.Tensor
    supplement_mask: torch.Tensor
    supplement_rows: torch.Tensor
    supplement_scores: torch.Tensor
    neighbor_rows: torch.Tensor
    neighbor_distances: torch.Tensor
    nearest_anchor_local_radius: torch.Tensor
    selected_scale_indices: torch.Tensor
    legacy_low_by_scale: torch.Tensor
    legacy_high_by_scale: torch.Tensor


def _cpu_float(value: torch.Tensor, *, name: str) -> torch.Tensor:
    result = torch.as_tensor(value).detach().float().cpu().contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


def _cpu_bool(value: torch.Tensor, *, count: int, name: str) -> torch.Tensor:
    result = torch.as_tensor(value).detach().bool().cpu().reshape(-1).contiguous()
    if result.shape != (count,):
        raise ValueError(f"{name} must be [{count}]")
    return result


def _validate_raw_pair(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    count: int,
    queries: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = _cpu_float(positive, name=f"{label} positive scores")
    neg = _cpu_float(negative, name=f"{label} negative scores")
    if pos.ndim != 3 or pos.shape[0] != count or pos.shape[2] != queries:
        raise ValueError(f"{label} positive scores must be [N,S,Q]")
    if neg.ndim != 3 or neg.shape[:2] != pos.shape[:2] or neg.shape[2] == 0:
        raise ValueError(f"{label} negative scores must share [N,S]")
    tolerance = 1e-4
    if bool((pos.abs() > 1.0 + tolerance).any()) or bool(
        (neg.abs() > 1.0 + tolerance).any()
    ):
        raise ValueError(f"{label} raw normalized cosine scores leave [-1,1]")
    return pos, neg


def _coverage_masks(
    geometry_valid: torch.Tensor,
    global_rows: torch.Tensor,
    legacy_teacher_valid: torch.Tensor,
    all_available_teacher_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = int(geometry_valid.numel())
    rows = torch.as_tensor(global_rows).detach().long().cpu().reshape(-1).contiguous()
    if rows.numel() == 0 or bool((rows < 0).any()) or bool((rows >= count).any()):
        raise ValueError("global_rows must be non-empty in-range row indices")
    if not torch.equal(rows, torch.unique_consecutive(rows)):
        raise ValueError("global_rows must be strictly increasing and unique")
    expected_rows = torch.where(geometry_valid)[0]
    if not torch.equal(rows, expected_rows):
        raise ValueError("teacher global_rows must exactly equal geometry-valid rows")
    legacy = _cpu_bool(
        legacy_teacher_valid,
        count=int(rows.numel()),
        name="legacy_teacher_valid",
    )
    all_available = _cpu_bool(
        all_available_teacher_valid,
        count=int(rows.numel()),
        name="all_available_teacher_valid",
    )
    if not bool(legacy.any()):
        raise ValueError("legacy teacher domain must contain at least one row")
    if not bool((all_available | ~legacy).all()):
        raise ValueError("all-available teacher validity must include legacy validity")
    supplement_local = ~legacy & all_available
    supplement_global = rows[supplement_local]
    supplement_mask = torch.zeros(count, dtype=torch.bool)
    supplement_mask[supplement_global] = True
    legacy_anchor_rows = rows[legacy]
    return supplement_mask, supplement_global, legacy_anchor_rows


def _query_legacy_anchor_neighbors(
    xyz: torch.Tensor,
    supplement_rows: torch.Tensor,
    legacy_anchor_rows: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(k) <= 0:
        raise ValueError("k must be positive")
    if supplement_rows.numel() == 0:
        return (
            torch.empty((0, min(int(k), int(legacy_anchor_rows.numel()))), dtype=torch.long),
            torch.empty((0, min(int(k), int(legacy_anchor_rows.numel()))), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )
    from sklearn.neighbors import NearestNeighbors

    neighbors = min(int(k), int(legacy_anchor_rows.numel()))
    model = NearestNeighbors(n_neighbors=neighbors).fit(
        xyz[legacy_anchor_rows].numpy()
    )
    distances, local = model.kneighbors(
        xyz[supplement_rows].numpy(), return_distance=True
    )
    local_tensor = torch.from_numpy(local).long()
    radius_neighbors = min(int(k) + 1, int(legacy_anchor_rows.numel()))
    unique_nearest, inverse = torch.unique(
        local_tensor[:, 0], sorted=True, return_inverse=True
    )
    anchor_distances = model.kneighbors(
        xyz[legacy_anchor_rows[unique_nearest]].numpy(),
        n_neighbors=radius_neighbors,
        return_distance=True,
    )[0]
    unique_radius = torch.from_numpy(anchor_distances[:, -1]).float()
    nearest_anchor_radius = unique_radius[inverse]
    return (
        legacy_anchor_rows[local_tensor].contiguous(),
        torch.from_numpy(distances).float().contiguous(),
        nearest_anchor_radius.contiguous(),
    )


def legacy_anchored_coverage_supplement(
    accepted_scores: torch.Tensor,
    legacy_positive_scores: torch.Tensor,
    legacy_negative_scores: torch.Tensor,
    all_available_positive_scores: torch.Tensor,
    all_available_negative_scores: torch.Tensor,
    xyz: torch.Tensor,
    geometry_valid: torch.Tensor,
    global_rows: torch.Tensor,
    legacy_teacher_valid: torch.Tensor,
    all_available_teacher_valid: torch.Tensor,
    *,
    k: int = 10,
    chunk_size: int = 65536,
    logit_scale: float = 10.0,
) -> LegacyAnchoredCoverageResult:
    """Fill only newly source-covered rows while preserving the legacy cache.

    The legacy readout is reconstructed first and must match ``accepted_scores``
    bit for bit.  Its smoothing extrema and selected scales are then frozen.
    Supplement rows query only legacy-teacher-valid neighbors.  Consequently,
    all-available values can never flow back into a legacy row or statistic.
    """

    accepted = _cpu_float(accepted_scores, name="accepted scores")
    points = _cpu_float(xyz, name="xyz")
    if accepted.ndim != 2:
        raise ValueError("accepted scores must be [N,Q]")
    count, queries = map(int, accepted.shape)
    if points.shape != (count, 3):
        raise ValueError("xyz must be [N,3]")
    if bool((accepted < 0.0).any()) or bool((accepted > 1.0).any()):
        raise ValueError("accepted scores must remain in [0,1]")
    valid = _cpu_bool(geometry_valid, count=count, name="geometry_valid")
    if not bool(valid.any()):
        raise ValueError("geometry_valid must keep at least one row")
    legacy_pos, legacy_neg = _validate_raw_pair(
        legacy_positive_scores,
        legacy_negative_scores,
        count=count,
        queries=queries,
        label="legacy",
    )
    all_pos, all_neg = _validate_raw_pair(
        all_available_positive_scores,
        all_available_negative_scores,
        count=count,
        queries=queries,
        label="all-available",
    )
    if legacy_pos.shape != all_pos.shape or legacy_neg.shape != all_neg.shape:
        raise ValueError("legacy/all-available raw score axes differ")
    if not math.isfinite(float(logit_scale)) or float(logit_scale) <= 0.0:
        raise ValueError("logit_scale must be finite and positive")

    supplement_mask, supplement_rows, legacy_anchor_rows = _coverage_masks(
        valid,
        global_rows,
        legacy_teacher_valid,
        all_available_teacher_valid,
    )

    # This reconstruction is the fail-closed proof that the frozen legacy
    # cache, normalization, and selected-scale contract are exactly known.
    reconstructed = legacy_readout.valid_domain_multiscale_readout(
        legacy_pos,
        legacy_neg,
        points,
        valid,
        k=int(k),
        chunk_size=int(chunk_size),
        logit_scale=float(logit_scale),
    )
    if not torch.equal(reconstructed.scores, accepted):
        mismatch = int((reconstructed.scores != accepted).sum())
        raise ValueError(
            "accepted legacy scores cannot be reconstructed bitwise "
            f"({mismatch} mismatched cells)"
        )

    legacy_low = reconstructed.smoothed_probability[valid].amin(dim=0)
    legacy_high = reconstructed.smoothed_probability[valid].amax(dim=0)
    neighbor_rows, neighbor_distances, nearest_anchor_local_radius = (
        _query_legacy_anchor_neighbors(
        points,
        supplement_rows,
        legacy_anchor_rows,
        k=int(k),
        )
    )
    if supplement_rows.numel() == 0:
        supplement_scores = torch.empty((0, queries), dtype=torch.float32)
    else:
        legacy_probability = legacy_readout.canonical_negative_probability(
            legacy_pos,
            legacy_neg,
            logit_scale=float(logit_scale),
        )
        all_probability = legacy_readout.canonical_negative_probability(
            all_pos,
            all_neg,
            logit_scale=float(logit_scale),
        )
        neighbor_mean = legacy_probability[neighbor_rows].mean(dim=1)
        smoothed = 0.5 * (all_probability[supplement_rows] + neighbor_mean)
        span = legacy_high - legacy_low
        normalized = torch.where(
            span > 1e-9,
            (smoothed - legacy_low) / span.clamp_min(1e-9),
            torch.zeros_like(smoothed),
        )
        remapped = (2.0 * normalized - 1.0).clamp_(0.0, 1.0)
        query_axis = torch.arange(queries)
        supplement_scores = remapped[
            :, reconstructed.selected_scale_indices, query_axis
        ].contiguous()

    output = accepted.clone()
    output[supplement_rows] = supplement_scores
    if not torch.equal(output[~supplement_mask], accepted[~supplement_mask]):
        raise AssertionError("all-available data changed a non-supplement row")
    if not bool(torch.isfinite(output).all()) or bool((output < 0.0).any()) or bool(
        (output > 1.0).any()
    ):
        raise ValueError("supplemented scores must be finite in [0,1]")
    return LegacyAnchoredCoverageResult(
        scores=output.contiguous(),
        supplement_mask=supplement_mask.contiguous(),
        supplement_rows=supplement_rows.contiguous(),
        supplement_scores=supplement_scores.contiguous(),
        neighbor_rows=neighbor_rows.contiguous(),
        neighbor_distances=neighbor_distances.contiguous(),
        nearest_anchor_local_radius=nearest_anchor_local_radius.contiguous(),
        selected_scale_indices=reconstructed.selected_scale_indices.contiguous(),
        legacy_low_by_scale=legacy_low.contiguous(),
        legacy_high_by_scale=legacy_high.contiguous(),
    )


__all__ = [
    "CONTRACT",
    "LegacyAnchoredCoverageResult",
    "legacy_anchored_coverage_supplement",
]
