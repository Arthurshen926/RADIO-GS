"""Fail-closed source-view domains for all-available LERF teachers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


CONTRACT = "radio_gs.lerf_all_available_source_view_domain.v1"


@dataclass(frozen=True)
class SourceViewDomainAudit:
    feature_frames: tuple[int, ...]
    excluded_frames: tuple[int, ...]
    all_available_frames: tuple[int, ...]
    legacy_frames: tuple[int, ...]
    omitted_frames: tuple[int, ...]
    legacy_coverage_fraction: float
    legacy_is_all_available: bool
    legacy_is_prefix: bool


def _unique_ordered(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{label} must be non-empty and unique")
    return result


def audit_source_view_domain(
    *,
    feature_frame_ids: Sequence[int],
    excluded_frame_ids: Sequence[int],
    legacy_frame_ids: Sequence[int],
) -> SourceViewDomainAudit:
    """Define all available as feature frames minus frozen held-out frames.

    Feature-manifest order is authoritative.  Sorting or resampling here would
    silently introduce another view-selection policy.  Legacy frames may be a
    deterministic uniform subset, but must preserve their relative manifest
    order and may never overlap held-out frames.
    """

    feature = _unique_ordered(feature_frame_ids, label="feature frame axis")
    excluded = tuple(int(value) for value in excluded_frame_ids)
    legacy = _unique_ordered(legacy_frame_ids, label="legacy frame axis")
    if len(set(excluded)) != len(excluded):
        raise ValueError("excluded frame axis must be unique")
    feature_set = set(feature)
    excluded_set = set(excluded)
    legacy_set = set(legacy)
    # Some frozen feature bundles were already extracted after held-out-frame
    # removal (Ramen), while older bundles still contain those rows.  Both are
    # valid only because exclusion is applied again as an idempotent set mask.
    all_available = tuple(value for value in feature if value not in excluded_set)
    if not all_available:
        raise ValueError("all source feature frames are held out")
    if not legacy_set <= set(all_available):
        raise ValueError("legacy source frames escape the available source domain")
    # The legacy authority must preserve source-manifest order even when it is
    # uniformly subsampled.  This rejects order-corrupted composite inputs.
    legacy_in_manifest_order = tuple(value for value in all_available if value in legacy_set)
    if legacy != legacy_in_manifest_order:
        raise ValueError("legacy source authority order differs from feature manifest")
    omitted = tuple(value for value in all_available if value not in legacy_set)
    return SourceViewDomainAudit(
        feature_frames=feature,
        excluded_frames=excluded,
        all_available_frames=all_available,
        legacy_frames=legacy,
        omitted_frames=omitted,
        legacy_coverage_fraction=len(legacy) / len(all_available),
        legacy_is_all_available=not omitted,
        legacy_is_prefix=legacy == all_available[: len(legacy)],
    )


def validate_composite_frame_axis(
    audit: SourceViewDomainAudit,
    supplemental_frame_ids: Sequence[int],
) -> tuple[int, ...]:
    """Require the supplemental authority to cover exactly omitted frames."""

    supplemental = tuple(int(value) for value in supplemental_frame_ids)
    if supplemental != audit.omitted_frames:
        raise ValueError("supplemental responsibility axis is not exactly omitted views")
    combined_set = set(audit.legacy_frames) | set(supplemental)
    combined = tuple(
        value for value in audit.all_available_frames if value in combined_set
    )
    if combined != audit.all_available_frames:
        raise ValueError("composite source authority is not all-available")
    return combined


__all__ = [
    "CONTRACT",
    "SourceViewDomainAudit",
    "audit_source_view_domain",
    "validate_composite_frame_axis",
]
