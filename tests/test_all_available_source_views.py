from __future__ import annotations

import pytest

from radio_gs.querying.all_available_source_views import (
    audit_source_view_domain,
    validate_composite_frame_axis,
)


def test_all_available_is_manifest_minus_frozen_exclusions() -> None:
    audit = audit_source_view_domain(
        feature_frame_ids=[1, 2, 3, 4, 5, 6],
        excluded_frame_ids=[2, 5],
        legacy_frame_ids=[1, 4],
    )
    assert audit.all_available_frames == (1, 3, 4, 6)
    assert audit.omitted_frames == (3, 6)
    assert audit.legacy_coverage_fraction == pytest.approx(0.5)
    assert audit.legacy_is_all_available is False
    assert audit.legacy_is_prefix is False
    assert validate_composite_frame_axis(audit, [3, 6]) == (1, 3, 4, 6)


def test_complete_legacy_domain_needs_no_supplement() -> None:
    audit = audit_source_view_domain(
        feature_frame_ids=[3, 7, 9],
        excluded_frame_ids=[7],
        legacy_frame_ids=[3, 9],
    )
    assert audit.legacy_is_all_available is True
    assert audit.omitted_frames == ()
    assert validate_composite_frame_axis(audit, []) == (3, 9)


@pytest.mark.parametrize("broken", ["duplicate_excluded", "legacy", "order", "supplement"])
def test_domain_fails_closed(broken: str) -> None:
    feature = [1, 2, 3, 4]
    excluded = [2]
    legacy = [1, 3]
    if broken == "duplicate_excluded":
        excluded = [2, 2]
    elif broken == "legacy":
        legacy = [1, 9]
    elif broken == "order":
        legacy = [3, 1]
    if broken == "supplement":
        audit = audit_source_view_domain(
            feature_frame_ids=feature,
            excluded_frame_ids=excluded,
            legacy_frame_ids=legacy,
        )
        with pytest.raises(ValueError, match="exactly omitted"):
            validate_composite_frame_axis(audit, [4, 3])
    else:
        with pytest.raises(ValueError):
            audit_source_view_domain(
                feature_frame_ids=feature,
                excluded_frame_ids=excluded,
                legacy_frame_ids=legacy,
            )
