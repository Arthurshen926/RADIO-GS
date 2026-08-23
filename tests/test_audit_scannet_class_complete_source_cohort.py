from __future__ import annotations

import pytest

from radio_gs.scripts.audit_scannet_class_complete_source_cohort import (
    class_coverage_counts,
    select_class_complete_cohort,
)


def test_select_class_complete_cohort_is_deterministic_and_complete() -> None:
    coverage = {
        "scene_c": {1, 2},
        "scene_b": {1, 2},
        "scene_a": {3},
        "scene_d": {4},
    }
    assert select_class_complete_cohort(coverage, [1, 2, 3]) == [
        "scene_b",
        "scene_a",
    ]


def test_select_class_complete_cohort_fails_closed_on_missing_class() -> None:
    with pytest.raises(ValueError, match="cannot cover"):
        select_class_complete_cohort({"scene_a": {1}}, [1, 2])


def test_class_coverage_counts_exposes_loso_unidentifiable_classes() -> None:
    assert class_coverage_counts({"a": {1, 2}, "b": {1}}, [1, 2, 3]) == {
        1: 2,
        2: 1,
        3: 0,
    }
