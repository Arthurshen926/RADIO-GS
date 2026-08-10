from __future__ import annotations

import pytest
import torch

from radio_gs.querying import legacy_anchored_coverage_supplement as candidate
from radio_gs.querying import valid_domain_knn_readout as legacy
from radio_gs.scripts import (
    materialize_lerf_legacy_anchored_coverage_supplement as materializer,
)


def _fixture() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    count, queries = 9, 3
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [1.3, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    geometry_valid = torch.tensor(
        [True, True, True, True, True, True, True, True, False]
    )
    legacy_positive = torch.rand(count, 3, queries, generator=generator) * 0.4 - 0.2
    legacy_negative = torch.rand(count, 3, 2, generator=generator) * 0.4 - 0.2
    accepted = legacy.valid_domain_multiscale_readout(
        legacy_positive,
        legacy_negative,
        xyz,
        geometry_valid,
        k=3,
        chunk_size=2,
    ).scores
    all_positive = legacy_positive.clone()
    all_negative = legacy_negative.clone()
    # Rows 2 and 6 gain source-teacher coverage.
    all_positive[2] += 0.15
    all_positive[6] -= 0.12
    all_negative[2] -= 0.05
    all_negative[6] += 0.04
    return {
        "accepted": accepted,
        "legacy_positive": legacy_positive,
        "legacy_negative": legacy_negative,
        "all_positive": all_positive,
        "all_negative": all_negative,
        "xyz": xyz,
        "geometry_valid": geometry_valid,
        "global_rows": torch.where(geometry_valid)[0],
        "legacy_teacher_valid": torch.tensor(
            [True, True, False, True, True, True, False, True]
        ),
        "all_teacher_valid": torch.ones(8, dtype=torch.bool),
    }


def _run(data: dict[str, torch.Tensor]) -> candidate.LegacyAnchoredCoverageResult:
    return candidate.legacy_anchored_coverage_supplement(
        data["accepted"],
        data["legacy_positive"],
        data["legacy_negative"],
        data["all_positive"],
        data["all_negative"],
        data["xyz"],
        data["geometry_valid"],
        data["global_rows"],
        data["legacy_teacher_valid"],
        data["all_teacher_valid"],
        k=3,
        chunk_size=2,
    )


def test_only_newly_covered_rows_can_change_bitwise() -> None:
    data = _fixture()
    result = _run(data)
    expected_mask = torch.zeros(9, dtype=torch.bool)
    expected_mask[[2, 6]] = True
    assert torch.equal(result.supplement_mask, expected_mask)
    assert result.supplement_rows.tolist() == [2, 6]
    assert torch.equal(
        result.scores[~expected_mask], data["accepted"][~expected_mask]
    )
    assert result.neighbor_rows.shape == (2, 3)
    assert not bool(result.supplement_mask[8])


def test_all_available_non_supplement_values_have_no_reverse_path() -> None:
    data = _fixture()
    reference = _run(data)
    perturbed = dict(data)
    perturbed["all_positive"] = data["all_positive"].clone()
    perturbed["all_negative"] = data["all_negative"].clone()
    non_supplement = torch.tensor([0, 1, 3, 4, 5, 7, 8])
    perturbed["all_positive"][non_supplement] = 0.95
    perturbed["all_negative"][non_supplement] = -0.95
    actual = _run(perturbed)
    assert torch.equal(actual.scores, reference.scores)
    assert torch.equal(
        actual.selected_scale_indices, reference.selected_scale_indices
    )
    assert torch.equal(actual.legacy_low_by_scale, reference.legacy_low_by_scale)
    assert torch.equal(actual.legacy_high_by_scale, reference.legacy_high_by_scale)


def test_no_new_coverage_is_exact_noop() -> None:
    data = _fixture()
    data["all_teacher_valid"] = data["legacy_teacher_valid"].clone()
    result = _run(data)
    assert result.supplement_rows.numel() == 0
    assert torch.equal(result.scores, data["accepted"])


def test_fails_closed_when_accepted_cache_cannot_be_reconstructed() -> None:
    data = _fixture()
    data["accepted"] = data["accepted"].clone()
    data["accepted"][0, 0] = min(1.0, float(data["accepted"][0, 0]) + 0.01)
    with pytest.raises(ValueError, match="cannot be reconstructed bitwise"):
        _run(data)


def test_fails_closed_when_all_available_loses_legacy_coverage() -> None:
    data = _fixture()
    data["all_teacher_valid"] = data["all_teacher_valid"].clone()
    data["all_teacher_valid"][0] = False
    with pytest.raises(ValueError, match="must include legacy validity"):
        _run(data)


def test_source_diagnostics_are_target_blind_and_report_one_way_invariance() -> None:
    data = _fixture()
    result = _run(data)
    generator = torch.Generator().manual_seed(9)
    legacy_teacher = torch.rand(8, 6, generator=generator)
    all_teacher = legacy_teacher.clone()
    all_teacher[[2, 6]] = torch.rand(2, 6, generator=generator)
    report = materializer.source_diagnostics(
        result,
        data["accepted"],
        data["global_rows"],
        legacy_teacher,
        all_teacher,
        torch.tensor([2, 2, 1, 2, 3, 1, 1, 2], dtype=torch.uint8),
    )
    assert report["supplement_rows"] == 2
    assert report["changed_cells_outside_supplement"] == 0
    assert report["legacy_rows_bitwise_unchanged"] is True
    assert report["frozen_threshold"] == 0.6
    assert 0.0 <= report["spatial_attachment"]["largest_component_fraction"] <= 1.0
    assert -1.0 <= report["source_teacher_neighbor_consistency"]["mean_cosine"] <= 1.0


def test_contract_has_no_tuned_parameters_or_metric_access() -> None:
    assert candidate.CONTRACT.endswith(".v1")
    assert materializer.KNN_K == 10
    assert materializer.LOGIT_SCALE == 10.0
    assert materializer.FROZEN_THRESHOLD == 0.6
    audit = materializer.access_audit()
    assert audit["benchmark_labels_opened"] is False
    assert audit["target_metrics_computed"] is False
    assert audit["result_dependent_parameters"] is False
