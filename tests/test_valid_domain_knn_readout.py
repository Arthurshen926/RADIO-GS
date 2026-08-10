from __future__ import annotations

import pytest
import torch

from radio_gs.querying import valid_domain_knn_readout as candidate
from radio_gs.scripts import eval_lerf_direct_3d_selection as frozen


def test_all_valid_candidate_matches_frozen_readout() -> None:
    generator = torch.Generator().manual_seed(11)
    positive = torch.rand(12, 3, 4, generator=generator) * 2.0 - 1.0
    negative = torch.rand(12, 3, 4, generator=generator) * 2.0 - 1.0
    xyz = torch.rand(12, 3, generator=generator)
    valid = torch.ones(12, dtype=torch.bool)

    actual = candidate.valid_domain_multiscale_readout(
        positive, negative, xyz, valid, k=5, chunk_size=3
    )
    probability = frozen.canonical_negative_relevancy_query_scores(
        positive, negative, logit_scale=10.0
    )
    expected = frozen.vala_multiscale_knn_peak_select_scores(
        probability, xyz, k=5, chunk_size=3, valid_mask=valid
    )
    assert torch.equal(actual.selected_scale_indices, expected.selected_scale_indices)
    assert torch.equal(actual.raw_smoothed_peaks, expected.raw_smoothed_peaks)
    assert torch.equal(actual.scores, expected.scores)


def test_invalid_rows_cannot_occupy_candidate_neighbor_slots() -> None:
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]
    )
    valid = torch.tensor([True, False, False, True, True])
    scores = torch.tensor([[1.0], [100.0], [-100.0], [0.0], [0.0]])

    actual = candidate.valid_domain_knn_smoothed_scores(
        scores, xyz, k=2, valid_mask=valid
    )
    filtered = candidate.valid_domain_knn_smoothed_scores(
        scores[valid], xyz[valid], k=2, valid_mask=torch.ones(3, dtype=torch.bool)
    )
    assert torch.equal(actual[valid], filtered)
    assert float(actual[0, 0]) == pytest.approx(0.75)
    assert torch.equal(actual[~valid], torch.zeros(2, 1))

    legacy = frozen.vala_knn_smoothed_scores(
        scores, xyz, k=2, valid_mask=valid
    )
    assert not torch.equal(actual[valid], legacy[valid])


def test_neighbor_audit_reports_lost_valid_slots() -> None:
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [10.0, 0.0, 0.0]]
    )
    audit = candidate.audit_legacy_neighbor_domain(
        xyz, torch.tensor([True, False, True]), k=2
    )
    assert audit.valid_rows == 2
    assert audit.effective_valid_k_min == 1
    assert audit.valid_rows_with_invalid_legacy_neighbor == 2
    assert audit.affected_valid_fraction == pytest.approx(1.0)


@pytest.mark.parametrize("broken", ["scores", "xyz", "range"])
def test_candidate_fails_closed_on_invalid_numerics(broken: str) -> None:
    positive = torch.zeros(2, 3, 1)
    negative = torch.zeros(2, 3, 1)
    xyz = torch.zeros(2, 3)
    if broken == "scores":
        positive[0, 0, 0] = torch.nan
    elif broken == "xyz":
        xyz[0, 0] = torch.nan
    else:
        positive[0, 0, 0] = 2.0
    with pytest.raises(ValueError):
        candidate.valid_domain_multiscale_readout(
            positive, negative, xyz, torch.ones(2, dtype=torch.bool), k=1
        )
