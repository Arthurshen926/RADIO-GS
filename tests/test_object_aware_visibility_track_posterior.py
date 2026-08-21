import torch

from radio_gs.scripts.build_lerf_object_aware_visibility_track_posterior import (
    _calibrated_association_logit,
)
from radio_gs.querying.object_aware_visibility_track_posterior import (
    object_aware_visibility_track_posterior,
)


def _run(observed: torch.Tensor, duplicate_candidates: bool = False):
    proposal_views = torch.tensor([0, 1, 1] if duplicate_candidates else [0, 1])
    proposal_areas = torch.full((len(proposal_views),), 0.1)
    # Seed covers both rows; view-1 candidate covers only row 0.
    rows = torch.tensor([0, 1, 0] + ([0] if duplicate_candidates else []))
    props = torch.tensor([0, 0, 1] + ([2] if duplicate_candidates else []))
    edge_right = torch.tensor([1, 2] if duplicate_candidates else [1])
    return object_aware_visibility_track_posterior(
        torch.tensor([[0.1], [0.1]]),
        torch.tensor([[0.9]] + [[0.1]] * (len(proposal_views) - 1)),
        torch.tensor([[True]] + [[False]] * (len(proposal_views) - 1)),
        torch.full((len(proposal_views), 1), 0.8),
        rows, props, torch.ones(len(rows)), proposal_views, proposal_areas,
        torch.zeros(len(edge_right), dtype=torch.long), edge_right,
        torch.full((len(edge_right),), 0.5),
        torch.full((len(edge_right),), -1, dtype=torch.int8),
        observed.float(), observed,
    )


def test_absence_without_visibility_is_unknown() -> None:
    invisible = _run(torch.tensor([[True, True], [True, False]]))
    visible = _run(torch.tensor([[True, True], [True, True]]))
    # Row 1 is absent from view 1.  Only the visible case adds denominator.
    assert invisible.visibility_denominator[1, 0] < visible.visibility_denominator[1, 0]
    assert invisible.probability[1, 0] > visible.probability[1, 0]


def test_count_correction_stabilizes_nonnull_mass() -> None:
    observed = torch.ones(2, 2, dtype=torch.bool)
    single = _run(observed, duplicate_candidates=False)
    duplicate = _run(observed, duplicate_candidates=True)
    torch.testing.assert_close(
        1.0 - single.null_probability[1, 0],
        1.0 - duplicate.null_probability[1, 0],
    )


def test_no_identity_seed_is_bitwise_v1_fallback() -> None:
    base = torch.tensor([[0.2], [0.7]])
    result = object_aware_visibility_track_posterior(
        base, torch.zeros(2, 1), torch.zeros(2, 1, dtype=torch.bool),
        torch.zeros(2, 1), torch.tensor([0, 1]), torch.tensor([0, 1]),
        torch.ones(2), torch.tensor([0, 1]), torch.tensor([0.1, 0.1]),
        torch.tensor([0]), torch.tensor([1]), torch.tensor([0.0]),
        torch.tensor([-1], dtype=torch.int8), torch.ones(2, 2),
        torch.ones(2, 2, dtype=torch.bool),
    )
    assert torch.equal(result.probability, base)
    assert bool(result.fallback.all())


def test_known_different_has_negligible_association() -> None:
    result = object_aware_visibility_track_posterior(
        torch.full((1, 1), 0.1), torch.tensor([[0.9], [0.1]]),
        torch.tensor([[True], [False]]), torch.full((2, 1), 0.8),
        torch.tensor([0, 0]), torch.tensor([0, 1]), torch.ones(2),
        torch.tensor([0, 1]), torch.tensor([0.1, 0.1]), torch.tensor([0]),
        torch.tensor([1]), torch.tensor([1.0]), torch.tensor([0], dtype=torch.int8),
        torch.ones(2, 1), torch.ones(2, 1, dtype=torch.bool),
    )
    assert result.association_probability[1, 0] < 0.001


def test_exact_membership_probability_is_not_proposal_max_normalized() -> None:
    result = object_aware_visibility_track_posterior(
        torch.zeros(1, 1), torch.tensor([[0.9]]), torch.tensor([[True]]),
        torch.tensor([[0.8]]), torch.tensor([0]), torch.tensor([0]),
        torch.tensor([0.5]), torch.tensor([0]), torch.tensor([0.1]),
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
        torch.empty(0), torch.empty(0, dtype=torch.int8), torch.ones(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
    )
    ratio = result.positive_evidence[0, 0] / result.visibility_denominator[0, 0]
    torch.testing.assert_close(ratio, torch.tensor(0.5))


def test_nested_calibration_applies_temperature_before_affinity_logit() -> None:
    calibrator = {
        "schema": "radio_gs.lerf_source_physical_track_calibrator_nested.v1",
        "feature_mean": torch.zeros(1), "feature_scale": torch.ones(1),
        "weight": torch.ones(1), "bias": torch.tensor(0.0),
        "temperature": 4.0, "jeffreys_strength": 0.0,
    }
    assert torch.allclose(
        _calibrated_association_logit(torch.tensor([[4.0]]), calibrator),
        torch.tensor([1.0]),
    )
