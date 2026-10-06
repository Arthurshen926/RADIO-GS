import pytest
import torch
from radio_gs.querying.component_reference_evidence import retain_reference_supported_components


def test_disconnected_supported_component_kept_without_restoring_unseen():
    keep, report = retain_reference_supported_components(
        torch.tensor([0, 1, 2, 3]), torch.ones(4, dtype=torch.bool),
        torch.tensor([True, False, False, False]),
        torch.tensor([0., 9.5, 0., .001]), torch.tensor([2., 10., 0., .001]),
    )
    assert keep.tolist() == [True, True, False, False]
    assert report['additional_components'] == 1


def test_component_mass_is_aggregated_before_ratio_not_equal_element_voting():
    keep, _ = retain_reference_supported_components(
        torch.tensor([0, 0, 1, 1]), torch.ones(4, dtype=torch.bool),
        torch.zeros(4, dtype=torch.bool),
        torch.tensor([.01, 0., .6, .6]), torch.tensor([.01, 10., .6, .6]),
    )
    assert keep.tolist() == [False, False, True, True]


def test_non_candidate_mass_cannot_admit_candidate_component():
    keep, _ = retain_reference_supported_components(
        torch.tensor([0, 0]), torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool),
        torch.tensor([0., 100.]), torch.tensor([1., 100.]),
    )
    assert not keep.any()


def test_existing_retention_and_inputs_are_preserved():
    labels = torch.tensor([0, 1]); candidate = torch.tensor([True, True])
    retained = torch.tensor([True, False]); mass = torch.tensor([0., 3.])
    result, _ = retain_reference_supported_components(labels, candidate, retained, mass, mass)
    assert result.all() and retained.tolist() == [True, False]
    assert mass.tolist() == [0., 3.]


def test_rejects_impossible_or_nonfinite_mass():
    arguments = (torch.tensor([0]), torch.tensor([True]), torch.tensor([False]))
    with pytest.raises(ValueError, match='exceed'):
        retain_reference_supported_components(*arguments, torch.tensor([2.]), torch.tensor([1.]))
    with pytest.raises(ValueError, match='finite'):
        retain_reference_supported_components(*arguments, torch.tensor([float('nan')]), torch.tensor([1.]))
