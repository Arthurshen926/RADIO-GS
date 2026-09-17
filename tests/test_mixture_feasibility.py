import pytest
import torch
from radio_gs.training.mixture_feasibility import mixture_error_floor


def test_shared_element_opposite_labels_force_error():
    result = mixture_error_floor(torch.tensor([1., 0.]), torch.tensor([1., 2.]))
    assert result['minimum_absolute_error_sum'] == 1
    assert result['minimum_mean_absolute_error'] == .25
    assert result['conflicting_elements'] == 1


def test_disjoint_carriers_can_fit_labels_exactly():
    assert mixture_error_floor(torch.tensor([2., 0.]), torch.tensor([0., 3.]))['minimum_mean_absolute_error'] == 0


def test_unknown_only_is_undefined():
    assert mixture_error_floor(torch.zeros(2), torch.zeros(2))['minimum_mean_absolute_error'] is None


def test_reject_invalid_evidence():
    with pytest.raises(ValueError):
        mixture_error_floor(torch.tensor([-1.]), torch.tensor([1.]))


def test_formula_matches_exhaustive_binary_posteriors_for_rendered_pixels():
    import itertools
    matrix = torch.tensor([[.2, .8, 0.], [.7, 0., .3], [0., .4, .6]])
    labels = torch.tensor([1., 0., 1.])
    positive = matrix.T @ labels
    negative = matrix.T @ (1 - labels)
    candidates = torch.tensor(list(itertools.product([0., 1.], repeat=3)))
    brute_force = ((matrix @ candidates.T - labels[:, None]).abs().sum(0)).min()
    result = mixture_error_floor(positive, negative)
    assert result['minimum_absolute_error_sum'] == pytest.approx(float(brute_force))
