import torch

from radio_gs.v4.evaluation.unknown_decomposition import _classify_unknown
from radio_gs.v4.evaluation.view_coverage_ladder import _greedy_indices, _uniform_indices


def test_unknown_reason_codes_are_mutually_exclusive():
    # E, A, B, C, D, and one committed control element.
    hard = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    visible = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.bool)
    covered = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.bool)
    associated = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool)
    committed = ~hard
    ground_truth = torch.tensor([0, -1, 0, 0, 0, 0])
    token_observed = torch.tensor([True])
    reason = _classify_unknown(
        hard_unknown=hard,
        visible=visible,
        mask_covered=covered,
        associated=associated,
        committed=committed,
        ground_truth_token=ground_truth,
        token_observed=token_observed,
    )
    assert reason.tolist() == [4, 0, 1, 2, 3, -1]


def test_view_selection_is_deterministic_and_geometry_only():
    visibility = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.bool).numpy()
    assert _greedy_indices(visibility, 2) == [1, 0]
    assert _uniform_indices(5, 3) == [0, 2, 4]
