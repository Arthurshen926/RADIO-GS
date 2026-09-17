import pytest
import torch
from radio_gs.geometry.coverage_selection import select_coverage_views


def test_selection_excludes_heldout_and_breaks_ties_by_frame():
    views = {1: torch.tensor([1,0,0,0]), 2: torch.tensor([0,1,0,0]),
             3: torch.tensor([0,0,1,0]), 4: torch.ones(4)}
    result = select_coverage_views(views, [1], [4], 3)
    assert [x['frame'] for x in result['selection']] == [2,3]
    assert result['final_covered_elements'] == 3
    assert views[1].tolist() == [1,0,0,0]


def test_initial_heldout_is_rejected():
    with pytest.raises(ValueError):
        select_coverage_views({1: torch.ones(2)}, [1], [1], 1)
