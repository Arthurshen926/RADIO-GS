import torch
import pytest

from radio_gs.scripts.build_lerf_multiscale_sam3_exact_mpr_memberships import (
    _frame_id,
    remap_parent_forest,
)


def test_frame_id_requires_canonical_source_identity():
    assert _frame_id("frame_00123") == 123
    with pytest.raises(ValueError):
        _frame_id("rgb_123")


def test_remap_parent_forest_preserves_roots_and_edges():
    parents, edges = remap_parent_forest(
        torch.tensor([-1, 0, 0, 1]),
        torch.tensor([[0, 1], [0, 2], [1, 3]]),
        offset=7,
    )
    assert torch.equal(parents, torch.tensor([-1, 7, 7, 8]))
    assert torch.equal(edges, torch.tensor([[7, 8], [7, 9], [8, 10]]))


def test_remap_parent_forest_rejects_inconsistent_edges():
    with pytest.raises(ValueError, match="differs"):
        remap_parent_forest(
            torch.tensor([-1, 0]), torch.tensor([[1, 0]]), offset=3
        )
