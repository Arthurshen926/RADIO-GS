from __future__ import annotations

import torch

from radio_gs.querying.latent_proposal_posterior import DIFFERENT_RELATION, SAME_RELATION, UNKNOWN_RELATION
from radio_gs.scripts.build_lerf_proposal_identity_reliability_authority import ternary_cross_view_edges


def test_ternary_edges_require_overlap_or_bidirectional_visibility() -> None:
    supports = [{1, 2}, {2, 3}, {4}, {5}]
    views = torch.tensor([0, 1, 1, 0])
    visibility = torch.zeros(4, 2)
    visibility[2, 0] = 1; visibility[3, 1] = 1
    left, right, relation, _ = ternary_cross_view_edges(supports, views, visibility)
    result = {(int(a), int(b)): int(r) for a, b, r in zip(left, right, relation)}
    assert result[(0, 1)] == SAME_RELATION
    assert result[(2, 3)] == DIFFERENT_RELATION
    assert result[(0, 2)] == UNKNOWN_RELATION
    assert (1, 2) not in result and (0, 3) not in result
