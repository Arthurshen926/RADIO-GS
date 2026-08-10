from __future__ import annotations

import torch

from radio_gs.interfaces.lerf_marginal_preserving_copula_residual import (
    marginal_preserving_primitive_query_scores,
)


def test_target_cache_policy_preserves_accepted_domain_and_missing_candidate_rows() -> None:
    accepted = torch.tensor(
        [
            [0.1, 0.8],
            [0.2, 0.7],
            [0.3, 0.6],
            [0.4, 0.5],
            [0.5, 0.4],
            [0.6, 0.3],
        ]
    )
    candidate = accepted.flip(0)
    accepted_valid = torch.tensor([True, True, True, True, True, False])
    candidate_valid = torch.tensor([True, True, False, True, True, True])
    reliable = accepted_valid & candidate_valid
    filled = torch.where(reliable[:, None], candidate, accepted)
    result = marginal_preserving_primitive_query_scores(
        accepted,
        filled,
        accepted_valid,
        strength=0.5,
        maximum_rank_fraction=0.4,
        reliability=reliable.float(),
    )
    assert torch.equal(result.scores[accepted_valid & ~candidate_valid], accepted[accepted_valid & ~candidate_valid])
    assert torch.equal(result.scores[~accepted_valid], accepted[~accepted_valid])
    for query in range(accepted.shape[1]):
        assert torch.equal(
            torch.sort(result.scores[accepted_valid, query]).values,
            torch.sort(accepted[accepted_valid, query]).values,
        )

