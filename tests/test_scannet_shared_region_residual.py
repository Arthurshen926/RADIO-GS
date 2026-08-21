import torch

from radio_gs.querying.scannet_shared_region_residual import (
    class_count_aware_unknown_alpha,
    shared_proposal_consensus_residual,
    shared_uncertainty_background_rejection,
)


def _inputs():
    scores = torch.tensor([[3.0, 0.0, -1.0], [0.2, 0.1, 0.0], [2.5, 0.0, -1.0]])
    rows = torch.tensor([0, 1, 2])
    proposals = torch.tensor([0, 0, 0])
    weights = torch.ones(3)
    return scores, rows, proposals, weights


def test_zero_alpha_is_bitwise_identity():
    scores, rows, proposals, weights = _inputs()
    output, stats = shared_proposal_consensus_residual(
        scores, rows, proposals, weights, num_proposals=1, alpha=0.0
    )
    assert torch.equal(output, scores)
    assert stats["changed_rows"] == 0


def test_class_and_proposal_permutations_commute():
    scores, rows, proposals, weights = _inputs()
    kwargs = dict(
        num_proposals=2,
        alpha=1.0,
        row_margin_threshold=0.2,
        proposal_margin_scale=0.0,
        minimum_row_mass=0.0,
    )
    proposals = torch.tensor([1, 1, 1])
    output, _ = shared_proposal_consensus_residual(
        scores, rows, proposals, weights, **kwargs
    )
    permutation = torch.tensor([2, 0, 1])
    permuted, _ = shared_proposal_consensus_residual(
        scores[:, permutation], rows.flip(0), proposals.flip(0), weights.flip(0), **kwargs
    )
    assert torch.allclose(permuted, output[:, permutation])


def test_high_margin_rows_are_immutable_decisions():
    scores, rows, proposals, weights = _inputs()
    output, stats = shared_proposal_consensus_residual(
        scores,
        rows,
        proposals,
        weights,
        num_proposals=1,
        alpha=1.0,
        row_margin_threshold=0.2,
        proposal_margin_scale=0.0,
        minimum_row_mass=0.0,
    )
    assert output[0].argmax() == scores[0].argmax()
    assert output[2].argmax() == scores[2].argmax()
    assert stats["eligible_rows"] == 1


def test_background_rejection_is_zero_init_and_class_equivariant():
    scores, *_ = _inputs()
    identity, stats = shared_uncertainty_background_rejection(scores, alpha=0.0)
    assert torch.equal(identity[:, :-1].argmax(dim=-1), scores.argmax(dim=-1))
    assert stats["rejected_rows"] == 0
    permutation = torch.tensor([2, 0, 1])
    original, _ = shared_uncertainty_background_rejection(
        scores,
        alpha=1.0,
        normalized_margin_threshold=0.5,
        normalized_entropy_threshold=0.0,
    )
    permuted, _ = shared_uncertainty_background_rejection(
        scores[:, permutation],
        alpha=1.0,
        normalized_margin_threshold=0.5,
        normalized_entropy_threshold=0.0,
    )
    assert torch.allclose(permuted[:, :-1], original[:, permutation])
    assert torch.equal(permuted[:, -1], original[:, -1])


def test_class_count_policy_replays_coarse_split_without_class_parameters():
    assert class_count_aware_unknown_alpha(19) == 1.0
    assert class_count_aware_unknown_alpha(15) == 1.0
    assert class_count_aware_unknown_alpha(10) == 0.0
