import torch

from radio_gs.querying.scannet_object_aware_category_voting import (
    object_aware_category_vote,
)


def _call(scores, rows, proposals, weights, **kwargs):
    return object_aware_category_vote(
        scores,
        rows,
        proposals,
        weights,
        num_proposals=2,
        class_ids=(1, 5, 7),
        strength=1.0,
        **kwargs,
    )


def test_zero_strength_is_bitwise_primitive_replay():
    scores = torch.tensor([[0.2, 0.4, 0.1], [0.1, 0.2, 0.5]])
    output, stats = object_aware_category_vote(
        scores,
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
        torch.ones(2),
        num_proposals=1,
        class_ids=(1, 5, 7),
        strength=0.0,
    )
    assert torch.equal(output, scores)
    assert stats["changed_rows"] == 0


def test_stuff_rows_are_immutable_and_thing_vote_is_bounded():
    scores = torch.tensor(
        [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.49, 0.51]]
    )
    output, stats = _call(
        scores,
        torch.tensor([0, 1, 2, 1, 2]),
        torch.tensor([0, 0, 0, 1, 1]),
        torch.ones(5),
    )
    assert output[0].argmax().item() == 0
    assert stats["maximum_mixing"] <= 0.25


def test_class_and_proposal_permutations_commute():
    scores = torch.tensor(
        [[0.1, 0.8, 0.2], [0.2, 0.7, 0.3], [0.1, 0.3, 0.5]]
    )
    rows = torch.tensor([0, 1, 1, 2])
    proposals = torch.tensor([0, 0, 1, 1])
    weights = torch.tensor([1.0, 0.8, 0.7, 1.0])
    output, _ = _call(scores, rows, proposals, weights)
    proposal_output, _ = _call(scores, rows, 1 - proposals, weights)
    torch.testing.assert_close(output, proposal_output)

    order = torch.tensor([2, 0, 1])
    permuted, _ = object_aware_category_vote(
        scores[:, order],
        rows,
        proposals,
        weights,
        num_proposals=2,
        class_ids=tuple((1, 5, 7)[index] for index in order.tolist()),
        strength=1.0,
    )
    torch.testing.assert_close(output[:, order], permuted)
