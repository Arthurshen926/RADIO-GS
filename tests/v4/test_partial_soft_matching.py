import torch

from radio_gs.v4.object_memory import ObservedObjectEvidence, partial_soft_match


def test_mask_exterior_stays_unknown_until_explicit_negative():
    evidence = ObservedObjectEvidence.from_positive_visibility(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.ones(1, 3),
        view_ids=torch.tensor([0]),
        quality=torch.tensor([0.9]),
    )
    assert evidence.negative.sum() == 0
    assert evidence.unknown.tolist() == [[0.0, 1.0, 1.0]]
    revised = evidence.with_explicit_negative(torch.tensor([[0.0, 1.0, 0.0]]))
    assert revised.negative.tolist() == [[0.0, 1.0, 0.0]]


def test_partial_matching_allows_multiple_parts_to_one_token_and_null():
    evidence = ObservedObjectEvidence.from_positive_visibility(
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        view_ids=torch.tensor([0, 0, 0]),
        quality=torch.ones(3),
    )
    membership = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    result = partial_soft_match(
        evidence,
        membership,
        element_visibility=torch.tensor([[1.0, 1.0, 0.0, 0.0]]).expand(3, -1),
        mask_centres=torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        token_centres=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        token_scales=torch.ones(2, 3),
        null_logit=0.5,
    )
    assert result.token_probability[:2].argmax(-1).tolist() == [0, 0]
    assert bool((result.token_probability[:2, 0] > result.null_probability[:2]).all())
    assert result.null_probability[2] > result.token_probability[2].max()
    assert torch.allclose(
        result.token_probability.sum(-1) + result.null_probability,
        torch.ones(3),
    )
