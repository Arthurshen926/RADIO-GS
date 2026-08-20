import torch

from radio_gs.querying import (
    DIFFERENT_RELATION,
    SAME_RELATION,
    UNKNOWN_RELATION,
    latent_proposal_null_posterior,
    ternary_comembership_authority,
)


def test_latent_proposal_null_posterior_is_exact_convex_marginal():
    primitive = torch.tensor([[0.2], [0.8], [0.4]])
    result = latent_proposal_null_posterior(
        primitive,
        row_indices=torch.tensor([0, 1, 1, 2]),
        proposal_indices=torch.tensor([0, 0, 1, 1]),
        membership_probability=torch.tensor([1.0, 0.5, 0.25, 1.0]),
        proposal_logits=torch.log(torch.tensor([[2.0], [1.0]])),
        null_logits=torch.log(torch.tensor([1.0])),
    )
    assert torch.allclose(result.null_probability, torch.tensor([0.25]))
    assert torch.allclose(result.proposal_probability[:, 0], torch.tensor([0.5, 0.25]))
    expected = torch.tensor([[0.55], [0.5125], [0.35]])
    assert torch.allclose(result.probability, expected)
    assert torch.allclose(
        result.null_probability + result.proposal_probability.sum(dim=0),
        torch.ones(1),
    )


def test_latent_proposal_null_posterior_has_bitwise_primitive_fallback():
    primitive = torch.tensor([[0.1234567, 0.9], [0.7654321, 0.1]])
    result = latent_proposal_null_posterior(
        primitive,
        row_indices=torch.tensor([0, 1]),
        proposal_indices=torch.tensor([0, 0]),
        membership_probability=torch.ones(2),
        proposal_logits=torch.zeros((1, 2)),
        null_logits=torch.zeros(2),
        proposal_valid=torch.tensor([[False, True]]),
    )
    assert torch.equal(result.probability[:, 0], primitive[:, 0])
    assert result.null_probability[0].item() == 1.0
    assert result.proposal_probability[0, 0].item() == 0.0


def test_ternary_comembership_keeps_unobserved_pairs_unknown():
    label = ternary_comembership_authority(
        jointly_visible=torch.tensor([True, True, False, True]),
        same_instance_evidence=torch.tensor([True, False, False, False]),
        different_instance_evidence=torch.tensor([False, True, False, False]),
    )
    assert label.tolist() == [
        SAME_RELATION,
        DIFFERENT_RELATION,
        UNKNOWN_RELATION,
        UNKNOWN_RELATION,
    ]


def test_ternary_comembership_rejects_relation_without_joint_visibility():
    try:
        ternary_comembership_authority(
            jointly_visible=torch.tensor([False]),
            same_instance_evidence=torch.tensor([True]),
            different_instance_evidence=torch.tensor([False]),
        )
    except ValueError as error:
        assert "joint visibility" in str(error)
    else:
        raise AssertionError("missing joint visibility must fail closed")
