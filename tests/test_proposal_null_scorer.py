import torch

from radio_gs.models.proposal_null_scorer import (
    ProposalNullScorer,
    proposal_null_proper_loss,
    proposal_null_set_proper_loss,
)


def _model() -> ProposalNullScorer:
    return ProposalNullScorer(torch.zeros(3), torch.ones(3), scene_feature_dimension=2)


def test_epoch_zero_prior_is_count_corrected_and_set_equivariant() -> None:
    model = _model()
    features = torch.randn(4, 2, 3)
    valid = torch.tensor([[1, 1], [1, 1], [1, 0], [1, 0]], dtype=torch.bool)
    scene = torch.randn(2, 2)
    result = model(features, valid, scene)
    torch.testing.assert_close(result.joint_probability[0], torch.full((2,), 0.5))
    torch.testing.assert_close(result.joint_probability[1:, 0], torch.full((4,), 0.125))
    torch.testing.assert_close(result.joint_probability[1:3, 1], torch.full((2,), 0.25))
    assert bool((result.joint_probability[3:, 1] == 0).all())

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = model(features[permutation], valid[permutation], scene)
    torch.testing.assert_close(permuted.joint_probability[0], result.joint_probability[0])
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(
        permuted.joint_probability[1:][inverse], result.joint_probability[1:]
    )


def test_proper_loss_ignores_unknown_queries_and_backpropagates() -> None:
    model = _model()
    result = model(
        torch.randn(3, 2, 3),
        torch.ones(3, 2, dtype=torch.bool),
        torch.randn(2, 2),
    )
    target = torch.zeros_like(result.joint_probability)
    target[2, 0] = 1.0
    target[0, 1] = 1.0
    loss = proposal_null_proper_loss(
        result.joint_probability, target, torch.tensor([True, False])
    )
    loss.backward()
    assert model.proposal_head.weight.grad is not None
    assert float(model.proposal_head.weight.grad.abs().sum()) > 0


def test_set_proper_loss_sums_equivalent_proposal_mass() -> None:
    probability = torch.tensor([[0.1], [0.3], [0.4], [0.2]], requires_grad=True)
    acceptable = torch.tensor([[False], [True], [True], [False]])
    loss = proposal_null_set_proper_loss(
        probability, acceptable, torch.tensor([True]), brier_weight=0.5
    )
    expected = -torch.log(torch.tensor(0.7)) + 0.5 * (1.0 - 0.7) ** 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert probability.grad is not None
    assert float(probability.grad[1]) == float(probability.grad[2])
