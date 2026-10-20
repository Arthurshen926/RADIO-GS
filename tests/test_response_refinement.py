import pytest
import torch

from radio_gs.querying.response_refinement import (
    PartialResponseEvidence, fuse_admitted_evidence, refine_response,
)


def evidence(target, positive, negative, compatible=True, confidence=1.):
    target = torch.tensor(target, dtype=torch.float32).reshape(-1, 1)
    return PartialResponseEvidence(torch.eye(len(target)).to_sparse(), target,
        torch.tensor(positive).reshape(-1, 1), torch.tensor(negative).reshape(-1, 1),
        torch.tensor(compatible), torch.tensor(confidence))


def test_unknown_and_rejected_evidence_keep_prior_exactly():
    prior = torch.tensor([[.2], [.8]])
    unknown = evidence([1, 0], [False, False], [False, False])
    rejected = evidence([0, 1], [False, True], [True, False], compatible=False)
    result, info = refine_response(prior, [unknown, rejected])
    assert torch.equal(prior, result)
    assert info["supported_entries"] == 0
    assert torch.equal(refine_response(prior, [])[0], prior)


def test_partial_positive_never_supplies_complement_negative():
    prior = torch.tensor([[.2], [.9]])
    result, _ = refine_response(prior, [evidence([1, 0], [True, False], [False, False])])
    assert torch.equal(result, torch.tensor([[1.], [.9]]))


def test_positive_disagreement_does_not_lower_prior():
    prior = torch.tensor([[.9]])
    result, _ = refine_response(prior, [evidence([.6], [True], [False])])
    assert torch.equal(result, prior)


def test_explicit_counterevidence_can_correct_high_prior():
    prior = torch.tensor([[.95]])
    result, info = refine_response(prior, [evidence([0], [False], [True])])
    assert result.item() == 0
    assert info["negative_entries"] == 1


def test_conflicting_views_are_fused_without_reusing_current_prediction():
    prior = torch.tensor([[.5]])
    a = evidence([1], [True], [False], confidence=3.)
    b = evidence([0], [False], [True])
    result, _ = refine_response(prior, [a, b])
    assert result.item() == .75
    assert torch.equal(result, refine_response(prior, [b, a])[0])


def test_zero_confidence_is_not_support():
    prior = torch.tensor([[.2]])
    result, info = refine_response(prior, [evidence([1], [True], [False], confidence=0.)])
    assert torch.equal(result, prior)
    assert info["positive_entries"] == 0


def test_ray_contributors_receive_updates_but_unsupported_elements_do_not():
    prior = torch.tensor([[.2], [.4], [.7]])
    item = PartialResponseEvidence(torch.tensor([[.25, .75, 0.]]).to_sparse(),
        torch.ones(1, 1), torch.tensor(True), torch.tensor(False), torch.tensor(True), torch.tensor(1.))
    result, _ = refine_response(prior, [item])
    assert torch.allclose(result[:2], torch.tensor([[.85], [1.]]))
    assert result[2] == prior[2]


def test_legacy_fusion_keeps_admitted_soft_response_and_unsupported_prior():
    prior = torch.tensor([[.1], [.7], [.9]])
    op = torch.tensor([[.2, .8, 0.], [.6, .4, 0.]]).to_sparse()
    target = torch.tensor([[.8], [.3]])
    weight = torch.tensor([[.75], [.5]])
    mass = torch.sparse.mm(op.T, weight)
    expected = torch.where(mass > 1e-8, torch.sparse.mm(op.T, weight * target) / mass.clamp_min(1e-8), prior)
    actual, actual_mass = fuse_admitted_evidence(prior, [(op, target, weight)])
    assert torch.equal(actual, expected) and torch.equal(actual_mass, mass)


@pytest.mark.parametrize("bad", [float("nan"), -1., 2.])
def test_unbounded_prior_rejected(bad):
    with pytest.raises(ValueError, match="bounded response"):
        refine_response(torch.tensor([[bad]]), [])


def test_contradictory_claim_is_rejected():
    with pytest.raises(ValueError, match="both signs"):
        refine_response(torch.tensor([[.5]]), [evidence([1], [True], [True])])


def test_query_columns_share_rule_without_cross_query_normalization():
    prior = torch.tensor([[.2, .8], [.6, .1]])
    item = PartialResponseEvidence(torch.eye(2).to_sparse(), torch.tensor([[1., 0.], [0., 1.]]),
        torch.tensor([[True, False], [False, True]]), torch.tensor([[False, True], [True, False]]),
        torch.tensor([[True, False]]), torch.tensor(1.))
    result, _ = refine_response(prior, [item])
    assert torch.equal(result[:, 0], torch.tensor([1., 0.]))
    assert torch.equal(result[:, 1], prior[:, 1])
