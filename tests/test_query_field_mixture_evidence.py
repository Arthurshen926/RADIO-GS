import numpy as np
import pytest
from scipy import sparse
import torch

from radio_gs.query_field.contracts import Support, QueryPacket
from radio_gs.query_field.mixture_paired_evidence import MixturePairedEvidence, enable
from radio_gs.query_field.soft_paired_evidence import SoftPairedEvidence
from radio_gs.query_field.confidence_mixture_evidence import ConfidenceMixtureEvidence, enable as enable_confidence
from radio_gs.query_field.storage import load_field, save_field
from radio_gs.querying.anchor_observation_evidence import joint_anchor_evidence
from test_query_field_paired_evidence import state
from test_query_preserving_field import example


@pytest.fixture(params=[MixturePairedEvidence, ConfidenceMixtureEvidence])
def relation_class(request):
    return request.param


def data(p, n, views):
    result = state(p, n, views)
    result.update({k.replace('positive_', 'association_'): v.clone()
                   for k, v in result.copy().items() if k.startswith('positive_')})
    return result


def test_unknown_does_not_veto_known_support_or_become_a_full_vote(relation_class):
    tensors = data([[.2, .2], [0, 0], [0, 0]], [[0, 0], [0, 0], [1, 1]], [0, 0])
    relation = relation_class(tensors, 3)
    support = Support(torch.tensor([0, 1]), torch.tensor([.25, .75]))
    assert not SoftPairedEvidence(tensors, 3).coefficients(support, 1.).any()
    # Only one quarter of the query has weak known evidence. It remains weak.
    torch.testing.assert_close(relation.coefficients(support, 1.), torch.tensor([.025, .025]))
    assert not relation.coefficients(Support(torch.tensor([2]), torch.ones(1)), 1.).any()
    logits = torch.tensor([.7, -.3, .2])
    assert relation.refine(logits, support, 1.)[1] == logits[1]


def test_lookup_then_mixture_preserves_uncertain_alternatives_and_duplicate_rows(relation_class):
    relation = relation_class(data([[1, 0], [0, 1]], [[0, 1], [1, 0]], [0, 0]), 2)
    for weights in [torch.tensor([.1, .9]), torch.tensor([.5, .5])]:
        support = Support(torch.tensor([0, 1]), weights)
        expected = sum(w * relation.coefficients(Support(torch.tensor([i]), torch.ones(1)), 1.)
                       for i, w in enumerate(weights))
        torch.testing.assert_close(relation.coefficients(support, 1.), expected)
    repeated = Support(torch.tensor([0, 1, 0]), torch.tensor([.2, .6, .2]))
    torch.testing.assert_close(relation.coefficients(repeated, 1.), torch.tensor([.4, .6]))


def test_binary_anchor_transfers_v10_joint_visibility_and_nested_disagreement(relation_class):
    membership = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 0, 0]], np.float32)
    visibility = np.array([[1, 1], [1, 1], [0, 1], [1, 0]], np.float32)
    frames = np.array([0, 0, 1])
    exclusion = visibility[:, frames] - membership
    relation = relation_class(data(membership, exclusion, frames), 4)
    support = Support(torch.tensor([0]), torch.ones(1))
    actual = relation.joint_features(support, 1.)
    expected = joint_anchor_evidence(sparse.csr_matrix(membership), sparse.csr_matrix(visibility),
                                     frames, 0, np.arange(4))
    torch.testing.assert_close(actual, torch.from_numpy(expected))
    torch.testing.assert_close(actual[[3, 1]], relation.joint_features(support, 1., torch.tensor([3, 1])))


def test_nested_duplicate_does_not_multiply_view_confidence(relation_class):
    first = relation_class(data([[.3], [1]], [[0], [0]], [0]), 2)
    twice = relation_class(data([[.3, .3], [1, 1]], [[0, 0], [0, 0]], [0, 0]), 2)
    support = Support(torch.tensor([0]), torch.ones(1))
    for a, b in zip(first.masses(support, 1.), twice.masses(support, 1.)):
        torch.testing.assert_close(a, b)


def test_mixture_export_uses_same_member_for_signed_and_inferred_queries(tmp_path, relation_class):
    field = example()
    tensors = data([[1, 1], [0, 1]] + [[0, 0]] * 10,
                   [[0, 0], [1, 0]] + [[0, 0]] * 10, [0, 0])
    (enable if relation_class is MixturePairedEvidence else enable_confidence)(field, tensors)
    save_field(field, tmp_path / 'field', {})
    loaded, manifest = load_field(tmp_path / 'field')
    for packet in [QueryPacket('point_2d', positive=Support(torch.tensor([0]), torch.ones(1)),
                               negative=Support(torch.tensor([1]), torch.ones(1))),
                   QueryPacket('text', semantic=field.semantic.decode(torch.tensor([0]))[0])]:
        assert torch.equal(field.query(packet).membership_scores, loaded.query(packet).membership_scores)
    assert manifest['external_scene_dependencies'] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_full_scribble_chunking_cpu_cuda_and_component_linearity(relation_class):
    generator = torch.Generator().manual_seed(9)
    p = torch.rand(1100, 7, generator=generator)
    n = (1 - p) * torch.rand(1100, 7, generator=generator)
    tensors = data(p, n, [0, 0, 0, 1, 1, 2, 2])
    support = Support(torch.arange(1100), torch.rand(1100, generator=generator))
    cpu = relation_class(tensors, 1100)
    gpu = relation_class(tensors, 1100).cuda()
    torch.testing.assert_close(cpu.coefficients(support, 1.), gpu.coefficients(support, 1.).cpu(), atol=1e-6, rtol=1e-5)
    rows = torch.tensor([0, 50, 777])
    for a, b in zip(cpu.masses(support, 1., rows), gpu.masses(support, 1., rows)):
        torch.testing.assert_close(a, b.cpu(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(cpu.joint_features(support, 1., rows), gpu.joint_features(support, 1., rows).cpu(),
                               atol=1e-6, rtol=1e-5)


def test_vanishing_regional_evidence_cannot_borrow_another_masks_confidence():
    support = Support(torch.tensor([0]), torch.ones(1))
    for weak in [1e-6, .001, .1]:
        tensors = data([[1., weak]], [[0., 0.]], [0, 0])
        old = MixturePairedEvidence(tensors, 1).coefficients(support, 1.)
        new = ConfidenceMixtureEvidence(tensors, 1).coefficients(support, 1.)
        assert old[1] == .5  # The source-only counterexample that stopped v1.
        assert new[1] <= weak
        torch.testing.assert_close(new.sum(), torch.tensor(1.))
    # An isolated weak observation must also remain weak after normalization.
    alone = ConfidenceMixtureEvidence(data([[.001]], [[0.]], [0]), 1)
    torch.testing.assert_close(alone.coefficients(support, 1.), torch.tensor([.001]))
