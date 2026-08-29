import torch

from radio_gs.v3.query.calibrated_posterior import NullCalibratedPosterior
from radio_gs.v3.query.interface import StructuredGaussianQueryInterface
from radio_gs.v3.training.materialize_clean_posterior_evidence import _pool_membership
from radio_gs.v3.training.fit_null_calibrated_posterior import _pool_hit_probability


def _reliability(unknown=(0.0, 0.0, 0.0, 0.0)):
    value = torch.tensor(unknown).float()
    return torch.stack((1 - value, 1 - value, value, 1 - value, 1 - value), dim=1)


def test_identical_siblings_do_not_glue_across_a_boundary():
    posterior = NullCalibratedPosterior()
    result = posterior(
        identity=torch.tensor([0.9, 0.1]),
        instance=torch.tensor([0.9, 0.9]),
        null=torch.zeros(2), negative=torch.zeros(2), unknown=torch.zeros(2),
        boundary=torch.tensor([0.0, 1.0]), reliability=_reliability()[:2],
    )

    assert result[0] > 0.9
    assert result[1] < 0.5


def test_multipart_instance_expands_when_local_identity_and_instance_agree():
    posterior = NullCalibratedPosterior()
    result = posterior(
        identity=torch.tensor([0.9, 0.7, 0.1]),
        instance=torch.tensor([0.9, 0.9, 0.1]),
        null=torch.zeros(3), negative=torch.zeros(3), unknown=torch.zeros(3),
        boundary=torch.tensor([0.0, 0.2, 0.0]), reliability=_reliability()[:3],
    )

    assert bool((result[:2] > 0.9).all())
    assert result[2] < 0.5


def test_d16_boundary_evidence_changes_the_final_posterior_without_hard_gate():
    posterior = NullCalibratedPosterior()
    common = dict(
        identity=torch.tensor([0.2]), instance=torch.tensor([0.9]),
        null=torch.zeros(1), negative=torch.zeros(1), unknown=torch.zeros(1),
        reliability=_reliability()[:1],
    )
    interior = posterior(boundary=torch.zeros(1), **common)
    boundary = posterior(boundary=torch.ones(1), **common)

    assert boundary < interior
    assert 0 < boundary < 1


def test_null_and_unknown_queries_abstain_near_zero():
    posterior = NullCalibratedPosterior()
    result = posterior(
        identity=torch.zeros(2), instance=torch.zeros(2),
        null=torch.tensor([1.0, 0.0]), negative=torch.zeros(2),
        unknown=torch.tensor([0.0, 1.0]), boundary=torch.zeros(2),
        reliability=_reliability((1.0, 1.0)),
    )

    assert bool((result < 0.02).all())


def test_2d_views_render_the_exact_same_3d_posterior_object():
    posterior = torch.tensor([0.9, 0.2, 0.7])
    first = StructuredGaussianQueryInterface.render_posterior(
        posterior, torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1]),
        torch.tensor([0.7, 0.3, 1.0]), num_pixels=2,
    )
    second = StructuredGaussianQueryInterface.render_posterior(
        posterior, torch.tensor([2, 0]), torch.tensor([0, 1]),
        torch.tensor([1.0, 1.0]), num_pixels=2,
    )

    assert first.shape == second.shape == (2,)
    torch.testing.assert_close(posterior, torch.tensor([0.9, 0.2, 0.7]))


def test_vectorized_proposal_pooling_preserves_weighted_evidence():
    value = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    pooled = _pool_membership(
        value,
        row_indices=torch.tensor([0, 1, 1, 2]),
        proposal_indices=torch.tensor([0, 0, 1, 1]),
        weights=torch.tensor([0.25, 0.75, 0.5, 0.5]),
        num_proposals=2,
    )

    torch.testing.assert_close(pooled, torch.tensor([[2.5, 5.0], [4.0, 8.0]]))


def test_calibrator_feature_path_matches_direct_forward():
    posterior = NullCalibratedPosterior()
    evidence = dict(
        identity=torch.tensor([0.8, 0.1]),
        instance=torch.tensor([0.7, 0.6]),
        null=torch.tensor([0.0, 0.5]),
        negative=torch.tensor([0.1, 0.9]),
        unknown=torch.zeros(2),
        boundary=torch.tensor([0.2, 0.8]),
        reliability=_reliability()[:2],
    )
    positive, negative = posterior.evidence_features(**evidence)

    torch.testing.assert_close(
        posterior(**evidence),
        torch.sigmoid(posterior.logit_from_features(positive, negative)),
    )


def test_hit_probability_is_pooled_after_sigmoid_like_deployment():
    result = _pool_hit_probability(
        torch.tensor([0.1, 0.9, 0.4]),
        torch.tensor([0, 0, 1]),
        torch.tensor([0.25, 0.75, 2.0]),
        2,
    )

    torch.testing.assert_close(result, torch.tensor([0.7, 0.4]))
