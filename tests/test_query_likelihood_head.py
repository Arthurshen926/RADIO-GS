import pytest
import torch

from radio_gs.querying.query_likelihood_head import (
    MonotoneLikelihoodRatioHead,
    MonotoneQueryLikelihoodHead,
    QueryLikelihoodInputs,
    fit_registered_2d_source_reconstruction_head,
    registered_2d_likelihood_inputs,
)
from radio_gs.querying.query_spec import PrimitiveUnaryEvidence


def _inputs(*, positive, negative, coverage=None, reliability=None, prior=None):
    positive_tensor = torch.tensor(positive, dtype=torch.float32)
    rows = positive_tensor.shape[0]
    return QueryLikelihoodInputs(
        positive_affinity=positive_tensor,
        negative_affinity=torch.tensor(negative, dtype=torch.float32),
        prior_probability=torch.full((rows,), 0.5) if prior is None else torch.tensor(prior),
        coverage=torch.ones(rows) if coverage is None else torch.tensor(coverage),
        reliability=torch.ones(rows) if reliability is None else torch.tensor(reliability),
    )


def test_query_likelihood_is_permutation_invariant_and_bounded():
    head = MonotoneQueryLikelihoodHead()
    inputs = _inputs(
        positive=[[0.2, 0.8, 0.4], [0.1, 0.3, 0.2]],
        negative=[[0.3, 0.1], [0.7, 0.2]],
    )
    permuted = _inputs(
        positive=[[0.4, 0.2, 0.8], [0.2, 0.1, 0.3]],
        negative=[[0.1, 0.3], [0.2, 0.7]],
    )
    first = head(inputs, source="test")
    second = head(permuted, source="test")
    torch.testing.assert_close(first.values, second.values)
    assert bool((first.foreground_probability >= 0).all())
    assert bool((first.foreground_probability <= 1).all())


def test_query_likelihood_has_monotone_evidence_directions():
    head = MonotoneQueryLikelihoodHead()
    weak = head(
        _inputs(positive=[[0.2]], negative=[[0.2]]), source="test"
    ).foreground_probability
    stronger_positive = head(
        _inputs(positive=[[0.8]], negative=[[0.2]]), source="test"
    ).foreground_probability
    stronger_negative = head(
        _inputs(positive=[[0.2]], negative=[[0.8]]), source="test"
    ).foreground_probability
    assert stronger_positive.item() > weak.item()
    assert stronger_negative.item() < weak.item()


def test_likelihood_ratio_is_monotone_and_inference_ignores_empirical_prior():
    head = MonotoneLikelihoodRatioHead(affinity_channel_count=1)
    weak_inputs = _inputs(positive=[[0.2]], negative=[[0.2]], prior=[0.01])
    strong_inputs = _inputs(positive=[[0.8]], negative=[[0.2]], prior=[0.99])
    weak = head.log_likelihood_ratio(weak_inputs)
    strong = head.log_likelihood_ratio(strong_inputs)
    assert strong.item() > weak.item()
    # The empirical prior is used only by the analytic source-training
    # posterior, never by the inference likelihood ratio.
    same_evidence_other_prior = _inputs(
        positive=[[0.2]], negative=[[0.2]], prior=[0.99]
    )
    torch.testing.assert_close(
        weak, head.log_likelihood_ratio(same_evidence_other_prior)
    )


def test_likelihood_ratio_analytic_prior_correction_is_exact():
    head = MonotoneLikelihoodRatioHead(affinity_channel_count=1)
    inputs = _inputs(positive=[[0.8], [0.3]], negative=[[0.1], [0.7]])
    prevalence = 0.08
    posterior = head.posterior_probability(
        inputs, foreground_prevalence=prevalence
    )
    recovered = torch.logit(posterior) - torch.logit(torch.tensor(prevalence))
    torch.testing.assert_close(recovered, head.log_likelihood_ratio(inputs))


def test_query_likelihood_preserves_exact_abstention():
    head = MonotoneQueryLikelihoodHead()
    evidence = head(
        _inputs(
            positive=[[1.0], [1.0]],
            negative=[[], []],
            coverage=[0.0, 1.0],
            reliability=[1.0, 0.25],
        ),
        source="world_click",
    )
    torch.testing.assert_close(evidence.confidence, torch.tensor([0.0, 0.25]))
    assert evidence.values[0].item() == pytest.approx(0.0)
    assert evidence.source == "world_click:monotone-query-likelihood-v1"


def test_query_likelihood_rejects_invalid_observations():
    head = MonotoneQueryLikelihoodHead()
    with pytest.raises(ValueError, match="positive_affinity"):
        head(
            _inputs(positive=[[1.1]], negative=[[]]),
            source="invalid",
        )


def test_registered_2d_adapter_separates_prior_coverage_and_reliability():
    observation = PrimitiveUnaryEvidence.from_probability(
        torch.tensor([0.9, 0.1, 0.8]),
        confidence=torch.tensor([0.5, 0.25, 0.0]),
        source="registered_test",
    )
    inputs = registered_2d_likelihood_inputs(
        observation,
        prior_probability=torch.tensor([0.2, 0.7, 0.6]),
        reliability=torch.tensor([0.8, 0.4, 0.3]),
    )
    torch.testing.assert_close(inputs.positive_affinity[:, 0, 0], torch.tensor([0.9, 0.1, 0.0]))
    torch.testing.assert_close(inputs.negative_affinity[:, 0, 0], torch.tensor([0.1, 0.9, 0.0]))
    torch.testing.assert_close(inputs.prior_probability, torch.tensor([0.2, 0.7, 0.6]))
    torch.testing.assert_close(inputs.coverage, torch.tensor([0.5, 0.25, 0.0]))
    torch.testing.assert_close(inputs.reliability, torch.tensor([0.8, 0.4, 0.3]))
    evidence = MonotoneQueryLikelihoodHead()(inputs, source="registered_2d")
    torch.testing.assert_close(evidence.confidence, torch.tensor([0.4, 0.1, 0.0]))
    assert evidence.values[2].item() == pytest.approx(0.0)


def test_registered_2d_source_reconstruction_is_deterministic_and_improves():
    observation = PrimitiveUnaryEvidence.from_probability(
        torch.tensor([0.95, 0.85, 0.15, 0.05, 0.5]),
        confidence=torch.tensor([1.0, 0.8, 0.9, 1.0, 0.0]),
        source="registered_test",
    )
    inputs = registered_2d_likelihood_inputs(
        observation,
        prior_probability=torch.tensor([0.8, 0.6, 0.4, 0.2, 0.9]),
    )
    positive = torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0])
    negative = torch.tensor([0.0, 0.0, 0.5, 1.0, 0.0])
    first, first_diagnostics = fit_registered_2d_source_reconstruction_head(
        inputs,
        positive_reference_mass=positive,
        negative_reference_mass=negative,
    )
    second, second_diagnostics = fit_registered_2d_source_reconstruction_head(
        inputs,
        positive_reference_mass=positive,
        negative_reference_mass=negative,
    )
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["final_balanced_bce"] < first_diagnostics["initial_balanced_bce"]
    evidence = first(inputs, source="registered_2d")
    assert evidence.confidence[4].item() == pytest.approx(0.0)
    assert evidence.values[4].item() == pytest.approx(0.0)
