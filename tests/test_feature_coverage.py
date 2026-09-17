import pytest
import torch
from radio_gs.training.feature_coverage import supervised_feature_rows


def test_visible_but_untouched_features_are_not_semantic_observations():
    gradient = torch.tensor([[0.,0.],[0.,2.],[0.,0.]])
    visible = torch.tensor([True,True,False])
    assert supervised_feature_rows(gradient,visible).tolist() == [False,True,False]


def test_nonfinite_and_outside_visibility_gradients_are_rejected():
    with pytest.raises(ValueError,match='nonfinite'):
        supervised_feature_rows(torch.tensor([[float('nan')]]),torch.tensor([True]))
    with pytest.raises(ValueError,match='unobserved'):
        supervised_feature_rows(torch.ones(1,2),torch.tensor([False]))
    with pytest.raises(ValueError,match='axes'):
        supervised_feature_rows(torch.ones(1,2),torch.ones(2,dtype=torch.bool))


def test_pixel_sampler_covers_each_view_without_duplicates_before_repeat():
    from radio_gs.training.feature_coverage import CoveragePixelSampler
    state = torch.random.get_rng_state()
    sampler = CoveragePixelSampler([torch.arange(11)*2, torch.tensor([99])])
    samples = torch.cat([sampler.sample(0,4) for _ in range(3)])
    assert sorted(samples.tolist()) == list(range(0,22,2))
    assert sampler.covered[0].all() and not sampler.covered[1].any()
    assert sampler.sample(1,4).tolist() == [99]
    assert len(sampler.sample(0,4)) == 4
    assert torch.equal(state,torch.random.get_rng_state())
