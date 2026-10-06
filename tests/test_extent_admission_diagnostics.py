import pytest
import torch
from radio_gs.training.extent_admission_diagnostics import admission_diagnostics


def test_all_allow_mean_gain_can_hide_severe_harm():
    n=torch.tensor([.9,0.,0.]);e=torch.tensor([0.,.6,.6])
    result=admission_diagnostics(n,e,torch.ones(3,dtype=torch.bool))
    assert e.mean()>n.mean()
    assert result['policy']=='always_extent'
    assert result['harmful_takeovers']==1
    assert result['harmful_takeover_rate']==1.
    assert result['native_regret_sum']==pytest.approx(.9)
    assert result['oracle_regret_mean']==pytest.approx(.3)


def test_ties_unknown_denominators_and_selective_misses():
    result=admission_diagnostics(torch.tensor([.1,.4,.6]),torch.tensor([.8,.2,.6]),torch.tensor([False,False,True]))
    assert result['policy']=='selective'
    assert result['missed_improvements']==1 and result['harmful_takeovers']==0
    assert result['missed_gain_sum']==pytest.approx(.7)
    empty=admission_diagnostics(torch.empty(0),torch.empty(0),torch.empty(0,dtype=torch.bool))
    assert empty['policy']=='empty' and empty['harmful_takeover_rate'] is None


def test_misaligned_and_nonfinite_inputs_fail():
    with pytest.raises(ValueError):admission_diagnostics(torch.zeros(2),torch.zeros(3),torch.zeros(2,dtype=torch.bool))
    with pytest.raises(ValueError):admission_diagnostics(torch.tensor([float('nan')]),torch.zeros(1),torch.zeros(1,dtype=torch.bool))
