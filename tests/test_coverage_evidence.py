import torch
import pytest
from radio_gs.querying.coverage_evidence import coverage_evidence


def test_confident_thin_foreground_keeps_fractional_mass():
    x=torch.full((4,4),-20.);x[:,0]=20.
    y,m=coverage_evidence(x,(1,1))
    assert y.item()==pytest.approx(.25,abs=1e-6)
    assert m.item()==pytest.approx(1.,abs=1e-6)


def test_uncertain_pixels_have_no_authority():
    x=torch.zeros(2,2);x[0,0]=20.
    y,m=coverage_evidence(x,(1,1))
    assert y.item()==1. and m.item()==.25


def test_uniform_uncertainty_is_unknown_not_negative():
    y,m=coverage_evidence(torch.zeros(4,4),(2,2))
    assert torch.equal(y,torch.full((4,1),.5)) and not m.any()


def test_invalid_input_and_upsampling_rejected():
    with pytest.raises(ValueError):coverage_evidence(torch.zeros(2,2),(4,4))
    with pytest.raises(ValueError):coverage_evidence(torch.full((2,2),float('nan')),(1,1))
