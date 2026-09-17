import pytest
import torch
from radio_gs.training.frozen_score_calibration import fit_histogram_readout


def test_source_fit_recovers_separable_high_cosine_classes():
    p=torch.zeros(1024);n=torch.zeros(1024)
    p[970:990]=10;n[830:900]=100
    result=fit_histogram_readout(p,n)
    assert result['positive_recall']==1
    assert result['negative_rejection']==1
    assert .76<result['cosine_threshold']<.90
    assert result['scale']>0


def test_balancing_does_not_invent_separation_for_identical_scores():
    p=torch.zeros(256);p[220]=1;n=100*p
    count=fit_histogram_readout(p,n);balanced=fit_histogram_readout(p,n,balanced=True)
    score=-1+(220.5)*2/256
    assert torch.sigmoid(torch.tensor(count['scale']*score+count['bias'])).item()==pytest.approx(1/101,abs=1e-5)
    assert torch.sigmoid(torch.tensor(balanced['scale']*score+balanced['bias'])).item()==pytest.approx(.5,abs=1e-5)


def test_empty_or_invalid_class_is_rejected():
    for negative in (torch.zeros(4),torch.tensor([1.,-1,2,1]),torch.full((4,),float('nan'))):
        with pytest.raises(ValueError):fit_histogram_readout(torch.ones(4),negative)
