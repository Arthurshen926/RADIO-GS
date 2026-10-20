import pytest
import torch

from radio_gs.querying.response_calibration import SourceReadoutHistogram


def test_source_rule_recovers_underconfident_foreground_without_changing_ranks():
    h=SourceReadoutHistogram(10)
    response=torch.tensor([.1,.15,.3,.4],dtype=torch.float64)
    h.add(response,torch.tensor([0.,0.,1.,1.]),torch.ones(4))
    result=h.result()
    assert result['threshold']==.3 and result['soft_iou']==1
    assert result['soft_iou_at_half']==0
    assert response.tolist()==[.1,.15,.3,.4]


def test_quantized_histogram_matches_direct_soft_counts():
    h=SourceReadoutHistogram(8)
    response=torch.tensor([0.,.125,.25,.5,.99,1.],dtype=torch.float64)
    target=torch.tensor([.0,.2,.4,.8,.9,1.],dtype=torch.float64)
    weight=torch.tensor([1.,2.,3.,4.,5.,0.],dtype=torch.float64)
    h.add(response,target,weight);r=h.result()
    predicted=response>=r['threshold']
    direct=(weight*target*predicted).sum()/((weight*target).sum()+(weight*(1-target)*predicted).sum())
    assert r['soft_iou']==pytest.approx(float(direct))
    assert r['observations']==5


def test_unknown_and_empty_positive_evidence_keep_default_readout():
    for target in (torch.ones(3),torch.zeros(3)):
        h=SourceReadoutHistogram();h.add(torch.ones(3)*.1,target,torch.zeros(3))
        assert h.result()['threshold']==.5
    h=SourceReadoutHistogram(8);h.add(torch.tensor([.75]),torch.ones(1),torch.ones(1))
    assert h.result()['threshold']==.5


def test_invalid_evidence_is_rejected():
    h=SourceReadoutHistogram()
    with pytest.raises(ValueError,match='finite'):
        h.add(torch.tensor([float('nan')]),torch.ones(1),torch.ones(1))
    with pytest.raises(ValueError,match='nonnegative'):
        h.add(torch.ones(1),torch.ones(1),-torch.ones(1))
    with pytest.raises(ValueError,match='shapes'):
        h.add(torch.ones(2),torch.ones(1),torch.ones(1))
