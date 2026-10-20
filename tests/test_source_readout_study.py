import pytest
import torch

from radio_gs.querying.response_calibration import SourceReadoutHistogram
from radio_gs.response_study.source_readout_study import verify_histogram


def test_stdlib_verifier_rederives_threshold_from_source_counts():
    h=SourceReadoutHistogram(512)
    g=torch.Generator().manual_seed(83)
    h.add(torch.rand(5000,generator=g),torch.rand(5000,generator=g),torch.rand(5000,generator=g))
    r=h.result()
    assert verify_histogram(r)==r['threshold']
    r['threshold']=.5
    with pytest.raises(ValueError,match='threshold'):verify_histogram(r)


def test_negative_or_nonfinite_histograms_cannot_pass():
    h=SourceReadoutHistogram(8);r=h.result()
    assert verify_histogram(r)==.5
    r['positive_histogram'][0]=float('nan')
    with pytest.raises(ValueError,match='histogram'):verify_histogram(r)
