import torch
from radio_gs.geometry.source_scale_calibration import fit_source_depth_scale


def test_valid_large_scale_is_not_rejected_and_outliers_do_not_dominate():
    p=torch.arange(1,101).float();r=p*1.7;r[:10]*=10
    fit=fit_source_depth_scale(p,r)
    assert fit['accepted'] and abs(fit['scale']-1.7)<1e-6


def test_insufficient_source_tracks_abstain():
    assert not fit_source_depth_scale(torch.ones(10),torch.ones(10))['accepted']


def test_shape_inconsistency_is_not_hidden_by_scale():
    p=torch.ones(100);r=torch.linspace(1,4,100)
    assert not fit_source_depth_scale(p,r)['accepted']
