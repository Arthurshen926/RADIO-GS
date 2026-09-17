import torch
from radio_gs.geometry.independent_view_scale import fit_independent_view_scale


def test_scale_is_not_bounded_to_arbitrary_scene_median_and_is_unit_covariant():
 ids=torch.arange(1000);p=torch.linspace(1.,8.,1000);r=p*.13;valid=torch.ones(1000,dtype=torch.bool)
 a=fit_independent_view_scale(p,r,ids,valid);b=fit_independent_view_scale(p,r*1000,ids,valid)
 assert a['accepted'] and b['accepted'] and abs(a['scale']-.13)<1e-6
 assert abs(b['scale']/a['scale']-1000)<1e-3


def test_audit_points_cannot_change_fit_or_admission_and_bad_gate_rejects():
 ids=torch.arange(1000);p=torch.linspace(1.,8.,1000);r=p*3;valid=torch.ones(1000,dtype=torch.bool)
 before=fit_independent_view_scale(p,r,ids,valid);r[ids%10==5]*=2
 after=fit_independent_view_scale(p,r,ids,valid)
 assert after['scale']==before['scale'] and after['accepted']==before['accepted'] and after['gate']==before['gate']
 assert after['audit']['median']>.49
 r[ids%10==0]*=2
 assert not fit_independent_view_scale(p,r,ids,valid)['accepted']


def test_invalid_predictions_and_insufficient_evidence_never_admitted():
 ids=torch.arange(1000);p=torch.ones(1000);r=p*2;valid=torch.ones(1000,dtype=torch.bool)
 p[:300]=float('nan');valid[300:500]=False;r[300:500]=999
 a=fit_independent_view_scale(p,r,ids,valid);assert a['accepted'] and a['scale']==2
 assert not fit_independent_view_scale(p[:20],r[:20],ids[:20],valid[:20])['accepted']
