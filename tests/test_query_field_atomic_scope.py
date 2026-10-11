from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from test_query_field_scope_topology import make
from radio_gs.query_field.atomic_scope import atomic_response_interval, assemble_atomic_scope
from radio_gs.query_field.extent_preparation import ExtentPreparationView, ExtentPreparation


def identity_view(count):
    return ExtentPreparationView(dict(index=torch.arange(count).repeat(2, 1), weight=torch.ones(count),
                                     valid=torch.ones(count, dtype=torch.bool), footprint=torch.ones(count)), count)


def test_parent_child_disagreement_does_not_authorize_whole_positive_or_negative():
    masks = torch.zeros(2, 5, 9, dtype=torch.bool)
    masks[0, :, :6] = True
    masks[1, :, 3:6] = True
    view = make(masks); prep = identity_view(45)
    with patch('builtins.open', side_effect=AssertionError('IO')), patch('torch.load', side_effect=AssertionError('IO')):
        scope, report = assemble_atomic_scope(view, prep, torch.zeros(45), torch.tensor([.7, .3]), torch.tensor(.4))
    assert scope['positive'].reshape(5, 9)[:, :2].all()
    assert not scope['positive'].reshape(5, 9)[:, 3:].any()
    assert not scope['negative'].any()  # conflict and no coverage are both unknown
    assert report['conflicting_valid_rays'] == 15 and report['uncovered_valid_rays'] == 15


def test_random_atomic_signs_match_independent_dense_raster():
    generator = torch.Generator().manual_seed(20261012)
    for _ in range(12):
        masks = torch.rand(5, 7, 11, generator=generator) > .7
        prob = torch.rand(5, generator=generator); view = make(masks)
        scope, _ = assemble_atomic_scope(view, identity_view(77), torch.zeros(77), prob, torch.tensor(.4))
        dense_p = torch.zeros(7, 11, dtype=torch.bool); dense_n = dense_p.clone()
        for y in range(7):
            for x in range(11):
                present = prob[masks[:, y, x]]
                dense_p[y, x] = bool(len(present) and (present > .5).all())
                dense_n[y, x] = bool(len(present) and (present < .5).all())
        erode = lambda v: (-F.max_pool2d(-v.float()[None, None], 3, 1, 1) > .5).flatten()
        assert torch.equal(scope['target'], dense_p.flatten().float())
        assert torch.equal(scope['positive'], erode(dense_p))
        assert torch.equal(scope['negative'], erode(dense_n))


def test_duplicate_proposals_do_not_strengthen_evidence_and_ties_abstain():
    masks = torch.tensor([[[True, True, False]], [[False, True, True]]])
    v = make(masks); repeated = make(torch.cat([masks, masks[:1]]))
    a = assemble_atomic_scope(v, identity_view(3), torch.zeros(3), torch.tensor([.8, .5]), torch.tensor(.6))[0]
    b = assemble_atomic_scope(repeated, identity_view(3), torch.zeros(3), torch.tensor([.8, .5, .8]), torch.tensor(.6))[0]
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not atomic_response_interval(v, torch.full((2,), .5))[0].any()
    assert not atomic_response_interval(v, torch.full((2,), .5))[1].any()


def test_unknown_target_does_not_change_initialization_cost_or_member_decision():
    from radio_gs.query_field.extent_decision_linked import LinkedExtentDecision
    from radio_gs.query_field.extent_factors import ExtentFactors
    pv = identity_view(3)
    prep = ExtentPreparation({'views.0.'+k: x for k,x in pv.state_dict().items()}, 3)
    factors = ExtentFactors({'views.0.'+k: x for k,x in dict(pointer=torch.arange(4), index=torch.arange(3),
        alpha=torch.full((3,), .8), visibility=torch.ones(3), atom=torch.zeros(3,dtype=torch.long)).items()}, 3, torch.full((3,1), .8))
    engine = LinkedExtentDecision(factors, 3)
    s = dict(target=torch.tensor([1., 0., 0.]), positive=torch.tensor([True, False, False]),
             negative=torch.tensor([False, True, False]), confidence=torch.tensor(.7))
    def run():
        initial,cost,_ = prep(torch.tensor([.6,.6,.6]),[s])
        selected, report = engine.solve((initial>.5).numpy(),s['target'].numpy(),s['positive'].numpy(),s['negative'].numpy(),
                                        np.array([.7]),cost.numpy())
        return initial,cost,selected,report['energy_trace']
    first=run();s['target'][2]=1;second=run()
    assert torch.equal(first[0],second[0]) and torch.equal(first[1],second[1])
    assert np.array_equal(first[2],second[2]) and first[3]==second[3]


def test_negative_permission_preserves_native_protection_and_rejects_invalid_values():
    v=make(torch.ones(1, 3, 3,dtype=torch.bool));pv=identity_view(9)
    prior=torch.zeros(9);prior[4]=1;pv.valid[0]=False
    s,_=assemble_atomic_scope(v,pv,prior,torch.tensor([.2]),torch.tensor(.2))
    assert int(s['negative'].sum())==7 and not s['negative'][4] and not s['negative'][0]
    for p in [torch.tensor([float('nan')]),torch.tensor([-1.]),torch.tensor([.2,.3])]:
        with pytest.raises(ValueError):atomic_response_interval(v,p)
