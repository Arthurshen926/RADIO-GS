import numpy as np
import pytest
import torch
from radio_gs.querying.category_response_transfer import covered_region_carrier,assert_shared_carrier,replay_menu
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS
from radio_gs.response_study.transfer_diagnosis import menu_capacity,instance_capacity


def test_region_carrier_keeps_nested_membership_and_view_coverage_separate():
    field=dict(shape=[3,4],crow=torch.tensor([0,2,4,4]),columns=torch.tensor([0,1,1,3]),
        weights=torch.tensor([.1,.9,.5,.5]),supported=torch.tensor([True,True,False]),geometry={'sha256':'g'})
    member,visible,frames=covered_region_carrier(field,[0,2,4])
    np.testing.assert_array_equal(member.toarray(),[[1,1,0,0],[0,1,0,1],[0,0,0,0]])
    np.testing.assert_array_equal(visible.toarray(),[[1,0],[1,1],[0,0]])
    np.testing.assert_array_equal(frames,[0,0,1,1])
    changed=dict(field,weights=field['weights']*.9)
    with pytest.raises(ValueError,match='transfer changed'):
        assert_shared_carrier(field,changed)


def test_replay_checks_submenus_even_if_complete_menu_winner_is_unchanged():
    classes=list(SPLITS['19']); original=torch.full((1,19),.01)
    outside=next(c for c in classes if c not in SPLITS['10']); subset=SPLITS['10'][:2]
    original[0,classes.index(outside)]=.9
    original[0,classes.index(subset[0])]=.4
    original[0,classes.index(subset[1])]=.4-1e-7
    reconstructed=original.clone(); reconstructed[0,classes.index(subset[1])]=.4+1e-7
    assert original.argmax()==reconstructed.argmax()
    with pytest.raises(ValueError,match='output labels'):
        replay_menu(original,reconstructed,classes,torch.tensor([True]))
    assert replay_menu(original,original.clone(),classes,torch.tensor([True]))['max_abs']==0


def test_menu_oracle_cannot_stitch_submenu_winners_or_hide_a_regression():
    # The high-mean answer sacrifices the final submenu. A protected whole-menu
    # oracle must retain the incumbent; a per-output oracle would hide the loss.
    values=[[.5,.5,.5],[.9,.9,.1],[.4,.4,.8]]
    unrestricted=menu_capacity(values,2)
    guarded=menu_capacity(values,2,protect_all=True)
    assert unrestricted['oracle_candidate']==1
    assert unrestricted['oracle_metrics']==values[1]
    assert guarded['oracle_candidate']==0
    assert guarded['oracle_metrics']==values[0]
    assert unrestricted['selected_mean_gain']==pytest.approx(
        unrestricted['available_mean_gain']-unrestricted['decision_mean_regret'])


def test_cached_capacity_requires_exact_candidates_and_fresh_score():
    current=dict(query_id='q',candidates=['a','b'],candidate_selection_sha256='sethash',
        arms={k:dict(candidate=v) for k,v in [('full',1),('query_blind',0),('permuted',1)]})
    historical=dict(query_id='q',candidates=['a','b'],candidate_iou=[.4,.6],scene='s',modality='image')
    result=instance_capacity(current,historical,'sethash',.6)
    assert result['decision_regret']==0 and not result['same_choice_as_blind']
    for digest,actual in [('changed',.6),('sethash',.59)]:
        with pytest.raises(ValueError):instance_capacity(current,historical,digest,actual)


def test_menu_decision_uses_one_global_margin_and_rejects_missing_evidence():
    from scripts.research.evaluate_tmm_multitask_transfer import decide_menu
    class Value:
        def predict(self,x):return np.asarray(x)[:,0]
    x=np.zeros((3,39));x[:,0]=[.1,.3,.5]
    report=dict(features=x.tolist(),permuted_features=x[[0,2,1]].tolist(),evidence={'verification_views':3})
    bundle=dict(models={'full':Value(),'query_blind':Value()},margins={'full':.3,'query_blind':None})
    result=decide_menu(report,bundle)
    assert result['full']['candidate']==2 and result['permuted']['candidate']==1
    assert result['query_blind']['candidate']==0
    report['evidence']['verification_views']=0
    assert all(v['candidate']==0 for v in decide_menu(report,bundle).values())


def test_no_null_control_uses_its_own_sealed_model_and_margin():
    from scripts.research.evaluate_tmm_multitask_transfer import decide_menu
    class Value:
        def __init__(self, column): self.column=column
        def predict(self,x):return np.asarray(x)[:,self.column]
    x=np.zeros((3,39));x[:,0]=[.1,.3,.5];x[:,1]=[.1,.7,.2]
    report=dict(features=x.tolist(),permuted_features=x.tolist(),evidence={'verification_views':3})
    bundle=dict(models={'full':Value(0),'query_blind':Value(0),'unregularized':Value(1)},
                margins={'full':.3,'query_blind':None,'unregularized':.5})
    result=decide_menu(report,bundle,arms=('full','unregularized'))
    assert result['full']['candidate']==2 and result['unregularized']['candidate']==1
    bundle['margins']['unregularized']=.7
    result=decide_menu(report,bundle,arms=('full','unregularized'))
    assert result['full']['candidate']==2 and result['unregularized']['candidate']==0
    assert 'unregularized' not in decide_menu(report,bundle)
